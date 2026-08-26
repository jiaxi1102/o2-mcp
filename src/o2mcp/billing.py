"""Price a Slurm submission before it is submitted.

Fair share is bought with *allocated* resources, not used ones, and the
weighted sum is floored — so memory is effectively sold in whole blocks and a
request sitting exactly on a block edge pays for a full block while forfeiting
the headroom inside it. Nothing in a job's own output reveals this, which is
why it is worth computing at the moment the request is written.

Everything here is pure arithmetic over a weight table. No SSH, no subprocess,
no Slurm: the only cluster-derived input is ``TRESBillingWeights``, which
changes rarely and is cached. That keeps pricing available while the O2 policy
is disabled and before any broker exists — precisely when a submission is being
composed.

Scope is deliberately narrow. This module answers "what will this cost and
where is the next boundary"; it does not estimate savings across a workload,
infer efficiency, or recommend reducing memory below what a job already holds.
Those are inferences over data that does not support them, and they belong to
the retrospective audit, not to a pre-submission check.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "o2mcp", "billing_weights.json")

# Env var so a test, or a second checkout, can point at its own table.
CACHE_PATH_ENV = "O2_BILLING_WEIGHTS_CACHE"


def cache_path(path: str | None = None) -> str:
    """Resolve the weight-cache location at call time, not at import."""
    if path:
        return path
    return os.environ.get(CACHE_PATH_ENV) or _DEFAULT_CACHE_PATH


# Memory suffixes expressed in GB. There is no "no unit" entry on purpose:
# sbatch documents an unsuffixed --mem as MEGABYTES, so `--mem=32000` is 31.25
# GB, not 32000 bytes. Reading it as bytes drops essentially the whole memory
# charge and every boundary derived from it.
_MEM_UNITS = {"K": 1.0 / (1024**2), "M": 1.0 / 1024, "G": 1.0, "T": 1024.0}
SBATCH_DEFAULT_MEM_UNIT = "M"


class BillingError(ValueError):
    """A pricing input that cannot be answered honestly."""


@dataclass(frozen=True)
class Weights:
    """One partition's TRESBillingWeights, plus the memory it defaults to.

    The defaults matter because omitting --mem does not allocate zero memory:
    Slurm applies DefMemPerCPU or DefMemPerNode and bills that allocation. They
    come from the same ``scontrol show partition`` output as the weights, so
    they are captured together or not at all.
    """

    cpu: float = 1.0
    mem_per_gb: float = 0.0
    gpu: float = 0.0
    def_mem_per_cpu_gb: float | None = None
    def_mem_per_node_gb: float | None = None

    def default_mem_gb(self, cpus: float, nodes: float = 1.0) -> float | None:
        """Memory Slurm would allocate for a request that names none."""
        if self.def_mem_per_cpu_gb is not None:
            return self.def_mem_per_cpu_gb * cpus
        if self.def_mem_per_node_gb is not None:
            return self.def_mem_per_node_gb * max(1.0, nodes)
        return None

    @property
    def block_gb(self) -> float | None:
        """GB of memory that costs one whole billing unit, if memory is billed."""
        if self.mem_per_gb <= 0:
            return None
        return 1.0 / self.mem_per_gb


@dataclass
class Request:
    """The resource shape of a submission, however it was expressed."""

    cpus: float = 1.0
    mem_gb: float = 0.0
    gpus: float = 0.0
    nodes: float = 1.0
    # True when --nodes gave a range: the allocation is not determined, so a
    # price would be a floor presented as a figure.
    nodes_is_range: bool = False
    # --exclusive bills every TRES on the allocated nodes, not what was asked
    # for. Pricing it needs per-node CPU/GPU topology the weight cache has no
    # way to hold.
    exclusive: bool = False
    # False when no --mem/--mem-per-cpu was given, so the partition default
    # applies and a price of "zero memory" would be a fiction.
    mem_specified: bool = True
    partition: str | None = None
    # Recorded because --mem-per-cpu multiplies out to a round total and so
    # lands on a block edge far more often than an absolute --mem does.
    mem_source: str = "default"
    warnings: list[str] = field(default_factory=list)


def to_gb(value: str | float | None, default_unit: str = SBATCH_DEFAULT_MEM_UNIT) -> float:
    """'32G' / '8192M' / '32000' -> gigabytes.

    An unsuffixed value takes ``default_unit``, which is megabytes because that
    is what sbatch documents for --mem and --mem-per-cpu. Callers reading a
    source with different conventions must say so explicitly.
    """
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) * _MEM_UNITS[default_unit.upper()]
    m = re.match(r"^\s*([0-9.]+)\s*([KMGTkmgt]?)", str(value))
    if not m:
        raise BillingError(f"could not read a memory size from {value!r}")
    suffix = m.group(2).upper() or default_unit.upper()
    return float(m.group(1)) * _MEM_UNITS[suffix]


def parse_weight_table(scontrol_output: str) -> dict[str, Weights]:
    """Read ``scontrol show partition -o`` into per-partition weights.

    A partition without TRESBillingWeights bills CPU only, which is Slurm's
    default and not an error.
    """
    table: dict[str, Weights] = {}
    for line in (scontrol_output or "").splitlines():
        name = None
        raw = None
        for token in line.split():
            if token.startswith("PartitionName="):
                name = token[len("PartitionName=") :]
            elif token.startswith("TRESBillingWeights="):
                raw = token[len("TRESBillingWeights=") :]
        if not name:
            continue
        cpu, mem, gpu = 1.0, 0.0, 0.0
        def_cpu = def_node = None
        for token in line.split():
            # scontrol prints these as megabytes, and 0/UNLIMITED means "not set".
            if token.startswith("DefMemPerCPU="):
                value = token[len("DefMemPerCPU=") :]
                if value.isdigit() and int(value) > 0:
                    def_cpu = int(value) / 1024.0
            elif token.startswith("DefMemPerNode="):
                value = token[len("DefMemPerNode=") :]
                if value.isdigit() and int(value) > 0:
                    def_node = int(value) / 1024.0
        if raw and raw not in ("(null)", ""):
            for part in raw.split(","):
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                key = key.strip().lower()
                number = re.sub(r"[A-Za-z]", "", value.strip())
                if not number:
                    continue
                if key == "cpu":
                    cpu = float(number)
                elif key == "mem":
                    # The weight is per UNIT of memory, and the unit is part of
                    # the value: Mem=0.0625G is 0.0625 per GB, but Mem=1M is one
                    # per MB -- 1024 per GB. Dropping the suffix underprices
                    # memory by that factor.
                    unit = re.sub(r"[^A-Za-z]", "", value.strip()).upper()
                    per_unit_gb = _MEM_UNITS.get(unit or SBATCH_DEFAULT_MEM_UNIT)
                    if per_unit_gb is None:
                        continue
                    mem = float(number) / per_unit_gb
                elif key.startswith("gres/gpu"):
                    gpu = float(number)
        table[name] = Weights(
            cpu=cpu,
            mem_per_gb=mem,
            gpu=gpu,
            def_mem_per_cpu_gb=def_cpu,
            def_mem_per_node_gb=def_node,
        )
    return table


# One GB below an edge is the practical step; finer precision invites node
# memory granularity rounding the request back up over the edge.
_EPSILON_GB = 1.0

_SBATCH = re.compile(r"^\s*#SBATCH\s+(.*?)\s*$")


def parse_sbatch(text: str) -> Request:
    """Extract the resource shape from a submission script's #SBATCH lines.

    Later directives win, matching sbatch itself. Only the fields that affect
    billing are read; --time is deliberately ignored because it is not billed.
    """
    req = Request()
    ntasks: float | None = None
    ntasks_per_node: float | None = None
    nodes = 1.0
    nodes_range = False
    cpus_per_task: float | None = None
    mem_per_cpu: float | None = None
    total_gpus: float | None = None
    gpus_per_node: float | None = None
    gpus_per_task: float | None = None
    saw_mem = False

    prologue: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        m = _SBATCH.match(line)
        if not m:
            # sbatch stops processing directives at the first non-comment,
            # non-blank line. A later #SBATCH in the body is inert, and honouring
            # it prices resources Slurm will never request.
            if stripped and not stripped.startswith("#"):
                break
            continue
        prologue.append(line)
        for key, value in _split_directives(m.group(1)):
            if key in ("--cpus-per-task", "-c"):
                cpus_per_task = float(value)
            elif key in ("--ntasks", "-n"):
                ntasks = float(value)
            elif key == "--ntasks-per-node":
                ntasks_per_node = float(value)
            elif key in ("--nodes", "-N"):
                text = str(value).strip()
                if "-" in text:
                    # A range: Slurm may allocate anything within it, so the
                    # CPU total -- and every figure downstream -- is unknown.
                    nodes_range = True
                    nodes = float(text.split("-")[0])
                else:
                    nodes = float(text)
            elif key == "--mem":
                req.mem_gb = to_gb(value)
                req.mem_source = "--mem"
                saw_mem = True
            elif key == "--mem-per-cpu":
                mem_per_cpu = to_gb(value)
            elif key == "--gres":
                # --gres is per node, like --gpus-per-node.
                gpu = re.match(r"^gpu(?::[^:]+)?:(\d+)$", value.strip())
                if gpu:
                    gpus_per_node = float(gpu.group(1))
            elif key == "--gpus":
                total_gpus = float(re.sub(r"^.*:", "", value.strip()))
            elif key == "--gpus-per-node":
                gpus_per_node = float(re.sub(r"^.*:", "", value.strip()))
            elif key in ("--partition", "-p"):
                req.partition = value.strip()
            elif key == "--gpus-per-task":
                gpus_per_task = float(re.sub(r"^.*:", "", value.strip()))

    # Total tasks: an explicit --ntasks wins; otherwise --ntasks-per-node
    # multiplied across --nodes; otherwise one.
    if ntasks is not None:
        total_tasks = ntasks
    elif ntasks_per_node is not None:
        total_tasks = ntasks_per_node * nodes
    else:
        total_tasks = 1.0
    # Valueless flags never take the key=value shape _split_directives yields.
    req.exclusive = any(re.search(r"(?:^|\s)--exclusive(?:$|[\s=])", ln) for ln in prologue)
    req.nodes = nodes
    req.nodes_is_range = nodes_range
    if total_gpus is not None:
        req.gpus = total_gpus
    elif gpus_per_task is not None:
        req.gpus = gpus_per_task * total_tasks
    elif gpus_per_node is not None:
        req.gpus = gpus_per_node * nodes
    req.cpus = total_tasks * (cpus_per_task if cpus_per_task is not None else 1.0)

    req.mem_specified = saw_mem or mem_per_cpu is not None
    if mem_per_cpu is not None and not saw_mem:
        req.mem_gb = mem_per_cpu * req.cpus
        req.mem_source = "--mem-per-cpu"
        req.warnings.append(
            f"--mem-per-cpu={mem_per_cpu:g} GB x {req.cpus:g} CPU resolves to {req.mem_gb:g} GB, which lands on a "
            "round total far more often than an absolute --mem. Write --mem "
            "directly so the value you choose is the value that is billed."
        )
    return req


def _split_directives(chunk: str) -> list[tuple[str, str]]:
    """'--mem=32G -c 4' -> [('--mem','32G'), ('-c','4')]."""
    out: list[tuple[str, str]] = []
    tokens = chunk.split()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("-"):
            i += 1
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            out.append((key, value))
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            out.append((token, tokens[i + 1]))
            i += 2
            continue
        i += 1
    return out


def weighted_sum(req: Request, w: Weights) -> float:
    return w.cpu * req.cpus + w.mem_per_gb * req.mem_gb + w.gpu * req.gpus


def billing_units(req: Request, w: Weights) -> int:
    """Slurm's billing TRES: the weighted sum, floored to a whole unit."""
    return int(math.floor(weighted_sum(req, w)))


def boundary(req: Request, w: Weights) -> dict[str, Any]:
    """Where this request sits relative to the price transitions.

    Derived from the COMPLETE weighted sum, not from memory alone. The tempting
    shortcut -- that ``floor(C + G/k)`` factors into ``C + floor(G/k)``, so
    transitions land on multiples of the block -- holds only when the CPU and
    GPU contribution is a whole number. On a discounted partition it is not:
    with cpu=0.1 and gpu=0.1, an 8-CPU/1-GPU request contributes 0.9, and the
    price rises at 176 GB rather than at a multiple of 160. Assuming otherwise
    turns a one-gigabyte edge shave into a seventeen-gigabyte cut.

    Price is ``floor(base + w_mem * G)``, so for a given number of units the
    admissible memory is the half-open band ``[(u - base)/w_mem,
    (u+1 - base)/w_mem)``. Everything below is that band's arithmetic.

    The cheapest SAFE request is the largest one below the next transition, not
    the smallest one that fits: inside a band the extra gigabytes are free, so
    rounding memory down past what a job needs buys nothing and only removes
    headroom.
    """
    if w.mem_per_gb <= 0:
        return {"billed": False, "note": "memory is not billed on this partition"}

    base = w.cpu * req.cpus + w.gpu * req.gpus
    units = billing_units(req, w)
    band_start = (units - base) / w.mem_per_gb  # smallest memory still priced at `units`
    band_end = (units + 1 - base) / w.mem_per_gb  # first memory priced one unit higher

    result: dict[str, Any] = {
        "billed": True,
        "mem_per_billing_unit_gb": round(1.0 / w.mem_per_gb, 6),
        "current_mem_gb": req.mem_gb,
        # True when the request sits exactly on a transition, i.e. it just
        # bought a whole unit and holds none of the band it paid for.
        "on_price_edge": band_start > 0 and abs(req.mem_gb - band_start) < 1e-6,
        "largest_same_price_mem_gb": round(band_end - _EPSILON_GB, 3),
        "free_headroom_gb": round(max(0.0, band_end - _EPSILON_GB - req.mem_gb), 3),
    }

    cheaper_gb = band_start - _EPSILON_GB
    if cheaper_gb >= 0 and cheaper_gb < req.mem_gb:
        cheaper = Request(cpus=req.cpus, mem_gb=cheaper_gb, gpus=req.gpus)
        cheaper_units = billing_units(cheaper, w)
        if cheaper_units < units:
            given_up = req.mem_gb - cheaper_gb
            result["next_cheaper"] = {
                "mem_gb": round(cheaper_gb, 3),
                "units": cheaper_units,
                "units_now": units,
                "reduction_pct": (round(100.0 * (units - cheaper_units) / units, 1) if units else 0.0),
                "mem_given_up_gb": round(given_up, 3),
                # An edge shave gives up ~nothing and is close to free. Anything
                # larger is a genuine reduction in what the job can hold, and an
                # OOM kill bills full elapsed AND forces a rerun -- so the two
                # must not be presented as the same offer.
                "kind": ("edge_shave" if given_up <= _EPSILON_GB + 1e-9 else "real_reduction"),
                "note": (
                    f"Costs {given_up:.3g} GB of headroom. Safe only if the "
                    "family's observed MAXIMUM RSS stays well under it -- a mean "
                    "will not tell you."
                    if given_up > _EPSILON_GB + 1e-9
                    else f"Gives up {given_up:.3g} GB, the price edge only. The "
                    "request keeps essentially the headroom it has now."
                ),
            }
    return result


def load_weight_cache(path: str | None = None) -> dict[str, Any] | None:
    try:
        with open(cache_path(path), encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or "partitions" not in payload:
        return None
    return payload


def save_weight_cache(table: dict[str, Weights], captured_at: float, path: str | None = None) -> None:
    path = cache_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "captured_at": captured_at,
        "partitions": {
            name: {
                "cpu": w.cpu,
                "mem_per_gb": w.mem_per_gb,
                "gpu": w.gpu,
                "def_mem_per_cpu_gb": w.def_mem_per_cpu_gb,
                "def_mem_per_node_gb": w.def_mem_per_node_gb,
            }
            for name, w in table.items()
        },
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def cache_to_table(payload: dict[str, Any]) -> dict[str, Weights]:
    table: dict[str, Weights] = {}
    for name, entry in (payload.get("partitions") or {}).items():
        try:
            def_cpu = entry.get("def_mem_per_cpu_gb")
            def_node = entry.get("def_mem_per_node_gb")
            table[name] = Weights(
                cpu=float(entry.get("cpu", 1.0)),
                mem_per_gb=float(entry.get("mem_per_gb", 0.0)),
                gpu=float(entry.get("gpu", 0.0)),
                def_mem_per_cpu_gb=None if def_cpu is None else float(def_cpu),
                def_mem_per_node_gb=None if def_node is None else float(def_node),
            )
        except (TypeError, ValueError):
            continue
    return table


def resolve_request(req: Request, table: dict[str, Weights], partition: str) -> Request:
    """Fill in what the submission left implicit, or refuse to price it.

    Two things must be settled before any number is produced, and both were
    previously resolved inside price() alone -- so alternatives() went on to
    compare a different, memory-less shape across partitions.

    A request that names no memory does not get none: Slurm applies
    DefMemPerCPU or DefMemPerNode and bills that allocation. A --nodes range
    leaves the allocation genuinely undetermined, and a price computed from its
    minimum is a floor wearing the clothes of a figure.
    """
    if partition not in table:
        raise BillingError(
            "no billing weights known for partition {!r}; refresh the weight "
            "cache before pricing (known: {})".format(partition, ", ".join(sorted(table)) or "none")
        )
    if req.exclusive:
        raise BillingError(
            "--exclusive allocates whole nodes and Slurm bills every TRES on "
            "them, not the CPUs or GPUs the script asked for. The weight cache "
            "holds no per-node topology, so the real charge cannot be computed "
            "here -- it is bounded below by this request and above by the node's "
            "full complement. Price it against the node specification instead."
        )
    if req.nodes_is_range:
        raise BillingError(
            "--nodes was given as a range, so Slurm may allocate anything within "
            "it and the CPU total is not determined. Pricing the minimum would "
            "understate every job that receives more. Fix the node count, or "
            "price a specific size explicitly."
        )
    if req.mem_specified:
        return req
    default = table[partition].default_mem_gb(req.cpus, req.nodes)
    if default is None:
        raise BillingError(
            f"{partition!r} was given no --mem and no DefMemPerCPU or DefMemPerNode is "
            "recorded for it, so the memory Slurm would actually allocate -- and "
            "bill -- is unknown. Refresh the weight cache while connected, or "
            "state the memory explicitly."
        )
    return Request(
        cpus=req.cpus,
        mem_gb=default,
        gpus=req.gpus,
        nodes=req.nodes,
        mem_specified=True,
        mem_source=f"partition default ({partition})",
        warnings=list(req.warnings),
    )


def price(
    req: Request, table: dict[str, Weights], partition: str, captured_at: float | None = None, now: float | None = None
) -> dict[str, Any]:
    """Price one request. Raises BillingError rather than guessing a weight."""
    if partition not in table:
        raise BillingError(
            "no billing weights known for partition {!r}; refresh the weight "
            "cache before pricing (known: {})".format(partition, ", ".join(sorted(table)) or "none")
        )
    w = table[partition]
    req = resolve_request(req, table, partition)
    units = billing_units(req, w)
    pre = weighted_sum(req, w)
    payload: dict[str, Any] = {
        "partition": partition,
        "request": {"cpus": req.cpus, "mem_gb": req.mem_gb, "gpus": req.gpus, "mem_source": req.mem_source},
        "billing_units": units,
        "breakdown": {
            "cpu": round(w.cpu * req.cpus, 6),
            "mem": round(w.mem_per_gb * req.mem_gb, 6),
            "gpu": round(w.gpu * req.gpus, 6),
            "pre_floor": round(pre, 6),
            "floor_discards": round(pre - units, 6),
        },
        "boundary": boundary(req, w),
        "weights": {"cpu": w.cpu, "mem_per_gb": w.mem_per_gb, "gpu": w.gpu},
        "caveats": [
            "Requested --time is not part of the billing formula; raising a "
            "wall limit costs no fair share and under-requesting destroys the "
            "run's output for nothing.",
            "Billing follows the ALLOCATION, not usage: idle cores and " "untouched memory are charged in full.",
            "Node memory granularity can round --mem back up. Confirm what was "
            "granted with: sacct -j <id> -o AllocTRES",
        ],
        "warnings": list(req.warnings),
    }
    if units == 0 and pre > 0:
        # Whether a site clamps a positive rate up to one whole unit is not
        # discoverable from scontrol, and no sub-1.0 shape has ever been
        # observed billed. Saying "free" here would be an assumption dressed as
        # a price.
        payload["warnings"].append(
            f"The weighted rate is {pre:.4f}, which floors to zero billing units. "
            "Some sites enforce a one-unit minimum instead; that is not visible "
            "in scontrol, so treat this as 0-or-1 rather than free until a job "
            "of this shape has been observed billed on this partition."
        )
    if captured_at is not None:
        stamp = now if now is not None else time.time()
        payload["weights"]["captured_at"] = captured_at
        payload["weights"]["age_hours"] = round((stamp - captured_at) / 3600.0, 2)
    return payload


def alternatives(req: Request, table: dict[str, Weights], current: str, limit: int = 4) -> list[dict[str, Any]]:
    """Cheaper partitions for the identical request, cheapest first.

    Reported as prices, not as advice: a discounted partition is usually
    preemptible, and whether that is acceptable is a property of the job that
    this module cannot see.
    """
    if current not in table:
        return []
    now_units = billing_units(req, table[current])
    rows = []
    for name, w in table.items():
        if name == current:
            continue
        units = billing_units(req, w)
        if units < now_units:
            rows.append({"partition": name, "units": units, "units_now": now_units})
    rows.sort(key=lambda r: (r["units"], r["partition"]))
    return rows[:limit]
