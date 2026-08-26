"""Price a Slurm allocation before it is submitted.

Fair share is bought with *allocated* resources, not used ones, and the weighted
TRES sum is floored -- so memory is effectively sold in whole blocks and a
request sitting exactly on a block edge pays for a full block while forfeiting
the headroom inside it. Nothing in a job's own output reveals this, which is why
it is worth computing while the request is being written.

Everything here is pure arithmetic over a weight table. No SSH, no subprocess,
no Slurm: the only cluster-derived input is ``TRESBillingWeights``, which changes
rarely and is cached. That keeps pricing available while the O2 policy is
disabled and before any broker exists -- precisely when a submission is being
composed.

The input is a resource SHAPE -- CPUs, memory, GPUs -- not a submission script.
That boundary is deliberate. Reading a script means reimplementing sbatch's
option semantics: per-task and per-GPU forms, GRES list grammar, attached short
options, values like ``--mem=0`` that mean "everything", partition caps that
silently raise the CPU count, heterogeneous components. Getting any of them
subtly wrong yields a confident number that is wrong, with no checkpoint. When
an agent reads the script instead and states the shape it found, that reading
lands in the plan a human approves, where it can be challenged.

Scope is narrow on purpose. This answers "what will this cost and where is the
next boundary"; it does not estimate savings across a workload, infer
efficiency, or recommend reducing memory below what a job already holds.
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
    # Eligibility as scontrol reports it. Group membership cannot be evaluated
    # here, so a restricted partition is "unknown", never "available".
    state_up: bool = True
    unrestricted: bool = True
    # Slurm caps --mem-per-cpu at this and adds CPUs to preserve the memory the
    # task asked for, so it changes the billed CPU count, not just the memory.
    max_mem_per_cpu_gb: float | None = None
    # Per-model GPU weights, when the site prices accelerators differently.
    gpu_by_model: dict[str, float] | None = None

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
        state_up, unrestricted = True, True
        max_mem_cpu = None
        gpu_by_model: dict[str, float] = {}
        for token in line.split():
            # scontrol prints these as megabytes, and 0/UNLIMITED means "not set".
            if token.startswith("DefMemPerCPU="):
                value = token[len("DefMemPerCPU=") :]
                if value.isdigit() and int(value) > 0:
                    def_cpu = int(value) / 1024.0
            elif token.startswith("State="):
                state_up = token[len("State=") :].upper() == "UP"
            elif token.startswith("AllowGroups="):
                unrestricted = token[len("AllowGroups=") :].upper() == "ALL"
            elif token.startswith("AllowAccounts="):
                value = token[len("AllowAccounts=") :]
                if value and value.upper() != "ALL":
                    unrestricted = False
            elif token.startswith("MaxMemPerCPU="):
                value = token[len("MaxMemPerCPU=") :]
                if value.isdigit() and int(value) > 0:
                    max_mem_cpu = int(value) / 1024.0
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
                elif key == "gres/gpu":
                    gpu = float(number)
                elif key.startswith("gres/gpu:"):
                    # A model-specific weight (GRES/gpu:a100=8.0) applies only to
                    # that model. Folding it into the generic term would price
                    # every GPU at one model's rate; recorded separately so a
                    # caller naming no model is never charged at it.
                    gpu_by_model[key.split(":", 1)[1]] = float(number)
        table[name] = Weights(
            cpu=cpu,
            mem_per_gb=mem,
            gpu=gpu,
            def_mem_per_cpu_gb=def_cpu,
            def_mem_per_node_gb=def_node,
            state_up=state_up,
            unrestricted=unrestricted,
            max_mem_per_cpu_gb=max_mem_cpu,
            gpu_by_model=gpu_by_model or None,
        )
    return table


# One GB below an edge is the practical step; finer precision invites node
# memory granularity rounding the request back up over the edge.
_EPSILON_GB = 1.0

_SBATCH = re.compile(r"^\s*#SBATCH\s+(.*?)\s*$")


# Options whose effect on the allocation cannot be computed from the directives
# alone: each depends on the hardware of the nodes Slurm happens to pick, and
# the weight cache holds no topology. Refusing is a complete answer; a price
# derived from a guessed socket count is not.
UNPRICEABLE_OPTIONS = {
    "--exclusive": "bills every TRES on the allocated nodes, not what was asked for",
    "--ntasks-per-socket": "needs the sockets-per-node of the chosen hardware",
    "--gpus-per-socket": "needs the sockets-per-node of the chosen hardware",
    "--sockets-per-node": "constrains node selection by hardware layout",
    "--cores-per-socket": "constrains node selection by hardware layout",
    "--threads-per-core": "changes how many CPUs a core contributes",
    "--extra-node-info": "specifies sockets/cores/threads of the chosen nodes",
    "--overcommit": "decouples the CPU allocation from the task count",
    "--mem=0": "means all memory on every allocated node, which needs node sizes",
    "hetjob": (
        "separates heterogeneous components, each with its own allocation; "
        "folding them into one set of numbers prices neither"
    ),
}


def weighted_sum(req: Request, w: Weights) -> float:
    return w.cpu * req.cpus + w.mem_per_gb * req.mem_gb + w.gpu * req.gpus


def billing_units(req: Request, w: Weights) -> int:
    """Slurm's billing TRES: the weighted sum, floored to a whole unit."""
    return int(math.floor(weighted_sum(req, w)))


def _step_for(w: Weights) -> float:
    """How far below a transition to sit.

    One gigabyte where blocks are large -- finer precision invites node memory
    granularity rounding the request back over the edge. But on a partition
    whose billing block is under a gigabyte, a 1 GB shave would skip several
    price levels and report a far larger cut than the one actually needed.
    """
    if w.mem_per_gb <= 0:
        return _EPSILON_GB
    return min(_EPSILON_GB, (1.0 / w.mem_per_gb) / 2.0)


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
        "largest_same_price_mem_gb": round(band_end - _step_for(w), 3),
        "free_headroom_gb": round(max(0.0, band_end - _step_for(w) - req.mem_gb), 3),
    }

    # The step must stay inside the band it is stepping out of: on a partition
    # whose billing block is under a gigabyte, a fixed 1 GB shave skips past
    # several price levels and reports a far larger cut than the one needed.
    cheaper_gb = band_start - _step_for(w)
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
                "kind": ("edge_shave" if given_up <= _step_for(w) + 1e-9 else "real_reduction"),
                "note": (
                    f"Costs {given_up:.3g} GB of headroom. Safe only if the "
                    "family's observed MAXIMUM RSS stays well under it -- a mean "
                    "will not tell you."
                    if given_up > _step_for(w) + 1e-9
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
                "state_up": w.state_up,
                "unrestricted": w.unrestricted,
                "max_mem_per_cpu_gb": w.max_mem_per_cpu_gb,
                "gpu_by_model": w.gpu_by_model,
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
                state_up=bool(entry.get("state_up", True)),
                unrestricted=bool(entry.get("unrestricted", True)),
                max_mem_per_cpu_gb=(
                    None if entry.get("max_mem_per_cpu_gb") is None else float(entry["max_mem_per_cpu_gb"])
                ),
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
        # Offering a partition the caller cannot submit to wastes their time and
        # invites a resubmission that will be rejected.
        if not w.state_up or not w.unrestricted:
            continue
        # A GPU request cannot run where GPUs are not a billed resource. That is
        # a proxy for "has no GPUs" rather than proof, so it can only exclude a
        # suggestion -- never manufacture one.
        if req.gpus > 0 and w.gpu <= 0:
            continue
        units = billing_units(req, w)
        if units < now_units:
            rows.append({"partition": name, "units": units, "units_now": now_units})
    rows.sort(key=lambda r: (r["units"], r["partition"]))
    return rows[:limit]
