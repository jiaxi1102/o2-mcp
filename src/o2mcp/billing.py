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


# sacct/scontrol memory suffixes, expressed in GB.
_MEM_UNITS = {"": 1.0 / (1024**3), "K": 1.0 / (1024**2), "M": 1.0 / 1024, "G": 1.0, "T": 1024.0}


class BillingError(ValueError):
    """A pricing input that cannot be answered honestly."""


@dataclass(frozen=True)
class Weights:
    """One partition's TRESBillingWeights."""

    cpu: float = 1.0
    mem_per_gb: float = 0.0
    gpu: float = 0.0

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
    partition: str | None = None
    # Recorded because --mem-per-cpu multiplies out to a round total and so
    # lands on a block edge far more often than an absolute --mem does.
    mem_source: str = "default"
    warnings: list[str] = field(default_factory=list)


def to_gb(value: str | float | None) -> float:
    """'32G' / '8192M' / 32 -> gigabytes."""
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    m = re.match(r"^\s*([0-9.]+)\s*([KMGTkmgt]?)", str(value))
    if not m:
        raise BillingError(f"could not read a memory size from {value!r}")
    return float(m.group(1)) * _MEM_UNITS[m.group(2).upper()]


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
                    # Written per-unit: 'Mem=0.0625G' is 0.0625 per GB.
                    mem = float(number)
                elif key.startswith("gres/gpu"):
                    gpu = float(number)
        table[name] = Weights(cpu=cpu, mem_per_gb=mem, gpu=gpu)
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
    ntasks = 1.0
    cpus_per_task: float | None = None
    mem_per_cpu: float | None = None
    saw_mem = False

    for line in (text or "").splitlines():
        m = _SBATCH.match(line)
        if not m:
            continue
        for key, value in _split_directives(m.group(1)):
            if key in ("--cpus-per-task", "-c"):
                cpus_per_task = float(value)
            elif key in ("--ntasks", "-n"):
                ntasks = float(value)
            elif key == "--mem":
                req.mem_gb = to_gb(value)
                req.mem_source = "--mem"
                saw_mem = True
            elif key == "--mem-per-cpu":
                mem_per_cpu = to_gb(value)
            elif key == "--gres":
                gpu = re.match(r"^gpu(?::[^:]+)?:(\d+)$", value.strip())
                if gpu:
                    req.gpus = float(gpu.group(1))
            elif key in ("--gpus", "--gpus-per-node"):
                req.gpus = float(re.sub(r"^.*:", "", value.strip()))
            elif key in ("--partition", "-p"):
                req.partition = value.strip()

    req.cpus = ntasks * (cpus_per_task if cpus_per_task is not None else 1.0)

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
    """Where this request sits relative to the memory block edges.

    The cheapest safe request is the LARGEST one below the next edge, not the
    smallest one that fits: inside a block the extra gigabytes are free, so
    rounding memory down past what a job needs buys nothing and only removes
    headroom.
    """
    block = w.block_gb
    if block is None:
        return {"billed": False, "note": "memory is not billed on this partition"}

    units = billing_units(req, w)
    blocks_used = math.floor(req.mem_gb / block)
    # Largest memory that still yields one fewer whole block.
    cheaper_gb = max(0.0, (blocks_used * block) - _EPSILON_GB) if blocks_used else None
    top_of_block = ((blocks_used + 1) * block) - _EPSILON_GB
    on_edge = abs(req.mem_gb - blocks_used * block) < 1e-9 and blocks_used > 0

    result: dict[str, Any] = {
        "billed": True,
        "block_gb": block,
        "current_mem_gb": req.mem_gb,
        "on_block_edge": on_edge,
        "free_headroom_gb": round(top_of_block - req.mem_gb, 3),
        "largest_same_price_mem_gb": round(top_of_block, 3),
    }
    if cheaper_gb is not None and cheaper_gb < req.mem_gb:
        cheaper = Request(cpus=req.cpus, mem_gb=cheaper_gb, gpus=req.gpus)
        cheaper_units = billing_units(cheaper, w)
        if cheaper_units < units:
            given_up = req.mem_gb - cheaper_gb
            result["next_cheaper"] = {
                "mem_gb": round(cheaper_gb, 3),
                "units": cheaper_units,
                "units_now": units,
                "reduction_pct": round(100.0 * (units - cheaper_units) / units, 1) if units else 0.0,
                "mem_given_up_gb": round(given_up, 3),
                # An edge shave gives up ~nothing and is close to free. Anything
                # larger is a genuine reduction in what the job can hold, and an
                # OOM kill bills full elapsed AND forces a rerun -- so the two
                # must not be presented as the same offer.
                "kind": "edge_shave" if given_up <= _EPSILON_GB + 1e-9 else "real_reduction",
                "note": (
                    (
                        f"Costs {given_up:.3g} GB of headroom. Safe only if the family's "
                        "observed MAXIMUM RSS stays well under it -- a mean will "
                        "not tell you."
                    )
                    if given_up > _EPSILON_GB + 1e-9
                    else (
                        f"Gives up {given_up:.3g} GB, the block edge only. The request keeps "
                        "essentially the headroom it has now."
                    )
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
        "partitions": {name: {"cpu": w.cpu, "mem_per_gb": w.mem_per_gb, "gpu": w.gpu} for name, w in table.items()},
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def cache_to_table(payload: dict[str, Any]) -> dict[str, Weights]:
    table: dict[str, Weights] = {}
    for name, entry in (payload.get("partitions") or {}).items():
        try:
            table[name] = Weights(
                cpu=float(entry.get("cpu", 1.0)),
                mem_per_gb=float(entry.get("mem_per_gb", 0.0)),
                gpu=float(entry.get("gpu", 0.0)),
            )
        except (TypeError, ValueError):
            continue
    return table


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
