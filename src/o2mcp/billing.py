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
import threading
import time
from dataclasses import dataclass, field, replace
from decimal import ROUND_FLOOR, Decimal
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
# Every suffix Slurm permits on a memory billing weight. A missing one is not
# harmless: an unrecognised unit left mem_per_gb at zero, so "Mem=1P" priced
# memory as free rather than as the largest term in the sum.
_MEM_UNITS = {
    "K": 1.0 / (1024**2),
    "M": 1.0 / 1024,
    "G": 1.0,
    "T": 1024.0,
    "P": 1024.0**2,
}
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
    # Weighted TRES types this calculator cannot charge for. Non-empty means
    # any price for this partition would be understated, so it refuses.
    unpriceable_tres: dict[str, float] | None = None
    # GPUs the partition HOLDS, from its TRES inventory rather than its billing
    # weights. None means the inventory was never captured.
    gpu_stock: float | None = None
    gpu_stock_by_model: dict[str, float] | None = None

    def default_mem_gb(self, cpus: float, nodes: float | None = None) -> float | None:
        """Memory Slurm would allocate for a request that names none.

        DefMemPerCPU needs only the CPU count. DefMemPerNode scales with the
        NODE count, which a resource shape does not have to state -- and
        assuming one node would underprice a multi-node allocation by the whole
        per-node default, so the caller is asked instead of guessed at.
        """
        if self.def_mem_per_cpu_gb is not None:
            return self.def_mem_per_cpu_gb * cpus
        if self.def_mem_per_node_gb is not None:
            if nodes is None:
                return None
            return self.def_mem_per_node_gb * nodes
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
    # Whether the caller actually said how many nodes. Only DefMemPerNode needs
    # it, and silently assuming one would underprice every multi-node job.
    nodes_stated: bool = False
    # The GPU model, when the shape names one and the site prices models apart.
    gpu_model: str | None = None
    # False when no --mem/--mem-per-cpu was given, so the partition default
    # applies and a price of "zero memory" would be a fiction.
    mem_specified: bool = True
    partition: str | None = None
    # Recorded because --mem-per-cpu multiplies out to a round total and so
    # lands on a block edge far more often than an absolute --mem does.
    mem_source: str = "default"
    warnings: list[str] = field(default_factory=list)


def _remem(req: "Request", mem_gb: float) -> "Request":
    """``req`` at a different memory size, with every other field intact.

    Repricing a shape means changing exactly one number. Rebuilding the request
    from named fields instead has twice dropped ``gpu_model``, which reprices a
    model-specific GPU at the generic weight and reports a saving for an
    allocation the caller never asked about.
    """
    return replace(req, mem_gb=mem_gb, warnings=list(req.warnings))


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
        # Slurm's CPU fallback applies only when TRESBillingWeights is ABSENT:
        # "If TRESBillingWeights is not defined then the job's billing TRES is
        # equal to the total CPUs allocated." Where a table exists but names no
        # CPU term, CPUs are not billed -- so seeding 1.0 unconditionally
        # charged them anyway, pricing 32 CPUs + 16 GB under "Mem=0.0625G" at
        # 33 units instead of the 1 the site configured.
        cpu = 1.0 if not (raw and raw not in ("(null)", "")) else 0.0
        mem, gpu = 0.0, 0.0
        def_cpu = def_node = None
        state_up, unrestricted = True, True
        unpriceable: dict[str, float] = {}
        # None until a TRES= token is seen: "no GPUs" and "never told" are
        # different answers, and only one of them permits a suggestion.
        gpu_stock: float | None = None
        gpu_stock_by_model: dict[str, float] = {}
        max_mem_cpu = None
        gpu_by_model: dict[str, float] = {}
        for token in line.split():
            # scontrol prints these as megabytes, and 0/UNLIMITED means "not set".
            if token.startswith("DefMemPerCPU="):
                value = token[len("DefMemPerCPU=") :]
                if value.isdigit() and int(value) > 0:
                    def_cpu = int(value) / 1024.0
            elif token.startswith("TRES="):
                # Distinct from TRESBillingWeights: this is the inventory the
                # partition actually holds. Weights say what a resource COSTS;
                # only inventory says whether it can run there at all.
                for part in token[len("TRES=") :].split(","):
                    if "=" not in part:
                        continue
                    tkey, tval = part.split("=", 1)
                    tkey = tkey.strip().lower()
                    if not tkey.startswith("gres/gpu"):
                        continue
                    try:
                        count = float(re.sub(r"[A-Za-z]", "", tval) or 0)
                    except ValueError:
                        continue
                    if tkey == "gres/gpu":
                        gpu_stock = count
                    else:
                        gpu_stock_by_model[tkey.split(":", 1)[1]] = count
                if gpu_stock is None:
                    gpu_stock = 0.0
            elif token.startswith("State="):
                state_up = token[len("State=") :].upper() == "UP"
            elif token.startswith("AllowGroups="):
                unrestricted = token[len("AllowGroups=") :].upper() == "ALL"
            elif token.startswith("AllowAccounts="):
                value = token[len("AllowAccounts=") :]
                if value and value.upper() != "ALL":
                    unrestricted = False
            elif token.startswith("AllowQos="):
                value = token[len("AllowQos=") :]
                if value and value.upper() != "ALL":
                    unrestricted = False
            elif token.startswith("DenyQos="):
                # Any deny list at all means some callers are turned away, and
                # this tool cannot tell whether the caller is one of them.
                value = token[len("DenyQos=") :]
                if value and value.upper() not in ("", "(NULL)", "NONE"):
                    unrestricted = False
            elif token.startswith("DenyAccounts="):
                value = token[len("DenyAccounts=") :]
                if value and value.upper() not in ("", "(NULL)", "NONE"):
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
                        # Recorded, not skipped: skipping leaves mem_per_gb at
                        # zero, which prices memory as free -- the same silent
                        # drop that "Node=10" used to get.
                        unpriceable["mem:" + unit] = float(number)
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
                else:
                    # TRESBillingWeights is an open list of TRES types. A
                    # nonzero weight this calculator cannot represent -- Node=10,
                    # a licence, a burst buffer -- is charged by Slurm and not by
                    # us, so every figure for the partition would be understated
                    # by it. Recorded and refused at pricing time rather than
                    # dropped: silently pricing what we only partly understand
                    # is the one failure this module exists to avoid.
                    try:
                        if float(number) != 0:
                            unpriceable[key] = float(number)
                    except ValueError:
                        pass
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
            unpriceable_tres=unpriceable or None,
            gpu_stock=gpu_stock,
            gpu_stock_by_model=gpu_stock_by_model or None,
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


def _unpriceable_gpu_reason(req: "Request", w: "Weights") -> str | None:
    """Why this partition cannot price the request's GPUs, or None if it can.

    gpu_weight_for() falls back to the generic weight whenever the named model
    misses, and that fallback is wrong in both directions: it charges one
    accelerator at another's rate, and where a site declares ONLY per-model
    rates the generic entry is zero, so an untyped request prices every GPU as
    free. Both are confident numbers, which is the failure that matters here.

    So the question is settled once, before any number is produced, and by ONE
    predicate -- alternatives() and resolve_request() disagreeing about it has
    already produced both an advertised move that could not be priced on
    arrival and a partition hidden as GPU-less while pricing perfectly well.
    """
    if req.gpus <= 0 or not w.gpu_by_model:
        return None
    if req.gpu_model:
        if req.gpu_model in w.gpu_by_model:
            return None
        known = ", ".join(sorted(w.gpu_by_model)) or "none"
        return (
            f"has no weight for GPU model {req.gpu_model!r} (priced: {known}). "
            "Name a priced model, or omit gpu_model where the partition has a "
            "generic GPU weight."
        )
    if w.gpu > 0:
        # A declared generic rate is a real rate, so an untyped request is
        # priceable even where models are also listed.
        return None
    known = ", ".join(sorted(w.gpu_by_model)) or "none"
    return (
        f"prices GPUs only per model (priced: {known}) and has no generic GPU "
        "weight, so an untyped GPU request would be charged nothing for its "
        "accelerators. Name the model the allocation will hold."
    )


def gpu_weight_for(req: Request, w: Weights) -> float:
    """The GPU weight that applies to this request.

    A site may price accelerators per model. Capturing those weights without
    consulting them charges an A100 at whatever the generic entry says -- or at
    another model's rate -- so the named model wins when one is given.
    """
    if req.gpu_model and w.gpu_by_model:
        by_model = w.gpu_by_model.get(req.gpu_model)
        if by_model is not None:
            return by_model
    return w.gpu


def _exact(value: float) -> Decimal:
    """The decimal a float was WRITTEN as, not the binary it became.

    str() gives the shortest repr that round-trips, so a weight configured as
    0.29 comes back as exactly 0.29 rather than 0.28999999999999998.
    """
    return Decimal(str(value))


def _weighted_sum_exact(req: Request, w: Weights) -> Decimal:
    """The pre-floor sum in decimal, because the floor is unforgiving.

    Binary arithmetic puts a weight that should land ON a unit boundary just
    under it: cpu=0.29 across 100 CPUs is 28.999999999999996, which floors to
    28 rather than 29. That off-by-one then propagates into the boundary
    figures and the alternatives, so it is settled once, here.
    """
    return (
        _exact(w.cpu) * _exact(req.cpus)
        + _exact(w.mem_per_gb) * _exact(req.mem_gb)
        + _exact(gpu_weight_for(req, w)) * _exact(req.gpus)
    )


def weighted_sum(req: Request, w: Weights) -> float:
    return float(_weighted_sum_exact(req, w))


def billing_units(req: Request, w: Weights) -> int:
    """Slurm's billing TRES: the weighted sum, floored to a whole unit."""
    return int(_weighted_sum_exact(req, w).to_integral_value(rounding=ROUND_FLOOR))


def _round_below(value: float, places: int = 9) -> float:
    """Round DOWN, so a suggested memory size cannot drift back over an edge.

    Rounding to a fixed precision is not safe here: where a billing block is
    smaller than that precision, the nearest value sits on the far side of the
    transition. With Mem=1M the cheaper size is 0.99951171875 GB, and rounding
    to three places returns 1.0 -- the very request being priced down from.
    """
    if value <= 0:
        return 0.0
    scale = 10**places
    return math.floor(value * scale) / scale


# The finest margin worth suggesting below a price edge. Anything smaller is
# noise against node memory granularity.
_MIN_STEP_GB = 0.001


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


def _largest_at_same_price(req: Request, w: Weights, band_end: float) -> float:
    """The largest memory we would recommend at the request's current price.

    A fixed step below band_end is a deliberate safety margin -- node memory
    granularity can round a request back over the edge -- but a request already
    sitting above that step lands BEHIND itself, and then reports none of the
    same-price capacity it genuinely still has. So the margin adapts: half the
    distance that actually remains, and finally a hair below the edge.
    """
    remaining = band_end - req.mem_gb
    for step in (_step_for(w), remaining / 2.0, _MIN_STEP_GB):
        if step <= 0:
            continue
        candidate = _round_below(band_end - step)
        if candidate >= req.mem_gb:
            return candidate
    return req.mem_gb


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

    base = w.cpu * req.cpus + gpu_weight_for(req, w) * req.gpus
    units = billing_units(req, w)
    band_start = (units - base) / w.mem_per_gb  # smallest memory still priced at `units`
    band_end = (units + 1 - base) / w.mem_per_gb  # first memory priced one unit higher

    largest_same_price = _largest_at_same_price(req, w, band_end)

    result: dict[str, Any] = {
        "billed": True,
        "mem_per_billing_unit_gb": round(1.0 / w.mem_per_gb, 6),
        "current_mem_gb": req.mem_gb,
        # True when the request sits exactly on a transition, i.e. it just
        # bought a whole unit and holds none of the band it paid for.
        "on_price_edge": band_start > 0 and abs(req.mem_gb - band_start) < 1e-6,
        # Never behind the current request: it demonstrably holds this price,
        # so a "largest at this price" below it is a contradiction. On a
        # sub-2 GB block a request in the upper half of its band produced
        # exactly that, and then reported zero headroom for it.
        "largest_same_price_mem_gb": largest_same_price,
        "free_headroom_gb": _round_below(max(0.0, largest_same_price - req.mem_gb)),
    }

    # The step must stay inside the band it is stepping out of: on a partition
    # whose billing block is under a gigabyte, a fixed 1 GB shave skips past
    # several price levels and reports a far larger cut than the one needed.
    cheaper_gb = band_start - _step_for(w)
    if cheaper_gb >= 0 and cheaper_gb < req.mem_gb:
        # replace(), not a fresh Request: only the memory changes, and naming
        # the surviving fields by hand is what silently dropped gpu_model here
        # -- a model-priced GPU then repriced at the generic weight, so the
        # quoted reduction described a different allocation than the caller's.
        cheaper = _remem(req, cheaper_gb)
        cheaper_units = billing_units(cheaper, w)
        if cheaper_units < units:
            given_up = req.mem_gb - cheaper_gb
            suggested = _round_below(cheaper_gb)
            # A suggestion that does not reprice lower is worse than none.
            if billing_units(_remem(req, suggested), w) >= units:
                suggested = cheaper_gb
            result["next_cheaper"] = {
                "mem_gb": suggested,
                "units": cheaper_units,
                "units_now": units,
                "reduction_pct": (round(100.0 * (units - cheaper_units) / units, 1) if units else 0.0),
                "mem_given_up_gb": _round_below(given_up),
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


def save_weight_cache(
    table: dict[str, Weights],
    captured_at: float,
    path: str | None = None,
    priority_flags: list[str] | None = None,
) -> None:
    path = cache_path(path)
    parent = os.path.dirname(path)
    if parent:  # a bare filename has no directory to create
        os.makedirs(parent, exist_ok=True)
    payload = {
        "captured_at": captured_at,
        # Cluster-global, not per-partition: PriorityFlags decides whether
        # Billing is the SUM of weighted TRES or their MAX, and this module
        # only implements the sum. None means the flags were never captured,
        # which is not the same as "no flags".
        "priority_flags": (
            None if priority_flags is None else sorted({f.strip().upper() for f in priority_flags if f.strip()})
        ),
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
                "unpriceable_tres": w.unpriceable_tres,
                "gpu_stock": w.gpu_stock,
                "gpu_stock_by_model": w.gpu_stock_by_model,
            }
            for name, w in table.items()
        },
    }
    # Unique per writer, not merely per process: two threads refreshing at once
    # would otherwise share a name and interleave into one corrupt file.
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _float_map(value: Any) -> dict[str, float] | None:
    """A cached {name: weight} map, or None. Never a partial one.

    A half-read weight map is worse than none: it prices some models and
    silently falls back for the rest.
    """
    if not isinstance(value, dict) or not value:
        return None
    out: dict[str, float] = {}
    for key, number in value.items():
        try:
            out[str(key)] = float(number)
        except (TypeError, ValueError):
            return None
    return out or None


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
                # Both of these decide a price, and both were written to the
                # cache but never read back -- so on the cache path, which is
                # the DEFAULT path, every per-model GPU weight silently
                # disappeared and an a100 was charged at the generic rate.
                gpu_by_model=_float_map(entry.get("gpu_by_model")),
                unpriceable_tres=_float_map(entry.get("unpriceable_tres")),
                gpu_stock=(None if entry.get("gpu_stock") is None else float(entry["gpu_stock"])),
                gpu_stock_by_model=_float_map(entry.get("gpu_stock_by_model")),
            )
        except (TypeError, ValueError):
            continue
    return table


def unsupported_billing_model(payload: dict[str, Any]) -> str | None:
    """Why the cached configuration cannot be priced here, or None.

    Everything in this module computes floor(weighted SUM). Under
    PriorityFlags=MAX_TRES Slurm bills the MAXIMUM weighted TRES instead, and
    the two give opposite advice: with CPU and memory contributions of four
    units each, the sum says eight and the max says four, and trimming a
    non-dominant TRES saves nothing at all. The partition weights alone cannot
    reveal this -- the flag is global -- so it is captured with them.

    An absent record is refused rather than assumed, because assuming is how a
    confident wrong number reaches an approval.
    """
    flags = payload.get("priority_flags")
    if flags is None:
        return (
            "the cached weights predate PriorityFlags capture, so whether this "
            "cluster bills the SUM or the MAX of weighted TRES is unknown -- and "
            "the two imply opposite advice about memory. Run "
            "o2_refresh_billing_weights to record it."
        )
    # Prefix, not equality: MAX_TRES_GRES is the same maximum-based calculation
    # with GRES folded in, and an exact match let it through to the sum.
    maxing = sorted(f for f in {str(x).upper() for x in flags} if f.startswith("MAX_TRES"))
    if maxing:
        return (
            f"this cluster sets PriorityFlags={','.join(maxing)}, so Slurm bills the MAXIMUM "
            "weighted TRES rather than their sum. This tool computes the sum, and "
            "the memory-boundary reasoning it exists to support does not hold "
            "under MAX_TRES: trimming a non-dominant TRES saves nothing and only "
            "removes headroom. Right-size the dominant TRES instead."
        )
    return None


def parse_priority_flags(text: str) -> list[str]:
    """PriorityFlags from `scontrol show config` output."""
    for line in text.splitlines():
        if line.strip().startswith("PriorityFlags"):
            _, _, value = line.partition("=")
            return [f.strip().upper() for f in value.split(",") if f.strip()]
    return []


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
    extra = table[partition].unpriceable_tres
    if extra:
        listed = ", ".join("%s=%g" % (k, v) for k, v in sorted(extra.items()))
        raise BillingError(
            f"{partition!r} bills weighted TRES this calculator cannot charge "
            f"for ({listed}). Every figure for it would be understated by that "
            "amount, so no price is offered rather than a low one."
        )
    reason = _unpriceable_gpu_reason(req, table[partition])
    if reason:
        raise BillingError(f"{partition!r} {reason}")
    # Slurm allocates whole CPUs and whole GPUs; a fractional count is not a
    # shape it can produce, and rounding one silently would price a job that
    # cannot exist.
    whole = [("cpus", req.cpus), ("gpus", req.gpus)]
    if req.nodes_stated:
        # Only when stated: the field defaults to 1 and an unstated default is
        # not a claim about the shape.
        whole.append(("nodes", req.nodes))
    for label, value in whole:
        if value != int(value):
            raise BillingError(
                f"{label}={value:g} is not a whole number, and Slurm allocates "
                f"whole {label}. Give the count the allocation will actually hold."
            )
    if req.mem_specified and req.mem_gb <= 0:
        raise BillingError(
            "a memory size of zero is not an allocation Slurm makes: sbatch "
            "reads --mem=0 as all memory on every allocated node, which depends "
            "on node sizes this weight table does not hold. Give the real size, "
            "or omit mem_gb to price the partition default."
        )
    if req.mem_specified:
        return req
    w = table[partition]
    default = w.default_mem_gb(req.cpus, req.nodes if req.nodes_stated else None)
    if default is None and w.def_mem_per_node_gb is not None and not req.nodes_stated:
        raise BillingError(
            f"{partition!r} defaults memory per NODE ({w.def_mem_per_node_gb:g} GB), "
            "so the allocation's memory depends on how many nodes it holds. State "
            "`nodes`, or give mem_gb explicitly."
        )
    if default is None:
        raise BillingError(
            f"{partition!r} was given no --mem and no DefMemPerCPU or DefMemPerNode is "
            "recorded for it, so the memory Slurm would actually allocate -- and "
            "bill -- is unknown. Refresh the weight cache while connected, or "
            "state the memory explicitly."
        )
    return replace(
        req,
        mem_gb=default,
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
        # gpu_model is echoed because it changes the price: a caller comparing
        # this response against their own shape cannot tell which accelerator
        # was charged without it.
        "request": {
            "cpus": req.cpus,
            "mem_gb": req.mem_gb,
            "gpus": req.gpus,
            "gpu_model": req.gpu_model,
            "mem_source": req.mem_source,
        },
        "billing_units": units,
        "breakdown": {
            "cpu": round(w.cpu * req.cpus, 6),
            "mem": round(w.mem_per_gb * req.mem_gb, 6),
            "gpu": round(gpu_weight_for(req, w) * req.gpus, 6),
            "pre_floor": round(pre, 6),
            "floor_discards": round(pre - units, 6),
        },
        "boundary": boundary(req, w),
        # The GPU entry is the weight this request was actually charged at, so
        # the breakdown above can be recomputed from it. Echoing the generic
        # weight beside a model-priced breakdown made the response disagree
        # with itself and left the caller unable to audit the charge.
        "weights": {
            "cpu": w.cpu,
            "mem_per_gb": w.mem_per_gb,
            "gpu": gpu_weight_for(req, w),
            "gpu_generic": w.gpu,
            "gpu_model": req.gpu_model,
        },
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


def _has_gpu_stock(req: "Request", w: "Weights") -> bool:
    """Does this partition hold enough of the GPU the request names?

    Answers False when the inventory was never captured: a suggestion has to be
    positively supported, and "we did not look" is not support.
    """
    if w.gpu_stock is None:
        return False
    if req.gpu_model:
        by_model = w.gpu_stock_by_model or {}
        if req.gpu_model in by_model:
            return by_model[req.gpu_model] >= req.gpus
        # A model-priced partition that never lists that model's stock cannot
        # be shown to hold it.
        return not (w.gpu_by_model or by_model) and w.gpu_stock >= req.gpus
    return w.gpu_stock >= req.gpus


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
        # Two different questions, and the weights answer only the first.
        #
        # Can it be PRICED here? A partition carrying only "GRES/gpu:a100=1"
        # has a generic weight of zero while pricing an a100 perfectly well, so
        # ask for the weight that would actually price THIS request.
        if req.gpus > 0 and gpu_weight_for(req, w) <= 0:
            continue
        # Can it RUN here? TRESBillingWeights defines charges; the partition's
        # TRES inventory defines what it holds. A positive weight is not
        # evidence of a single GPU, so advertising on it offered moves that
        # would be rejected on arrival. An uncaptured inventory is unknown, and
        # unknown may not manufacture a suggestion either.
        if req.gpus > 0 and not _has_gpu_stock(req, w):
            continue
        # Never advertise a partition that pricing this same request directly
        # would refuse: the units would come from another model's weight.
        if _unpriceable_gpu_reason(req, w) or w.unpriceable_tres:
            continue
        units = billing_units(req, w)
        if units < now_units:
            rows.append({"partition": name, "units": units, "units_now": now_units})
    rows.sort(key=lambda r: (r["units"], r["partition"]))
    return rows[:limit]
