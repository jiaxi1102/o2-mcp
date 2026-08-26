"""Pricing arithmetic for a Slurm submission.

The shape table below is not synthetic: every row was observed on HMS O2 as an
``AllocTRES`` -> ``billing`` pair. They matter because they discriminate between
the two billing models Slurm supports -- ``floor(weighted sum)`` and
``floor(max weighted TRES)`` -- which imply opposite advice about memory. Six of
these rows refute the max model outright.
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from o2mcp import billing

SHORT = billing.Weights(cpu=1.0, mem_per_gb=0.0625, gpu=0.0)
GPU_QUAD = billing.Weights(
    cpu=1.0, mem_per_gb=0.0625, gpu=5.0, stock={"cpu": 400, "mem": 4000, "node": 10, "gres/gpu": 40}
)
GPU_REQUEUE = billing.Weights(
    cpu=0.1, mem_per_gb=0.00625, gpu=0.1, stock={"cpu": 1080, "mem": 10000, "node": 27, "gres/gpu": 108}
)

# (weights, cpus, mem_gb, gpus, observed billing)
OBSERVED = [
    (SHORT, 2, 16, 0, 3),
    (SHORT, 4, 8, 0, 4),
    (SHORT, 4, 48, 0, 7),
    (SHORT, 4, 32, 0, 6),
    (SHORT, 4, 16, 0, 5),
    (SHORT, 4, 24, 0, 5),
    (SHORT, 2, 8, 0, 2),
    (SHORT, 1, 4, 0, 1),
    (SHORT, 2, 32, 0, 4),
    (SHORT, 8, 24, 0, 9),
    (SHORT, 1, 16, 0, 2),
    (SHORT, 8, 64, 0, 12),
    (SHORT, 4, 96, 0, 10),
    (SHORT, 4, 64, 0, 8),
    (SHORT, 8, 16, 0, 9),
    (GPU_QUAD, 4, 6, 1, 9),
    (GPU_REQUEUE, 8, 24, 1, 1),
    (GPU_REQUEUE, 8, 96, 1, 1),
    (GPU_REQUEUE, 12, 120, 1, 2),
    (GPU_REQUEUE, 16, 64, 1, 2),
    (GPU_REQUEUE, 20, 128, 2, 3),
    (GPU_REQUEUE, 32, 64, 2, 3),
]


@pytest.mark.parametrize("weights,cpus,mem,gpus,expected", OBSERVED)
def test_observed_shapes_reproduce(weights, cpus, mem, gpus, expected):
    request = billing.Request(cpus=cpus, mem_gb=mem, gpus=gpus)
    assert billing.billing_units(request, weights) == expected


def test_the_max_model_is_refuted_by_real_shapes():
    # If billing were max(weighted TRES) rather than the sum, memory would be
    # free until it exceeded the CPU term and every boundary claim below would
    # be wrong. These rows say otherwise.
    refuting = [
        (SHORT, 2, 16, 0, 3),
        (SHORT, 4, 48, 0, 7),
        (SHORT, 4, 32, 0, 6),
        (SHORT, 4, 16, 0, 5),
        (SHORT, 4, 24, 0, 5),
        (GPU_QUAD, 4, 6, 1, 9),
    ]
    for weights, cpus, mem, gpus, observed in refuting:
        request = billing.Request(cpus=cpus, mem_gb=mem, gpus=gpus)
        as_max = math.floor(max(weights.cpu * cpus, weights.mem_per_gb * mem, weights.gpu * gpus))
        assert billing.billing_units(request, weights) == observed
        assert as_max != observed


def test_memory_is_sold_in_whole_blocks():
    # 16 GB costs exactly one CPU at these weights, and the floor makes the
    # price a step function rather than a slope.
    prices = [billing.billing_units(billing.Request(cpus=4, mem_gb=g), SHORT) for g in (0, 15, 16, 31, 32, 47, 48)]
    assert prices == [4, 4, 5, 5, 6, 6, 7]


def test_largest_same_price_memory_is_reported():
    # The cheapest SAFE request is the largest one below the next edge; rounding
    # down past what a job needs buys nothing and only removes headroom.
    info = billing.boundary(billing.Request(cpus=4, mem_gb=20), SHORT)
    assert info["billed"] is True
    assert info["largest_same_price_mem_gb"] == pytest.approx(31.0)
    assert info["free_headroom_gb"] == pytest.approx(11.0)
    assert info["mem_per_billing_unit_gb"] == pytest.approx(16.0)


def test_request_on_a_block_edge_is_flagged_with_the_cheaper_value():
    info = billing.boundary(billing.Request(cpus=4, mem_gb=32), SHORT)
    assert info["on_price_edge"] is True
    assert info["next_cheaper"]["mem_gb"] == pytest.approx(31.0)
    assert info["next_cheaper"]["units"] == 5
    assert info["next_cheaper"]["units_now"] == 6


def test_edge_shave_and_real_reduction_are_distinguished():
    # 32 -> 31 GB gives up a gigabyte and keeps the job's headroom. 20 -> 15 GB
    # also saves a unit, but costs a quarter of the allocation -- presenting
    # both as the same offer is how a pricing tool causes an OOM.
    edge = billing.boundary(billing.Request(cpus=4, mem_gb=32), SHORT)["next_cheaper"]
    assert edge["kind"] == "edge_shave"
    assert edge["mem_given_up_gb"] == pytest.approx(1.0)

    real = billing.boundary(billing.Request(cpus=4, mem_gb=20), SHORT)["next_cheaper"]
    assert real["kind"] == "real_reduction"
    assert real["mem_given_up_gb"] == pytest.approx(5.0)
    assert "MAXIMUM RSS" in real["note"]


def test_partition_without_memory_billing():
    info = billing.boundary(billing.Request(cpus=4, mem_gb=32), billing.Weights(cpu=1.0, mem_per_gb=0.0))
    assert info["billed"] is False


class TestWeightTable:
    SCONTROL = (
        "PartitionName=short AllowGroups=ALL TRESBillingWeights=CPU=1.0,Mem=0.0625G MaxTime=12:00:00\n"
        "PartitionName=gpu_quad AllowGroups=ALL TRESBillingWeights=CPU=1.0,Mem=0.0625G,GRES/gpu=5.0"
        " TRES=cpu=400,mem=4000G,node=10,gres/gpu=40\n"
        "PartitionName=gpu_requeue TRESBillingWeights=CPU=0.1,Mem=0.00625G,GRES/gpu=0.1"
        " TRES=cpu=1080,mem=10000G,node=27,gres/gpu=108\n"
        "PartitionName=plain MaxTime=1:00:00\n"
    )

    def test_parses_each_partition(self):
        table = billing.parse_weight_table(self.SCONTROL)
        assert table["short"] == SHORT
        assert table["gpu_quad"] == GPU_QUAD
        assert table["gpu_requeue"] == GPU_REQUEUE

    def test_partition_without_weights_bills_cpu_only(self):
        table = billing.parse_weight_table(self.SCONTROL)
        assert table["plain"] == billing.Weights(cpu=1.0, mem_per_gb=0.0, gpu=0.0)

    def test_round_trips_through_the_cache(self):
        table = billing.parse_weight_table(self.SCONTROL)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "billing_weights.json")
            billing.save_weight_cache(table, captured_at=1000.0, path=path)
            loaded = billing.load_weight_cache(path)
            assert billing.cache_to_table(loaded) == table
            assert loaded["captured_at"] == 1000.0

    def test_default_path_is_resolved_at_call_time(self, monkeypatch):
        # Binding the default at import made the location impossible to
        # override and left open(None) reachable -- a TypeError the caller's
        # except clause does not catch.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weights.json")
            monkeypatch.setenv(billing.CACHE_PATH_ENV, path)
            assert billing.cache_path() == path
            assert billing.load_weight_cache() is None  # absent, not a crash
            billing.save_weight_cache(billing.parse_weight_table(self.SCONTROL), 5.0)
            assert billing.load_weight_cache()["captured_at"] == 5.0

    def test_explicit_path_beats_the_environment(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv(billing.CACHE_PATH_ENV, os.path.join(tmp, "env.json"))
            explicit = os.path.join(tmp, "explicit.json")
            assert billing.cache_path(explicit) == explicit

    def test_corrupt_cache_reads_as_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "billing_weights.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            assert billing.load_weight_cache(path) is None


class TestPrice:
    TABLE = {"short": SHORT, "gpu_quad": GPU_QUAD, "gpu_requeue": GPU_REQUEUE}

    def test_unknown_partition_refuses_rather_than_guessing(self):
        with pytest.raises(billing.BillingError):
            billing.price(billing.Request(cpus=1, mem_gb=4), self.TABLE, "nonexistent")

    def test_breakdown_reconciles_to_the_billed_units(self):
        payload = billing.price(billing.Request(cpus=4, mem_gb=6), self.TABLE, "short")
        b = payload["breakdown"]
        assert b["cpu"] + b["mem"] + b["gpu"] == pytest.approx(b["pre_floor"])
        assert b["pre_floor"] - b["floor_discards"] == pytest.approx(payload["billing_units"])

    def test_stale_cache_reports_its_age(self):
        payload = billing.price(billing.Request(cpus=1, mem_gb=4), self.TABLE, "short", captured_at=0.0, now=7200.0)
        assert payload["weights"]["age_hours"] == pytest.approx(2.0)

    def test_zero_units_is_flagged_not_sold_as_free(self):
        # Whether a site clamps a positive rate up to one unit is not visible in
        # scontrol, and no sub-1.0 shape has ever been observed billed.
        payload = billing.price(billing.Request(cpus=4, mem_gb=1, gpus=1), self.TABLE, "gpu_requeue")
        assert payload["billing_units"] == 0
        assert any("one-unit minimum" in w for w in payload["warnings"])

    def test_time_caveat_is_always_present(self):
        payload = billing.price(billing.Request(cpus=1, mem_gb=4), self.TABLE, "short")
        assert any("--time" in c for c in payload["caveats"])

    def test_alternatives_are_cheaper_and_ordered(self):
        request = billing.Request(cpus=4, mem_gb=6, gpus=1)
        rows = billing.alternatives(request, self.TABLE, "gpu_quad")
        assert rows and rows[0]["partition"] == "gpu_requeue"
        assert rows[0]["units"] < rows[0]["units_now"]

    def test_no_alternatives_when_already_cheapest(self):
        request = billing.Request(cpus=4, mem_gb=6, gpus=1)
        assert billing.alternatives(request, self.TABLE, "gpu_requeue") == []


class TestReviewFindings:
    """Each case is a defect the review of PR #26 identified, kept as a guard."""

    TABLE = {
        "short": SHORT,
        "gpu_requeue": GPU_REQUEUE,
        "defaulted": billing.Weights(cpu=1.0, mem_per_gb=0.0625, def_mem_per_cpu_gb=4.0),
    }

    def test_suffixed_memory_is_unaffected(self):
        assert billing.to_gb("32G") == pytest.approx(32.0)
        assert billing.to_gb("8192M") == pytest.approx(8.0)

    def test_boundaries_come_from_the_whole_weighted_sum(self):
        # With cpu=0.1/gpu=0.1 an 8-CPU/1-GPU request contributes 0.9, so the
        # price rises at 176 GB -- not at a multiple of 1/mem_per_gb = 160.
        # The old shortcut reported 159 GB and turned a 1 GB shave into 17 GB.
        req = billing.Request(cpus=8, mem_gb=176, gpus=1)
        info = billing.boundary(req, GPU_REQUEUE)
        assert billing.billing_units(req, GPU_REQUEUE) == 2
        assert info["on_price_edge"] is True
        assert info["next_cheaper"]["mem_gb"] == pytest.approx(175.0)
        assert info["next_cheaper"]["units"] == 1
        assert info["next_cheaper"]["kind"] == "edge_shave"

    def test_largest_same_price_respects_a_fractional_base(self):
        info = billing.boundary(billing.Request(cpus=8, mem_gb=160, gpus=1), GPU_REQUEUE)
        assert info["largest_same_price_mem_gb"] == pytest.approx(175.0)

    def test_integer_base_behaviour_is_unchanged(self):
        info = billing.boundary(billing.Request(cpus=4, mem_gb=32), SHORT)
        assert info["next_cheaper"]["mem_gb"] == pytest.approx(31.0)
        assert info["largest_same_price_mem_gb"] == pytest.approx(47.0)

    def test_def_mem_per_node_is_read_and_scaled(self):
        table = billing.parse_weight_table(
            "PartitionName=p TRESBillingWeights=CPU=1.0,Mem=0.0625G DefMemPerNode=8192\n"
        )
        assert table["p"].def_mem_per_node_gb == pytest.approx(8.0)
        assert table["p"].default_mem_gb(cpus=4, nodes=2) == pytest.approx(16.0)

    def test_zero_or_unset_defaults_are_not_treated_as_a_default(self):
        table = billing.parse_weight_table("PartitionName=p TRESBillingWeights=CPU=1.0,Mem=0.0625G DefMemPerCPU=0\n")
        assert table["p"].def_mem_per_cpu_gb is None


class TestSecondReviewFindings:
    """Guards for the second round on PR #26 -- four of the five were
    consequences of the first round's own fixes."""

    SC = (
        "PartitionName=short TRESBillingWeights=CPU=1.0,Mem=0.0625G DefMemPerCPU=4096"
        " TRES=cpu=400,mem=4000G,node=10\n"
        "PartitionName=cheap TRESBillingWeights=CPU=0.1,Mem=0.00625G DefMemPerCPU=4096"
        " TRES=cpu=400,mem=4000G,node=10\n"
        "PartitionName=bare TRESBillingWeights=CPU=1.0,Mem=0.0625G"
        " TRES=cpu=400,mem=4000G,node=10\n"
    )

    def table(self):
        return billing.parse_weight_table(self.SC)

    def test_cache_preserves_partition_memory_defaults(self):
        # The defaults were parsed but never serialised, so they survived only
        # the call that refreshed them -- and the cache is the primary path.
        table = self.table()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.json")
            billing.save_weight_cache(table, 1.0, path)
            restored = billing.cache_to_table(billing.load_weight_cache(path))
        assert restored["short"].def_mem_per_cpu_gb == pytest.approx(4.0)
        assert restored["bare"].def_mem_per_cpu_gb is None
        assert restored == table

    def test_alternatives_compare_the_resolved_allocation(self):
        # Comparing partitions with an unresolved request priced every
        # alternative as holding no memory at all.
        table = self.table()
        unresolved = billing.Request(cpus=4, mem_specified=False)
        resolved = billing.resolve_request(unresolved, table, "short")
        assert resolved.mem_gb == pytest.approx(16.0)
        assert billing.alternatives(resolved, table, "short")[0]["units_now"] == 5
        assert billing.alternatives(unresolved, table, "short")[0]["units_now"] == 4

    def test_resolution_is_refused_when_no_default_is_recorded(self):
        with pytest.raises(billing.BillingError, match="DefMemPerCPU"):
            billing.resolve_request(billing.Request(cpus=4, mem_specified=False), self.table(), "bare")


class TestThirdReviewFindings:
    """Round three on PR #26: sbatch semantics the parser had not covered."""


class TestShapeOnlySurface:
    """Round eight, the first after the parser was removed. Every finding is
    now about the shape interface itself rather than sbatch semantics."""

    PER_NODE = billing.parse_weight_table("PartitionName=n TRESBillingWeights=CPU=1.0,Mem=0.0625G DefMemPerNode=8192\n")
    PER_CPU = billing.parse_weight_table("PartitionName=c TRESBillingWeights=CPU=1.0,Mem=0.0625G DefMemPerCPU=4096\n")

    def test_per_node_default_requires_a_node_count(self):
        # DefMemPerNode scales with nodes; assuming one would underprice every
        # multi-node allocation by the whole per-node default.
        req = billing.Request(cpus=4, mem_specified=False)
        with pytest.raises(billing.BillingError, match="per NODE"):
            billing.resolve_request(req, self.PER_NODE, "n")

    def test_stated_node_count_resolves_the_per_node_default(self):
        req = billing.Request(cpus=4, mem_specified=False, nodes=3, nodes_stated=True)
        assert billing.resolve_request(req, self.PER_NODE, "n").mem_gb == pytest.approx(24.0)

    def test_per_cpu_default_needs_no_node_count(self):
        req = billing.Request(cpus=4, mem_specified=False)
        assert billing.resolve_request(req, self.PER_CPU, "c").mem_gb == pytest.approx(16.0)

    def test_explicit_zero_memory_is_refused(self):
        # sbatch reads a zero size as all memory on every allocated node, so it
        # is not the "no memory term" a caller might intend.
        req = billing.Request(cpus=4, mem_gb=0, mem_specified=True)
        with pytest.raises(billing.BillingError, match="zero is not an allocation"):
            billing.resolve_request(req, self.PER_CPU, "c")

    def test_omitted_memory_is_still_the_partition_default(self):
        req = billing.Request(cpus=4, mem_gb=0.0, mem_specified=False)
        assert billing.resolve_request(req, self.PER_CPU, "c").mem_gb == pytest.approx(16.0)

    def test_cache_override_without_a_directory_component(self):
        # os.makedirs("") raises; a bare filename is a legitimate override.
        import contextlib

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                billing.save_weight_cache(self.PER_CPU, 1.0, "bare.json")
                assert billing.cache_to_table(billing.load_weight_cache("bare.json")) == self.PER_CPU
            finally:
                with contextlib.suppress(OSError):
                    os.chdir(cwd)


class TestSuggestionFidelity:
    """Round nine: a suggestion has to survive being priced again."""

    FINE = billing.Weights(cpu=1.0, mem_per_gb=1024.0)  # Mem=1M: 1/1024 GB blocks

    def test_suggested_memory_reprices_to_the_quoted_units(self):
        # Rounding to a fixed precision moved the value back over the edge:
        # with a 1/1024 GB block the cheaper size is 0.99951171875 GB, and three
        # decimals returned 1.0 -- the request being priced down from.
        req = billing.Request(cpus=1, mem_gb=1.0)
        info = billing.boundary(req, self.FINE)
        nc = info["next_cheaper"]
        repriced = billing.billing_units(billing.Request(cpus=1, mem_gb=nc["mem_gb"]), self.FINE)
        assert repriced == nc["units"]
        assert repriced < billing.billing_units(req, self.FINE)

    def test_coarse_blocks_still_read_cleanly(self):
        info = billing.boundary(billing.Request(cpus=4, mem_gb=32), SHORT)
        assert info["next_cheaper"]["mem_gb"] == pytest.approx(31.0)
        assert info["largest_same_price_mem_gb"] == pytest.approx(47.0)

    def test_rounding_never_rounds_up(self):
        assert billing._round_below(0.99951171875) < 1.0
        assert billing._round_below(31.0) == pytest.approx(31.0)
        assert billing._round_below(0.0) == 0.0

    @pytest.mark.parametrize(
        "restriction,expected",
        [
            ("AllowGroups=ALL", True),
            ("AllowGroups=labonly", False),
            ("AllowAccounts=x", False),
            ("AllowQos=high", False),
            ("DenyQos=low", False),
            ("DenyAccounts=x", False),
        ],
    )
    def test_every_access_restriction_marks_a_partition_uncertain(self, restriction, expected):
        # Advertising a cheaper partition the caller will be rejected from
        # wastes a resubmission; this tool cannot evaluate membership, so any
        # restriction at all means "unknown", never "available".
        table = billing.parse_weight_table(f"PartitionName=p TRESBillingWeights=CPU=1.0,Mem=0.0625G {restriction}\n")
        assert table["p"].unrestricted is expected

    def test_restricted_partitions_stay_out_of_alternatives(self):
        table = billing.parse_weight_table(
            "PartitionName=here TRESBillingWeights=CPU=1.0,Mem=0.0625G AllowGroups=ALL\n"
            "PartitionName=qos TRESBillingWeights=CPU=0.1,Mem=0.00625G AllowQos=high\n"
            "PartitionName=deny TRESBillingWeights=CPU=0.1,Mem=0.00625G DenyAccounts=x\n"
        )
        assert billing.alternatives(billing.Request(cpus=8, mem_gb=32), table, "here") == []


class TestModelWeightsAndWholeCounts:
    """Round ten: a captured weight that was never consulted, and shapes Slurm
    cannot allocate."""

    TYPED = billing.parse_weight_table(
        "PartitionName=p TRESBillingWeights=CPU=1.0,Mem=0.0625G,GRES/gpu=5.0,GRES/gpu:a100=10.0\n"
    )
    PLAIN = billing.parse_weight_table("PartitionName=q TRESBillingWeights=CPU=1.0,Mem=0.0625G,GRES/gpu=5.0\n")

    def test_named_model_is_priced_at_its_own_weight(self):
        # The per-model weights were captured two rounds ago and never read, so
        # an A100 was charged the generic rate.
        generic = billing.Request(cpus=4, mem_gb=16, gpus=1)
        a100 = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="a100")
        assert billing.billing_units(generic, self.TYPED["p"]) == 10
        assert billing.billing_units(a100, self.TYPED["p"]) == 15

    def test_unnamed_model_uses_the_generic_weight(self):
        req = billing.Request(cpus=4, mem_gb=16, gpus=1)
        assert billing.billing_units(req, self.TYPED["p"]) == 10

    def test_model_reaches_the_breakdown_and_the_boundary(self):
        a100 = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="a100")
        payload = billing.price(a100, self.TYPED, "p")
        assert payload["breakdown"]["gpu"] == pytest.approx(10.0)
        # base = 4 CPU + 10 GPU, so the memory band starts one unit up from 14
        assert payload["boundary"]["on_price_edge"] is True

    def test_unpriced_model_is_refused_not_defaulted(self):
        req = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="h100")
        with pytest.raises(billing.BillingError, match="no weight for"):
            billing.resolve_request(req, self.TYPED, "p")

    def test_model_is_ignored_where_the_site_prices_gpus_uniformly(self):
        req = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="a100")
        assert billing.resolve_request(req, self.PLAIN, "q") is req
        assert billing.billing_units(req, self.PLAIN["q"]) == 10

    @pytest.mark.parametrize("field,value", [("cpus", 2.5), ("gpus", 0.5)])
    def test_fractional_counts_are_refused(self, field, value):
        # Slurm allocates whole CPUs and GPUs; rounding one silently would price
        # a job that cannot exist.
        kwargs = {"cpus": 4, "mem_gb": 16, field: value}
        with pytest.raises(billing.BillingError, match="whole number"):
            billing.resolve_request(billing.Request(**kwargs), self.PLAIN, "q")

    def test_whole_counts_expressed_as_floats_are_fine(self):
        req = billing.Request(cpus=4.0, mem_gb=16.0, gpus=1.0)
        assert billing.resolve_request(req, self.PLAIN, "q") is req

    def test_temp_cache_name_is_unique_per_writer(self):
        # Two threads refreshing at once would otherwise share a name and
        # interleave into one corrupt file.
        with open(billing.__file__, encoding="utf-8") as handle:
            assert "threading.get_ident()" in handle.read()


# A repriced shape must stay the SAME shape apart from its memory. Both paths
# below rebuilt the request from named fields and omitted gpu_model, so a
# model-priced GPU was silently repriced at the generic weight. The quoted
# figure then described an allocation the caller never asked about -- and it
# was cheaper, which is the direction that gets acted on.
A100 = billing.Weights(cpu=1.0, mem_per_gb=0.0625, gpu=5.0, gpu_by_model={"a100": 10.0})


def test_boundary_reprices_with_the_model_weight():
    req = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="a100")
    out = billing.boundary(req, A100)
    cheaper = out["next_cheaper"]
    assert cheaper["units_now"] == billing.billing_units(req, A100)
    assert cheaper["units"] == billing.billing_units(
        billing.Request(cpus=4, mem_gb=cheaper["mem_gb"], gpus=1, gpu_model="a100"),
        A100,
    )
    # The generic weight is 5 lower per GPU; a dropped model shows up here.
    assert cheaper["units"] > 4 + 5


def test_partition_default_memory_keeps_the_model():
    table = {
        "gpu": billing.Weights(
            cpu=1.0,
            mem_per_gb=0.0625,
            gpu=5.0,
            gpu_by_model={"a100": 10.0},
            def_mem_per_cpu_gb=4.0,
        )
    }
    req = billing.Request(cpus=2, gpus=1, gpu_model="a100", mem_specified=False)
    resolved = billing.resolve_request(req, table, "gpu")
    assert resolved.gpu_model == "a100"
    assert resolved.mem_gb == 8.0
    # 2 CPU + 8 GB*0.0625 + 1 a100*10 = 12.5 -> 12. Generic would give 7.
    assert billing.billing_units(resolved, table["gpu"]) == 12


def test_fractional_nodes_are_refused_like_cpus_and_gpus():
    # Slurm allocates whole nodes; 1.5 is not a shape it can produce, and
    # pricing it would answer confidently about a job that cannot exist.
    table = {"short": billing.Weights(cpu=1.0, mem_per_gb=0.0625)}
    req = billing.Request(cpus=2, mem_gb=8, nodes=1.5, nodes_stated=True)
    with pytest.raises(billing.BillingError, match="nodes"):
        billing.resolve_request(req, table, "short")


def test_unstated_node_default_is_not_read_as_a_claim():
    # nodes defaults to 1.0 and is whole anyway, but the guard must key on
    # nodes_stated so an unstated default never becomes a refusable "claim".
    table = {"short": billing.Weights(cpu=1.0, mem_per_gb=0.0625)}
    req = billing.Request(cpus=2, mem_gb=8)
    assert billing.resolve_request(req, table, "short").nodes == 1.0


def test_alternatives_never_offer_a_partition_that_cannot_price_the_model():
    # An A100 request must not be advertised on a partition that prices only
    # H100: gpu_weight_for() would quote it at the H100 rate, while pricing the
    # same request there directly is refused as unpriceable. A suggestion that
    # cannot be taken is worse than none.
    table = {
        "gpu_a": billing.Weights(
            cpu=1.0,
            mem_per_gb=0.0625,
            gpu=5.0,
            gpu_by_model={"a100": 10.0},
            stock={"cpu": 100, "mem": 1000, "node": 4, "gres/gpu": 8, "gres/gpu:a100": 8},
        ),
        "gpu_h": billing.Weights(
            cpu=0.5,
            mem_per_gb=0.03,
            gpu=1.0,
            gpu_by_model={"h100": 20.0},
            stock={"cpu": 100, "mem": 1000, "node": 4, "gres/gpu": 8, "gres/gpu:h100": 8},
        ),
        # Prices GPUs generically AND is known to hold a100s. Both halves
        # matter: the pricing filter must not exclude it, and the inventory
        # filter needs typed evidence before it may be offered.
        "gpu_generic": billing.Weights(
            cpu=0.5,
            mem_per_gb=0.03,
            gpu=1.0,
            stock={"cpu": 100, "mem": 1000, "node": 4, "gres/gpu": 8, "gres/gpu:a100": 8},
        ),
    }
    req = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="a100")
    offered = {r["partition"] for r in billing.alternatives(req, table, "gpu_a")}
    assert "gpu_h" not in offered
    # A partition with no per-model weight table still prices the request
    # generically, so the PRICING filter must not exclude it.
    assert "gpu_generic" in offered
    with pytest.raises(billing.BillingError):
        billing.resolve_request(req, table, "gpu_h")


def test_a_partition_priced_only_per_model_is_still_offered():
    # "GRES/gpu:a100=1" leaves the generic weight at zero while pricing an a100
    # perfectly well. Reading the generic entry hid such a partition as
    # GPU-less, which withholds a genuinely cheaper option.
    table = {
        "gpu_now": billing.Weights(
            cpu=1.0,
            mem_per_gb=0.0625,
            gpu=5.0,
            gpu_by_model={"a100": 10.0},
            stock={"cpu": 100, "mem": 1000, "node": 4, "gres/gpu": 8, "gres/gpu:a100": 8},
        ),
        "gpu_cheap": billing.Weights(
            cpu=1.0,
            mem_per_gb=0.0625,
            gpu=0.0,
            gpu_by_model={"a100": 1.0},
            stock={"cpu": 100, "mem": 1000, "node": 4, "gres/gpu": 8, "gres/gpu:a100": 8},
        ),
    }
    req = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="a100")
    offered = {r["partition"] for r in billing.alternatives(req, table, "gpu_now")}
    assert "gpu_cheap" in offered
    # And it must be priceable there, not merely advertised.
    assert billing.resolve_request(req, table, "gpu_cheap").gpu_model == "a100"


def test_the_reported_weight_is_the_one_the_charge_used():
    # A breakdown computed at the a100 rate beside a reported generic weight is
    # a response that disagrees with itself, and cannot be audited.
    table = {"gpu": billing.Weights(cpu=1.0, mem_per_gb=0.0625, gpu=5.0, gpu_by_model={"a100": 10.0})}
    req = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="a100")
    out = billing.price(req, table, "gpu")
    assert out["weights"]["gpu"] == 10.0
    assert out["weights"]["gpu_generic"] == 5.0
    assert out["weights"]["gpu_model"] == "a100"
    assert out["request"]["gpu_model"] == "a100"
    # The breakdown must reconcile against the weight that is reported.
    assert out["breakdown"]["gpu"] == out["weights"]["gpu"] * req.gpus


def test_untyped_gpus_are_refused_where_only_models_are_priced():
    # gpu_by_model={"a100": 10} with no generic entry means the generic weight
    # is 0, so an untyped GPU request would price every accelerator as free --
    # 2 units for 1 CPU + 16 GB + 1 GPU, with the GPU charge silently absent.
    table = {"gpu": billing.Weights(cpu=1.0, mem_per_gb=0.0625, gpu=0.0, gpu_by_model={"a100": 10.0})}
    req = billing.Request(cpus=1, mem_gb=16, gpus=1)
    with pytest.raises(billing.BillingError, match="only per model"):
        billing.resolve_request(req, table, "gpu")
    # Naming the model makes it priceable, at the model's rate.
    named = billing.Request(cpus=1, mem_gb=16, gpus=1, gpu_model="a100")
    assert billing.billing_units(billing.resolve_request(named, table, "gpu"), table["gpu"]) == 12


def test_a_declared_generic_rate_still_prices_an_untyped_request():
    # Models listed ALONGSIDE a real generic weight are not a refusal: the
    # generic rate is declared, so an untyped request has a true price.
    table = {"gpu": billing.Weights(cpu=1.0, mem_per_gb=0.0625, gpu=5.0, gpu_by_model={"a100": 10.0})}
    req = billing.Request(cpus=1, mem_gb=16, gpus=1)
    assert billing.billing_units(billing.resolve_request(req, table, "gpu"), table["gpu"]) == 7


def test_a_gpuless_request_is_unaffected_by_model_tables():
    # The guard keys on gpus > 0; a CPU job must never be refused because the
    # partition happens to list accelerator models.
    table = {"gpu": billing.Weights(cpu=1.0, mem_per_gb=0.0625, gpu=0.0, gpu_by_model={"a100": 10.0})}
    req = billing.Request(cpus=2, mem_gb=16)
    assert billing.resolve_request(req, table, "gpu").cpus == 2


def test_decimal_weights_land_on_the_boundary_they_were_written_as():
    # cpu=0.29 across 100 CPUs is 29 exactly as configured, but 28.999999999999996
    # in binary -- and the floor turns that into a whole unit lost, which then
    # propagates into the boundary figures and the alternatives.
    w = billing.Weights(cpu=0.29, mem_per_gb=0.0)
    assert billing.billing_units(billing.Request(cpus=100, mem_gb=0), w) == 29
    # A weight genuinely below the boundary must still floor down.
    w2 = billing.Weights(cpu=0.289, mem_per_gb=0.0)
    assert billing.billing_units(billing.Request(cpus=100, mem_gb=0), w2) == 28


def test_a_weighted_tres_we_cannot_charge_is_refused_not_dropped():
    # TRESBillingWeights is an open list. Dropping Node=10 prices a two-node
    # request 20 units under what Slurm bills, with nothing to show for it.
    table = billing.parse_weight_table("PartitionName=odd TRESBillingWeights=CPU=1,Node=10 State=UP AllowGroups=ALL")
    assert table["odd"].unpriceable_tres == {"node": 10.0}
    with pytest.raises(billing.BillingError, match="cannot charge"):
        billing.resolve_request(billing.Request(cpus=2, mem_gb=8), table, "odd")
    # And it must never be advertised as a cheaper destination either.
    full = dict(table)
    full["short"] = billing.Weights(cpu=1.0, mem_per_gb=0.0625)
    offered = {r["partition"] for r in billing.alternatives(billing.Request(cpus=2, mem_gb=8), full, "short")}
    assert "odd" not in offered


def test_per_model_gpu_weights_survive_the_cache():
    # The cache path is the DEFAULT path. gpu_by_model was written to it and
    # never read back, so an a100 was charged at the generic rate by every
    # caller that did not pass refresh_weights.
    import time as _time

    table = {"gpu": billing.Weights(cpu=1.0, mem_per_gb=0.0625, gpu=5.0, gpu_by_model={"a100": 10.0})}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.json")
        billing.save_weight_cache(table, _time.time(), path)
        payload = billing.load_weight_cache(path)
    back = billing.cache_to_table(payload)
    assert back["gpu"].gpu_by_model == {"a100": 10.0}
    req = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="a100")
    assert billing.billing_units(req, back["gpu"]) == billing.billing_units(req, table["gpu"])


def test_cpu_is_billed_only_when_the_table_says_so():
    # Slurm: "If TRESBillingWeights is not defined then the job's billing TRES
    # is equal to the total CPUs allocated." A table that exists but names no
    # CPU term therefore bills no CPUs -- seeding 1.0 regardless priced 32 CPUs
    # + 16 GB at 33 units where the site had configured 1.
    explicit = billing.parse_weight_table(
        "PartitionName=memonly TRESBillingWeights=Mem=0.0625G State=UP AllowGroups=ALL"
    )
    assert explicit["memonly"].cpu == 0.0
    assert billing.billing_units(billing.Request(cpus=32, mem_gb=16), explicit["memonly"]) == 1
    # With no table at all the CPU fallback still applies.
    absent = billing.parse_weight_table("PartitionName=plain State=UP AllowGroups=ALL")
    assert absent["plain"].cpu == 1.0
    assert billing.billing_units(billing.Request(cpus=32, mem_gb=16), absent["plain"]) == 32


def test_same_price_headroom_is_never_behind_the_current_request():
    # Blocks under 2 GB: at CPU=1, Mem=1G, 10.75 GB prices at 11 units and stays
    # there until 11 GB. Stepping half a block back from band_end reported 10.5
    # -- a "largest at this price" BELOW a request already held at that price.
    w = billing.Weights(cpu=1.0, mem_per_gb=1.0)
    req = billing.Request(cpus=1, mem_gb=10.75)
    out = billing.boundary(req, w)
    assert out["largest_same_price_mem_gb"] >= req.mem_gb
    assert out["free_headroom_gb"] >= 0.0
    # The reported value must genuinely still cost what the request costs.
    assert billing.billing_units(
        billing.Request(cpus=1, mem_gb=out["largest_same_price_mem_gb"]), w
    ) == billing.billing_units(req, w)


def test_headroom_below_a_band_edge_is_still_offered():
    # The guard must not flatten real headroom: a request low in its band still
    # has room to grow at the same price.
    w = billing.Weights(cpu=1.0, mem_per_gb=0.0625)
    req = billing.Request(cpus=4, mem_gb=17)
    out = billing.boundary(req, w)
    assert out["free_headroom_gb"] > 0
    assert billing.billing_units(
        billing.Request(cpus=4, mem_gb=out["largest_same_price_mem_gb"]), w
    ) == billing.billing_units(req, w)


def test_max_tres_is_refused_rather_than_summed():
    # Under PriorityFlags=MAX_TRES Slurm bills the MAXIMUM weighted TRES, not
    # the sum: CPU and memory contributions of 4 each are 8 by this module's
    # arithmetic and 4 by Slurm's. The two imply opposite advice about memory,
    # so a sum-shaped price here would be confidently wrong.
    assert billing.unsupported_billing_model({"priority_flags": ["MAX_TRES"]}) is not None
    assert "MAX_TRES" in billing.unsupported_billing_model({"priority_flags": ["MAX_TRES"]})
    assert billing.unsupported_billing_model({"priority_flags": ["NO_FAIR_TREE"]}) is None
    assert billing.unsupported_billing_model({"priority_flags": []}) is None


def test_uncaptured_priority_flags_are_refused_not_assumed():
    # A cache written before the flags were captured cannot say which model
    # applies. Assuming the sum is how a confident wrong number reaches an
    # approval, so it refuses and names the one command that fixes it.
    reason = billing.unsupported_billing_model({"captured_at": 1.0})
    assert reason is not None
    assert "o2_refresh_billing_weights" in reason


def test_priority_flags_are_parsed_from_scontrol_config():
    text = (
        "AccountingStorageType    = accounting_storage/slurmdbd\n"
        "PriorityFlags           = NO_FAIR_TREE,MAX_TRES\n"
        "PriorityType            = priority/multifactor\n"
    )
    assert billing.parse_priority_flags(text) == ["NO_FAIR_TREE", "MAX_TRES"]
    assert billing.parse_priority_flags("PriorityType = priority/basic\n") == []


def test_flags_round_trip_through_the_weight_cache():
    import time as _time

    table = {"short": billing.Weights(cpu=1.0, mem_per_gb=0.0625)}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.json")
        billing.save_weight_cache(table, _time.time(), path, priority_flags=["max_tres"])
        payload = billing.load_weight_cache(path)
    assert payload["priority_flags"] == ["MAX_TRES"]
    assert billing.unsupported_billing_model(payload) is not None


def test_same_price_headroom_is_preserved_above_the_fixed_step():
    # 10.75 GB at CPU=1, Mem=1G costs 11 units and stays there until 11 GB. A
    # fixed half-block margin lands behind the request; clamping to the request
    # merely reported zero headroom for capacity that genuinely exists.
    w = billing.Weights(cpu=1.0, mem_per_gb=1.0)
    req = billing.Request(cpus=1, mem_gb=10.75)
    out = billing.boundary(req, w)
    assert out["largest_same_price_mem_gb"] > req.mem_gb
    assert out["free_headroom_gb"] > 0
    assert billing.billing_units(
        billing.Request(cpus=1, mem_gb=out["largest_same_price_mem_gb"]), w
    ) == billing.billing_units(req, w)


def test_max_tres_gres_is_refused_like_max_tres():
    # Slurm documents MAX_TRES_GRES as the same maximum-based calculation with
    # GRES folded in. An exact "MAX_TRES" match let it through to the sum.
    reason = billing.unsupported_billing_model({"priority_flags": ["MAX_TRES_GRES"]})
    assert reason is not None
    assert "MAX_TRES_GRES" in reason


def test_petabyte_memory_weights_are_converted_not_dropped():
    # Slurm permits K, M, G, T and P. An unrecognised suffix left mem_per_gb at
    # zero, which prices memory as FREE rather than as the dominant term.
    table = billing.parse_weight_table("PartitionName=big TRESBillingWeights=CPU=1,Mem=1P State=UP AllowGroups=ALL")
    # 1 per PB is 1/1048576 per GB -- the same direction as T=1024.
    assert table["big"].mem_per_gb == pytest.approx(1.0 / (1024**2))
    assert not table["big"].unpriceable_tres


def test_an_unknown_memory_suffix_is_refused_not_treated_as_free():
    table = billing.parse_weight_table("PartitionName=odd TRESBillingWeights=CPU=1,Mem=1Z State=UP AllowGroups=ALL")
    assert table["odd"].unpriceable_tres
    with pytest.raises(billing.BillingError, match="cannot charge"):
        billing.resolve_request(billing.Request(cpus=2, mem_gb=8), table, "odd")


def test_gpu_alternatives_need_inventory_not_just_a_weight():
    # TRESBillingWeights says what a GPU COSTS; the partition's TRES inventory
    # says whether it holds one. A positive weight on a GPU-less partition
    # advertised a move that would be rejected on arrival.
    table = billing.parse_weight_table(
        "PartitionName=now TRESBillingWeights=CPU=1,Mem=0.0625G,GRES/gpu=5"
        " TRES=cpu=100,mem=1000G,node=4,gres/gpu=8 State=UP AllowGroups=ALL\n"
        "PartitionName=cheap_nogpu TRESBillingWeights=CPU=0.1,Mem=0.00625G,GRES/gpu=0.1"
        " TRES=cpu=100,mem=1000G,node=4 State=UP AllowGroups=ALL\n"
        "PartitionName=cheap_gpu TRESBillingWeights=CPU=0.1,Mem=0.00625G,GRES/gpu=0.1"
        " TRES=cpu=100,mem=1000G,node=4,gres/gpu=8 State=UP AllowGroups=ALL\n"
    )
    assert table["cheap_nogpu"].stock.get("gres/gpu", 0.0) == 0.0
    req = billing.Request(cpus=4, mem_gb=16, gpus=1)
    offered = {r["partition"] for r in billing.alternatives(req, table, "now")}
    assert "cheap_nogpu" not in offered
    assert "cheap_gpu" in offered
    # A CPU-only request is unaffected by any of this.
    cpu_only = billing.Request(cpus=4, mem_gb=16)
    assert "cheap_nogpu" in {r["partition"] for r in billing.alternatives(cpu_only, table, "now")}


def test_uncaptured_gpu_inventory_does_not_manufacture_a_suggestion():
    # "We never looked" is not support for a recommendation.
    table = {
        "now": billing.Weights(
            cpu=1.0,
            mem_per_gb=0.0625,
            gpu=5.0,
            stock={"cpu": 100, "mem": 1000, "node": 4, "gres/gpu": 8},
        ),
        "unknown": billing.Weights(cpu=0.1, mem_per_gb=0.00625, gpu=0.1),
    }
    req = billing.Request(cpus=4, mem_gb=16, gpus=1)
    assert "unknown" not in {r["partition"] for r in billing.alternatives(req, table, "now")}


def test_alternatives_check_cpu_memory_and_node_capacity():
    # Inventory answers "can it run here" for every resource, not just GPUs. A
    # one-node, 8-CPU, 32 GB partition was advertised for a 64-CPU/128-GB job
    # because only the GPU count was ever consulted.
    table = billing.parse_weight_table(
        "PartitionName=now TRESBillingWeights=CPU=1,Mem=0.0625G"
        " TRES=cpu=400,mem=4000G,node=10 State=UP AllowGroups=ALL\n"
        "PartitionName=tiny TRESBillingWeights=CPU=0.1,Mem=0.00625G"
        " TRES=cpu=8,mem=32G,node=1 State=UP AllowGroups=ALL\n"
        "PartitionName=roomy TRESBillingWeights=CPU=0.1,Mem=0.00625G"
        " TRES=cpu=400,mem=4000G,node=10 State=UP AllowGroups=ALL\n"
    )
    big = billing.Request(cpus=64, mem_gb=128, nodes=2, nodes_stated=True)
    offered = {r["partition"] for r in billing.alternatives(big, table, "now")}
    assert "tiny" not in offered
    assert "roomy" in offered
    # A shape the small partition CAN hold is still offered there.
    small = billing.Request(cpus=4, mem_gb=16)
    assert "tiny" in {r["partition"] for r in billing.alternatives(small, table, "now")}


def test_a_typed_gpu_request_needs_typed_inventory():
    # "gres/gpu=4" says four accelerators exist, not that any is an a100.
    table = billing.parse_weight_table(
        "PartitionName=now TRESBillingWeights=CPU=1,Mem=0.0625G,GRES/gpu=5"
        " TRES=cpu=100,mem=1000G,node=4,gres/gpu=8,gres/gpu:a100=8 State=UP AllowGroups=ALL\n"
        "PartitionName=untyped TRESBillingWeights=CPU=0.1,Mem=0.00625G,GRES/gpu=0.1"
        " TRES=cpu=100,mem=1000G,node=4,gres/gpu=4 State=UP AllowGroups=ALL\n"
        "PartitionName=typed TRESBillingWeights=CPU=0.1,Mem=0.00625G,GRES/gpu=0.1"
        " TRES=cpu=100,mem=1000G,node=4,gres/gpu=4,gres/gpu:a100=4 State=UP AllowGroups=ALL\n"
    )
    req = billing.Request(cpus=4, mem_gb=16, gpus=1, gpu_model="a100")
    offered = {r["partition"] for r in billing.alternatives(req, table, "now")}
    assert "untyped" not in offered
    assert "typed" in offered
    # An UNtyped request is happy with the aggregate count.
    any_gpu = billing.Request(cpus=4, mem_gb=16, gpus=1)
    assert "untyped" in {r["partition"] for r in billing.alternatives(any_gpu, table, "now")}


def test_a_reservation_only_partition_is_not_advertised():
    # ReqResv=YES means a plain submission is rejected, and the pricing input
    # carries no reservation -- the same "eligibility unknown" treatment the
    # QoS and account restrictions already get.
    table = billing.parse_weight_table(
        "PartitionName=now TRESBillingWeights=CPU=1,Mem=0.0625G"
        " TRES=cpu=400,mem=4000G,node=10 State=UP AllowGroups=ALL ReqResv=NO\n"
        "PartitionName=resv TRESBillingWeights=CPU=0.1,Mem=0.00625G"
        " TRES=cpu=400,mem=4000G,node=10 State=UP AllowGroups=ALL ReqResv=YES\n"
    )
    assert table["resv"].unrestricted is False
    assert table["now"].unrestricted is True
    req = billing.Request(cpus=4, mem_gb=16)
    assert "resv" not in {r["partition"] for r in billing.alternatives(req, table, "now")}
