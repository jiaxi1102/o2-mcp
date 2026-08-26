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
GPU_QUAD = billing.Weights(cpu=1.0, mem_per_gb=0.0625, gpu=5.0)
GPU_REQUEUE = billing.Weights(cpu=0.1, mem_per_gb=0.00625, gpu=0.1)

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


class TestSbatchParsing:
    def test_absolute_mem_and_cpus_per_task(self):
        req = billing.parse_sbatch(
            "#!/bin/bash\n"
            "#SBATCH -p short\n"
            "#SBATCH --cpus-per-task=4\n"
            "#SBATCH --mem=31G\n"
            "#SBATCH --time=4:00:00\n"
        )
        assert req.cpus == 4
        assert req.mem_gb == pytest.approx(31.0)
        assert req.partition == "short"
        assert req.warnings == []

    def test_ntasks_multiplies_cpus_per_task(self):
        req = billing.parse_sbatch("#SBATCH --ntasks=2\n#SBATCH -c 4\n")
        assert req.cpus == 8

    def test_ntasks_alone_counts_as_cpus(self):
        req = billing.parse_sbatch("#SBATCH -n 3\n")
        assert req.cpus == 3

    def test_mem_per_cpu_resolves_and_warns(self):
        # This is the trap that silently reverts a boundary fix: it multiplies
        # out to a round total and lands back on an edge.
        req = billing.parse_sbatch("#SBATCH -c 4\n#SBATCH --mem-per-cpu=8G\n")
        assert req.mem_gb == pytest.approx(32.0)
        assert req.mem_source == "--mem-per-cpu"
        assert any("--mem-per-cpu" in w for w in req.warnings)
        assert billing.boundary(req, SHORT)["on_price_edge"] is True

    def test_absolute_mem_wins_over_mem_per_cpu(self):
        req = billing.parse_sbatch("#SBATCH -c 4\n#SBATCH --mem=31G\n" "#SBATCH --mem-per-cpu=8G\n")
        assert req.mem_gb == pytest.approx(31.0)

    def test_gres_gpu_is_read(self):
        req = billing.parse_sbatch("#SBATCH --gres=gpu:1\n#SBATCH -c 4\n" "#SBATCH --mem=6G\n")
        assert req.gpus == 1
        assert billing.billing_units(req, GPU_QUAD) == 9

    def test_typed_gres_gpu_is_read(self):
        req = billing.parse_sbatch("#SBATCH --gres=gpu:teslaV100:2\n")
        assert req.gpus == 2

    def test_later_directive_wins(self):
        req = billing.parse_sbatch("#SBATCH --mem=16G\n#SBATCH --mem=31G\n")
        assert req.mem_gb == pytest.approx(31.0)

    def test_time_is_ignored_entirely(self):
        # --time is not in the billing formula. Treating it as a cost lever is
        # the most expensive misconception about Slurm accounting.
        short = billing.parse_sbatch("#SBATCH -c 4\n#SBATCH --mem=31G\n#SBATCH -t 1:00:00\n")
        long = billing.parse_sbatch("#SBATCH -c 4\n#SBATCH --mem=31G\n#SBATCH -t 11:00:00\n")
        assert billing.billing_units(short, SHORT) == billing.billing_units(long, SHORT)


class TestWeightTable:
    SCONTROL = (
        "PartitionName=short AllowGroups=ALL TRESBillingWeights=CPU=1.0,Mem=0.0625G MaxTime=12:00:00\n"
        "PartitionName=gpu_quad AllowGroups=ALL TRESBillingWeights=CPU=1.0,Mem=0.0625G,GRES/gpu=5.0\n"
        "PartitionName=gpu_requeue TRESBillingWeights=CPU=0.1,Mem=0.00625G,GRES/gpu=0.1\n"
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
            billing.price(billing.Request(cpus=1), self.TABLE, "nonexistent")

    def test_breakdown_reconciles_to_the_billed_units(self):
        payload = billing.price(billing.Request(cpus=4, mem_gb=6), self.TABLE, "short")
        b = payload["breakdown"]
        assert b["cpu"] + b["mem"] + b["gpu"] == pytest.approx(b["pre_floor"])
        assert b["pre_floor"] - b["floor_discards"] == pytest.approx(payload["billing_units"])

    def test_stale_cache_reports_its_age(self):
        payload = billing.price(billing.Request(cpus=1), self.TABLE, "short", captured_at=0.0, now=7200.0)
        assert payload["weights"]["age_hours"] == pytest.approx(2.0)

    def test_zero_units_is_flagged_not_sold_as_free(self):
        # Whether a site clamps a positive rate up to one unit is not visible in
        # scontrol, and no sub-1.0 shape has ever been observed billed.
        payload = billing.price(billing.Request(cpus=4, mem_gb=1, gpus=1), self.TABLE, "gpu_requeue")
        assert payload["billing_units"] == 0
        assert any("one-unit minimum" in w for w in payload["warnings"])

    def test_time_caveat_is_always_present(self):
        payload = billing.price(billing.Request(cpus=1), self.TABLE, "short")
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

    def test_unsuffixed_memory_is_megabytes_not_bytes(self):
        # sbatch documents an unsuffixed --mem as MB. Reading 32000 as bytes
        # priced a 31.25 GB job at ~0.00003 GB and dropped the whole memory
        # charge along with every boundary derived from it.
        assert billing.to_gb("32000") == pytest.approx(31.25)
        req = billing.parse_sbatch("#SBATCH -c 4\n#SBATCH --mem=32000\n")
        assert req.mem_gb == pytest.approx(31.25)
        assert billing.billing_units(req, SHORT) == 5

    def test_suffixed_memory_is_unaffected(self):
        assert billing.to_gb("32G") == pytest.approx(32.0)
        assert billing.to_gb("8192M") == pytest.approx(8.0)

    def test_nodes_and_ntasks_per_node_reach_the_cpu_total(self):
        # 2 nodes x 4 tasks x 2 CPUs = 16, not the 2 that ignoring the first two
        # directives would give.
        req = billing.parse_sbatch("#SBATCH --nodes=2\n#SBATCH --ntasks-per-node=4\n#SBATCH --cpus-per-task=2\n")
        assert req.cpus == 16
        assert req.nodes == 2

    def test_explicit_ntasks_wins_over_per_node(self):
        req = billing.parse_sbatch("#SBATCH --nodes=2\n#SBATCH --ntasks-per-node=4\n#SBATCH --ntasks=3\n")
        assert req.cpus == 3

    def test_node_range_takes_the_guaranteed_minimum(self):
        req = billing.parse_sbatch("#SBATCH --nodes=2-8\n#SBATCH --ntasks-per-node=2\n")
        assert req.cpus == 4

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

    def test_partition_default_memory_is_applied(self):
        # Omitting --mem does not allocate zero memory; Slurm applies the
        # configured default and bills it.
        req = billing.parse_sbatch("#SBATCH -c 4\n")
        assert req.mem_specified is False
        payload = billing.price(req, self.TABLE, "defaulted")
        assert payload["request"]["mem_gb"] == pytest.approx(16.0)
        assert "partition default" in payload["request"]["mem_source"]
        assert payload["billing_units"] == 5

    def test_unknown_default_memory_is_refused_not_priced_as_zero(self):
        req = billing.parse_sbatch("#SBATCH -c 4\n")
        with pytest.raises(billing.BillingError, match="DefMemPerCPU"):
            billing.price(req, self.TABLE, "short")

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
        "PartitionName=short TRESBillingWeights=CPU=1.0,Mem=0.0625G DefMemPerCPU=4096\n"
        "PartitionName=cheap TRESBillingWeights=CPU=0.1,Mem=0.00625G DefMemPerCPU=4096\n"
        "PartitionName=bare TRESBillingWeights=CPU=1.0,Mem=0.0625G\n"
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

    def test_gres_and_gpus_per_node_scale_across_nodes(self):
        assert billing.parse_sbatch("#SBATCH --nodes=3\n#SBATCH --gres=gpu:2\n").gpus == 6
        assert billing.parse_sbatch("#SBATCH --nodes=3\n#SBATCH --gpus-per-node=2\n").gpus == 6

    def test_total_gpus_is_not_scaled(self):
        assert billing.parse_sbatch("#SBATCH --nodes=3\n#SBATCH --gpus=2\n").gpus == 2

    def test_node_range_is_refused_not_priced_at_its_minimum(self):
        req = billing.parse_sbatch("#SBATCH --nodes=2-8\n#SBATCH --mem=16G\n")
        assert req.nodes_is_range is True
        with pytest.raises(billing.BillingError, match="range"):
            billing.resolve_request(req, self.table(), "short")

    def test_fixed_node_count_still_prices(self):
        req = billing.parse_sbatch("#SBATCH --nodes=2\n#SBATCH --mem=16G\n")
        assert req.nodes_is_range is False
        assert billing.resolve_request(req, self.table(), "short") is req

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

    def test_exclusive_is_refused_not_priced_as_the_request(self):
        # --exclusive bills every TRES on the allocated nodes. Without node
        # topology the charge is unknowable, and `--nodes=2 --exclusive` would
        # otherwise be underpriced by nearly two whole nodes.
        req = billing.parse_sbatch("#SBATCH --nodes=2\n#SBATCH --exclusive\n#SBATCH --mem=16G\n")
        assert req.exclusive is True
        with pytest.raises(billing.BillingError, match="exclusive"):
            billing.resolve_request(req, {"p": SHORT}, "p")

    def test_gpus_per_task_multiplies_by_the_task_count(self):
        req = billing.parse_sbatch("#SBATCH --ntasks=4\n#SBATCH --gpus-per-task=1\n")
        assert req.gpus == 4

    def test_gpus_per_task_respects_nodes_and_per_node_tasks(self):
        req = billing.parse_sbatch("#SBATCH --nodes=2\n#SBATCH --ntasks-per-node=3\n#SBATCH --gpus-per-task=1\n")
        assert req.gpus == 6

    def test_directives_stop_at_the_first_executable_line(self):
        # Slurm stops reading #SBATCH at the first non-comment, non-blank line;
        # honouring a later one prices memory it will never request.
        req = billing.parse_sbatch("#!/bin/bash\n# a comment\n#SBATCH --mem=16G\n\nsrun ./work\n#SBATCH --mem=128G\n")
        assert req.mem_gb == pytest.approx(16.0)

    def test_shebang_and_comments_do_not_end_the_prologue(self):
        req = billing.parse_sbatch("#!/bin/bash\n\n# note\n#SBATCH --mem=31G\n")
        assert req.mem_gb == pytest.approx(31.0)

    def test_memory_weight_units_are_normalised_to_gb(self):
        # Mem=1M is one weight per MB, i.e. 1024 per GB. Dropping the suffix
        # underpriced memory by that factor.
        assert billing.parse_weight_table("PartitionName=p TRESBillingWeights=CPU=1.0,Mem=1M\n")[
            "p"
        ].mem_per_gb == pytest.approx(1024.0)
        assert billing.parse_weight_table("PartitionName=p TRESBillingWeights=CPU=1.0,Mem=0.0625G\n")[
            "p"
        ].mem_per_gb == pytest.approx(0.0625)
        assert billing.parse_weight_table("PartitionName=p TRESBillingWeights=CPU=1.0,Mem=1T\n")[
            "p"
        ].mem_per_gb == pytest.approx(1.0 / 1024)


class TestDeliberateOptionPass:
    """One pass over sbatch's resource options instead of one per review round.

    The contract this pins is not "every option is parsed" but "every option is
    either computed or refused" -- an unpriceable script is a complete answer, a
    wrongly-priced one is not.
    """

    TABLE = {"p": SHORT}

    # --- computed -----------------------------------------------------------

    def test_node_only_request_runs_one_task_per_node(self):
        assert billing.parse_sbatch("#SBATCH --nodes=4\n").cpus == 4

    def test_gres_list_with_other_resources(self):
        assert billing.parse_sbatch("#SBATCH --gres=gpu:2,scratch:100\n").gpus == 2
        assert billing.parse_sbatch("#SBATCH --gres=scratch:100,gpu:a100:4\n").gpus == 4
        assert billing.parse_sbatch("#SBATCH --gres=scratch:100\n").gpus == 0

    def test_mem_per_gpu_scales_with_gpus(self):
        req = billing.parse_sbatch("#SBATCH --gres=gpu:2\n#SBATCH --mem-per-gpu=8G\n")
        assert req.mem_gb == pytest.approx(16.0)
        assert req.mem_source == "--mem-per-gpu"

    def test_cpus_per_gpu_scales_with_gpus(self):
        assert billing.parse_sbatch("#SBATCH --gpus=4\n#SBATCH --cpus-per-gpu=3\n").cpus == 12

    def test_ntasks_per_gpu_drives_the_task_count(self):
        assert billing.parse_sbatch("#SBATCH --gpus=2\n#SBATCH --ntasks-per-gpu=3\n").cpus == 6

    def test_memory_precedence_mem_beats_per_cpu_beats_per_gpu(self):
        both = billing.parse_sbatch(
            "#SBATCH -c 4\n#SBATCH --gres=gpu:2\n"
            "#SBATCH --mem=31G\n#SBATCH --mem-per-cpu=8G\n#SBATCH --mem-per-gpu=8G\n"
        )
        assert both.mem_gb == pytest.approx(31.0)
        per_cpu = billing.parse_sbatch(
            "#SBATCH -c 4\n#SBATCH --gres=gpu:2\n#SBATCH --mem-per-cpu=2G\n#SBATCH --mem-per-gpu=8G\n"
        )
        assert per_cpu.mem_gb == pytest.approx(8.0)

    def test_array_is_recorded_because_each_task_bills_separately(self):
        assert billing.parse_sbatch("#SBATCH --array=1-100\n").array_spec == "1-100"

    # --- refused ------------------------------------------------------------

    @pytest.mark.parametrize(
        "directive,expected",
        [
            ("#SBATCH --exclusive\n", "--exclusive"),
            ("#SBATCH --ntasks-per-socket=2\n", "--ntasks-per-socket"),
            ("#SBATCH --gpus-per-socket=1\n", "--gpus-per-socket"),
            ("#SBATCH --sockets-per-node=2\n", "--sockets-per-node"),
            ("#SBATCH --cores-per-socket=8\n", "--cores-per-socket"),
            ("#SBATCH --threads-per-core=1\n", "--threads-per-core"),
            ("#SBATCH --overcommit\n", "--overcommit"),
            ("#SBATCH -O\n", "--overcommit"),
            ("#SBATCH -B 2:4:1\n", "--extra-node-info"),
        ],
    )
    def test_topology_dependent_options_are_refused(self, directive, expected):
        req = billing.parse_sbatch(directive + "#SBATCH --mem=16G\n")
        assert expected in [o for o, _ in req.unpriceable]
        with pytest.raises(billing.BillingError, match="cannot be priced"):
            billing.resolve_request(req, self.TABLE, "p")

    def test_an_ordinary_script_is_not_refused(self):
        req = billing.parse_sbatch("#SBATCH -c 4\n#SBATCH --mem=31G\n")
        assert req.unpriceable == []
        assert billing.resolve_request(req, self.TABLE, "p") is req

    def test_refusal_names_the_option_and_the_reason(self):
        req = billing.parse_sbatch("#SBATCH --exclusive\n#SBATCH --mem=16G\n")
        with pytest.raises(billing.BillingError) as exc:
            billing.resolve_request(req, self.TABLE, "p")
        assert "--exclusive" in str(exc.value)
        assert "cpus/mem_gb/gpus" in str(exc.value)

    # --- round-four remainder ----------------------------------------------

    def test_boundary_step_stays_inside_a_sub_gigabyte_block(self):
        # A fixed 1 GB shave would skip several price levels where a block is
        # a quarter of a gigabyte, reporting a far larger cut than needed.
        tiny = billing.Weights(cpu=1.0, mem_per_gb=4.0)  # 0.25 GB per unit
        info = billing.boundary(billing.Request(cpus=1, mem_gb=1.0), tiny)
        assert info["next_cheaper"]["mem_given_up_gb"] == pytest.approx(0.125)
        assert info["next_cheaper"]["units"] < billing.billing_units(billing.Request(cpus=1, mem_gb=1.0), tiny)

    def test_alternatives_exclude_ineligible_partitions(self):
        table = billing.parse_weight_table(
            "PartitionName=open TRESBillingWeights=CPU=1.0,Mem=0.0625G State=UP AllowGroups=ALL\n"
            "PartitionName=priv TRESBillingWeights=CPU=0.1,Mem=0.00625G State=UP AllowGroups=labonly\n"
            "PartitionName=down TRESBillingWeights=CPU=0.1,Mem=0.00625G State=DOWN AllowGroups=ALL\n"
            "PartitionName=acct TRESBillingWeights=CPU=0.1,Mem=0.00625G State=UP AllowAccounts=x\n"
        )
        assert table["priv"].unrestricted is False
        assert table["down"].state_up is False
        assert table["acct"].unrestricted is False
        assert billing.alternatives(billing.Request(cpus=8, mem_gb=32), table, "open") == []

    def test_eligibility_survives_the_cache(self):
        table = billing.parse_weight_table(
            "PartitionName=priv TRESBillingWeights=CPU=1.0,Mem=0.0625G AllowGroups=labonly\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.json")
            billing.save_weight_cache(table, 1.0, path)
            restored = billing.cache_to_table(billing.load_weight_cache(path))
        assert restored["priv"].unrestricted is False


class TestSpecialValuesAndConfigInteractions:
    """Round five: special option VALUES and partition-config interactions --
    a different class from the option-surface pass, and one the same
    parse-or-refuse framework absorbs."""

    CAPPED = billing.parse_weight_table("PartitionName=p TRESBillingWeights=CPU=1.0,Mem=0.0625G MaxMemPerCPU=8192\n")

    def test_zero_memory_is_refused_in_every_spelling(self):
        # Slurm reads a zero size as "all memory on every allocated node".
        for script in ("#SBATCH --mem=0\n", "#SBATCH --mem 0\n", "#SBATCH --mem=0G\n"):
            req = billing.parse_sbatch(script)
            assert [o for o, _ in req.unpriceable] == ["--mem=0"], script
            with pytest.raises(billing.BillingError, match="cannot be priced"):
                billing.resolve_request(req, self.CAPPED, "p")

    def test_a_real_memory_size_is_not_mistaken_for_zero(self):
        assert billing.parse_sbatch("#SBATCH --mem=10G\n").unpriceable == []

    def test_heterogeneous_components_are_refused_not_merged(self):
        # Folding components into one set of scalars prices neither: later
        # directives simply overwrite earlier ones.
        req = billing.parse_sbatch("#SBATCH -c 4\n#SBATCH --mem=16G\n#SBATCH hetjob\n#SBATCH --gres=gpu:1\n")
        assert [o for o, _ in req.unpriceable] == ["hetjob"]
        with pytest.raises(billing.BillingError, match="hetjob"):
            billing.resolve_request(req, self.CAPPED, "p")

    def test_max_mem_per_cpu_raises_the_billed_cpu_count(self):
        # Slurm preserves the memory per task and adds CPUs, so the CPU term is
        # billed at the raised count rather than the one written.
        req = billing.parse_sbatch("#SBATCH --ntasks=1\n#SBATCH --mem-per-cpu=64G\n")
        assert req.cpus == 1
        resolved = billing.resolve_request(req, self.CAPPED, "p")
        assert resolved.cpus == 8
        assert any("MaxMemPerCPU" in w for w in resolved.warnings)

    def test_request_under_the_cap_is_untouched(self):
        req = billing.parse_sbatch("#SBATCH --ntasks=2\n#SBATCH --mem-per-cpu=4G\n")
        assert billing.resolve_request(req, self.CAPPED, "p").cpus == 2

    def test_cap_scales_with_the_task_count(self):
        req = billing.parse_sbatch("#SBATCH --ntasks=3\n#SBATCH --mem-per-cpu=16G\n")
        assert billing.resolve_request(req, self.CAPPED, "p").cpus == 6

    def test_max_mem_per_cpu_survives_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.json")
            billing.save_weight_cache(self.CAPPED, 1.0, path)
            restored = billing.cache_to_table(billing.load_weight_cache(path))
        assert restored["p"].max_mem_per_cpu_gb == pytest.approx(8.0)

    def test_gpu_request_skips_partitions_that_do_not_bill_gpus(self):
        table = billing.parse_weight_table(
            "PartitionName=cpu TRESBillingWeights=CPU=0.1,Mem=0.00625G\n"
            "PartitionName=gpu TRESBillingWeights=CPU=1.0,Mem=0.0625G,GRES/gpu=5.0\n"
        )
        assert billing.alternatives(billing.Request(cpus=4, mem_gb=8, gpus=1), table, "gpu") == []
        # A CPU-only request may still legitimately move to the cheap partition.
        assert billing.alternatives(billing.Request(cpus=4, mem_gb=8), table, "gpu")[0]["partition"] == "cpu"

    def test_array_price_is_labelled_per_element(self):
        # Slurm creates a job record per index, so equal-runtime elements accrue
        # the quoted figure once each.
        payload = billing.price(
            billing.parse_sbatch("#SBATCH --array=1-100\n#SBATCH -c 2\n#SBATCH --mem=15G\n"),
            self.CAPPED,
            "p",
        )
        assert payload["array"]["spec"] == "1-100"
        assert "PER ELEMENT" in payload["array"]["note"]

    def test_non_array_result_carries_no_array_block(self):
        payload = billing.price(billing.parse_sbatch("#SBATCH -c 2\n#SBATCH --mem=15G\n"), self.CAPPED, "p")
        assert "array" not in payload
