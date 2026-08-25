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


def test_request_on_a_block_edge_is_flagged_with_the_cheaper_value():
    info = billing.boundary(billing.Request(cpus=4, mem_gb=32), SHORT)
    assert info["on_block_edge"] is True
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
        assert billing.boundary(req, SHORT)["on_block_edge"] is True

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
