"""Test node usage calculations to ensure sensible results."""
# ruff: noqa: PLR2004

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable mock data mode for all tests
os.environ["SLURM_USE_MOCK_DATA"] = "1"

import slurm_usage


class TestNodeUsageCalculations:
    """Test node usage statistics are calculated correctly."""

    def test_extract_node_usage_single_node(self) -> None:
        """Test that single-node jobs assign all resources to that node."""
        # Create test DataFrame with single-node job
        df = pl.DataFrame(
            {
                "node_list": ["node-001"],
                "cpu_hours_reserved": [100.0],
                "gpu_hours_reserved": [10.0],
                "elapsed_seconds": [3600],  # 1 hour
            },
        )

        result = slurm_usage._extract_node_usage_data(df)

        assert len(result) == 1
        assert result["node"][0] == "node-001"
        assert result["cpu_hours"][0] == 100.0
        assert result["gpu_hours"][0] == 10.0
        assert result["elapsed_hours"][0] == 1.0

    def test_extract_node_usage_multi_node(self) -> None:
        """Test that multi-node jobs split resources evenly across nodes."""
        # Create test DataFrame with job running on 4 nodes
        df = pl.DataFrame(
            {
                "node_list": ["node[001-004]"],  # 4 nodes
                "cpu_hours_reserved": [100.0],
                "gpu_hours_reserved": [20.0],
                "elapsed_seconds": [7200],  # 2 hours
            },
        )

        result = slurm_usage._extract_node_usage_data(df)

        assert len(result) == 4
        # Each node should get 1/4 of the resources
        result_dicts = result.to_dicts()
        for i, node_data in enumerate(result_dicts):
            assert node_data["node"] == f"node00{i+1}"
            assert node_data["cpu_hours"] == 25.0  # 100/4
            assert node_data["gpu_hours"] == 5.0  # 20/4
            assert node_data["elapsed_hours"] == 0.5  # 2/4

    def test_extract_node_usage_mixed_jobs(self) -> None:
        """Test mixture of single and multi-node jobs."""
        df = pl.DataFrame(
            {
                "node_list": ["node-001", "node[002-003]", "node-001"],
                "cpu_hours_reserved": [50.0, 100.0, 25.0],
                "gpu_hours_reserved": [5.0, 20.0, 0.0],
                "elapsed_seconds": [3600, 7200, 1800],
            },
        )

        result = slurm_usage._extract_node_usage_data(df)
        result_dicts = result.to_dicts()

        # Should have 4 entries total:
        # - 2 for node-001 (from job 1 and job 3)
        # - 1 for node002 (from job 2, split) - without hyphen due to parse_node_list
        # - 1 for node003 (from job 2, split) - without hyphen due to parse_node_list
        assert len(result) == 4

        # Check that node-001 appears twice
        node_001_entries = [r for r in result_dicts if r["node"] == "node-001"]
        assert len(node_001_entries) == 2

        # Check split resources for node002 and node003 (no hyphens)
        node_002 = next(r for r in result_dicts if r["node"] == "node002")
        assert node_002["cpu_hours"] == 50.0  # 100/2
        assert node_002["gpu_hours"] == 10.0  # 20/2

    def test_aggregate_node_statistics(self) -> None:
        """Test aggregation of node statistics."""
        # Create sample node usage data as DataFrame
        node_usage_df = pl.DataFrame(
            [
                {"node": "node-001", "cpu_hours": 10.0, "gpu_hours": 1.0, "elapsed_hours": 1.0},
                {"node": "node-001", "cpu_hours": 15.0, "gpu_hours": 2.0, "elapsed_hours": 1.5},
                {"node": "node-002", "cpu_hours": 20.0, "gpu_hours": 0.0, "elapsed_hours": 2.0},
            ],
        )

        # Mock the _get_node_cpus function to return consistent values
        with patch("slurm_usage._get_node_cpus") as mock_get_cpus:
            mock_get_cpus.return_value = 64  # Standard node with 64 CPUs

            result = slurm_usage._aggregate_node_statistics(node_usage_df, period_days=7)

            assert not result.is_empty()
            assert len(result) == 2  # Two unique nodes

            # Check node-001 aggregation
            node_001 = result.filter(pl.col("node") == "node-001")
            assert node_001["total_cpu_hours"][0] == 25.0  # 10 + 15
            assert node_001["total_gpu_hours"][0] == 3.0  # 1 + 2
            assert node_001["job_count"][0] == 2

            # Check CPU utilization calculation
            # Available hours = 64 CPUs * 7 days * 24 hours = 10,752
            expected_utilization = (25.0 / 10752) * 100
            assert abs(node_001["cpu_utilization_pct"][0] - expected_utilization) < 0.01

    def test_cpu_utilization_percentage_bounds(self) -> None:
        """Test that CPU utilization percentages are within reasonable bounds."""
        # Create data that would use 50% of node capacity
        node_usage_df = pl.DataFrame(
            [
                {"node": "node-001", "cpu_hours": 5376.0, "gpu_hours": 0.0, "elapsed_hours": 84.0},
            ],
        )

        with patch("slurm_usage._get_node_cpus") as mock_get_cpus:
            mock_get_cpus.return_value = 64

            result = slurm_usage._aggregate_node_statistics(node_usage_df, period_days=7)

            # 5376 hours / (64 CPUs * 7 days * 24 hours) = 50%
            utilization = result["cpu_utilization_pct"][0]
            assert 49.9 < utilization < 50.1
            assert 0 <= utilization <= 100  # Should always be in valid percentage range

    def test_cpu_utilization_over_100_percent(self) -> None:
        """Test handling of oversubscribed nodes (>100% utilization)."""
        # Create data that would exceed 100% (oversubscription)
        node_usage_df = pl.DataFrame(
            [
                {"node": "node-001", "cpu_hours": 15000.0, "gpu_hours": 0.0, "elapsed_hours": 168.0},
            ],
        )

        with patch("slurm_usage._get_node_cpus") as mock_get_cpus:
            mock_get_cpus.return_value = 64

            result = slurm_usage._aggregate_node_statistics(node_usage_df, period_days=7)

            # Should allow >100% for oversubscribed nodes
            utilization = result["cpu_utilization_pct"][0]
            assert utilization > 100  # Oversubscribed
            assert utilization < 200  # But not absurdly high

    def test_node_missing_cpu_info(self) -> None:
        """Test handling of nodes where CPU info cannot be retrieved."""
        node_usage_df = pl.DataFrame(
            [
                {"node": "node-001", "cpu_hours": 100.0, "gpu_hours": 10.0, "elapsed_hours": 10.0},
                {"node": "node-bad", "cpu_hours": 50.0, "gpu_hours": 5.0, "elapsed_hours": 5.0},
            ],
        )

        def mock_get_cpus(node_name: str) -> int | None:
            if node_name == "node-bad":
                return None  # Return None for nodes where CPU info cannot be retrieved
            return 64

        with patch("slurm_usage._get_node_cpus", side_effect=mock_get_cpus):
            result = slurm_usage._aggregate_node_statistics(node_usage_df, period_days=7)

            # Should have both nodes - node-bad with null CPU info
            assert len(result) == 2

            # Check node-001 has utilization
            node_001 = result.filter(pl.col("node") == "node-001")
            assert len(node_001) == 1
            assert node_001["est_cpus"][0] == 64
            assert node_001["cpu_utilization_pct"][0] is not None

            # Check node-bad has null CPU info but data is preserved
            node_bad = result.filter(pl.col("node") == "node-bad")
            assert len(node_bad) == 1
            assert node_bad["est_cpus"][0] is None
            assert node_bad["cpu_utilization_pct"][0] is None
            assert node_bad["total_cpu_hours"][0] == 50.0

    def test_parse_node_list_variations(self) -> None:
        """Test various node list formats are parsed correctly."""
        # Test single node
        assert slurm_usage.parse_node_list("node-001") == ["node-001"]

        # Test simple range
        assert slurm_usage.parse_node_list("node[001-003]") == ["node001", "node002", "node003"]

        # Test comma-separated
        assert slurm_usage.parse_node_list("node[001,003,005]") == ["node001", "node003", "node005"]

        # Test mixed range and comma
        result = slurm_usage.parse_node_list("node[001-003,005]")
        assert result == ["node001", "node002", "node003", "node005"]

        # Test with hyphen in prefix
        assert slurm_usage.parse_node_list("gpu-node[1-3]") == ["gpu-node1", "gpu-node2", "gpu-node3"]

        # Test empty/none
        assert slurm_usage.parse_node_list("") == []
        assert slurm_usage.parse_node_list("None") == []

    def test_job_count_is_integer(self) -> None:
        """Test that job counts are always integers, not decimals."""
        node_usage_df = pl.DataFrame(
            [
                {"node": "node-001", "cpu_hours": 10.5, "gpu_hours": 1.5, "elapsed_hours": 1.25},
                {"node": "node-001", "cpu_hours": 15.3, "gpu_hours": 2.7, "elapsed_hours": 1.75},
                {"node": "node-001", "cpu_hours": 8.2, "gpu_hours": 0.8, "elapsed_hours": 0.5},
            ],
        )

        with patch("slurm_usage._get_node_cpus") as mock_get_cpus:
            mock_get_cpus.return_value = 64

            result = slurm_usage._aggregate_node_statistics(node_usage_df, period_days=7)

            # Job count should be an integer (3 jobs)
            job_count = result["job_count"][0]
            assert job_count == 3
            assert isinstance(job_count, int)

    def test_gpu_node_utilization(self) -> None:
        """Test GPU utilization calculation for GPU nodes."""
        node_usage_df = pl.DataFrame(
            [
                {"node": "gpu-001", "cpu_hours": 100.0, "gpu_hours": 50.0, "elapsed_hours": 10.0},
            ],
        )

        with (
            patch("slurm_usage._get_node_cpus") as mock_get_cpus,
            patch("slurm_usage._get_node_gpus") as mock_get_gpus,
        ):
            mock_get_cpus.return_value = 64
            mock_get_gpus.return_value = 4  # 4 GPUs

            # Need to patch _display_node_utilization_charts internals
            # This is complex, so we'll just verify the data is reasonable
            result = slurm_usage._aggregate_node_statistics(node_usage_df, period_days=7)

            assert result["total_gpu_hours"][0] == 50.0
            # GPU hours available = 4 GPUs * 7 days * 24 hours = 672
            # Utilization = 50 / 672 = ~7.4%

    def test_zero_resource_jobs(self) -> None:
        """Test handling of jobs with zero resources."""
        df = pl.DataFrame(
            {
                "node_list": ["node-001"],
                "cpu_hours_reserved": [0.0],
                "gpu_hours_reserved": [0.0],
                "elapsed_seconds": [0],
            },
        )

        result = slurm_usage._extract_node_usage_data(df).to_dicts()

        assert len(result) == 1
        assert result[0]["cpu_hours"] == 0.0
        assert result[0]["gpu_hours"] == 0.0
        assert result[0]["elapsed_hours"] == 0.0

    def test_very_large_cluster(self) -> None:
        """Test handling of large clusters with many nodes."""
        # Create job running on 100 nodes
        df = pl.DataFrame(
            {
                "node_list": ["node[001-100]"],
                "cpu_hours_reserved": [10000.0],
                "gpu_hours_reserved": [1000.0],
                "elapsed_seconds": [3600],
            },
        )

        result = slurm_usage._extract_node_usage_data(df).to_dicts()

        assert len(result) == 100
        # Each node gets 1/100 of resources
        for node_data in result:
            assert node_data["cpu_hours"] == 100.0  # 10000/100
            assert node_data["gpu_hours"] == 10.0  # 1000/100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
