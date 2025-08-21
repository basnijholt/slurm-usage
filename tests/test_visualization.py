"""Tests for visualization and analysis functions."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable mock data mode for all tests
os.environ["SLURM_USE_MOCK_DATA"] = "1"

import slurm_usage

UTC = timezone.utc


class TestVisualizationFunctions:
    """Test visualization helper functions."""

    @patch("slurm_usage.console.print")
    def test_create_bar_chart(self, mock_print: MagicMock) -> None:
        """Test bar chart creation."""
        labels = ["user1", "user2", "user3"]
        values = [100.0, 75.0, 50.0]

        slurm_usage._create_bar_chart(
            labels,
            values,
            "Test Chart",
            width=50,
            top_n=10,
            unit="hours",
            show_percentage=True,
        )

        # Check that console.print was called
        assert mock_print.called

    @patch("slurm_usage.console.print")
    def test_create_bar_chart_empty(self, mock_print: MagicMock) -> None:
        """Test bar chart with empty data."""
        slurm_usage._create_bar_chart([], [], "Empty Chart")
        # Should handle empty data gracefully
        assert not mock_print.called or mock_print.call_count == 0

    @patch("slurm_usage.console.print")
    def test_create_bar_chart_with_zeros(self, mock_print: MagicMock) -> None:
        """Test bar chart filtering out zeros."""
        labels = ["user1", "user2", "user3"]
        values = [100.0, 0.0, 50.0]

        slurm_usage._create_bar_chart(labels, values, "Test Chart")
        # Should filter out the zero value
        assert mock_print.called


class TestNodeUsageAnalysis:
    """Test node usage analysis functions."""

    def test_extract_node_usage_data(self) -> None:
        """Test extracting node usage data from jobs."""
        # Create test data
        test_data = [
            {
                "node_list": "node-001",
                "cpu_hours_reserved": 10.0,
                "gpu_hours_reserved": 2.0,
                "elapsed_seconds": 3600,
            },
            {
                "node_list": "node-[002-003]",
                "cpu_hours_reserved": 20.0,
                "gpu_hours_reserved": 0.0,
                "elapsed_seconds": 7200,
            },
            {
                "node_list": "",  # Empty node list
                "cpu_hours_reserved": 5.0,
                "gpu_hours_reserved": 0.0,
                "elapsed_seconds": 1800,
            },
        ]

        df = pl.DataFrame(test_data)
        result = slurm_usage._extract_node_usage_data(df)
        result_dicts = result.to_dicts()

        # Should have 3 nodes (node-001, node-002, node-003)
        expected_nodes = 3
        assert len(result) == expected_nodes
        assert result_dicts[0]["node"] == "node-001"
        assert result_dicts[1]["node"] == "node-002"
        assert result_dicts[2]["node"] == "node-003"

    def test_extract_node_usage_data_empty(self) -> None:
        """Test extracting node usage data from empty DataFrame."""
        df = pl.DataFrame()
        result = slurm_usage._extract_node_usage_data(df)
        assert result.is_empty()

    def test_aggregate_node_statistics(self) -> None:
        """Test aggregating node statistics."""
        node_usage_df = pl.DataFrame(
            [
                {"node": "node-001", "cpu_hours": 10.0, "gpu_hours": 2.0, "elapsed_hours": 1.0},
                {"node": "node-001", "cpu_hours": 15.0, "gpu_hours": 0.0, "elapsed_hours": 1.5},
                {"node": "node-002", "cpu_hours": 20.0, "gpu_hours": 4.0, "elapsed_hours": 2.0},
            ],
        )

        with patch("slurm_usage._get_node_cpus", return_value=64):
            result = slurm_usage._aggregate_node_statistics(node_usage_df, period_days=1)

            assert not result.is_empty()
            expected_unique_nodes = 2
            assert len(result) == expected_unique_nodes  # Two unique nodes

            # Check aggregated values for node-001
            node1_stats = result.filter(pl.col("node") == "node-001")
            expected_cpu_hours = 25.0
            expected_job_count = 2
            assert node1_stats["total_cpu_hours"][0] == expected_cpu_hours
            assert node1_stats["job_count"][0] == expected_job_count

    def test_aggregate_node_statistics_empty(self) -> None:
        """Test aggregating empty node statistics."""
        result = slurm_usage._aggregate_node_statistics(pl.DataFrame(), period_days=1)
        assert result.is_empty()

    def test_calculate_analysis_period_days(self, test_dates: dict[str, str]) -> None:
        """Test calculating analysis period in days."""
        # Calculate dates relative to test_dates instead of hardcoding
        today = datetime.fromisoformat(f"{test_dates['today']}T00:00:00")
        tomorrow = today + timedelta(days=1)
        day_after_tomorrow = today + timedelta(days=2)

        test_data = [
            {
                "submit_time": datetime.fromisoformat(f"{test_dates['today']}T10:00:00"),
                "end_time": tomorrow,  # 1 day after today
            },
            {
                "submit_time": datetime.fromisoformat(f"{test_dates['tomorrow']}T10:00:00"),
                "end_time": day_after_tomorrow,  # 2 days after today
            },
        ]

        df = pl.DataFrame(test_data)
        days = slurm_usage._calculate_analysis_period_days(df)
        # Period from min(submit_time) = today to max(end_time) = day_after_tomorrow
        # That's 2 days: today and tomorrow
        expected_days = 2
        assert days == expected_days

    def test_calculate_analysis_period_days_default(self) -> None:
        """Test default analysis period when dates are missing."""
        # Create DataFrame with string columns (as they would be in real data)
        # but with None values
        test_data = {"submit_time": [None], "end_time": [None]}
        df = pl.DataFrame(test_data, schema={"submit_time": pl.Utf8, "end_time": pl.Utf8})
        days = slurm_usage._calculate_analysis_period_days(df)
        default_days = 7
        assert days == default_days  # Default value

    @patch("slurm_usage.console.print")
    def test_display_node_usage_table(self, mock_print: MagicMock) -> None:
        """Test displaying node usage table."""
        # The function expects columns in this order based on _aggregate_node_statistics output:
        # node, total_cpu_hours, total_gpu_hours, job_count, total_elapsed_hours, est_cpus, cpu_hours_available, cpu_utilization_pct
        test_data = {
            "node": ["node-001", "node-002"],
            "total_cpu_hours": [100.0, 50.0],
            "total_gpu_hours": [20.0, 0.0],
            "job_count": [10, 5],
            "total_elapsed_hours": [10.0, 5.0],
            "est_cpus": [64, 64],
            "cpu_hours_available": [1536.0, 1536.0],
            "cpu_utilization_pct": [6.5, 3.25],
        }

        node_stats = pl.DataFrame(test_data)
        slurm_usage._display_node_usage_table(node_stats)

        assert mock_print.called

    @patch("slurm_usage.console.print")
    def test_display_node_usage_table_empty(self, mock_print: MagicMock) -> None:
        """Test displaying empty node usage table."""
        node_stats = pl.DataFrame()
        slurm_usage._display_node_usage_table(node_stats)
        # Should handle empty data gracefully
        assert not mock_print.called

    @patch("slurm_usage.console.print")
    @patch("slurm_usage._create_bar_chart")
    def test_display_node_utilization_charts(
        self,
        mock_bar_chart: MagicMock,
        mock_print: MagicMock,
    ) -> None:
        """Test displaying node utilization charts."""
        test_data = {
            "node": ["node-001", "node-002"],
            "cpu_utilization_pct": [50.0, 75.0],
            "total_gpu_hours": [10.0, 20.0],
        }

        node_stats = pl.DataFrame(test_data)
        slurm_usage._display_node_utilization_charts(node_stats, period_days=7)

        # Should create CPU utilization chart
        assert mock_bar_chart.called
        assert mock_print.called

    @patch("slurm_usage.console.print")
    @patch("slurm_usage._extract_node_usage_data")
    @patch("slurm_usage._aggregate_node_statistics")
    @patch("slurm_usage._display_node_usage_table")
    @patch("slurm_usage._display_node_utilization_charts")
    def test_create_node_usage_stats(
        self,
        mock_charts: MagicMock,
        mock_table: MagicMock,
        mock_aggregate: MagicMock,
        mock_extract: MagicMock,
        mock_print: MagicMock,  # noqa: ARG002
        test_dates: dict[str, str],
    ) -> None:
        """Test the main node usage stats function."""
        # Setup mocks
        mock_extract.return_value = pl.DataFrame(
            [
                {"node": "node-001", "cpu_hours": 10.0, "gpu_hours": 0.0, "elapsed_hours": 1.0},
            ],
        )
        mock_aggregate.return_value = pl.DataFrame(
            {"node": ["node-001"], "total_cpu_hours": [10.0]},
        )

        # Create test DataFrame with datetime objects

        test_data = [
            {
                "node_list": "node-001",
                "cpu_hours_reserved": 10.0,
                "gpu_hours_reserved": 0.0,
                "elapsed_seconds": 3600,
                "submit_time": datetime.fromisoformat(f"{test_dates['today']}T10:00:00"),
                "end_time": datetime.fromisoformat(f"{test_dates['today']}T11:00:00"),
            },
        ]
        df = pl.DataFrame(test_data)

        slurm_usage._create_node_usage_stats(df)

        assert mock_extract.called
        assert mock_aggregate.called
        assert mock_table.called
        assert mock_charts.called

    @patch("slurm_usage.console.print")
    def test_create_node_usage_stats_empty(self, mock_print: MagicMock) -> None:  # noqa: ARG002
        """Test node usage stats with empty data."""
        df = pl.DataFrame()
        slurm_usage._create_node_usage_stats(df)
        # Should handle empty data gracefully


class TestSummaryStatistics:
    """Test summary statistics generation."""

    @patch("slurm_usage.console.print")
    @patch("slurm_usage._create_bar_chart")
    @patch("slurm_usage._create_node_usage_stats")
    def test_create_summary_stats(
        self,
        mock_node_stats: MagicMock,
        mock_bar_chart: MagicMock,
        mock_print: MagicMock,
        tmp_path: Path,
        test_dates: dict[str, str],
    ) -> None:
        """Test creating summary statistics."""
        config = slurm_usage.Config.create(data_dir=tmp_path)

        # Create test data
        test_data = [
            {
                "job_id": "job1",
                "user": "user1",
                "state": "COMPLETED",
                "cpu_hours_reserved": 10.0,
                "memory_gb_hours_reserved": 20.0,
                "gpu_hours_reserved": 0.0,
                "cpu_hours_wasted": 2.0,
                "memory_gb_hours_wasted": 5.0,
                "elapsed_seconds": 3600,
                "alloc_cpus": 4,
                "req_mem_mb": 4096,
                "cpu_efficiency": 80.0,
                "memory_efficiency": 70.0,
                "submit_time": datetime.fromisoformat(f"{test_dates['today']}T10:00:00"),
                "start_time": datetime.fromisoformat(f"{test_dates['today']}T10:05:00"),
                "end_time": datetime.fromisoformat(f"{test_dates['today']}T11:05:00"),
                "partition": "partition-01",
                "job_name": "test_job",
                "node_list": "node-001",
                "total_cpu_seconds": 2880.0,
                "max_rss_mb": 2048.0,
                "alloc_gpus": 0,
            },
        ]

        df = pl.DataFrame(test_data)
        slurm_usage._create_summary_stats(df, config)

        # Should call print and chart functions
        assert mock_print.called
        assert mock_bar_chart.called
        assert mock_node_stats.called

    @patch("slurm_usage.console.print")
    def test_create_summary_stats_empty(self, mock_print: MagicMock, tmp_path: Path) -> None:  # noqa: ARG002
        """Test summary stats with empty data."""
        config = slurm_usage.Config.create(data_dir=tmp_path)
        df = pl.DataFrame()
        slurm_usage._create_summary_stats(df, config)
        # Should handle empty data gracefully

    @patch("slurm_usage.console.print")
    def test_create_summary_stats_with_groups(self, mock_print: MagicMock, tmp_path: Path, test_dates: dict[str, str]) -> None:
        """Test summary stats with user groups."""
        config = slurm_usage.Config.create(
            data_dir=tmp_path,
            groups={"group1": ["user1", "user2"], "group2": ["user3"]},
        )

        # Create test data with multiple users
        test_data = []
        for i, user in enumerate(["user1", "user2", "user3", "user4"]):
            test_data.append(
                {
                    "job_id": f"job{i}",
                    "user": user,
                    "state": "COMPLETED",
                    "cpu_hours_reserved": 10.0 * (i + 1),
                    "memory_gb_hours_reserved": 20.0,
                    "gpu_hours_reserved": 5.0 if i == 2 else 0.0,  # noqa: PLR2004
                    "cpu_hours_wasted": 2.0,
                    "memory_gb_hours_wasted": 5.0,
                    "elapsed_seconds": 3600,
                    "alloc_cpus": 4,
                    "req_mem_mb": 4096,
                    "cpu_efficiency": 80.0 - i * 10,
                    "memory_efficiency": 70.0,
                    "submit_time": datetime.fromisoformat(f"{test_dates['today']}T10:00:00"),
                    "start_time": datetime.fromisoformat(f"{test_dates['today']}T10:05:00"),
                    "end_time": datetime.fromisoformat(f"{test_dates['today']}T11:05:00"),
                    "partition": "partition-01",
                    "job_name": f"job_{user}",
                    "node_list": "node-001",
                    "total_cpu_seconds": 2880.0,
                    "max_rss_mb": 2048.0,
                    "alloc_gpus": 0,
                },
            )

        df = pl.DataFrame(test_data)
        slurm_usage._create_summary_stats(df, config)

        # Should handle groups properly
        assert mock_print.called


class TestErrorHandling:
    """Test error handling in various functions."""

    def test_parse_node_list_error_handling(self) -> None:
        """Test node list parsing with invalid formats."""
        # Should handle malformed brackets
        result = slurm_usage.parse_node_list("node-[001-")
        assert len(result) >= 1

        # Should handle invalid ranges
        result = slurm_usage.parse_node_list("node-[abc-def]")
        assert len(result) >= 1

    def test_get_node_cpus_error(self) -> None:
        """Test error handling when node not found."""
        # Now returns None instead of raising exception
        result = slurm_usage._get_node_cpus("nonexistent-node")
        assert result is None

    def test_get_node_gpus_error(self) -> None:
        """Test error handling when node not found."""
        # Now returns None instead of raising exception
        result = slurm_usage._get_node_gpus("nonexistent-node")
        assert result is None

    @patch("slurm_usage.run_sacct")
    def test_fetch_raw_records_error(self, mock_sacct: MagicMock, test_dates: dict[str, str]) -> None:
        """Test error handling in fetch_raw_records."""
        # Simulate sacct failure
        mock_sacct.return_value = slurm_usage.CommandResult(
            stdout="",
            stderr="Error",
            returncode=1,
            command="sacct",
        )

        result = slurm_usage._fetch_raw_records_from_slurm(test_dates["today"])
        assert result == []

    def test_load_config_file_errors(self, tmp_path: Path) -> None:
        """Test config file loading with errors."""
        # Test with invalid YAML
        config_dir = tmp_path / ".config" / "slurm-usage"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("invalid: yaml: [")

        original_xdg = os.environ.get("XDG_CONFIG_HOME")
        original_home = os.environ.get("HOME")
        try:
            # Set HOME to tmp_path to prevent finding user's config
            os.environ["HOME"] = str(tmp_path)
            os.environ["XDG_CONFIG_HOME"] = str(tmp_path / ".config")

            # Create a mock for Path.exists that only returns True for our test file
            def mock_exists(self: Path) -> bool:
                # Only allow our test config file to exist
                return str(self) == str(config_file)

            with patch.object(Path, "exists", mock_exists):
                result, path = slurm_usage._load_config_file()
                assert result == {}  # Should return empty dict on error
                assert path is None  # No valid config path due to error
        finally:
            if original_xdg:
                os.environ["XDG_CONFIG_HOME"] = original_xdg
            else:
                os.environ.pop("XDG_CONFIG_HOME", None)
            if original_home:
                os.environ["HOME"] = original_home
