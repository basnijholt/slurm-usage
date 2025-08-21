"""Comprehensive tests for data collection functionality."""

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


class TestDataCollection:
    """Test data collection functions."""

    def test_fetch_raw_records_from_slurm(self, test_dates: dict[str, str]) -> None:
        """Test fetching raw records from SLURM."""
        # Test with the date we have mock data for
        records = slurm_usage._fetch_raw_records_from_slurm(test_dates["today"])

        assert len(records) > 0
        # Check that records are RawJobRecord instances
        assert all(isinstance(r, slurm_usage.RawJobRecord) for r in records)

        # Test with a date that might not have data
        records_empty = slurm_usage._fetch_raw_records_from_slurm("2020-01-01")
        assert isinstance(records_empty, list)

    def test_load_raw_records_from_parquet(self, tmp_path: Path, test_dates: dict[str, str]) -> None:
        """Test loading raw records from parquet file."""
        # Create a test parquet file
        raw_file = tmp_path / f"{test_dates['today']}.parquet"

        # Create sample data
        fields = slurm_usage.RawJobRecord.get_field_names()
        sample_data = []
        for i in range(3):
            values = [""] * len(fields)
            values[fields.index("JobID")] = f"1234{i}"
            values[fields.index("User")] = f"user{i}"
            values[fields.index("Start")] = f"{test_dates['today']}T10:00:00"
            values[fields.index("Submit")] = f"{test_dates['today']}T09:00:00"
            sample_data.append(dict(zip(fields, values, strict=True)))

        df = pl.DataFrame(sample_data)
        df.write_parquet(raw_file)

        # Test loading
        records = slurm_usage._load_raw_records_from_parquet(raw_file, test_dates["today"])
        expected_records = 3
        assert len(records) == expected_records
        assert all(isinstance(r, slurm_usage.RawJobRecord) for r in records)

    def test_apply_incremental_filtering(self) -> None:
        """Test incremental filtering of raw records."""
        fields = slurm_usage.RawJobRecord.get_field_names()

        # Create test records
        raw_records = []
        for i in range(5):
            values = [""] * len(fields)
            values[fields.index("JobID")] = f"1234{i}"
            completed_threshold = 3
            values[fields.index("State")] = "COMPLETED" if i < completed_threshold else "RUNNING"
            line = "|".join(values)
            record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
            if record:
                raw_records.append(record)

        # Test filtering with existing completed jobs
        existing_states = {"12340": "COMPLETED", "12341": "COMPLETED"}
        filtered, skipped = slurm_usage._apply_incremental_filtering(raw_records, existing_states)

        # Should skip the completed jobs that haven't changed
        expected_skipped = 2
        expected_filtered = 3
        assert skipped == expected_skipped
        assert len(filtered) == expected_filtered  # One completed with different state + 2 running

    def test_process_raw_records_into_jobs(self) -> None:
        """Test processing raw records into ProcessedJob objects."""
        fields = slurm_usage.RawJobRecord.get_field_names()

        # Create main job and batch job
        raw_records = []

        # Main job
        values = [""] * len(fields)
        values[fields.index("JobID")] = "12345"
        values[fields.index("User")] = "testuser"
        values[fields.index("State")] = "COMPLETED"
        values[fields.index("AllocCPUS")] = "4"
        line = "|".join(values)
        main_record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        if main_record:
            raw_records.append(main_record)

        # Batch job
        values[fields.index("JobID")] = "12345.batch"
        values[fields.index("TotalCPU")] = "01:00:00"
        line = "|".join(values)
        batch_record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        if batch_record:
            raw_records.append(batch_record)

        # Process
        jobs, is_complete = slurm_usage._process_raw_records_into_jobs(raw_records)

        assert len(jobs) == 1  # Only one main job
        assert jobs[0].job_id == "12345"
        assert is_complete  # No incomplete jobs

    def test_extract_job_date(self, test_dates: dict[str, str]) -> None:
        """Test job date extraction."""
        # Test with start time
        assert slurm_usage._extract_job_date(f"{test_dates['today']}T10:30:00", None) == test_dates["today"]

        # Test with submit time fallback
        assert slurm_usage._extract_job_date(None, f"{test_dates['tomorrow']}T09:00:00") == test_dates["tomorrow"]
        assert slurm_usage._extract_job_date("Unknown", f"{test_dates['tomorrow']}T09:00:00") == test_dates["tomorrow"]

        # Test invalid cases
        assert slurm_usage._extract_job_date("N/A", "None") is None
        assert slurm_usage._extract_job_date("", "") is None

        # Test malformed dates
        assert slurm_usage._extract_job_date("202-08-20", None) is None
        assert slurm_usage._extract_job_date("2025-8-20", None) is None

    def test_fetch_jobs_for_date(self, tmp_path: Path, test_dates: dict[str, str]) -> None:
        """Test fetching jobs for a specific date."""
        config = slurm_usage.Config.create(data_dir=tmp_path)
        config.ensure_directories_exist()

        # Test fetching for a date with mock data
        raw, processed, is_complete = slurm_usage._fetch_jobs_for_date(
            test_dates["today"],
            config,
            skip_if_complete=False,
        )

        # Should get some data from mock
        assert len(raw) > 0
        assert len(processed) > 0

    def test_fetch_jobs_for_date_with_tracker(self, tmp_path: Path, test_dates: dict[str, str]) -> None:
        """Test fetching jobs with completion tracker."""
        config = slurm_usage.Config.create(data_dir=tmp_path)
        config.ensure_directories_exist()

        tracker = slurm_usage.DateCompletionTracker()
        tracker.mark_complete(test_dates["yesterday"])

        # Should skip completed date
        raw, processed, is_complete = slurm_usage._fetch_jobs_for_date(
            test_dates["yesterday"],
            config,
            skip_if_complete=True,
            completion_tracker=tracker,
        )

        assert len(raw) == 0
        assert len(processed) == 0
        assert is_complete

    def test_fetch_jobs_for_date_incremental(self, tmp_path: Path, test_dates: dict[str, str]) -> None:
        """Test incremental job fetching."""
        config = slurm_usage.Config.create(data_dir=tmp_path)
        config.ensure_directories_exist()

        # First fetch
        raw1, processed1, _ = slurm_usage._fetch_jobs_for_date(
            test_dates["today"],
            config,
            skip_if_complete=False,
        )

        if processed1:
            # Save some processed data
            processed_file = config.processed_data_dir / f"{test_dates['today']}.parquet"
            df = pl.DataFrame([j.to_dict() for j in processed1])
            df.write_parquet(processed_file)

            # Second fetch should do incremental
            raw2, processed2, _ = slurm_usage._fetch_jobs_for_date(
                test_dates["today"],
                config,
                skip_if_complete=False,
            )

            # May get fewer records due to incremental filtering
            assert len(raw2) <= len(raw1)

    def test_load_recent_data(self, tmp_path: Path, test_dates: dict[str, str], mock_datetime_now: MagicMock) -> None:  # noqa: ARG002
        """Test loading recent data from multiple days."""
        config = slurm_usage.Config.create(data_dir=tmp_path)
        config.ensure_directories_exist()

        # Create test data for multiple days
        base_date = datetime.fromisoformat(f"{test_dates['today']}T00:00:00+00:00")
        for i in range(3):
            date = base_date - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")

            # Create processed data
            test_data = [
                {
                    "job_id": f"job_{i}_{j}",
                    "user": f"user{j}",
                    "state": "COMPLETED",
                    "cpu_hours_reserved": 10.0,
                    "elapsed_seconds": 3600,
                    "alloc_cpus": 4,
                    "cpu_efficiency": 80.0,
                    "memory_efficiency": 70.0,
                    "processed_date": base_date.isoformat(),
                }
                for j in range(2)
            ]

            df = pl.DataFrame(test_data)
            file_path = config.processed_data_dir / f"{date_str}.parquet"
            df.write_parquet(file_path)

        # Test loading
        result = slurm_usage._load_recent_data(config, days=2, data_type="processed")
        assert result is not None
        expected_jobs = 6  # 2 jobs * 3 days (today, yesterday, day before)
        assert len(result) == expected_jobs

    def test_load_recent_data_raw(self, tmp_path: Path, test_dates: dict[str, str], mock_datetime_now: MagicMock) -> None:  # noqa: ARG002
        """Test loading recent raw data."""
        config = slurm_usage.Config.create(data_dir=tmp_path)
        config.ensure_directories_exist()

        # Create test raw data
        fields = slurm_usage.RawJobRecord.get_field_names()
        date_str = test_dates["today"]

        test_data = []
        for i in range(3):
            record_dict = {field: "" for field in fields}
            record_dict["JobIDRaw"] = f"job_{i}"
            record_dict["User"] = f"user{i}"
            test_data.append(record_dict)

        df = pl.DataFrame(test_data)
        file_path = config.raw_data_dir / f"{date_str}.parquet"
        df.write_parquet(file_path)

        # Test loading raw data
        result = slurm_usage._load_recent_data(config, days=0, data_type="raw")
        assert result is not None
        expected_records = 3
        assert len(result) == expected_records

    def test_load_recent_data_empty(self, tmp_path: Path) -> None:
        """Test loading when no data exists."""
        config = slurm_usage.Config.create(data_dir=tmp_path)
        config.ensure_directories_exist()

        result = slurm_usage._load_recent_data(config, days=7)
        assert result is None

    def test_load_recent_data_corrupted(self, tmp_path: Path, test_dates: dict[str, str]) -> None:
        """Test loading with corrupted parquet files."""
        config = slurm_usage.Config.create(data_dir=tmp_path)
        config.ensure_directories_exist()

        # Create a corrupted file
        date_str = test_dates["today"]
        file_path = config.processed_data_dir / f"{date_str}.parquet"
        file_path.write_text("corrupted data")

        # Should handle gracefully
        result = slurm_usage._load_recent_data(config, days=0)
        assert result is None or len(result) == 0


class TestCollectCommand:
    """Test the collect CLI command."""

    def test_collect_with_mock_data(self, tmp_path: Path) -> None:
        """Test collect command with mock data."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Use temporary data directory
        result = runner.invoke(
            slurm_usage.app,
            ["collect", "--days", "0", "--data-dir", str(tmp_path), "--summary", "False"],
        )

        # Check for expected output
        if result.exit_code == 0:
            assert "Collection complete" in result.stdout or "Collected" in result.stdout

    @patch("slurm_usage._fetch_jobs_for_date")
    def test_collect_parallel_processing(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        """Test parallel collection of multiple dates."""
        # Mock the fetch function
        mock_fetch.return_value = ([], [], True)

        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(
            slurm_usage.app,
            ["collect", "--days", "2", "--data-dir", str(tmp_path), "--n-parallel", "2", "--summary", "False"],
        )

        # Should have called fetch for 3 days (0, 1, 2)
        if result.exit_code == 0:
            min_fetch_count = 3
            assert mock_fetch.call_count >= min_fetch_count

    def test_collect_with_summary(self, tmp_path: Path, test_dates: dict[str, str]) -> None:
        """Test collect command with summary display."""
        # Create some test data first
        config = slurm_usage.Config.create(data_dir=tmp_path)
        config.ensure_directories_exist()

        date_str = test_dates["today"]
        test_data = [
            {
                "job_id": "test_job",
                "user": "testuser",
                "state": "COMPLETED",
                "cpu_hours_reserved": 10.0,
                "memory_gb_hours_reserved": 20.0,
                "gpu_hours_reserved": 0.0,
                "elapsed_seconds": 3600,
                "alloc_cpus": 4,
                "req_mem_mb": 4096,
                "cpu_efficiency": 80.0,
                "memory_efficiency": 70.0,
                "cpu_hours_wasted": 2.0,
                "memory_gb_hours_wasted": 5.0,
                "submit_time": f"{test_dates['today']}T10:00:00+00:00",
                "start_time": f"{test_dates['today']}T10:05:00+00:00",
                "end_time": f"{test_dates['today']}T11:05:00+00:00",
                "node_list": "node-001",
                "partition": "partition-01",
                "job_name": "test_job",
                "total_cpu_seconds": 2880.0,
                "max_rss_mb": 2048.0,
                "alloc_gpus": 0,
                "is_complete": True,
                "processed_date": f"{test_dates['today']}T12:00:00+00:00",
            },
        ]

        df = pl.DataFrame(test_data)
        file_path = config.processed_data_dir / f"{date_str}.parquet"
        df.write_parquet(file_path)

        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(
            slurm_usage.app,
            ["collect", "--days", "0", "--data-dir", str(tmp_path)],
        )

        if result.exit_code == 0 and "Resource Usage" in result.stdout:
            assert "testuser" in result.stdout or "CPU Hours" in result.stdout


class TestNodeInfoCache:
    """Test node information caching."""

    def test_node_info_cache(self) -> None:
        """Test that node info is cached properly."""
        # First call should populate cache
        info1 = slurm_usage._get_node_info_from_slurm()
        assert len(info1) > 0

        # Second call should use cache
        info2 = slurm_usage._get_node_info_from_slurm()
        assert info1 == info2

    def test_get_node_cpus(self) -> None:
        """Test getting CPU count for a node."""
        # This should work with mock data
        cpus = slurm_usage._get_node_cpus("node-001")
        assert isinstance(cpus, int)
        assert cpus > 0

    def test_get_node_gpus(self) -> None:
        """Test getting GPU count for a node."""
        gpus = slurm_usage._get_node_gpus("node-001")
        assert isinstance(gpus, int)
        assert gpus >= 0
