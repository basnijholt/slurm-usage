"""Test data processing and models."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable mock data mode for all tests
os.environ["SLURM_USE_MOCK_DATA"] = "1"

import slurm_usage


class TestRawJobRecord:
    """Test RawJobRecord model."""

    def test_field_names(self) -> None:
        """Test that field names are correctly defined."""
        fields = slurm_usage.RawJobRecord.get_field_names()

        assert "JobID" in fields
        assert "User" in fields
        assert "JobName" in fields
        assert "State" in fields
        assert "AllocCPUS" in fields
        assert "NodeList" in fields
        expected_field_count = 61
        assert len(fields) == expected_field_count  # Expected number of fields

    def test_from_sacct_line(self) -> None:
        """Test parsing a sacct output line."""
        fields = slurm_usage.RawJobRecord.get_field_names()

        # Create a sample line with pipe-separated values
        values = [""] * len(fields)
        values[fields.index("JobID")] = "12345"
        values[fields.index("JobIDRaw")] = "12345"
        values[fields.index("User")] = "alice"
        values[fields.index("JobName")] = "test_job"
        values[fields.index("State")] = "COMPLETED"
        values[fields.index("AllocCPUS")] = "4"

        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)

        assert record is not None
        assert record.JobID == "12345"
        assert record.User == "alice"
        assert record.JobName == "test_job"
        assert record.State == "COMPLETED"
        assert record.AllocCPUS == "4"

    def test_job_id_base(self) -> None:
        """Test job_id_base property."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)

        # Test regular job
        values[fields.index("JobID")] = "12345"
        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert record is not None
        assert record.job_id_base == "12345"

        # Test batch job
        values[fields.index("JobID")] = "12345.batch"
        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert record is not None
        assert record.job_id_base == "12345"

        # Test array job
        values[fields.index("JobID")] = "12345.0"
        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert record is not None
        assert record.job_id_base == "12345"

    def test_is_batch_step(self) -> None:
        """Test is_batch_step property."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)

        # Regular job
        values[fields.index("JobID")] = "12345"
        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert record is not None
        assert record.is_batch_step is False

        # Batch step
        values[fields.index("JobID")] = "12345.batch"
        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert record is not None
        assert record.is_batch_step is True

    def test_is_main_job(self) -> None:
        """Test is_main_job property."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)

        # Main job
        values[fields.index("JobID")] = "12345"
        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert record is not None
        assert record.is_main_job is True

        # Not main job (batch step)
        values[fields.index("JobID")] = "12345.batch"
        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert record is not None
        assert record.is_main_job is False

    def test_is_finished(self) -> None:
        """Test is_finished property."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)
        values[fields.index("JobID")] = "12345"

        # Finished states
        for state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
            values[fields.index("State")] = state
            line = "|".join(values)
            record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
            assert record is not None
            assert record.is_finished is True

        # Unfinished states
        for state in ["RUNNING", "PENDING", "SUSPENDED"]:
            values[fields.index("State")] = state
            line = "|".join(values)
            record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
            assert record is not None
            assert record.is_finished is False


class TestProcessedJob:
    """Test ProcessedJob model."""

    def test_from_raw_records(self, test_dates: dict[str, str]) -> None:
        """Test creating ProcessedJob from RawJobRecord."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)

        # Set required fields
        values[fields.index("JobID")] = "12345"
        values[fields.index("JobIDRaw")] = "12345"
        values[fields.index("User")] = "alice"
        values[fields.index("JobName")] = "test_job"
        values[fields.index("State")] = "COMPLETED"
        values[fields.index("Partition")] = "partition-01"
        values[fields.index("NodeList")] = "node-001"
        values[fields.index("AllocCPUS")] = "4"
        values[fields.index("ElapsedRaw")] = "3600"  # 1 hour
        values[fields.index("ReqMem")] = "4G"
        values[fields.index("MaxRSS")] = "2G"
        values[fields.index("TotalCPU")] = "01:30:00"  # 1.5 hours
        values[fields.index("Submit")] = f"{test_dates['today']}T10:00:00"
        values[fields.index("Start")] = f"{test_dates['today']}T10:05:00"
        values[fields.index("End")] = f"{test_dates['today']}T11:05:00"
        values[fields.index("AllocTRES")] = "cpu=4,mem=4G"

        line = "|".join(values)
        raw_record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert raw_record is not None

        processed = slurm_usage.ProcessedJob.from_raw_records(raw_record)

        assert processed.job_id == "12345"
        assert processed.user == "alice"
        assert processed.job_name == "test_job"
        assert processed.state == "COMPLETED"
        assert processed.partition == "partition-01"
        expected_cpus = 4
        expected_elapsed = 3600
        assert processed.alloc_cpus == expected_cpus
        assert processed.elapsed_seconds == expected_elapsed
        assert processed.is_complete is True

    def test_efficiency_calculations(self) -> None:
        """Test CPU and memory efficiency calculations."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)

        # Set up a job with known efficiency values
        values[fields.index("JobID")] = "12345"
        values[fields.index("User")] = "alice"
        values[fields.index("State")] = "COMPLETED"
        values[fields.index("AllocCPUS")] = "4"
        values[fields.index("ElapsedRaw")] = "3600"  # 1 hour
        values[fields.index("TotalCPU")] = "02:00:00"  # 2 CPU hours used
        values[fields.index("ReqMem")] = "4096M"
        values[fields.index("MaxRSS")] = "2048M"

        line = "|".join(values)
        raw_record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert raw_record is not None
        processed = slurm_usage.ProcessedJob.from_raw_records(raw_record)

        # CPU efficiency: 2 CPU hours used / (4 CPUs * 1 hour) = 50%
        assert processed.cpu_efficiency == pytest.approx(50.0, rel=0.1)

        # Memory efficiency: 2048M used / 4096M requested = 50%
        assert processed.memory_efficiency == pytest.approx(50.0, rel=0.1)

    def test_waste_calculations(self) -> None:
        """Test resource waste calculations."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)

        values[fields.index("JobID")] = "12345"
        values[fields.index("User")] = "alice"
        values[fields.index("State")] = "COMPLETED"
        values[fields.index("AllocCPUS")] = "4"
        values[fields.index("ElapsedRaw")] = "3600"  # 1 hour
        values[fields.index("TotalCPU")] = "01:00:00"  # 1 CPU hour used
        values[fields.index("ReqMem")] = "4096M"
        values[fields.index("MaxRSS")] = "1024M"

        line = "|".join(values)
        raw_record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert raw_record is not None
        processed = slurm_usage.ProcessedJob.from_raw_records(raw_record)

        # CPU waste: (4 CPUs * 1 hour) - 1 CPU hour = 3 CPU hours
        assert processed.cpu_hours_wasted == pytest.approx(3.0, rel=0.1)

        # Memory waste: (4096M - 1024M) * 1 hour / 1024 = ~3 GB-hours
        assert processed.memory_gb_hours_wasted == pytest.approx(3.0, rel=0.1)


class TestParsers:
    """Test parsing utility functions."""

    def test_parse_memory_mb(self) -> None:
        """Test memory parsing to MB."""
        mb_in_gb = 1024.0
        assert slurm_usage._parse_memory_mb("1024M") == mb_in_gb
        assert slurm_usage._parse_memory_mb("1G") == mb_in_gb
        expected_mb = 2.0
        assert slurm_usage._parse_memory_mb("2048K") == expected_mb
        mb_in_tb = 1024 * 1024.0
        assert slurm_usage._parse_memory_mb("1T") == mb_in_tb
        assert slurm_usage._parse_memory_mb("") == 0.0
        assert slurm_usage._parse_memory_mb("N/A") == 0.0

    def test_parse_cpu_seconds(self) -> None:
        """Test CPU time parsing to seconds."""
        one_hour = 3600.0
        half_hour = 1800.0
        one_day = 86400.0
        assert slurm_usage._parse_cpu_seconds("01:00:00") == one_hour
        assert slurm_usage._parse_cpu_seconds("00:30:00") == half_hour
        assert slurm_usage._parse_cpu_seconds("1-00:00:00") == one_day
        assert slurm_usage._parse_cpu_seconds("") == 0.0
        assert slurm_usage._parse_cpu_seconds("INVALID") == 0.0

    def test_parse_int(self) -> None:
        """Test integer parsing."""
        expected_value = 123
        assert slurm_usage._parse_int("123") == expected_value
        assert slurm_usage._parse_int("0") == 0
        assert slurm_usage._parse_int("") == 0
        assert slurm_usage._parse_int("abc") == 0

    def test_parse_gpu_count(self) -> None:
        """Test GPU count parsing from AllocTRES."""
        expected_gpus = 2
        assert slurm_usage._parse_gpu_count("cpu=4,mem=8G,gres/gpu=2") == expected_gpus
        assert slurm_usage._parse_gpu_count("cpu=4,mem=8G") == 0
        assert slurm_usage._parse_gpu_count("") == 0
        single_gpu = 1
        assert slurm_usage._parse_gpu_count("gres/gpu=1") == single_gpu


class TestNodeListParsing:
    """Test node list parsing."""

    def test_parse_single_node(self) -> None:
        """Test parsing single node."""
        assert slurm_usage.parse_node_list("node-001") == ["node-001"]
        assert slurm_usage.parse_node_list("compute-01") == ["compute-01"]

    def test_parse_node_range(self) -> None:
        """Test parsing node range."""
        result = slurm_usage.parse_node_list("node-[001-003]")
        assert result == ["node-001", "node-002", "node-003"]

        result = slurm_usage.parse_node_list("node-[1-3]")
        assert result == ["node-1", "node-2", "node-3"]

    def test_parse_node_list_with_commas(self) -> None:
        """Test parsing node list with commas."""
        result = slurm_usage.parse_node_list("node-[001,003,005]")
        assert result == ["node-001", "node-003", "node-005"]

    def test_parse_complex_node_list(self) -> None:
        """Test parsing complex node list."""
        result = slurm_usage.parse_node_list("node-[001-003,005,007-009]")
        expected = ["node-001", "node-002", "node-003", "node-005", "node-007", "node-008", "node-009"]
        assert result == expected

    def test_parse_empty_node_list(self) -> None:
        """Test parsing empty or invalid node list."""
        assert slurm_usage.parse_node_list("") == []
        assert slurm_usage.parse_node_list("None") == []
        assert slurm_usage.parse_node_list("N/A") == []


class TestSqueueParsing:
    """Test squeue output parsing."""

    def test_slurm_job_from_line(self) -> None:
        """Test creating SlurmJob from squeue line."""
        line = "alice/R/1/partition-01/4/node-001/OK"
        job = slurm_usage.SlurmJob.from_line(line)

        assert job.user == "alice"
        assert job.status == "R"
        assert job.nnodes == 1
        assert job.partition == "partition-01"
        expected_cores = 4
        assert job.cores == expected_cores
        assert job.node == "node-001"
        assert job.oversubscribe == "OK"

    def test_squeue_output(self) -> None:
        """Test parsing full squeue output."""
        jobs = slurm_usage.squeue_output()

        assert len(jobs) > 0

        # Check first job
        if jobs:
            job = jobs[0]
            assert job.user != ""
            assert job.status in ["R", "PD"]
            assert job.partition != ""


class TestGresParsingWithSocket:
    """Test GRES parsing with socket information (PR #12)."""

    def test_parse_gres_with_socket_info(self) -> None:
        """Test parsing GRES string with socket information like 'gpu:8(S:0-1)'."""
        # Test case from PR #12 - GPU count with socket information
        gres_with_socket = "gpu:8(S:0-1)"
        expected_gpu_count = 8
        # Remove socket information using the regex from PR #12
        cleaned_gres = re.sub(r"\(S:[0-9-]+\)", "", gres_with_socket)
        gpu_parts = cleaned_gres.split(":")

        # Verify the GPU count is parsed correctly
        assert gpu_parts[-1] == str(expected_gpu_count)
        assert int(gpu_parts[-1]) == expected_gpu_count

    def test_parse_gres_without_socket(self) -> None:
        """Test that regular GRES strings still work."""
        # Regular format without socket info
        gres_regular = "gpu:4"
        expected_gpu_count = 4
        cleaned_gres = re.sub(r"\(S:[0-9-]+\)", "", gres_regular)
        gpu_parts = cleaned_gres.split(":")

        assert gpu_parts[-1] == str(expected_gpu_count)
        assert int(gpu_parts[-1]) == expected_gpu_count

    def test_parse_gres_with_model_and_socket(self) -> None:
        """Test GRES with GPU model and socket info."""
        # Format with GPU model and socket info
        gres_with_model = "gpu:v100:8(S:0-1)"
        expected_gpu_count = 8
        cleaned_gres = re.sub(r"\(S:[0-9-]+\)", "", gres_with_model)
        gpu_parts = cleaned_gres.split(":")

        # The GPU count is still the last part
        assert gpu_parts[-1] == str(expected_gpu_count)
        assert int(gpu_parts[-1]) == expected_gpu_count
        assert gpu_parts[1] == "v100"  # Model name preserved

    def test_parse_gres_multiple_sockets(self) -> None:
        """Test GRES with different socket patterns."""
        # Different socket patterns that might appear
        test_cases = [
            ("gpu:16(S:0-3)", 16),
            ("gpu:a100:4(S:0)", 4),
            ("gpu:2(S:1)", 2),
            ("gpu:v100:32(S:0-7)", 32),
        ]

        for gres, expected_count in test_cases:
            cleaned_gres = re.sub(r"\(S:[0-9-]+\)", "", gres)
            gpu_parts = cleaned_gres.split(":")
            assert int(gpu_parts[-1]) == expected_count


class TestDatetimeSchemaConsistency:
    """Test datetime schema consistency when saving and loading data."""

    def test_ensure_utc_datetimes(self) -> None:
        """Test that _ensure_utc_datetimes converts all datetime columns to UTC."""
        # Create DataFrame with mixed datetime schemas
        df = pl.DataFrame(
            {
                "job_id": ["job1", "job2"],
                "naive_datetime": [
                    datetime(2025, 9, 20, 10, 0, 0),
                    datetime(2025, 9, 20, 11, 0, 0),
                ],
                "utc_datetime": [
                    datetime(2025, 9, 20, 10, 0, 0, tzinfo=timezone.utc),
                    datetime(2025, 9, 20, 11, 0, 0, tzinfo=timezone.utc),
                ],
                "value": [1.0, 2.0],
            }
        )

        # Apply the function
        result = slurm_usage._ensure_utc_datetimes(df)

        # Check all datetime columns are UTC
        for col in ["naive_datetime", "utc_datetime"]:
            col_type = result[col].dtype
            assert isinstance(col_type, pl.Datetime)
            assert col_type.time_zone == "UTC", f"Column {col} should be UTC"

    def test_parse_datetime_returns_utc(self) -> None:
        """Test that _parse_datetime returns UTC timezone-aware datetimes."""
        # Test with ISO format string
        dt = slurm_usage._parse_datetime("2025-09-20T10:30:00")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc

        # Test with None
        assert slurm_usage._parse_datetime(None) is None
        assert slurm_usage._parse_datetime("Unknown") is None

    def test_processed_jobs_to_dataframe_utc(self) -> None:
        """Test that processed jobs are saved with UTC datetimes."""
        # Create test ProcessedJob with datetime fields
        from slurm_usage import ProcessedJob

        job = ProcessedJob(
            job_id="test123",
            user="alice",
            job_name="test_job",
            partition="gpus",
            state="COMPLETED",
            submit_time=datetime(2025, 9, 20, 9, 0, 0),  # Naive
            start_time=datetime(2025, 9, 20, 10, 0, 0, tzinfo=timezone.utc),  # UTC
            end_time=datetime(2025, 9, 20, 11, 0, 0),  # Naive
            node_list="node-001",
            elapsed_seconds=3600,
            alloc_cpus=4,
            req_mem_mb=4096,
            max_rss_mb=2048,
            total_cpu_seconds=7200,
            alloc_gpus=1,
            cpu_efficiency=50.0,
            memory_efficiency=50.0,
            cpu_hours_wasted=1.0,
            memory_gb_hours_wasted=2.0,
            cpu_hours_reserved=2.0,
            memory_gb_hours_reserved=4.0,
            gpu_hours_reserved=1.0,
            is_complete=True,
        )

        # Convert to DataFrame
        df = slurm_usage._processed_jobs_to_dataframe([job])

        # Check that all datetime columns are UTC
        for col in ["submit_time", "start_time", "end_time", "processed_date"]:
            if col in df.columns:
                col_type = df[col].dtype
                if isinstance(col_type, pl.Datetime):
                    assert col_type.time_zone == "UTC", f"Column {col} should be UTC"

    def test_load_recent_data_handles_empty_directory(self) -> None:
        """Test that _load_recent_data handles empty directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = slurm_usage.Config(
                data_dir=Path(tmpdir),
                groups={},
                user_to_group={},
            )

            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Should return None for empty directory
            result = slurm_usage._load_recent_data(config, days=1)
            assert result is None

    def test_diagonal_concat_handles_schema_differences(self) -> None:
        """Test that diagonal_relaxed concat handles schema differences gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = slurm_usage.Config(
                data_dir=Path(tmpdir),
                groups={},
                user_to_group={},
            )

            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            # DataFrame with different columns
            df1 = pl.DataFrame(
                {
                    "job_id": ["job1"],
                    "user": ["alice"],
                    "cpu_hours_used": [1.0],
                    "processed_date": [datetime(2025, 9, 20, 10, 0, 0)],
                    "is_complete": [True],
                }
            )

            # DataFrame with additional column
            df2 = pl.DataFrame(
                {
                    "job_id": ["job2"],
                    "user": ["bob"],
                    "cpu_hours_used": [2.0],
                    "gpu_hours_used": [0.5],  # Additional column
                    "processed_date": [datetime(2025, 9, 21, 10, 0, 0)],
                    "is_complete": [True],
                }
            )

            df1.write_parquet(processed_dir / "2025-09-20.parquet")
            df2.write_parquet(processed_dir / "2025-09-21.parquet")

            # Should handle schema differences with diagonal_relaxed
            result = slurm_usage._load_recent_data(config, days=2)

            assert result is not None
            assert len(result) == 2
            # The gpu_hours_used column should exist with null for first row
            assert "gpu_hours_used" in result.columns

    def test_multiple_datetime_columns_converted(self) -> None:
        """Test that all datetime columns are converted to UTC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = slurm_usage.Config(
                data_dir=Path(tmpdir),
                groups={},
                user_to_group={},
            )

            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Use today's date for the filename
            from datetime import date

            today = date.today()

            # DataFrame with multiple datetime columns
            df = pl.DataFrame(
                {
                    "job_id": ["job1"],
                    "user": ["alice"],
                    "submit_time": [datetime(2025, 9, 20, 9, 0, 0)],  # Naive
                    "start_time": [datetime(2025, 9, 20, 10, 0, 0, tzinfo=timezone.utc)],  # UTC
                    "end_time": [datetime(2025, 9, 20, 11, 0, 0)],  # Naive
                    "processed_date": [datetime(2025, 9, 20, 12, 0, 0)],  # Naive
                    "is_complete": [True],
                }
            )

            df.write_parquet(processed_dir / f"{today}.parquet")

            result = slurm_usage._load_recent_data(config, days=1)

            assert result is not None

            # Check all datetime columns are UTC
            for col in ["submit_time", "start_time", "end_time", "processed_date"]:
                if col in result.columns:
                    col_type = result[col].dtype
                    if isinstance(col_type, pl.Datetime):
                        assert col_type.time_zone == "UTC", f"Column {col} should be UTC"
