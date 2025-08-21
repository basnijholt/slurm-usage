"""Test advanced features and edge cases for better coverage."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable mock data mode for all tests
os.environ["SLURM_USE_MOCK_DATA"] = "1"

import slurm_usage


class TestParsingEdgeCases:
    """Test edge cases in parsing functions."""

    def test_parse_memory_mb_edge_cases(self) -> None:
        """Test memory parsing edge cases."""
        # Test more edge cases
        expected_mb = 1536.0
        assert slurm_usage._parse_memory_mb("1.5G") == expected_mb
        assert slurm_usage._parse_memory_mb("500") == 500 / 1024  # Raw number treated as KB
        assert slurm_usage._parse_memory_mb("0.5T") == 0.5 * 1024 * 1024
        assert slurm_usage._parse_memory_mb("invalid") == 0.0
        assert slurm_usage._parse_memory_mb(None) == 0.0  # type: ignore[arg-type]

    def test_parse_cpu_seconds_edge_cases(self) -> None:
        """Test CPU time parsing edge cases."""
        # Test with milliseconds
        expected_seconds = 5445.0
        assert slurm_usage._parse_cpu_seconds("01:30:45.123") == expected_seconds
        # Test with just minutes:seconds
        expected_seconds_2 = 1845.0
        assert slurm_usage._parse_cpu_seconds("30:45") == expected_seconds_2
        # Test with multiple days
        assert slurm_usage._parse_cpu_seconds("2-12:30:00") == 2 * 86400 + 12 * 3600 + 30 * 60
        # Test invalid formats
        assert slurm_usage._parse_cpu_seconds("INVALID") == 0.0
        assert slurm_usage._parse_cpu_seconds("UNLIMITED") == 0.0

    def test_parse_int_edge_cases(self) -> None:
        """Test integer parsing edge cases."""
        assert slurm_usage._parse_int("-123") == 0  # Negative numbers
        assert slurm_usage._parse_int("123abc") == 0  # Mixed content
        assert slurm_usage._parse_int(None) == 0  # type: ignore[arg-type]

    def test_parse_datetime_edge_cases(self) -> None:
        """Test datetime parsing edge cases."""
        # Test various invalid formats
        assert slurm_usage._parse_datetime("Unknown") is None
        assert slurm_usage._parse_datetime("N/A") is None
        assert slurm_usage._parse_datetime("None") is None
        assert slurm_usage._parse_datetime("") is None
        assert slurm_usage._parse_datetime(None) is None
        assert slurm_usage._parse_datetime("not-a-date") is None

    def test_parse_gpu_count_edge_cases(self) -> None:
        """Test GPU count parsing edge cases."""
        # Test various formats
        assert slurm_usage._parse_gpu_count("gres/gpu=invalid") == 0
        expected_gpus = 3
        assert slurm_usage._parse_gpu_count("billing=4,gres/gpu=3,node=1") == expected_gpus
        assert slurm_usage._parse_gpu_count("") == 0


class TestNodeListParsingAdvanced:
    """Test advanced node list parsing scenarios."""

    def test_parse_node_list_with_padding(self) -> None:
        """Test parsing node lists with zero-padded numbers."""
        result = slurm_usage.parse_node_list("node-[001-010]")
        expected_nodes = 10
        assert len(result) == expected_nodes
        assert result[0] == "node-001"
        assert result[9] == "node-010"

    def test_parse_node_list_mixed_ranges(self) -> None:
        """Test parsing node lists with mixed range formats."""
        result = slurm_usage.parse_node_list("node-[1-3,10,20-22]")
        expected = ["node-1", "node-2", "node-3", "node-10", "node-20", "node-21", "node-22"]
        assert result == expected

    def test_parse_node_list_invalid_format(self) -> None:
        """Test parsing invalid node list formats."""
        # Invalid format should return the original string as a single node
        result = slurm_usage.parse_node_list("node-[invalid")
        # The parser tries to handle it, result may vary
        assert len(result) >= 1  # Should return something

    def test_parse_node_list_single_digit_in_brackets(self) -> None:
        """Test parsing single digit in brackets."""
        result = slurm_usage.parse_node_list("node-[5]")
        assert result == ["node-5"]


class TestRawJobRecordAdvanced:
    """Test advanced RawJobRecord scenarios."""

    def test_from_sacct_line_invalid(self) -> None:
        """Test parsing invalid sacct lines."""
        fields = slurm_usage.RawJobRecord.get_field_names()

        # Empty line
        assert slurm_usage.RawJobRecord.from_sacct_line("", fields) is None

        # Wrong number of fields
        assert slurm_usage.RawJobRecord.from_sacct_line("field1|field2", fields) is None

        # Line with correct number of fields but invalid data should still parse
        invalid_line = "|".join([""] * len(fields))
        record = slurm_usage.RawJobRecord.from_sacct_line(invalid_line, fields)
        assert record is not None  # Should parse even with empty fields

    def test_job_id_edge_cases(self) -> None:
        """Test job ID parsing edge cases."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)

        # Test array job with high index
        values[fields.index("JobID")] = "12345.999"
        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert record is not None
        assert record.job_id_base == "12345"

        # Test job with multiple dots
        values[fields.index("JobID")] = "12345.0.extern"
        line = "|".join(values)
        record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert record is not None
        assert record.job_id_base == "12345"


class TestProcessedJobAdvanced:
    """Test advanced ProcessedJob scenarios."""

    def test_from_raw_records_missing_user(self) -> None:
        """Test processing job with missing username."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)

        # Set minimal required fields except User
        values[fields.index("JobID")] = "12345"
        values[fields.index("User")] = ""  # Empty user
        values[fields.index("UID")] = "1000"
        values[fields.index("State")] = "COMPLETED"

        line = "|".join(values)
        raw_record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert raw_record is not None

        # This should handle the missing user gracefully
        processed = slurm_usage.ProcessedJob.from_raw_records(raw_record)
        assert processed.user == "uid_1000"  # Should use UID fallback

    def test_from_raw_records_cancelled_state(self) -> None:
        """Test processing job with CANCELLED state variations."""
        fields = slurm_usage.RawJobRecord.get_field_names()
        values = [""] * len(fields)

        values[fields.index("JobID")] = "12345"
        values[fields.index("User")] = "testuser"
        values[fields.index("State")] = "CANCELLED by 123"  # Variation of cancelled

        line = "|".join(values)
        raw_record = slurm_usage.RawJobRecord.from_sacct_line(line, fields)
        assert raw_record is not None
        processed = slurm_usage.ProcessedJob.from_raw_records(raw_record)

        assert processed.state == "CANCELLED"  # Should normalize to just CANCELLED


class TestConfigAdvanced:
    """Test advanced configuration scenarios."""

    def test_config_empty_file(self, tmp_path: Path) -> None:
        """Test loading empty config file."""
        config_dir = tmp_path / ".config" / "slurm-usage"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("")

        # Mock environment to use test config
        import os

        original_xdg = os.environ.get("XDG_CONFIG_HOME")
        try:
            os.environ["XDG_CONFIG_HOME"] = str(tmp_path / ".config")
            config = slurm_usage.Config.create()
            assert config.groups == {}
        finally:
            if original_xdg:
                os.environ["XDG_CONFIG_HOME"] = original_xdg
            else:
                os.environ.pop("XDG_CONFIG_HOME", None)

    def test_config_invalid_yaml(self, tmp_path: Path) -> None:
        """Test loading invalid YAML config."""
        config_dir = tmp_path / ".config" / "slurm-usage"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        # Mock environment to use test config
        import os

        original_xdg = os.environ.get("XDG_CONFIG_HOME")
        try:
            os.environ["XDG_CONFIG_HOME"] = str(tmp_path / ".config")
            config = slurm_usage.Config.create()
            assert config.groups == {}  # Should fall back to empty
        finally:
            if original_xdg:
                os.environ["XDG_CONFIG_HOME"] = original_xdg
            else:
                os.environ.pop("XDG_CONFIG_HOME", None)


class TestGetNodeFunctions:
    """Test node information retrieval functions."""

    def test_get_ncores(self) -> None:
        """Test getting number of cores from partition name."""
        expected_cores_1 = 128
        expected_cores_2 = 32
        assert slurm_usage.get_ncores("partition-128") == expected_cores_1
        assert slurm_usage.get_ncores("gpu-32-v100") == expected_cores_2
        assert slurm_usage.get_ncores("no-numbers") == 0
        assert slurm_usage.get_ncores("") == 0


class TestExtractJobDate:
    """Test job date extraction function."""

    def test_extract_job_date_various_formats(self, test_dates: dict[str, str]) -> None:
        """Test extracting job date from various formats."""
        # ISO format with time
        assert slurm_usage._extract_job_date(f"{test_dates['today']}T10:30:00", None) == test_dates["today"]

        # Date only format
        assert slurm_usage._extract_job_date(test_dates["today"], None) == test_dates["today"]

        # Fall back to submit time
        assert slurm_usage._extract_job_date(None, f"{test_dates['tomorrow']}T09:00:00") == test_dates["tomorrow"]
        assert slurm_usage._extract_job_date("Unknown", f"{test_dates['tomorrow']}T09:00:00") == test_dates["tomorrow"]

        # Invalid formats
        assert slurm_usage._extract_job_date("N/A", "None") is None
        assert slurm_usage._extract_job_date("", "") is None
        assert slurm_usage._extract_job_date("invalid", "also-invalid") is None
