"""Test SLURM command execution with mock data."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable mock data mode for all tests
os.environ["SLURM_USE_MOCK_DATA"] = "1"

import slurm_usage


class TestSlurmCommands:
    """Test all SLURM command wrappers."""

    def test_run_sacct_version(self) -> None:
        """Test sacct version command."""
        result = slurm_usage.run_sacct_version()
        assert result.returncode == 0
        assert "slurm" in result.stdout.lower()
        assert result.command == "sacct --version"

    def test_run_squeue(self) -> None:
        """Test squeue command."""
        result = slurm_usage.run_squeue()
        assert result.returncode == 0
        assert "USER/ST/NODES/PARTITION" in result.stdout
        assert result.command == "squeue -ro %u/%t/%D/%P/%C/%N/%h"

        # Parse the output
        lines = result.stdout.strip().split("\n")
        assert len(lines) > 1  # Should have header + data

        # Check first data line format
        if len(lines) > 1:
            parts = lines[1].split("/")
            expected_parts = 7  # user/status/nodes/partition/cpus/nodelist/oversubscribe
            assert len(parts) == expected_parts

    def test_run_sinfo_cpus(self) -> None:
        """Test sinfo CPU information command."""
        result = slurm_usage.run_sinfo_cpus()
        assert result.returncode == 0
        assert result.command == "sinfo -h -N --format='%N,%c'"

        # Check output format
        lines = result.stdout.strip().split("\n")
        assert len(lines) > 0

        for line in lines[:5]:  # Check first few lines
            if line:
                parts = line.split(",")
                expected_fields = 2  # node_name,cpu_count
                assert len(parts) == expected_fields
                # Node names can vary (e.g., "node-001")
                assert len(parts[0]) > 0  # Just check that node name is not empty
                assert parts[1].isdigit()

    def test_run_sinfo_gpus(self) -> None:
        """Test sinfo GPU information command."""
        result = slurm_usage.run_sinfo_gpus()
        assert result.returncode == 0
        assert result.command == "sinfo -h -N --format='%N,%G'"

        # Check output format
        lines = result.stdout.strip().split("\n")
        assert len(lines) > 0

        for line in lines[:5]:  # Check first few lines
            if line:
                parts = line.split(",")
                expected_fields = 2  # node_name,gpu_info
                assert len(parts) == expected_fields
                # Node names can vary (e.g., "node-001")
                assert len(parts[0]) > 0  # Just check that node name is not empty

    def test_run_scontrol_show_node(self) -> None:
        """Test scontrol show node command."""
        # Mock the scontrol command since we don't have mock data for it
        mock_output = """NodeName=node-001 Arch=x86_64 CoresPerSocket=1
        CPUAlloc=0 CPUTot=64 CPULoad=0.00
        State=IDLE"""

        with patch("slurm_usage._run") as mock_run:
            mock_run.return_value = slurm_usage.CommandResult(
                stdout=mock_output,
                stderr="",
                returncode=0,
                command="scontrol show node node-001",
            )

            result = slurm_usage.run_scontrol_show_node("node-001")
            assert result.returncode == 0
            assert result.command == "scontrol show node node-001"
            assert "NodeName=node-001" in result.stdout
            assert "CPUTot=64" in result.stdout

    def test_run_sacct(self, test_dates: dict[str, str]) -> None:
        """Test sacct command for a specific date."""
        fields = slurm_usage.RawJobRecord.get_field_names()

        # Use the exact dates from our mock data
        result = slurm_usage.run_sacct(test_dates["today"], fields)

        assert result.returncode == 0
        assert result.command.startswith(f"sacct -a -S {test_dates['today']}T00:00:00")

        # Parse output
        lines = [line for line in result.stdout.strip().split("\n") if line]
        assert len(lines) > 0  # Should have job records

        # Try parsing a record
        if lines:
            record = slurm_usage.RawJobRecord.from_sacct_line(lines[0], fields)
            assert record is not None
            assert record.User != ""  # Should have anonymized user
            assert record.JobName != ""  # Should have anonymized job name


class TestCommandResult:
    """Test CommandResult named tuple."""

    def test_command_result_fields(self) -> None:
        """Test CommandResult has all expected fields."""
        result = slurm_usage.CommandResult(stdout="test output", stderr="", returncode=0, command="test command")

        assert result.stdout == "test output"
        assert result.stderr == ""
        assert result.returncode == 0
        assert result.command == "test command"


class TestMockDataSystem:
    """Test the mock data loading system."""

    def test_mock_data_enabled(self) -> None:
        """Test that mock data mode is enabled."""
        assert os.environ.get("SLURM_USE_MOCK_DATA") == "1"
        assert slurm_usage.USE_MOCK_DATA

    def test_data_directory_explicit(self) -> None:
        """Test that data directory can be explicitly set for mock data."""
        mock_data_dir = Path(__file__).parent.parent / "tests" / "mock_data"
        # Create a Config instance with explicit mock data directory
        config = slurm_usage.Config.create(data_dir=mock_data_dir)
        assert mock_data_dir == config.data_dir

    def test_exact_command_matching(self) -> None:
        """Test that exact command matching works."""
        # This command should match exactly
        result = slurm_usage.run_sacct_version()
        assert result.returncode == 0

        # A slightly different command should not match and would fail
        # (but we can't test this without actually running sacct)

    def test_snapshot_files_exist(self) -> None:
        """Test that snapshot files are accessible."""
        snapshot_dir = Path(__file__).parent / "snapshots"

        assert snapshot_dir.exists()
        assert (snapshot_dir / "command_map.json").exists()
        assert (snapshot_dir / "metadata.json").exists()
        assert (snapshot_dir / "squeue_output.txt").exists()
        assert (snapshot_dir / "sacct_version_output.txt").exists()
