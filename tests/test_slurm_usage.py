"""Tests for slurm_usage module."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from slurm_usage import current

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Enable mock data mode
os.environ["SLURM_USE_MOCK_DATA"] = "1"

from slurm_usage import (  # noqa: E402
    SlurmJob,
    combine_statuses,
    get_max_lengths,
    process_data,
    squeue_output,
    summarize_status,
)

# Mock squeue output
squeue_mock_output = """USER/ST/NODES/PARTITION
bas.nijholt/PD/1/mypartition-10
bas.nijholt/PD/1/mypartition-20
bas.nijholt/PD/1/mypartition-20"""


@pytest.fixture
def mock_subprocess_output() -> Generator[MagicMock, None, None]:
    """Mock subprocess output for testing."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = squeue_mock_output
        mock_run.return_value.returncode = 0
        yield mock_run


def test_squeue_output(mock_subprocess_output: MagicMock) -> None:  # noqa: ARG001
    """Test squeue_output function."""
    output = squeue_output()
    # With mock data enabled, we get the actual mock data
    assert len(output) > 0  # Should have some jobs
    assert all(isinstance(job, SlurmJob) for job in output)
    # Check that jobs have the expected structure
    if output:
        job = output[0]
        assert hasattr(job, "user")
        assert hasattr(job, "status")
        assert hasattr(job, "partition")


def test_process_data() -> None:
    """Test process_data function."""
    from slurm_usage import SlurmJob

    # Create proper SlurmJob objects instead of strings
    output = [
        SlurmJob("user1", "R", 2, "partition1", 10, "node1", "YES"),
        SlurmJob("user2", "PD", 1, "partition2", 5, "node2", "YES"),
        SlurmJob("user1", "PD", 1, "partition1", 5, "node3", "YES"),
    ]
    data, total_partition, totals = process_data(output, "nodes")
    expected_r_count = 2
    expected_pd_single = 1
    expected_pd_total = 2
    assert data["user1"]["partition1"]["R"] == expected_r_count
    assert data["user2"]["partition2"]["PD"] == expected_pd_single
    assert totals["PD"] == expected_pd_total
    assert totals["R"] == expected_r_count


def test_summarize_status() -> None:
    """Test summarize_status function."""
    status_dict = {"R": 2, "PD": 1}
    summary = summarize_status(status_dict)
    assert summary == "R=2 / PD=1"


def test_combine_statuses() -> None:
    """Test combine_statuses function."""
    statuses = {"partition1": {"R": 2, "PD": 1}, "partition2": {"R": 1}}
    combined = combine_statuses(statuses)
    assert combined == {"R": 3, "PD": 1}


def test_get_max_lengths() -> None:
    """Test get_max_lengths function."""
    rows = [["user1", "R=2 / PD=1"], ["user2", "R=3"]]
    lengths = get_max_lengths(rows)
    expected_user_len = 5
    expected_status_len = 10
    assert lengths == [expected_user_len, expected_status_len]


@patch("getpass.getuser", return_value="user1")
def test_main(mock_getuser: MagicMock, mock_subprocess_output: MagicMock) -> None:  # noqa: ARG001
    """Test main function (which is actually the current command)."""
    with patch("rich.console.Console.print") as mock_print:
        # The main function is now a callback, we test current instead

        current()
        assert mock_print.called
