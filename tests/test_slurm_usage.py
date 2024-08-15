import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from slurm_usage import (
    combine_statuses,
    get_max_lengths,
    main,
    process_data,
    squeue_output,
    summarize_status,
)

# Mock squeue output
squeue_mock_output = """USER/ST/NODES/PARTITION
bas.nijholt/PD/1/mypartition-10
bas.nijholt/PD/1/mypartition-20
bas.nijholt/PD/1/mypartition-20"""


@pytest.fixture()
def mock_subprocess_output():
    with patch("subprocess.getoutput", return_value=squeue_mock_output) as mock_output:
        yield mock_output


def test_squeue_output(mock_subprocess_output) -> None:
    output = squeue_output()
    assert len(output) == 3  # Based on the mocked output
    assert output[0] == "bas.nijholt/PD/1/mypartition-10"


def test_process_data() -> None:
    output = ["user1/R/2/partition1", "user2/PD/1/partition2", "user1/PD/1/partition1"]
    data, total_partition, totals = process_data(output, "nodes")
    assert data["user1"]["partition1"]["R"] == 2
    assert data["user2"]["partition2"]["PD"] == 1
    assert totals["PD"] == 2
    assert totals["R"] == 2


def test_summarize_status() -> None:
    status_dict = {"R": 2, "PD": 1}
    summary = summarize_status(status_dict)
    assert summary == "R=2 / PD=1"


def test_combine_statuses() -> None:
    statuses = {"partition1": {"R": 2, "PD": 1}, "partition2": {"R": 1}}
    combined = combine_statuses(statuses)
    assert combined == {"R": 3, "PD": 1}


def test_get_max_lengths() -> None:
    rows = [["user1", "R=2 / PD=1"], ["user2", "R=3"]]
    lengths = get_max_lengths(rows)
    assert lengths == [5, 10]


@patch("getpass.getuser", return_value="user1")
def test_main(mock_getuser, mock_subprocess_output) -> None:
    with patch("rich.console.Console.print") as mock_print:
        main()
        assert mock_print.called
