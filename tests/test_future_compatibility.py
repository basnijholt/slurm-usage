"""Test that mock data works correctly regardless of the current date."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable mock data mode for all tests
os.environ["SLURM_USE_MOCK_DATA"] = "1"

import slurm_usage

UTC = timezone.utc


class TestFutureCompatibility:
    """Test that the mock data system works with future dates."""

    def test_mock_data_with_future_date(self) -> None:
        """Test that mock data works even if current date is in the future."""
        # Simulate running the test in the year 2030
        future_date = datetime(2030, 1, 15, 10, 0, 0, tzinfo=UTC)

        with patch("slurm_usage.datetime") as mock_dt:
            mock_dt.now.return_value = future_date
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)  # noqa: DTZ001

            # The mock data should still work because it uses exact command matching
            # The commands in command_map.json use specific dates (2025-08-20)
            result = slurm_usage.run_sacct(
                "2025-08-20",
                slurm_usage.RawJobRecord.get_field_names(),
            )

            assert result.returncode == 0
            assert result.stdout != ""
            assert "sacct -a -S 2025-08-20T00:00:00" in result.command

    def test_fetch_jobs_with_future_current_date(self, tmp_path: Path) -> None:
        """Test that fetching jobs works even when 'today' is in the future."""
        # Simulate the test running in 2030 but fetching data from 2025
        future_date = datetime(2030, 1, 15, 10, 0, 0, tzinfo=UTC)

        with patch("slurm_usage.datetime") as mock_dt:
            mock_dt.now.return_value = future_date
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)  # noqa: DTZ001

            # Create config
            config = slurm_usage.Config.create(data_dir=tmp_path)
            config.ensure_directories_exist()

            # Fetch jobs for the exact date in our mock data
            # This should work because mock data uses exact command matching
            raw, processed, is_complete = slurm_usage._fetch_jobs_for_date(
                "2025-08-20",  # The date in our mock data
                config,
                skip_if_complete=False,
            )

            # Should get data from mock
            assert len(raw) > 0
            assert len(processed) > 0

    def test_fetch_raw_records_with_exact_date(self) -> None:
        """Test that fetching records with exact date always works."""
        # This should work regardless of current date because we're
        # asking for the exact date that exists in mock data
        records = slurm_usage._fetch_raw_records_from_slurm("2025-08-20")

        assert len(records) > 0
        # Verify we got the expected mock data
        assert all(isinstance(r, slurm_usage.RawJobRecord) for r in records)

    def test_squeue_command_always_works(self) -> None:
        """Test that squeue command works regardless of date."""
        # squeue doesn't use dates, so it should always work
        result = slurm_usage.run_squeue()

        assert result.returncode == 0
        assert "USER/ST/NODES/PARTITION" in result.stdout

    def test_sinfo_commands_always_work(self) -> None:
        """Test that sinfo commands work regardless of date."""
        # sinfo commands don't use dates
        result = slurm_usage.run_sinfo_cpus()
        assert result.returncode == 0
        # Node names can vary
        assert len(result.stdout) > 0

        result = slurm_usage.run_sinfo_gpus()
        assert result.returncode == 0
        # Node names can vary
        assert len(result.stdout) > 0

    def test_date_independent_snapshots(self) -> None:
        """Verify that snapshot files don't contain hardcoded future dates."""
        snapshot_dir = Path(__file__).parent / "snapshots"

        # Check that snapshot files exist
        assert (snapshot_dir / "command_map.json").exists()
        assert (snapshot_dir / "metadata.json").exists()
        assert (snapshot_dir / "squeue_output.txt").exists()

        # The command map uses exact command matching, so dates in commands
        # are preserved exactly as they were captured
        import json

        with open(snapshot_dir / "command_map.json") as f:
            command_map = json.load(f)

        # Verify we have the expected commands
        # The reference date is 2025-08-21 (when snapshots were captured)
        sacct_today_cmd = "sacct -a -S 2025-08-21T00:00:00 -E 2025-08-21T23:59:59 --format=JobID,JobIDRaw,JobName,User,UID,Group,GID,Account,Partition,QOS,State,ExitCode,Submit,Eligible,Start,End,Elapsed,ElapsedRaw,CPUTime,CPUTimeRAW,TotalCPU,UserCPU,SystemCPU,AllocCPUS,AllocNodes,NodeList,ReqCPUS,ReqMem,ReqNodes,Timelimit,TimelimitRaw,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,AveRSS,AveCPU,AveVMSize,ConsumedEnergy,ConsumedEnergyRaw,Priority,Reservation,ReservationId,WorkDir,Cluster,ReqTRES,AllocTRES,Comment,Constraints,Container,DerivedExitCode,Flags,Layout,MaxRSSNode,MaxVMSizeNode,MinCPU,NCPUS,NNodes,NTasks,Reason,SubmitLine -P -n"  # noqa: E501
        assert sacct_today_cmd in command_map
        assert command_map[sacct_today_cmd] == "sacct_day_0"
