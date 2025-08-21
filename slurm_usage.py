#!/usr/bin/env python
"""SLURM Job Usage - Optimized with incremental processing.

Collects and analyzes SLURM job efficiency metrics and displays current queue status.

Part of the [slurm-usage](https://github.com/basnijholt/slurm-usage) library.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "typer>=0.9",
#     "pydantic>=2.0",
#     "rich>=13.0",
#     "polars>=0.20",
#     "pyyaml>=6.0",
# ]
# ///

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from getpass import getuser
from pathlib import Path
from typing import Annotated, Any, NamedTuple

import polars as pl
import typer
import yaml
from pydantic import BaseModel, Field, computed_field, field_validator
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

UTC = timezone.utc

app = typer.Typer(help="SLURM Job Monitor - Collect and analyze job efficiency metrics")
console = Console()

# Use different data directory when using mock data
USE_MOCK_DATA = os.environ.get("SLURM_USE_MOCK_DATA")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """SLURM Job Monitor - displays current cluster usage by default.

    Use 'slurm_monitor.py COMMAND --help' for more information on a specific command.
    """
    if ctx.invoked_subcommand is None:
        # Run current command by default
        current()


# Job states that indicate a job is still active or may change
# These jobs need to be re-collected until they reach a final state
# Based on: https://slurm.schedmd.com/job_state_codes.html
INCOMPLETE_JOB_STATES = [
    # Primary states that are active/pending
    "RUNNING",  # Allocated resources and executing
    "PENDING",  # Queued and waiting for initiation
    "SUSPENDED",  # Allocated resources but execution suspended
    # Transitional states (temporary)
    "COMPLETING",  # Finished/cancelled, performing cleanup tasks
    "CONFIGURING",  # Allocated nodes, waiting for them to boot
    "POWER_UP_NODE",  # Allocated powered down nodes, waiting for boot
    "STAGE_OUT",  # Staging out data (burst buffer)
    "SIGNALING",  # Outgoing signal to job is pending
    "STOPPED",  # Received SIGSTOP, suspended without releasing resources
    "UPDATE_DB",  # Sending update about job to database
    # Requeue states (will run again)
    "REQUEUED",  # Job is being requeued
    "REQUEUE_FED",  # Requeued due to federated sibling job
    "REQUEUE_HOLD",  # Requeued but held from scheduling
    "SPECIAL_EXIT",  # Special requeue hold situation
    # Other transitional states
    "RESIZING",  # Size of job is changing
    "RESV_DEL_HOLD",  # Held due to deleted reservation
    "REVOKED",  # Revoked due to federated sibling job
    # Note: These are considered FINAL states (don't re-collect):
    # BOOT_FAIL - terminated due to node boot failure
    # CANCELLED - cancelled by user or administrator
    # COMPLETED - completed successfully (exit code 0)
    # DEADLINE - terminated, reached latest acceptable start time
    # FAILED - completed unsuccessfully (non-zero exit)
    # LAUNCH_FAILED - failed to launch on chosen nodes
    # NODE_FAIL - terminated due to node failure
    # OUT_OF_MEMORY - experienced out of memory error
    # PREEMPTED - terminated due to preemption (if not requeued)
    # RECONFIG_FAIL - node configuration for job failed
    # TIMEOUT - terminated due to reaching time limit
]


# ============================================================================
# Configuration Management
# ============================================================================


def _get_config_path() -> Path | None:
    """Get the first existing configuration file path.

    Searches in priority order:
    1. $XDG_CONFIG_HOME/slurm-usage/config.yaml
    2. ~/.config/slurm-usage/config.yaml
    3. /etc/slurm-usage/config.yaml

    Returns the first existing path, or None if none exist.
    """
    config_paths = []

    # XDG_CONFIG_HOME or default ~/.config
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        config_paths.append(Path(xdg_config_home) / "slurm-usage" / "config.yaml")
    else:
        config_paths.append(Path.home() / ".config" / "slurm-usage" / "config.yaml")

    # Global system configuration
    config_paths.append(Path("/etc/slurm-usage/config.yaml"))

    for config_path in config_paths:
        if config_path.exists():
            return config_path

    console.print("[yellow]No configuration file found. Using defaults.[/yellow]")
    console.print("[dim]Expected locations:[/dim]")
    console.print("  [dim]$XDG_CONFIG_HOME/slurm-usage/config.yaml[/dim]")
    console.print("  [dim]~/.config/slurm-usage/config.yaml[/dim]")
    console.print("  [dim]/etc/slurm-usage/config.yaml[/dim]")
    return None


def _load_config_file() -> tuple[dict[str, Any], Path | None]:
    """Load configuration from the first existing config file.

    Returns a tuple of (config dict, config file path).
    Returns ({}, None) if no configuration file is found.
    """
    config_path = _get_config_path()

    if config_path is None:
        return {}, None

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            if config:
                console.print(f"[green]Loaded configuration from {config_path}[/green]")
                return config, config_path
            console.print(f"[yellow]Warning: Empty configuration file at {config_path}[/yellow]")
            return {}, config_path
    except (OSError, yaml.YAMLError) as e:
        console.print(f"[yellow]Warning: Failed to load {config_path}: {e}[/yellow]")
        return {}, None


class Config(BaseModel):
    """Unified configuration for SLURM monitor.

    Handles both configuration file loading and data directory management.
    """

    data_dir: Path
    groups: dict[str, list[str]] = Field(default_factory=dict)
    user_to_group: dict[str, str] = Field(default_factory=dict, exclude=True)

    @classmethod
    def create(cls, data_dir: Path | None = None, groups: dict[str, list[str]] | None = None) -> Config:
        """Create a Config instance with proper data directory resolution.

        Args:
            data_dir: Explicit data directory path (overrides all defaults)
            groups: Explicit groups configuration (overrides config file)

        Returns:
            Configured Config instance

        """
        # Load configuration from file
        file_config, config_path = _load_config_file()

        # Get groups - prefer parameter over config file
        if groups is None:
            groups = file_config.get("groups", {})

        # Build user-to-group mapping
        user_to_group = {}
        for group, users in groups.items():
            for user in users:
                user_to_group[user] = group

        # Determine data_dir with priority:
        # 1. Explicit parameter
        # 2. Config file value
        # 3. Default: ./data (current directory)
        if data_dir is None:
            # Check config file, otherwise use default ./data
            data_dir = Path(file_config["data_dir"]) if "data_dir" in file_config and file_config["data_dir"] is not None else Path("data")

        return cls(
            data_dir=data_dir,
            groups=groups,
            user_to_group=user_to_group,
        )

    def get_user_group(self, user: str) -> str:
        """Get the group for a user, returning 'ungrouped' if not found."""
        return self.user_to_group.get(user, "ungrouped")

    @property
    def raw_data_dir(self) -> Path:
        """Get the raw data directory path."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Get the processed data directory path."""
        return self.data_dir / "processed"

    def ensure_directories_exist(self) -> None:
        """Ensure all data directories exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SLURM Commands
# ============================================================================


class CommandResult(NamedTuple):
    """Result from a command execution."""

    stdout: str
    stderr: str
    returncode: int
    command: str = ""  # The command that was executed


def _run(cmd: str | list[str], *, shell: bool = False) -> CommandResult:
    """Run a command or return mock data if configured."""
    if (r := _maybe_run_mock(cmd)) is not None:
        return r

    # Run the actual command
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, check=False)  # noqa: S603
    return CommandResult(result.stdout, result.stderr, result.returncode, cmd_str)


def run_sacct(
    date_str: str,
    fields: list[str],
) -> CommandResult:
    """Run sacct command to get job accounting data for a specific date.

    Args:
        date_str: Date string in format YYYY-MM-DD
        fields: List of field names to retrieve

    Returns:
        CommandResult with stdout, stderr, and return code

    """
    start_date = f"{date_str}T00:00:00"
    end_date = f"{date_str}T23:59:59"
    fields_str = ",".join(fields)
    cmd = f"sacct -a -S {start_date} -E {end_date} --format={fields_str} -P -n"
    return _run(cmd, shell=True)  # noqa: S604


def run_squeue() -> CommandResult:
    """Run squeue to get current queue status.

    Returns:
        CommandResult with stdout, stderr, and return code

    """
    cmd = ["squeue", "-ro", "%u/%t/%D/%P/%C/%N/%h"]
    return _run(cmd)


def run_scontrol_show_node(node_name: str) -> CommandResult:
    """Run scontrol to get node information.

    Args:
        node_name: Name of the node to query

    Returns:
        CommandResult with stdout, stderr, and return code

    """
    cmd = ["scontrol", "show", "node", node_name]
    return _run(cmd)


def run_sinfo_cpus() -> CommandResult:
    """Run sinfo to get CPU information for all nodes.

    Returns:
        CommandResult with stdout, stderr, and return code

    """
    cmd = "sinfo -h -N --format='%N,%c'"
    return _run(cmd, shell=True)  # noqa: S604


def run_sinfo_gpus() -> CommandResult:
    """Run sinfo to get GPU (GRES) information for all nodes.

    Returns:
        CommandResult with stdout, stderr, and return code

    """
    cmd = "sinfo -h -N --format='%N,%G'"
    return _run(cmd, shell=True)  # noqa: S604


def run_sacct_version() -> CommandResult:
    """Run sacct --version to check if sacct is available.

    Returns:
        CommandResult with stdout, stderr, and return code

    """
    cmd = "sacct --version"
    return _run(cmd, shell=True)  # noqa: S604


# ============================================================================
# Mock Data Handling from tests/snapshots
# ============================================================================


def _get_mock_file_prefix(cmd_str: str, command_map: dict[str, str]) -> tuple[str | None, bool]:
    """Get the mock file prefix for a command.

    Args:
        cmd_str: The command string to match
        command_map: Dictionary mapping commands to file prefixes

    Returns:
        Tuple of (file_prefix, is_sacct_outside_range)
        - file_prefix: File prefix string if found, None otherwise
        - is_sacct_outside_range: True if sacct command but date outside snapshot range

    """
    # Try exact matching first
    if cmd_str in command_map:
        return command_map[cmd_str], False

    # For sacct commands, calculate offset from the reference date in metadata
    if cmd_str.startswith("sacct -a -S "):
        # Extract the date from the command
        date_match = re.search(r"-S (\d{4}-\d{2}-\d{2})T", cmd_str)
        if date_match:
            requested_date_str = date_match.group(1)

            # Load metadata to get the reference date
            snapshot_dir = Path(__file__).parent / "tests" / "snapshots"
            metadata_file = snapshot_dir / "metadata.json"
            with metadata_file.open() as f:
                metadata = json.load(f)

            # Get the reference date (the "today" when snapshots were captured)
            reference_date_str = metadata["reference_date"]

            # Calculate offset from reference date
            reference_date = datetime.strptime(reference_date_str, "%Y-%m-%d").date()  # noqa: DTZ007
            requested_date = datetime.strptime(requested_date_str, "%Y-%m-%d").date()  # noqa: DTZ007
            days_offset = (reference_date - requested_date).days

            # Map to the appropriate sacct_day_N file
            # We have sacct_day_0 through sacct_day_7 (8 days total)
            if 0 <= days_offset <= 7:  # noqa: PLR2004
                return f"sacct_day_{days_offset}", False
            # Date is outside our snapshot range
            return None, True

    return None, False


def _maybe_run_mock(cmd: str | list[str]) -> CommandResult | None:
    if USE_MOCK_DATA:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        snapshot_dir = Path(__file__).parent / "tests" / "snapshots"
        command_map_file = snapshot_dir / "command_map.json"
        with command_map_file.open() as f:
            command_map = json.load(f)

        file_prefix, is_sacct_outside_range = _get_mock_file_prefix(cmd_str, command_map)

        if file_prefix is None:
            if is_sacct_outside_range:
                # sacct command for a date outside our snapshot range - return empty success
                return CommandResult("", "", 0, cmd_str)
            # Command not found in map
            return CommandResult("", "", 1, cmd_str)

        stdout_file = snapshot_dir / f"{file_prefix}_output.txt"
        stdout = stdout_file.read_text()
        rc_file = snapshot_dir / f"{file_prefix}_returncode.txt"
        err_file = snapshot_dir / f"{file_prefix}_stderr.txt"
        returncode = int(rc_file.read_text().strip())
        stderr = err_file.read_text()
        return CommandResult(stdout, stderr, returncode, cmd_str)
    return None


# ============================================================================
# Pydantic Models with Built-in Processing
# ============================================================================


class RawJobRecord(BaseModel):
    """Raw job record from sacct with automatic parsing."""

    JobID: str
    JobIDRaw: str
    JobName: str
    User: str
    UID: str
    Group: str
    GID: str
    Account: str
    Partition: str
    QOS: str
    State: str
    ExitCode: str
    Submit: str
    Eligible: str
    Start: str
    End: str
    Elapsed: str
    ElapsedRaw: str
    CPUTime: str
    CPUTimeRAW: str
    TotalCPU: str
    UserCPU: str
    SystemCPU: str
    AllocCPUS: str
    AllocNodes: str
    NodeList: str
    ReqCPUS: str
    ReqMem: str
    ReqNodes: str
    Timelimit: str
    TimelimitRaw: str
    MaxRSS: str
    MaxVMSize: str
    MaxDiskRead: str
    MaxDiskWrite: str
    AveRSS: str
    AveCPU: str
    AveVMSize: str
    ConsumedEnergy: str
    ConsumedEnergyRaw: str
    Priority: str
    Reservation: str
    ReservationId: str
    WorkDir: str
    Cluster: str
    ReqTRES: str
    AllocTRES: str
    Comment: str
    Constraints: str
    Container: str
    DerivedExitCode: str
    Flags: str
    Layout: str
    MaxRSSNode: str
    MaxVMSizeNode: str
    MinCPU: str
    NCPUS: str
    NNodes: str
    NTasks: str
    Reason: str
    SubmitLine: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def job_id_base(self) -> str:
        """Extract base job ID without suffixes."""
        return self.JobID.split(".")[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_batch_step(self) -> bool:
        """Check if this is a batch step."""
        return ".batch" in self.JobID

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_main_job(self) -> bool:
        """Check if this is a main job record (not a step or array task)."""
        # Only jobs without any suffix are main jobs
        # Array job steps like "123.0" are NOT main jobs
        # Batch steps like "123.batch" are NOT main jobs
        return "." not in self.JobID

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_finished(self) -> bool:
        """Check if job has finished (not RUNNING or PENDING)."""
        return self.State not in ["RUNNING", "PENDING", "REQUEUED", "SUSPENDED", "PREEMPTED"]

    @classmethod
    def get_field_names(cls) -> list[str]:
        """Get ordered field names for sacct query."""
        return list(cls.model_fields.keys())

    @classmethod
    def from_sacct_line(cls, line: str, fields: list[str]) -> RawJobRecord | None:
        """Parse a single sacct output line."""
        if not line:
            return None

        parts = line.split("|")
        if len(parts) != len(fields):
            return None

        try:
            data = {fields[i]: parts[i] for i in range(len(fields))}
            return cls(**data)
        except (ValueError, KeyError, IndexError, TypeError):
            # Expected parsing errors - invalid field values or missing fields
            return None


class ProcessedJob(BaseModel):
    """Processed job with calculated metrics and validation."""

    # Identity
    job_id: str
    user: str
    job_name: str = Field(max_length=50)

    # Job info
    partition: str
    state: str
    submit_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None
    node_list: str  # Nodes where job ran

    # Resources
    elapsed_seconds: int = Field(ge=0)
    alloc_cpus: int = Field(ge=0)
    req_mem_mb: float = Field(ge=0)
    max_rss_mb: float = Field(ge=0)
    total_cpu_seconds: float = Field(ge=0)
    alloc_gpus: int = Field(ge=0)  # Number of allocated GPUs

    # Calculated metrics
    cpu_efficiency: float = Field(ge=0, le=100)
    memory_efficiency: float = Field(ge=0, le=100)
    cpu_hours_wasted: float = Field(ge=0)
    memory_gb_hours_wasted: float = Field(ge=0)

    # Total resource usage (not just waste)
    cpu_hours_reserved: float = Field(ge=0)
    memory_gb_hours_reserved: float = Field(ge=0)
    gpu_hours_reserved: float = Field(ge=0)

    # Metadata
    processed_date: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_complete: bool = True  # Flag to indicate if job has finished

    @field_validator("cpu_efficiency", "memory_efficiency")
    @classmethod
    def clamp_percentage(cls, v: float) -> float:
        """Ensure percentages are within 0-100."""
        return min(max(v, 0), 100)

    @classmethod
    def from_raw_records(
        cls,
        main_job: RawJobRecord,
        batch_job: RawJobRecord | None = None,
    ) -> ProcessedJob:
        """Create processed job from raw records, merging batch data if available."""
        # Clean up the state - normalize all CANCELLED variants
        state = main_job.State
        if state.startswith("CANCELLED"):
            state = "CANCELLED"

        # Clean up empty usernames - this should rarely happen for main jobs
        user = main_job.User
        if not user or user.strip() == "":
            # This is unexpected for main jobs - log it
            console.print(
                f"[red]WARNING: Main job {main_job.JobID} has no username! "
                f"UID={main_job.UID}, State={main_job.State}, JobName={main_job.JobName}[/red]",
            )
            # Still provide a fallback but make it clear something is wrong
            user = f"uid_{main_job.UID}" if main_job.UID and main_job.UID.strip() else "MISSING_USER"

        # Use batch data for actual usage if available
        total_cpu = batch_job.TotalCPU if batch_job and batch_job.TotalCPU else main_job.TotalCPU
        max_rss = batch_job.MaxRSS if batch_job and batch_job.MaxRSS else main_job.MaxRSS

        # Parse numeric values
        elapsed_seconds = _parse_int(main_job.ElapsedRaw)
        alloc_cpus = _parse_int(main_job.AllocCPUS)
        req_mem_mb = _parse_memory_mb(main_job.ReqMem)
        max_rss_mb = _parse_memory_mb(max_rss)
        total_cpu_seconds = _parse_cpu_seconds(total_cpu)
        alloc_gpus = _parse_gpu_count(main_job.AllocTRES)

        # Calculate efficiency
        cpu_efficiency = 0.0
        if elapsed_seconds > 0 and alloc_cpus > 0:
            cpu_efficiency = min((total_cpu_seconds / (elapsed_seconds * alloc_cpus)) * 100, 100)

        memory_efficiency = 0.0
        if req_mem_mb > 0 and max_rss_mb > 0:
            memory_efficiency = min((max_rss_mb / req_mem_mb) * 100, 100)

        # Calculate waste
        cpu_hours_wasted = max(0, (elapsed_seconds * alloc_cpus - total_cpu_seconds) / 3600)
        memory_gb_hours_wasted = max(0, (req_mem_mb - max_rss_mb) * elapsed_seconds / (1024 * 3600))

        # Calculate total reserved resources
        elapsed_hours = elapsed_seconds / 3600
        cpu_hours_reserved = alloc_cpus * elapsed_hours
        memory_gb_hours_reserved = (req_mem_mb / 1024) * elapsed_hours
        gpu_hours_reserved = alloc_gpus * elapsed_hours

        # Keep datetime as strings (already in ISO format from SLURM)
        # Only parse to validate and reformat if needed
        submit_dt = _parse_datetime(main_job.Submit)
        start_dt = _parse_datetime(main_job.Start)
        end_dt = _parse_datetime(main_job.End)

        return cls(
            job_id=main_job.job_id_base,
            user=user,
            job_name=main_job.JobName[:50],
            partition=main_job.Partition,
            state=state,
            submit_time=submit_dt,
            start_time=start_dt,
            end_time=end_dt,
            node_list=main_job.NodeList,
            elapsed_seconds=elapsed_seconds,
            alloc_cpus=alloc_cpus,
            req_mem_mb=req_mem_mb,
            max_rss_mb=max_rss_mb,
            total_cpu_seconds=total_cpu_seconds,
            alloc_gpus=alloc_gpus,
            cpu_efficiency=cpu_efficiency,
            memory_efficiency=memory_efficiency,
            cpu_hours_wasted=cpu_hours_wasted,
            memory_gb_hours_wasted=memory_gb_hours_wasted,
            cpu_hours_reserved=cpu_hours_reserved,
            memory_gb_hours_reserved=memory_gb_hours_reserved,
            gpu_hours_reserved=gpu_hours_reserved,
            is_complete=main_job.is_finished,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return self.model_dump()


class DateCompletionTracker(BaseModel):
    """Tracks which dates have been fully processed and don't need re-collection."""

    completed_dates: set[str] = Field(default_factory=set)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def mark_complete(self, date_str: str) -> None:
        """Mark a date as complete."""
        self.completed_dates.add(date_str)
        self.last_updated = datetime.now(UTC)

    def is_complete(self, date_str: str) -> bool:
        """Check if a date is marked as complete."""
        return date_str in self.completed_dates

    def save(self, path: Path) -> None:
        """Save tracker to JSON file."""
        data = self.model_dump()
        # Convert set to list for JSON serialization
        data["completed_dates"] = list(data["completed_dates"])
        with open(path, "w") as f:
            json.dump(data, f, default=str, indent=2)

    @classmethod
    def load(cls, path: Path) -> DateCompletionTracker:
        """Load tracker from JSON file."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                # Convert list back to set
                data["completed_dates"] = set(data.get("completed_dates", []))
                return cls.model_validate(data)
        return cls()


# ============================================================================
# Parsing and Utility Functions
# ============================================================================


def _parse_memory_mb(mem_str: str) -> float:  # noqa: PLR0911
    """Parse memory string to MB."""
    if not mem_str or mem_str in ["", "N/A", "0"]:
        return 0.0

    mem_str = mem_str.strip().rstrip("nc")

    try:
        if mem_str.endswith("K"):
            return float(mem_str[:-1]) / 1024
        if mem_str.endswith("M"):
            return float(mem_str[:-1])
        if mem_str.endswith("G"):
            return float(mem_str[:-1]) * 1024
        if mem_str.endswith("T"):
            return float(mem_str[:-1]) * 1024 * 1024
        if mem_str.replace(".", "").isdigit():
            return float(mem_str) / 1024
    except (ValueError, AttributeError):
        pass
    return 0.0


def _parse_cpu_seconds(time_str: str) -> float:
    """Parse CPU time string to seconds."""
    if not time_str or time_str in ["", "INVALID", "UNLIMITED"]:
        return 0.0

    total_seconds = 0.0

    try:
        # Remove milliseconds
        if "." in time_str:
            time_str = time_str.split(".")[0]

        # Handle DD-HH:MM:SS
        if "-" in time_str:
            days, time_part = time_str.split("-")
            total_seconds += float(days) * 86400
            time_str = time_part

        # Handle HH:MM:SS
        if ":" in time_str:
            parts = time_str.split(":")
            if len(parts) == 3:  # noqa: PLR2004
                total_seconds += float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:  # noqa: PLR2004
                total_seconds += float(parts[0]) * 60 + float(parts[1])
    except (ValueError, AttributeError):
        pass

    return total_seconds


def _parse_int(value: str) -> int:
    """Safely parse string to int."""
    if value and value.isdigit():
        return int(value)
    return 0


def _parse_datetime(date_str: str | None) -> datetime | None:
    """Parse datetime from SLURM format string."""
    if not date_str or date_str in ("Unknown", "N/A", "None"):
        return None
    try:
        # SLURM uses ISO format: 2025-08-19T10:30:00
        return datetime.fromisoformat(date_str)
    except (ValueError, AttributeError):
        return None


def _parse_gpu_count(alloc_tres: str) -> int:
    """Parse GPU count from AllocTRES string.

    AllocTRES format: "cpu=4,mem=8G,node=1,billing=4,gres/gpu=2"
    """
    if not alloc_tres or alloc_tres == "":
        return 0

    # Look for gres/gpu in the TRES string
    for item in alloc_tres.split(","):
        if "gres/gpu=" in item:
            parts = item.split("=")
            if len(parts) > 1 and parts[1].isdigit():
                return int(parts[1])
            return 0
    return 0


@functools.lru_cache(maxsize=256)
def parse_node_list(node_list: str) -> list[str]:
    """Parse SLURM node list format into individual node names.

    Handles formats like:
    - "node001" -> ["node001"]
    - "node[001-003]" -> ["node001", "node002", "node003"]
    - "node[001,003,005]" -> ["node001", "node003", "node005"]
    - "node[001-003,005,007-009]" -> ["node001", "node002", "node003", "node005", "node007", "node008", "node009"]
    - "rig-[3-4,6-7]" -> ["rig-3", "rig-4", "rig-6", "rig-7"]

    Args:
        node_list: SLURM node list string

    Returns:
        List of individual node names

    """
    if not node_list or node_list in ["None", "N/A", ""]:
        return []

    nodes = []

    # Handle simple single node
    if "[" not in node_list:
        return [node_list]

    # Extract prefix and range part
    try:
        prefix = node_list.split("[")[0]
        range_part = node_list.split("[")[1].split("]")[0]

        # Split by comma to handle multiple ranges/values
        for part in range_part.split(","):
            part = part.strip()  # noqa: PLW2901
            if "-" in part:
                # Range like "001-003" or "3-4"
                try:
                    start_str, end_str = part.split("-", 1)  # Only split on first hyphen

                    # Determine if we need zero-padding
                    pad_width = len(start_str) if start_str[0] == "0" else 0

                    start = int(start_str)
                    end = int(end_str)

                    for i in range(start, end + 1):
                        node_name = f"{prefix}{str(i).zfill(pad_width)}" if pad_width > 0 else f"{prefix}{i}"
                        nodes.append(node_name)
                except (ValueError, IndexError):
                    # If parsing fails, add the original part as-is
                    nodes.append(f"{prefix}{part}")
            # Single value like "001" or "5"
            elif part.isdigit():
                # Preserve zero-padding if present
                if part[0] == "0" and len(part) > 1:
                    nodes.append(f"{prefix}{part}")
                else:
                    nodes.append(f"{prefix}{part}")
            else:
                nodes.append(f"{prefix}{part}")

    except (IndexError, ValueError):
        # If parsing fails completely, return original
        return [node_list]

    return nodes


# ============================================================================
# Current usage metrics (via squeue)
# ============================================================================


class SlurmJob(NamedTuple):
    """Represents a SLURM job with its properties."""

    user: str
    status: str
    nnodes: int
    partition: str
    cores: int
    node: str
    oversubscribe: str

    @classmethod
    def from_line(cls, line: str) -> SlurmJob:
        """Create a SlurmJob from a squeue output line."""
        user, status, nnodes, partition, cores, node, oversubscribe = line.split("/")
        return cls(
            user,
            status,
            int(nnodes),
            partition,
            int(cores),
            node,
            oversubscribe,
        )


def squeue_output() -> list[SlurmJob]:
    """Get current SLURM queue status."""
    # Get the output and skip the header
    result = run_squeue()
    output = result.stdout.split("\n")[1:]
    return [SlurmJob.from_line(line) for line in output if line.strip()]


def get_total_cores(node_name: str) -> int:
    """Get total number of cores for a given node."""
    result = run_scontrol_show_node(node_name)
    output = result.stdout

    # Find the line with "CPUTot" which indicates the total number of CPUs (cores)
    for line in output.splitlines():
        if "CPUTot" in line:
            # Extract the number after "CPUTot="
            return int(line.split("CPUTot=")[1].split()[0])

    return 0  # Return 0 if not found


def process_data(
    output: list[SlurmJob],
    cores_or_nodes: str,
) -> tuple[
    defaultdict[str, defaultdict[str, defaultdict[str, int]]],
    defaultdict[str, defaultdict[str, int]],
    defaultdict[str, int],
]:
    """Process SLURM job data and aggregate statistics."""
    data: defaultdict[str, defaultdict[str, defaultdict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)),
    )
    total_partition: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int),
    )
    totals: defaultdict[str, int] = defaultdict(int)

    # Track which nodes have been counted for each user
    counted_nodes: defaultdict[str, set[str]] = defaultdict(set)

    for s in output:
        if s.oversubscribe in ["NO", "USER"]:
            if s.node not in counted_nodes[s.user]:
                n = get_total_cores(s.node)  # Get total cores in the node
                # Mark this node as counted for this user
                counted_nodes[s.user].add(s.node)
            else:
                continue  # Skip this job to prevent double-counting
        else:
            n = s.nnodes if cores_or_nodes == "nodes" else s.cores

        # Update the data structures with the correct values
        data[s.user][s.partition][s.status] += n
        total_partition[s.partition][s.status] += n
        totals[s.status] += n

    return data, total_partition, totals


def summarize_status(d: dict[str, int]) -> str:
    """Summarize status dictionary into a readable string."""
    return " / ".join([f"{status}={n}" for status, n in d.items()])


def combine_statuses(d: dict[str, Any]) -> dict[str, int]:
    """Combine multiple status dictionaries into one."""
    tot: defaultdict[str, int] = defaultdict(int)
    for dct in d.values():
        for status, n in dct.items():
            tot[status] += n
    return dict(tot)


def get_max_lengths(rows: list[list[str]]) -> list[int]:
    """Get maximum lengths for each column in a list of rows."""
    max_lengths = [0] * len(rows[0])
    for row in rows:
        for i, entry in enumerate(row):
            max_lengths[i] = max(len(entry), max_lengths[i])
    return max_lengths


def get_ncores(partition: str) -> int:
    """Extract number of cores from partition name."""
    numbers = re.findall(r"\d+", partition)
    try:
        return int(numbers[0])
    except IndexError:
        return 0


# ============================================================================
# Visualization Helpers
# ============================================================================


def _create_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    width: int = 50,
    top_n: int = 20,
    unit: str = "",
    show_percentage: bool = False,  # noqa: FBT001, FBT002
    item_type: str = "users",  # Type of items being shown (users, nodes, dates, etc.)
) -> None:
    """Create a horizontal bar chart using Rich."""
    if not values or len(values) == 0:
        return

    # Combine labels and values, sort by value, take top N
    data = list(zip(labels, values, strict=False))
    data = [(label, v) for label, v in data if v > 0]  # Filter out zeros
    data.sort(key=lambda x: x[1], reverse=True)

    # Calculate total for percentage
    total = sum(v for _, v in data)

    # Take top N after calculating total
    data = data[:top_n]

    if not data:
        return

    max_val = max(v for _, v in data)

    # Create the bar chart table
    table = Table(title=title, box=box.SIMPLE, show_header=False)
    table.add_column(item_type.capitalize(), style="cyan", width=15)
    table.add_column("Bar")
    table.add_column("Value", justify="right")

    for label, value in data:
        # Truncate long labels
        display_label = label[:15]

        # Create bar
        bar_length = int((value / max_val) * width) if max_val > 0 else 0
        bar = "█" * bar_length

        # Color based on value
        if value > max_val * 0.66:
            bar_color = "red"
        elif value > max_val * 0.33:
            bar_color = "yellow"
        else:
            bar_color = "green"

        # Format value with percentage if requested
        if show_percentage and total > 0:
            percentage = (value / total) * 100
            value_str = f"{value:,.0f} {unit} ({percentage:.1f}%)"
        else:
            value_str = f"{value:,.0f} {unit}"

        table.add_row(display_label, f"[{bar_color}]{bar}[/{bar_color}]", value_str)

    console.print(table)
    console.print(f"[dim]Showing top {len(data)} {item_type}[/dim]\n")


# ============================================================================
# Data Loading Functions
# ============================================================================


def _load_raw_records_from_parquet(raw_file: Path, date_str: str) -> list[RawJobRecord]:
    """Load raw records from a parquet file.

    Args:
        raw_file: Path to the raw parquet file
        date_str: Date string in YYYY-MM-DD format to filter records

    Returns:
        List of RawJobRecord objects for the specified date

    """
    try:
        raw_df = pl.read_parquet(raw_file)
        console.print(f"[cyan]Loading {len(raw_df)} records from raw file for {date_str}[/cyan]")

        raw_records = []
        for row in raw_df.iter_rows(named=True):
            try:
                record = RawJobRecord(**row)
                # Only include jobs that actually belong to this date
                job_date = _extract_job_date(record.Start, record.Submit)
                if job_date == date_str:
                    raw_records.append(record)
            except (ValueError, KeyError, TypeError):  # noqa: PERF203
                # Skip records with invalid field types or missing fields
                continue

        console.print(f"[cyan]Loaded {len(raw_records)} valid records for {date_str}[/cyan]")
    except (OSError, pl.exceptions.ComputeError) as e:
        console.print(f"[yellow]Could not load raw file for {date_str}: {e}[/yellow]")
        return []
    else:
        return raw_records


def _load_recent_data(
    config: Config,
    days: int,
    data_type: str = "processed",
) -> pl.DataFrame | None:
    """Load recent data files efficiently from daily directories.

    Args:
        config: Configuration object
        days: Number of days to look back
        data_type: "processed" or "raw" to specify which data to load

    """
    # Choose the appropriate directory
    base_dir = config.raw_data_dir if data_type == "raw" else config.processed_data_dir

    if not base_dir.exists():
        return None

    end_date = datetime.now(UTC)

    # Find all parquet files in date range
    parquet_files: list[Path] = []
    for day_offset in range(days + 1):
        check_date = end_date - timedelta(days=day_offset)
        date_str = check_date.strftime("%Y-%m-%d")
        file_path = base_dir / f"{date_str}.parquet"

        if file_path.exists():
            parquet_files.append(file_path)

    if not parquet_files:
        return None

    # Read all files at once with polars
    dfs = []
    for f in parquet_files:
        try:
            df = pl.read_parquet(f)
            dfs.append(df)
        except (OSError, pl.exceptions.ComputeError) as e:  # noqa: PERF203
            console.print(f"[yellow]Warning: Could not read {f}: {e}[/yellow]")
            continue

    if not dfs:
        return None

    combined = pl.concat(dfs)

    # Deduplicate by job_id if processing processed data
    if data_type == "processed" and "job_id" in combined.columns:
        return combined.sort("processed_date", descending=True).unique(subset=["job_id"], keep="first")
    if data_type == "raw" and "JobIDRaw" in combined.columns:
        # Deduplicate raw data by JobIDRaw
        return combined.unique(subset=["JobIDRaw"], keep="last")

    return combined


# ============================================================================
# Incremental Data Collection
# ============================================================================


def _fetch_raw_records_from_slurm(date_str: str) -> list[RawJobRecord]:
    """Fetch raw records from SLURM for a specific date.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        List of RawJobRecord objects for the specified date

    """
    fields = RawJobRecord.get_field_names()

    # Query jobs that started or were submitted on this date
    result = run_sacct(date_str, fields)

    if result.returncode != 0:
        console.print(f"[red]Error fetching data for {date_str}: {result.stderr}[/red]")
        return []

    lines = result.stdout.strip().split("\n")
    raw_records = []

    for line in lines:
        if not line:
            continue
        record = RawJobRecord.from_sacct_line(line, fields)
        if record is not None:
            # Only include jobs that actually belong to this date
            job_date = _extract_job_date(record.Start, record.Submit)
            if job_date == date_str:
                raw_records.append(record)

    return raw_records


def _apply_incremental_filtering(
    raw_records: list[RawJobRecord],
    existing_job_states: dict[str, str],
) -> tuple[list[RawJobRecord], int]:
    """Apply incremental filtering to raw records.

    Args:
        raw_records: List of raw records to filter
        existing_job_states: Dictionary of existing job IDs to their states

    Returns:
        Tuple of (filtered records, number of skipped records)

    """
    filtered_records = []
    skipped_count = 0

    for record in raw_records:
        # Skip records for jobs that haven't changed
        if record.job_id_base in existing_job_states:
            old_state = existing_job_states[record.job_id_base]

            # Skip finished jobs that haven't changed
            if old_state not in ["RUNNING", "PENDING", "SUSPENDED"]:
                if record.is_main_job:
                    # For main jobs: skip only if state unchanged
                    if record.State == old_state:
                        skipped_count += 1
                        continue
                else:
                    # For batch/step jobs: ALWAYS skip if main job is finished
                    skipped_count += 1
                    continue

        filtered_records.append(record)

    return filtered_records, skipped_count


def _process_raw_records_into_jobs(
    raw_records: list[RawJobRecord],
) -> tuple[list[ProcessedJob], bool]:
    """Process raw records into ProcessedJob objects.

    Args:
        raw_records: List of raw records to process

    Returns:
        Tuple of (processed jobs, is_complete flag)

    """
    # Group into main and batch jobs
    main_jobs: dict[str, RawJobRecord] = {}
    batch_jobs: dict[str, RawJobRecord] = {}

    for record in raw_records:
        if record.is_batch_step:
            batch_jobs[record.job_id_base] = record
        elif record.is_main_job:
            main_jobs[record.job_id_base] = record

    # Create processed jobs
    processed_jobs: list[ProcessedJob] = []
    has_incomplete = False

    for job_id, main_job in main_jobs.items():
        batch_job = batch_jobs.get(job_id)
        processed_job = ProcessedJob.from_raw_records(main_job, batch_job)
        processed_jobs.append(processed_job)

        # Check if this job is incomplete
        if processed_job.state in INCOMPLETE_JOB_STATES:
            has_incomplete = True

    # Date is complete if no incomplete jobs
    is_complete = not has_incomplete
    return processed_jobs, is_complete


def _extract_job_date(start_time: str | None, submit_time: str | None) -> str | None:
    """Extract the job date from start or submit time.

    Args:
        start_time: Job start time in ISO format (e.g., "2025-08-19T10:30:00")
        submit_time: Job submit time in ISO format

    Returns:
        Date string in "YYYY-MM-DD" format, or None if no valid date found

    """
    # Try start time first (when the job actually ran)
    if start_time and start_time not in ["Unknown", "N/A", "", "None"]:
        try:
            # Handle both full ISO format and date-only format
            date_part = start_time.split("T")[0] if "T" in start_time else start_time[:10]
            # Validate it looks like a date
            if len(date_part) == 10 and date_part.count("-") == 2:  # noqa: PLR2004
                return date_part
        except (IndexError, AttributeError):
            pass

    # Fall back to submit time (for PENDING jobs)
    if submit_time and submit_time not in ["Unknown", "N/A", "", "None"]:
        try:
            date_part = submit_time.split("T")[0] if "T" in submit_time else submit_time[:10]
            if len(date_part) == 10 and date_part.count("-") == 2:  # noqa: PLR2004
                return date_part
        except (IndexError, AttributeError):
            pass

    return None


def _processed_jobs_to_dataframe(
    processed_jobs: list[ProcessedJob],
) -> pl.DataFrame:
    """Convert a list of ProcessedJob objects to a DataFrame.

    Args:
        processed_jobs: List of ProcessedJob objects

    Returns:
        DataFrame with job data

    """
    return pl.DataFrame(
        [j.to_dict() for j in processed_jobs],
        infer_schema_length=None,
    )


def _save_processed_jobs_to_parquet(
    processed_jobs: list[ProcessedJob],
    file_path: Path,
) -> None:
    """Save a list of ProcessedJob objects to a parquet file.

    Args:
        processed_jobs: List of ProcessedJob objects to save
        file_path: Path to the parquet file

    """
    df = _processed_jobs_to_dataframe(processed_jobs)
    df.write_parquet(file_path)


class FetchJobsResult(NamedTuple):
    """Result from fetching jobs for a date."""

    raw_records: list[RawJobRecord]
    processed_jobs: list[ProcessedJob]
    is_complete: bool


def _fetch_jobs_for_date(  # noqa: PLR0912
    date_str: str,
    config: Config,
    skip_if_complete: bool = True,  # noqa: FBT001, FBT002
    completion_tracker: DateCompletionTracker | None = None,
) -> FetchJobsResult:
    """Fetch and process jobs for a specific date.

    Args:
        date_str: Date in YYYY-MM-DD format
        config: Configuration object with data directories
        skip_if_complete: Skip if the date is marked as complete
        completion_tracker: Tracker for completed dates

    Returns:
        Tuple of (raw_records, processed_jobs, is_complete)

    """
    processed_file = config.processed_data_dir / f"{date_str}.parquet"
    raw_file = config.raw_data_dir / f"{date_str}.parquet"

    # Check if this date is already marked as complete
    if skip_if_complete and completion_tracker and completion_tracker.is_complete(date_str):
        return FetchJobsResult(raw_records=[], processed_jobs=[], is_complete=True)

    # Check if we need to re-collect this date
    if skip_if_complete and processed_file.exists():
        try:
            df = pl.read_parquet(processed_file)
            incomplete = df.filter(pl.col("state").is_in(INCOMPLETE_JOB_STATES)).height
            if incomplete == 0:
                if completion_tracker:
                    completion_tracker.mark_complete(date_str)
                return FetchJobsResult(raw_records=[], processed_jobs=[], is_complete=True)
        except (OSError, pl.exceptions.ComputeError):
            # If we can't read the file, re-collect
            pass

    # Load existing processed data to track what we've seen
    existing_job_states: dict[str, str] = {}
    if processed_file.exists():
        try:
            existing_df = pl.read_parquet(processed_file)
            for row in existing_df.iter_rows(named=True):
                existing_job_states[row["job_id"]] = row["state"]
        except (OSError, pl.exceptions.ComputeError):
            # Failed to read existing data - continue with empty state
            pass

    # Get raw records - try raw file first if processed is missing
    raw_records_unfiltered: list[RawJobRecord] = []

    if not processed_file.exists() and raw_file.exists():
        raw_records_unfiltered = _load_raw_records_from_parquet(raw_file, date_str)

    # If no records from raw file, fetch from SLURM
    if not raw_records_unfiltered:
        raw_records_unfiltered = _fetch_raw_records_from_slurm(date_str)
        if not raw_records_unfiltered:
            # SLURM error occurred
            return FetchJobsResult(raw_records=[], processed_jobs=[], is_complete=False)

    # Apply incremental filtering
    raw_records, skipped_count = _apply_incremental_filtering(
        raw_records_unfiltered,
        existing_job_states,
    )

    if not raw_records:
        # No new records, but check if this date is now complete
        is_complete = False
        if processed_file.exists():
            try:
                df = pl.read_parquet(processed_file)
                incomplete = df.filter(pl.col("state").is_in(INCOMPLETE_JOB_STATES)).height
                is_complete = incomplete == 0
                if is_complete and completion_tracker:
                    completion_tracker.mark_complete(date_str)
            except (OSError, pl.exceptions.ComputeError):
                # Failed to check completion status - assume incomplete
                pass
        return FetchJobsResult(raw_records=[], processed_jobs=[], is_complete=is_complete)

    # Process raw records into jobs
    processed_jobs, is_complete = _process_raw_records_into_jobs(raw_records)

    if is_complete and completion_tracker:
        completion_tracker.mark_complete(date_str)

    return FetchJobsResult(raw_records=raw_records, processed_jobs=processed_jobs, is_complete=is_complete)


# ============================================================================
# Node Usage Analysis Functions
# ============================================================================


def _extract_node_usage_data(df: pl.DataFrame) -> pl.DataFrame:
    """Extract node usage data from job dataframe.

    Args:
        df: DataFrame with job data including node_list column

    Returns:
        DataFrame with columns: node, cpu_hours, gpu_hours, elapsed_hours

    """
    if df.is_empty() or "node_list" not in df.columns:
        return pl.DataFrame()

    # Filter out jobs without node assignments
    jobs_with_nodes = df.filter(pl.col("node_list") != "")

    if jobs_with_nodes.is_empty():
        return pl.DataFrame()

    # OPTIMIZATION: Pre-compute parsing for unique node patterns only
    # This is much faster than calling parse_node_list for every row
    unique_node_lists = jobs_with_nodes["node_list"].unique().to_list()
    node_list_mapping = {node_list: parse_node_list(node_list) for node_list in unique_node_lists}

    # Create a DataFrame with the mapping for joining
    mapping_df = pl.DataFrame(
        {
            "node_list": list(node_list_mapping.keys()),
            "parsed_nodes": list(node_list_mapping.values()),
        },
    )

    # Join to get parsed nodes and continue with vectorized operations
    jobs_with_nodes = (
        jobs_with_nodes.join(mapping_df, on="node_list", how="left")
        .with_columns(
            pl.col("elapsed_seconds").truediv(3600).alias("elapsed_hours"),
        )
        .with_columns(
            pl.col("parsed_nodes").list.len().alias("num_nodes"),
        )
        .filter(pl.col("num_nodes") > 0)
        .with_columns(
            [
                (pl.col("cpu_hours_reserved") / pl.col("num_nodes")).alias("cpu_hours_per_node"),
                (pl.col("gpu_hours_reserved") / pl.col("num_nodes")).alias("gpu_hours_per_node"),
                (pl.col("elapsed_hours") / pl.col("num_nodes")).alias("elapsed_hours_per_node"),
            ],
        )
    )

    # Explode the parsed_nodes list to create one row per node
    return jobs_with_nodes.explode("parsed_nodes").select(
        [
            pl.col("parsed_nodes").alias("node"),
            pl.col("cpu_hours_per_node").alias("cpu_hours"),
            pl.col("gpu_hours_per_node").alias("gpu_hours"),
            pl.col("elapsed_hours_per_node").alias("elapsed_hours"),
        ],
    )


@functools.lru_cache
def _get_node_info_from_slurm() -> dict[str, dict[str, int]]:  # noqa: PLR0912
    """Get node CPU and GPU information from SLURM using sinfo.

    Returns:
        Dictionary mapping node names to their CPU and GPU counts

    """
    node_info = {}

    try:
        # Get CPU information for all nodes
        # Format: NODELIST:20,CPUS:5
        result = run_sinfo_cpus()

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.strip().split(",")
                    if len(parts) == 2:  # noqa: PLR2004
                        node_name = parts[0]
                        try:
                            cpus = int(parts[1])
                            node_info[node_name] = {"cpus": cpus, "gpus": 0}
                        except ValueError:
                            continue

        # Get GPU information using GRES
        # Format: NODELIST:20,GRES:30
        result = run_sinfo_gpus()

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.strip().split(",")
                    if len(parts) == 2:  # noqa: PLR2004
                        node_name = parts[0]
                        gres = parts[1]

                        # Parse GRES string (e.g., "gpu:4" or "gpu:v100:4")
                        if "gpu:" in gres:
                            # Handle formats like "gpu:4" or "gpu:v100:4"
                            gpu_parts = gres.split(":")
                            # The GPU count is always the last number
                            if gpu_parts and gpu_parts[-1].isdigit():
                                gpu_count = int(gpu_parts[-1])
                                if node_name in node_info:
                                    node_info[node_name]["gpus"] = gpu_count
                                else:
                                    # Shouldn't happen, but handle gracefully
                                    node_info[node_name] = {"cpus": 64, "gpus": gpu_count}

    except subprocess.SubprocessError as e:
        console.print(f"[yellow]Warning: Could not get node info from sinfo: {e}[/yellow]")

    return node_info


@functools.lru_cache(maxsize=256)
def _get_node_cpus(node_name: str) -> int | None:
    """Get number of CPUs for a node from SLURM.

    Args:
        node_name: Name of the node

    Returns:
        Number of CPUs or None if not found.


    """
    node_info = _get_node_info_from_slurm()

    if node_name in node_info:
        return node_info[node_name]["cpus"]

    return None


@functools.lru_cache(maxsize=256)
def _get_node_gpus(node_name: str) -> int | None:
    """Get number of GPUs for a node from SLURM.

    Args:
        node_name: Name of the node

    Returns:
        Number of GPUs or None if not found.

    """
    node_info = _get_node_info_from_slurm()

    if node_name in node_info:
        return node_info[node_name]["gpus"]

    console.print(
        f"[yellow]Warning: Could not get GPU info for node {node_name}[/yellow]",
    )
    return None


def _calculate_analysis_period_days(df: pl.DataFrame) -> int:
    """Calculate the analysis period in days from the DataFrame.

    Args:
        df: DataFrame with submit_time and end_time columns

    Returns:
        Number of days in the analysis period

    """
    min_date = df["submit_time"].min()
    max_date = df["end_time"].max()

    if min_date and max_date:
        return int(max((max_date - min_date).days + 1, 1))

    return 7  # Default


def _aggregate_node_statistics(
    node_df: pl.DataFrame,
    period_days: int,
) -> pl.DataFrame:
    """Aggregate node usage data and calculate statistics.

    Args:
        node_df: DataFrame with node usage data (columns: node, cpu_hours, gpu_hours, elapsed_hours)
        period_days: Number of days in the analysis period

    Returns:
        DataFrame with aggregated node statistics including utilization

    """
    if node_df.is_empty():
        return pl.DataFrame()

    # Aggregate by node and chain all operations for better performance
    return (
        node_df.group_by("node")
        .agg(
            [
                pl.col("cpu_hours").sum().alias("total_cpu_hours"),
                pl.col("gpu_hours").sum().alias("total_gpu_hours"),
                pl.col("elapsed_hours").sum().alias("total_elapsed_hours"),
                pl.len().alias("job_count"),
            ],
        )
        .sort("total_cpu_hours", descending=True)
        .with_columns(
            pl.col("node").map_elements(_get_node_cpus, return_dtype=pl.Int64).alias("est_cpus"),
        )
        .with_columns(
            pl.when(pl.col("est_cpus").is_not_null()).then(pl.col("est_cpus") * period_days * 24).otherwise(None).alias("cpu_hours_available"),
        )
        .with_columns(
            pl.when(pl.col("cpu_hours_available").is_not_null())
            .then(pl.col("total_cpu_hours") / pl.col("cpu_hours_available") * 100)
            .otherwise(None)
            .alias("cpu_utilization_pct"),
        )
    )


def _display_node_usage_table(node_stats: pl.DataFrame) -> None:
    """Display node usage statistics in a table.

    Args:
        node_stats: DataFrame with node statistics

    """
    if node_stats.is_empty():
        return

    # Report nodes with missing CPU info
    missing_cpu_nodes = node_stats.filter(pl.col("est_cpus").is_null())
    if not missing_cpu_nodes.is_empty():
        console.print("[yellow]Warning: The following nodes have missing CPU information:[/yellow]")
        for row in missing_cpu_nodes.iter_rows(named=True):
            console.print(f"[yellow]  - {row['node']}: {row['total_cpu_hours']:,.0f} CPU hours (utilization cannot be calculated)[/yellow]")
        console.print()

    node_table = Table(title="Node Resource Usage", box=box.ROUNDED)
    node_table.add_column("Node", style="cyan")
    node_table.add_column("Jobs", justify="right")
    node_table.add_column("CPU Hours", justify="right", style="yellow")
    node_table.add_column("GPU Hours", justify="right", style="green")
    node_table.add_column("CPU Util %", justify="right")

    for row in node_stats.head(20).iter_rows(named=True):
        # Format utilization - show N/A if missing
        util_str = f"{row['cpu_utilization_pct']:.1f}%" if row["cpu_utilization_pct"] is not None else "N/A"

        node_table.add_row(
            row["node"],
            f"{row['job_count']:,}",
            f"{row['total_cpu_hours']:,.0f}",
            f"{row['total_gpu_hours']:,.0f}",
            util_str,
        )

    console.print(node_table)


def _display_node_utilization_charts(node_stats: pl.DataFrame, period_days: int) -> None:
    """Display bar charts for node utilization.

    Args:
        node_stats: DataFrame with node statistics
        period_days: Number of days in the analysis period

    """
    if node_stats.is_empty():
        return

    # Create bar chart for node CPU utilization (only for nodes with CPU info)
    nodes_with_cpu = node_stats.filter(pl.col("cpu_utilization_pct").is_not_null())
    if not nodes_with_cpu.is_empty():
        nodes = nodes_with_cpu["node"].to_list()
        cpu_util = nodes_with_cpu["cpu_utilization_pct"].to_list()

        console.print("\n")
        _create_bar_chart(
            nodes,
            cpu_util,
            f"Node CPU Utilization (% of {period_days} days)",
            width=50,
            top_n=20,
            unit="%",
            item_type="nodes",
        )

    # Show GPU node utilization if any
    gpu_nodes = node_stats.filter(pl.col("total_gpu_hours") > 0)
    if not gpu_nodes.is_empty():
        gpu_nodes = gpu_nodes.with_columns(
            pl.col("node").map_elements(_get_node_gpus, return_dtype=pl.Int64).alias("est_gpus"),
        )

        # Filter out nodes where we couldn't get GPU info or have 0 GPUs
        gpu_nodes = gpu_nodes.filter((pl.col("est_gpus").is_not_null()) & (pl.col("est_gpus") > 0))

        # Calculate GPU hours available
        gpu_nodes = gpu_nodes.with_columns(
            (pl.col("est_gpus") * period_days * 24).alias("gpu_hours_available"),
        )

        # Add GPU utilization percentage
        gpu_nodes = gpu_nodes.with_columns(
            (pl.col("total_gpu_hours") / pl.col("gpu_hours_available") * 100).alias(
                "gpu_utilization_pct",
            ),
        )

        gpu_node_names = gpu_nodes["node"].to_list()
        gpu_util = gpu_nodes["gpu_utilization_pct"].to_list()

        _create_bar_chart(
            gpu_node_names,
            gpu_util,
            f"GPU Node Utilization (% of {period_days} days)",
            width=50,
            top_n=10,
            unit="%",
            item_type="nodes",
        )


def _create_node_usage_stats(df: pl.DataFrame) -> None:
    """Analyze and display node usage statistics.

    This function orchestrates the node usage analysis by:
    1. Extracting node usage data from jobs
    2. Aggregating statistics per node
    3. Displaying tables and charts

    Args:
        df: DataFrame containing job data with node_list column

    """
    if df.is_empty() or "node_list" not in df.columns:
        return

    console.print(Panel.fit("Node Usage Analysis", style="bold cyan", box=box.DOUBLE_EDGE))

    # Step 1: Extract node usage data from jobs
    node_usage_df = _extract_node_usage_data(df)

    if node_usage_df.is_empty():
        console.print("[yellow]No node usage data available[/yellow]")
        return

    # Step 2: Calculate the analysis period
    period_days = _calculate_analysis_period_days(df)

    # Step 3: Aggregate node statistics with utilization calculations
    node_stats = _aggregate_node_statistics(node_usage_df, period_days)

    if node_stats.is_empty():
        console.print("[yellow]Could not calculate node statistics[/yellow]")
        return

    # Step 4: Display the node usage table
    _display_node_usage_table(node_stats)

    # Step 5: Display utilization charts
    _display_node_utilization_charts(node_stats, period_days)


# ============================================================================
# Analysis and Reporting Functions
# ============================================================================


def _prepare_dataframe_for_analysis(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Prepare DataFrame with additional columns needed for analysis.

    Args:
        df: DataFrame with job data
        config: Config object for group mappings

    Returns:
        DataFrame with group, reserved resource, and wait time columns added

    """
    if df.is_empty():
        return df

    # Check which columns are available
    has_reserved_cols = all(col in df.columns for col in ["cpu_hours_reserved", "memory_gb_hours_reserved", "gpu_hours_reserved"])

    # Add group column for aggregation
    # OPTIMIZATION: Pre-compute user-to-group mapping for unique users only
    unique_users = df["user"].unique().to_list()
    user_group_mapping = {user: config.get_user_group(user) for user in unique_users}

    # Create mapping DataFrame and join
    mapping_df = pl.DataFrame(
        {"user": list(user_group_mapping.keys()), "group": list(user_group_mapping.values())},
    )

    df = df.join(mapping_df, on="user", how="left")

    # If we don't have the new columns, calculate them from existing data
    if not has_reserved_cols:
        df = df.with_columns(
            [
                (pl.col("elapsed_seconds") * pl.col("alloc_cpus") / 3600).alias("cpu_hours_reserved"),
                (pl.col("req_mem_mb") * pl.col("elapsed_seconds") / (1024 * 3600)).alias("memory_gb_hours_reserved"),
                pl.lit(0.0).alias("gpu_hours_reserved"),  # No GPU data in old files
            ],
        )

    # Calculate wait time (in seconds) for jobs that have both submit and start times
    return df.with_columns(
        pl.when((pl.col("submit_time").is_not_null()) & (pl.col("start_time").is_not_null()))
        .then((pl.col("start_time") - pl.col("submit_time")).dt.total_seconds())
        .otherwise(None)
        .alias("wait_seconds"),
    )


def _create_user_statistics_section(df: pl.DataFrame) -> None:
    """Create and display user statistics table and charts.

    Args:
        df: Prepared DataFrame with user and resource data

    """
    # Calculate per-user statistics
    user_stats = (
        df.group_by("user")
        .agg(
            [
                pl.len().alias("job_count"),
                pl.col("cpu_hours_reserved").sum().alias("total_cpu_hours"),
                pl.col("memory_gb_hours_reserved").sum().alias("total_memory_gb_hours"),
                pl.col("gpu_hours_reserved").sum().alias("total_gpu_hours"),
                pl.col("cpu_hours_wasted").sum().alias("cpu_hours_wasted"),
                pl.col("memory_gb_hours_wasted").sum().alias("memory_gb_hours_wasted"),
                # Total RAM allocated (not GB-hours, just GB)
                (pl.col("req_mem_mb").sum() / 1024).alias("total_ram_gb"),
                # Average wait time in hours
                (pl.col("wait_seconds").filter(pl.col("wait_seconds").is_not_null()).mean() / 3600).alias("avg_wait_hours"),
                # Calculate average efficiency for completed jobs only
                pl.col("cpu_efficiency").filter(pl.col("state") == "COMPLETED").mean().alias("avg_cpu_efficiency"),
                pl.col("memory_efficiency").filter(pl.col("state") == "COMPLETED").mean().alias("avg_mem_efficiency"),
            ],
        )
        .sort("total_cpu_hours", descending=True)
    )

    # Display per-user resource usage
    console.print(Panel.fit("Resource Usage by User", style="bold cyan", box=box.DOUBLE_EDGE))

    user_table = Table(title="Top 15 Users by CPU Hours", box=box.ROUNDED)
    user_table.add_column("User", style="cyan")
    user_table.add_column("Jobs", justify="right")
    user_table.add_column("CPU Hours", justify="right", style="yellow")
    user_table.add_column("GPU Hours", justify="right", style="green")
    user_table.add_column("RAM GB", justify="right")
    user_table.add_column("Wait (h)", justify="right", style="magenta")
    user_table.add_column("CPU %", justify="right")
    user_table.add_column("Mem %", justify="right")

    for row in user_stats.head(15).iter_rows(named=True):
        user_table.add_row(
            row["user"][:15],  # user (truncated)
            f"{row['job_count']:,}",
            f"{row['total_cpu_hours']:,.0f}",
            f"{row['total_gpu_hours']:,.0f}",
            f"{row['total_ram_gb']:,.0f}",
            f"{row['avg_wait_hours']:.1f}" if row["avg_wait_hours"] is not None else "0.0",
            f"{row['avg_cpu_efficiency']:.1f}" if row["avg_cpu_efficiency"] is not None else "N/A",
            f"{row['avg_mem_efficiency']:.1f}" if row["avg_mem_efficiency"] is not None else "N/A",
        )

    console.print(user_table)

    # Create bar chart for CPU hours per user
    users = user_stats["user"].to_list()
    cpu_hours = user_stats["total_cpu_hours"].to_list()

    console.print("\n")
    _create_bar_chart(
        users,
        cpu_hours,
        "CPU Hours by User",
        width=50,
        top_n=20,
        unit="hours",
        show_percentage=True,
        item_type="users",
    )

    # Create bar chart for GPU hours per user (if any GPU usage)
    gpu_hours = user_stats["total_gpu_hours"].to_list()
    gpu_users_with_hours = [(u, h) for u, h in zip(users, gpu_hours, strict=False) if h > 0]

    if gpu_users_with_hours:
        gpu_users, gpu_values = zip(*gpu_users_with_hours, strict=False)
        _create_bar_chart(
            list(gpu_users),
            list(gpu_values),
            "GPU Hours by User",
            width=50,
            top_n=10,
            unit="hours",
            show_percentage=True,
            item_type="users",
        )

    # Create bar charts for efficiency metrics
    cpu_eff = user_stats["avg_cpu_efficiency"].to_list()
    cpu_eff_with_users = [(u, e) for u, e in zip(users, cpu_eff, strict=False) if e is not None and e > 0]

    if cpu_eff_with_users:
        eff_users, eff_values = zip(*cpu_eff_with_users, strict=False)
        _create_bar_chart(
            list(eff_users),
            list(eff_values),
            "CPU Efficiency by User",
            width=50,
            top_n=20,
            unit="%",
            item_type="users",
        )

    mem_eff = user_stats["avg_mem_efficiency"].to_list()
    mem_eff_with_users = [(u, e) for u, e in zip(users, mem_eff, strict=False) if e is not None and e > 0]

    if mem_eff_with_users:
        eff_users, eff_values = zip(*mem_eff_with_users, strict=False)
        _create_bar_chart(
            list(eff_users),
            list(eff_values),
            "Memory Efficiency by User",
            width=50,
            top_n=20,
            unit="%",
            item_type="users",
        )


def _create_group_statistics_section(df: pl.DataFrame) -> None:
    """Create and display group statistics table and charts.

    Args:
        df: Prepared DataFrame with group and resource data

    """
    # Calculate per-group statistics
    group_stats = (
        df.group_by("group")
        .agg(
            [
                pl.col("user").n_unique().alias("unique_users"),
                pl.len().alias("job_count"),
                pl.col("cpu_hours_reserved").sum().alias("total_cpu_hours"),
                pl.col("memory_gb_hours_reserved").sum().alias("total_memory_gb_hours"),
                pl.col("gpu_hours_reserved").sum().alias("total_gpu_hours"),
                pl.col("cpu_hours_wasted").sum().alias("cpu_hours_wasted"),
                pl.col("memory_gb_hours_wasted").sum().alias("memory_gb_hours_wasted"),
            ],
        )
        .sort("total_cpu_hours", descending=True)
    )

    # Only show group statistics if there are actual groups (not just "ungrouped")
    groups = group_stats["group"].to_list()
    if len(groups) > 1 or (len(groups) == 1 and groups[0] != "ungrouped"):
        # Display per-group resource usage
        console.print(Panel.fit("Resource Usage by Group", style="bold cyan", box=box.DOUBLE_EDGE))

        group_table = Table(title="Research Group Statistics", box=box.ROUNDED)
        group_table.add_column("Group", style="magenta")
        group_table.add_column("Users", justify="right")
        group_table.add_column("Jobs", justify="right")
        group_table.add_column("CPU Hours", justify="right", style="yellow")
        group_table.add_column("Memory GB-hrs", justify="right", style="blue")
        group_table.add_column("GPU Hours", justify="right", style="green")

        for row in group_stats.iter_rows(named=True):
            group_table.add_row(
                row["group"],
                f"{row['unique_users']}",
                f"{row['job_count']:,}",
                f"{row['total_cpu_hours']:,.0f}",
                f"{row['total_memory_gb_hours']:,.0f}",
                f"{row['total_gpu_hours']:,.0f}",
            )

        console.print(group_table)

        # Create bar charts for group-based CPU and GPU usage
        group_cpu_hours = group_stats["total_cpu_hours"].to_list()
        group_gpu_hours = group_stats["total_gpu_hours"].to_list()

        console.print("\n")
        _create_bar_chart(
            groups,
            group_cpu_hours,
            "CPU Hours by Research Group",
            width=50,
            top_n=15,
            unit="hours",
            show_percentage=True,
            item_type="groups",
        )

        # Show GPU hours by group if any GPU usage
        group_gpu_with_hours = [(g, h) for g, h in zip(groups, group_gpu_hours, strict=False) if h > 0]
        if group_gpu_with_hours:
            gpu_groups, gpu_group_values = zip(*group_gpu_with_hours, strict=False)
            _create_bar_chart(
                list(gpu_groups),
                list(gpu_group_values),
                "GPU Hours by Research Group",
                width=50,
                top_n=10,
                unit="hours",
                show_percentage=True,
                item_type="groups",
            )


def _create_efficiency_analysis_section(df: pl.DataFrame) -> None:
    """Create and display efficiency analysis for completed jobs.

    Args:
        df: DataFrame with job data

    """
    # Show efficiency summary for completed jobs
    completed = df.filter(pl.col("state") == "COMPLETED")

    if not completed.is_empty():
        console.print(Panel.fit("Efficiency Analysis (Completed Jobs)", style="bold cyan", box=box.DOUBLE_EDGE))

        # Calculate stats in one pass
        stats = completed.select(
            [
                pl.col("cpu_efficiency").mean().alias("cpu_mean"),
                pl.col("cpu_efficiency").median().alias("cpu_median"),
                pl.col("cpu_efficiency").min().alias("cpu_min"),
                pl.col("cpu_efficiency").max().alias("cpu_max"),
                pl.col("memory_efficiency").mean().alias("mem_mean"),
                pl.col("memory_efficiency").median().alias("mem_median"),
                pl.col("memory_efficiency").min().alias("mem_min"),
                pl.col("memory_efficiency").max().alias("mem_max"),
            ],
        ).row(0)

        # Display efficiency summary
        summary_table = Table(title="Overall Efficiency Metrics", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Mean", justify="right")
        summary_table.add_column("Median", justify="right")
        summary_table.add_column("Min", justify="right")
        summary_table.add_column("Max", justify="right")

        summary_table.add_row(
            "CPU Efficiency (%)",
            f"{stats[0]:.1f}",
            f"{stats[1]:.1f}",
            f"{stats[2]:.1f}",
            f"{stats[3]:.1f}",
        )
        summary_table.add_row(
            "Memory Efficiency (%)",
            f"{stats[4]:.1f}",
            f"{stats[5]:.1f}",
            f"{stats[6]:.1f}",
            f"{stats[7]:.1f}",
        )

        console.print(summary_table)

        # Top wasters
        user_waste = (
            completed.group_by("user")
            .agg(
                [
                    pl.col("cpu_hours_wasted").sum().alias("cpu_wasted"),
                    pl.col("memory_gb_hours_wasted").sum().alias("mem_wasted"),
                ],
            )
            .sort("cpu_wasted", descending=True)
            .head(10)
        )

        if not user_waste.is_empty():
            console.print("\n")
            waste_table = Table(title="Top 10 Resource Wasters", box=box.SIMPLE)
            waste_table.add_column("User", style="red")
            waste_table.add_column("CPU Hours Wasted", justify="right")
            waste_table.add_column("Memory GB-Hours Wasted", justify="right")

            for row in user_waste.iter_rows(named=True):
                waste_table.add_row(row["user"], f"{row['cpu_wasted']:,.0f}", f"{row['mem_wasted']:,.0f}")

            console.print(waste_table)


def _create_cluster_summary_section(df: pl.DataFrame) -> None:
    """Create and display cluster-wide summary statistics.

    Args:
        df: DataFrame with job data

    """
    # Display cluster-wide summary statistics
    console.print(Panel.fit("Cluster-Wide Summary", style="bold cyan", box=box.DOUBLE_EDGE))

    cluster_summary = Table(title="Total Resource Usage", box=box.ROUNDED)
    cluster_summary.add_column("Metric", style="cyan")
    cluster_summary.add_column("Value", justify="right", style="yellow")

    # Calculate totals
    total_jobs = len(df)
    total_cpu_hours = df["cpu_hours_reserved"].sum()
    total_gpu_hours = df["gpu_hours_reserved"].sum()
    total_ram_gb = df["req_mem_mb"].sum() / 1024
    avg_wait_hours = (df["wait_seconds"].filter(df["wait_seconds"].is_not_null()).mean() / 3600) if "wait_seconds" in df.columns else None
    total_users = df["user"].n_unique()

    cluster_summary.add_row("Total Jobs", f"{total_jobs:,}")
    cluster_summary.add_row("Unique Users", f"{total_users:,}")
    cluster_summary.add_row("Total CPU Hours", f"{total_cpu_hours:,.0f}")
    cluster_summary.add_row("Total GPU Hours", f"{total_gpu_hours:,.0f}")
    cluster_summary.add_row("Total RAM Allocated (GB)", f"{total_ram_gb:,.0f}")
    cluster_summary.add_row(
        "Average Wait Time (hours)",
        f"{avg_wait_hours:.1f}" if avg_wait_hours is not None else "N/A",
    )

    # Add job state distribution
    state_counts = df.group_by("state").agg(pl.len().alias("count")).sort("count", descending=True)
    top_states = state_counts.head(3)
    states_str = ", ".join([f"{row['state']}: {row['count']:,}" for row in top_states.iter_rows(named=True)])
    cluster_summary.add_row("Top Job States", states_str)

    console.print(cluster_summary)


def _create_summary_stats(df: pl.DataFrame, config: Config) -> None:
    """Create and display comprehensive resource usage statistics.

    Args:
        df: DataFrame with job data
        config: Config object for group mappings

    """
    if df.is_empty():
        return

    prepared_df = _prepare_dataframe_for_analysis(df, config)
    _create_user_statistics_section(prepared_df)
    _create_group_statistics_section(prepared_df)
    _create_node_usage_stats(prepared_df)
    _create_efficiency_analysis_section(prepared_df)
    _create_cluster_summary_section(prepared_df)


def _create_daily_usage_chart(df: pl.DataFrame) -> None:
    """Create and display daily resource usage bar chart.

    Args:
        df: DataFrame with job data including start_time or submit_time

    """
    if df.is_empty():
        return

    # Extract date from start_time (or submit_time if start_time is null)
    # Now working with datetime objects directly
    df_with_date = df.with_columns(
        pl.when(pl.col("start_time").is_not_null())
        .then(pl.col("start_time").dt.date())  # Extract date from datetime
        .otherwise(pl.col("submit_time").dt.date())
        .alias("job_date"),
    ).filter(pl.col("job_date").is_not_null())

    if df_with_date.is_empty():
        return

    # Aggregate by date
    daily_stats = (
        df_with_date.group_by("job_date")
        .agg(
            [
                pl.len().alias("job_count"),
                pl.col("cpu_hours_reserved").sum().alias("cpu_hours"),
                pl.col("gpu_hours_reserved").sum().alias("gpu_hours"),
                pl.col("memory_gb_hours_reserved").sum().alias("memory_gb_hours"),
                pl.col("user").n_unique().alias("unique_users"),
            ],
        )
        .sort("job_date")
    )

    if daily_stats.is_empty():
        return

    console.print(Panel.fit("Daily Resource Usage", style="bold cyan", box=box.DOUBLE_EDGE))

    # Display daily usage table
    daily_table = Table(title="Resource Usage by Day", box=box.ROUNDED)
    daily_table.add_column("Date", style="cyan")
    daily_table.add_column("Jobs", justify="right")
    daily_table.add_column("Users", justify="right")
    daily_table.add_column("CPU Hours", justify="right", style="yellow")
    daily_table.add_column("GPU Hours", justify="right", style="green")
    daily_table.add_column("Memory GB-hrs", justify="right", style="blue")

    for row in daily_stats.tail(14).iter_rows(named=True):  # Show last 14 days max
        daily_table.add_row(
            str(row["job_date"]),  # Convert date object to string for display
            f"{row['job_count']:,}",
            str(row["unique_users"]),
            f"{row['cpu_hours']:,.0f}",
            f"{row['gpu_hours']:,.0f}",
            f"{row['memory_gb_hours']:,.0f}",
        )

    console.print(daily_table)

    # Create bar chart for CPU hours per day
    dates = [str(d) for d in daily_stats["job_date"].to_list()]  # Convert dates to strings
    cpu_hours = daily_stats["cpu_hours"].to_list()

    if len(dates) > 1:
        console.print("\n")
        _create_bar_chart(
            dates[-14:],  # Show last 14 days max
            cpu_hours[-14:],
            "Daily CPU Hours Usage",
            width=50,
            top_n=14,
            unit="CPU-hrs",
            item_type="days",
        )

    # Show GPU usage if any
    gpu_hours = daily_stats["gpu_hours"].to_list()
    if any(h > 0 for h in gpu_hours):
        _create_bar_chart(
            dates[-14:],
            gpu_hours[-14:],
            "Daily GPU Hours Usage",
            width=50,
            top_n=14,
            unit="GPU-hrs",
            item_type="days",
        )


# ============================================================================
# CLI Commands
# ============================================================================


@app.command()
def collect(  # noqa: PLR0912, PLR0915
    days: Annotated[int, typer.Option("--days", "-d", help="Days to look back")] = 1,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Data directory (default: ./data)")] = None,
    show_summary: Annotated[bool, typer.Option("--summary/--no-summary", help="Show summary after collection")] = True,  # noqa: FBT002
    n_parallel: Annotated[int, typer.Option("--n-parallel", "-n", help="Number of parallel workers for date-based collection")] = 4,
) -> None:
    """Collect job data from SLURM using parallel date-based queries."""
    mode_text = "[yellow]MOCK DATA MODE[/yellow]\n" if USE_MOCK_DATA else ""
    console.print(
        Panel.fit(
            f"[bold cyan]SLURM Job Monitor[/bold cyan]\n{mode_text}Parallel collection with {n_parallel} workers",
            border_style="cyan",
        ),
    )

    # Create config and ensure directories exist
    config = Config.create(data_dir=data_dir)
    config.ensure_directories_exist()

    if USE_MOCK_DATA:
        console.print(f"[yellow]Using mock data directory: {config.data_dir}[/yellow]")

    # Load or create completion tracker
    tracker_file = config.data_dir / ".date_completion_tracker.json"
    completion_tracker = DateCompletionTracker.load(tracker_file)

    # Get list of dates to collect
    dates_to_collect = []
    end_date = datetime.now(UTC).date()
    for i in range(days + 1):
        date = end_date - timedelta(days=i)
        dates_to_collect.append(date.strftime("%Y-%m-%d"))

    dates_to_collect = sorted(dates_to_collect)  # Chronological order

    # Report how many dates are already complete
    already_complete = sum(1 for d in dates_to_collect if completion_tracker.is_complete(d))
    console.print(
        f"[cyan]Collecting data for {len(dates_to_collect)} dates ({already_complete} already complete) using {n_parallel} workers[/cyan]",
    )

    # Collect data in parallel
    total_raw = 0
    total_processed = 0
    successful_dates = []
    completed_dates = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Collecting {len(dates_to_collect)} dates...",
            total=len(dates_to_collect),
        )

        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            # Submit all tasks with completion tracker
            future_to_date = {
                executor.submit(
                    _fetch_jobs_for_date,
                    date,
                    config,
                    skip_if_complete=True,
                    completion_tracker=completion_tracker,
                ): date
                for date in dates_to_collect
            }

            # Process completed tasks
            for future in as_completed(future_to_date):
                date_str = future_to_date[future]
                try:
                    result = future.result()
                    if result.raw_records:
                        # Save raw data (keeping for archival - SLURM might purge old data)
                        raw_file = config.raw_data_dir / f"{date_str}.parquet"
                        raw_df = pl.DataFrame([r.model_dump() for r in result.raw_records])
                        raw_df.write_parquet(raw_file)
                        total_raw += len(result.raw_records)

                    # Always ensure processed file exists if we have any data (raw or processed)
                    processed_file = config.processed_data_dir / f"{date_str}.parquet"
                    raw_file = config.raw_data_dir / f"{date_str}.parquet"

                    if result.processed_jobs:
                        # We have new processed jobs to save/merge
                        if processed_file.exists():
                            # Load existing data
                            existing_df = pl.read_parquet(processed_file)
                            new_df = _processed_jobs_to_dataframe(result.processed_jobs)

                            # Merge: keep the most recent version of each job
                            # This updates job states for existing jobs and adds new ones
                            merged_df = pl.concat([existing_df, new_df])
                            merged_df = merged_df.sort("processed_date", descending=True).unique(subset=["job_id"], keep="first")
                            merged_df.write_parquet(processed_file)
                            total_processed += len(new_df) - len(existing_df) + len(merged_df)
                        else:
                            # First time collecting this date
                            _save_processed_jobs_to_parquet(result.processed_jobs, processed_file)
                            total_processed += len(result.processed_jobs)

                        successful_dates.append(date_str)
                    elif not processed_file.exists() and raw_file.exists():
                        # No new jobs, but we have a raw file and no processed file
                        # Process the raw file to create the processed file
                        loaded_raw_records = _load_raw_records_from_parquet(raw_file, date_str)
                        if loaded_raw_records:
                            jobs, _ = _process_raw_records_into_jobs(loaded_raw_records)
                            if jobs:
                                _save_processed_jobs_to_parquet(jobs, processed_file)
                                total_processed += len(jobs)

                    if result.is_complete:
                        completed_dates += 1
                        status_icon = "✓"
                    else:
                        status_icon = "↻"

                    status = (
                        f"[green]{len(result.processed_jobs)} jobs {status_icon}[/green]"
                        if result.processed_jobs
                        else f"[dim]skipped {status_icon}[/dim]"
                    )
                    progress.update(task, advance=1, description=f"Collected {date_str}: {status}")

                except (OSError, pl.exceptions.ComputeError, ValueError) as e:
                    # Expected errors: I/O issues, corrupt parquet files, data parsing
                    console.print(f"[red]Error collecting {date_str}: {e}[/red]")
                    progress.update(task, advance=1)

    # Save the completion tracker
    completion_tracker.save(tracker_file)

    console.print(f"[green]✓ Collected {total_processed} jobs from {total_raw} records[/green]")
    console.print(
        f"[green]✓ {completed_dates} dates marked as complete (no more updates needed)[/green]",
    )

    if show_summary:
        # Load and analyze ALL data for the requested period, not just updated dates
        dfs = []
        for date_str in dates_to_collect:  # Use all requested dates, not just successful ones
            file_path = config.processed_data_dir / f"{date_str}.parquet"
            if file_path.exists():
                with contextlib.suppress(Exception):
                    dfs.append(pl.read_parquet(file_path))

        if dfs:
            df = pl.concat(dfs).unique(subset=["job_id"], keep="last")
            console.print(
                f"\n[cyan]Analyzing {len(df):,} unique jobs from {len(dates_to_collect)} days[/cyan]\n",
            )
            # Show daily usage trends first
            _create_daily_usage_chart(df)
            _create_summary_stats(df, config)

    console.print("\n[bold green]✓ Collection complete[/bold green]")
    console.print(f"  Total records processed: {total_processed:,}")


@app.command()
def analyze(
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Data directory (default: ./data)")] = None,
    days: Annotated[int, typer.Option("--days", "-d", help="Days to analyze")] = 7,
) -> None:
    """Analyze collected job data."""
    console.print(
        Panel.fit(
            f"[bold cyan]Job Efficiency Analysis[/bold cyan]\nAnalyzing last {days} days",
            border_style="cyan",
        ),
    )

    config = Config.create(data_dir=data_dir)
    df = _load_recent_data(config, days)

    if df is None or df.is_empty():
        console.print("[yellow]No data found for specified period[/yellow]")
        raise typer.Exit(1)

    console.print(f"[green]Loaded {len(df):,} unique job records[/green]\n")

    # Show daily usage trends first
    _create_daily_usage_chart(df)

    _create_summary_stats(df, config)

    # State distribution - optimized
    console.print("\n[bold]State Distribution:[/bold]")
    state_stats = df.group_by("state").agg([pl.len().alias("count"), (pl.len() / len(df) * 100).alias("percentage")]).sort("count", descending=True)

    state_table = Table(box=box.SIMPLE)
    state_table.add_column("State", style="yellow")
    state_table.add_column("Count", justify="right")
    state_table.add_column("Percentage", justify="right")

    for row in state_stats.iter_rows(named=True):
        state_table.add_row(row["state"], f"{row['count']:,}", f"{row['percentage']:.1f}%")

    console.print(state_table)


@app.command()
def status(
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Data directory (default: ./data)")] = None,
) -> None:
    """Show monitoring system status."""
    console.print(Panel.fit("[bold cyan]SLURM Job Monitor Status[/bold cyan]", border_style="cyan"))

    config = Config.create(data_dir=data_dir)

    if not config.data_dir.exists():
        console.print("[yellow]No data directory found[/yellow]")
        return

    # Count all parquet files
    data_files: list[Path] = []
    for item in config.data_dir.iterdir():
        if item.is_dir():
            data_files.extend(item.glob("*.parquet"))
        elif item.suffix == ".parquet":
            data_files.append(item)

    data_files = sorted(data_files)

    status_table = Table(title="System Status", box=box.ROUNDED)
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", justify="center")
    status_table.add_column("Details", style="dim")

    status_table.add_row(
        "Data Directory",
        "[green]✓[/green]",
        str(config.data_dir),
    )

    status_table.add_row(
        "Data Files",
        f"[green]{len(data_files)}[/green]",
        f"Latest: {data_files[-1].name if data_files else 'None'}",
    )

    # Quick count of recent records
    total_records = sum(len(pl.read_parquet(f)) for f in data_files[-5:]) if data_files else 0

    status_table.add_row("Recent Records", f"[cyan]{total_records:,}[/cyan]", "From last 5 files")

    console.print(status_table)

    if data_files:
        total_size = sum(f.stat().st_size for f in data_files)
        console.print(f"\n[bold]Disk Usage:[/bold] {total_size / (1024**2):.1f} MB")


@app.command()
def current() -> None:
    """Display current cluster usage statistics from squeue."""
    output = squeue_output()
    me = getuser()
    for which in ["cores", "nodes"]:
        data, total_partition, totals = process_data(output, which)
        table = Table(title=f"SLURM statistics [b]{which}[/]", show_footer=True)
        partitions = sorted(total_partition.keys())
        table.add_column("User", f"{len(data)} users", style="cyan")
        for partition in partitions:
            tot = summarize_status(total_partition[partition])
            table.add_column(partition, tot, style="magenta")
        table.add_column("Total", summarize_status(totals), style="magenta")

        for user, _stats in sorted(data.items()):
            kw = {"style": "bold italic"} if user == me else {}
            partition_stats = [summarize_status(_stats[p]) if p in _stats else "-" for p in partitions]
            table.add_row(user, *partition_stats, summarize_status(combine_statuses(_stats)), **kw)
        console.print(table, justify="center")


@app.command()
def nodes() -> None:
    """Display node information from SLURM."""
    console.print(Panel.fit("[bold cyan]SLURM Node Information[/bold cyan]", border_style="cyan"))

    # Get node information
    node_info = _get_node_info_from_slurm()

    if not node_info:
        console.print("[yellow]No node information available[/yellow]")
        return

    # Create table
    node_table = Table(title="Cluster Nodes", box=box.ROUNDED)
    node_table.add_column("Node", style="cyan")
    node_table.add_column("CPUs", justify="right", style="yellow")
    node_table.add_column("GPUs", justify="right", style="green")

    # Sort nodes by name
    for node_name in sorted(node_info.keys()):
        info = node_info[node_name]
        node_table.add_row(
            node_name,
            str(info["cpus"]),
            str(info["gpus"]) if info["gpus"] > 0 else "-",
        )

    console.print(node_table)

    # Summary statistics
    total_nodes = len(node_info)
    total_cpus = sum(info["cpus"] for info in node_info.values())
    total_gpus = sum(info["gpus"] for info in node_info.values())
    gpu_nodes = sum(1 for info in node_info.values() if info["gpus"] > 0)

    summary_table = Table(title="Summary", box=box.SIMPLE)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Total Nodes", str(total_nodes))
    summary_table.add_row("Total CPUs", f"{total_cpus:,}")
    summary_table.add_row("Total GPUs", str(total_gpus))
    summary_table.add_row("GPU Nodes", str(gpu_nodes))

    console.print("\n")
    console.print(summary_table)


@app.command()
def test() -> None:
    """Run a quick test of the system."""
    console.print("[cyan]Running system test...[/cyan]")

    # Test sacct access
    try:
        result = run_sacct_version()
        if result.returncode == 0:
            console.print("[green]✓ sacct is accessible[/green]")
        else:
            console.print("[red]✗ sacct not accessible[/red]")
            raise typer.Exit(1)  # noqa: TRY301
    except Exception as e:
        console.print(f"[red]✗ Error testing sacct: {e}[/red]")
        raise typer.Exit(1) from e

    # Test data collection for today
    console.print("\n[cyan]Testing data collection for today...[/cyan]")

    config = Config.create()
    config.ensure_directories_exist()
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    fetch_result = _fetch_jobs_for_date(
        today,
        config,
        skip_if_complete=False,
    )

    if fetch_result.raw_records:
        console.print(f"[green]✓ Found {len(fetch_result.raw_records)} raw records[/green]")
        console.print(f"[green]✓ Processed {len(fetch_result.processed_jobs)} jobs[/green]")
        if fetch_result.raw_records:
            sample = fetch_result.raw_records[0]
            console.print(f"  Sample user: {sample.User}")
            console.print(f"  Sample job: {sample.JobName}")
            console.print(f"  Main job: {sample.is_main_job}")
            console.print(f"  Batch step: {sample.is_batch_step}")
            console.print(f"  Is finished: {sample.is_finished}")
    else:
        console.print("[yellow]⚠ No data found (this may be normal if no recent jobs)[/yellow]")

    console.print("\n[bold green]✓ System test complete[/bold green]")


if __name__ == "__main__":
    app()
