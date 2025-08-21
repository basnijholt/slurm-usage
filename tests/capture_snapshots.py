#!/usr/bin/env python3
"""Capture snapshots of Slurm command outputs for testing.

This script captures real Slurm command outputs and saves them as test fixtures.
Run this on a system with Slurm installed to generate test data.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import slurm_usage

TEST_FOLDER = Path(__file__).parent


def capture_snapshots() -> None:  # noqa: PLR0915
    """Capture snapshots of all Slurm commands and save them."""
    snapshots_dir = TEST_FOLDER / "snapshots_raw"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    print("Capturing Slurm command snapshots...")

    command_map = {}

    # 1. Capture squeue output
    print("  - Capturing squeue output...")
    squeue_result = slurm_usage.run_squeue()
    with open(snapshots_dir / "squeue_output.txt", "w", encoding="utf-8") as f:
        f.write(squeue_result.stdout)
    with open(snapshots_dir / "squeue_stderr.txt", "w", encoding="utf-8") as f:
        f.write(squeue_result.stderr)
    with open(snapshots_dir / "squeue_returncode.txt", "w", encoding="utf-8") as f:
        f.write(str(squeue_result.returncode))
    command_map[squeue_result.command] = "squeue"

    # 2. Capture sinfo CPU output
    print("  - Capturing sinfo CPU output...")
    sinfo_cpu_result = slurm_usage.run_sinfo_cpus()
    with open(snapshots_dir / "sinfo_cpus_output.txt", "w", encoding="utf-8") as f:
        f.write(sinfo_cpu_result.stdout)
    with open(snapshots_dir / "sinfo_cpus_stderr.txt", "w", encoding="utf-8") as f:
        f.write(sinfo_cpu_result.stderr)
    with open(snapshots_dir / "sinfo_cpus_returncode.txt", "w", encoding="utf-8") as f:
        f.write(str(sinfo_cpu_result.returncode))
    command_map[sinfo_cpu_result.command] = "sinfo_cpus"

    # 3. Capture sinfo GPU output
    print("  - Capturing sinfo GPU output...")
    sinfo_gpu_result = slurm_usage.run_sinfo_gpus()
    with open(snapshots_dir / "sinfo_gpus_output.txt", "w", encoding="utf-8") as f:
        f.write(sinfo_gpu_result.stdout)
    with open(snapshots_dir / "sinfo_gpus_stderr.txt", "w", encoding="utf-8") as f:
        f.write(sinfo_gpu_result.stderr)
    with open(snapshots_dir / "sinfo_gpus_returncode.txt", "w", encoding="utf-8") as f:
        f.write(str(sinfo_gpu_result.returncode))
    command_map[sinfo_gpu_result.command] = "sinfo_gpus"

    # 4. Capture sacct output for individual days
    print("  - Capturing sacct outputs for individual days...")

    # Get field names from RawJobRecord
    fields = slurm_usage.RawJobRecord.get_field_names()

    # Capture for today and the past week (individual days)
    today = datetime.now()
    days_to_capture = 8  # Today plus 7 days back

    for i in range(days_to_capture):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        print(f"    - Capturing data for {date_str}...")
        sacct_result = slurm_usage.run_sacct(date_str, fields)

        # Use consistent naming for all days
        file_prefix = f"sacct_day_{i}"

        with open(snapshots_dir / f"{file_prefix}_output.txt", "w", encoding="utf-8") as f:
            f.write(sacct_result.stdout)
        with open(snapshots_dir / f"{file_prefix}_stderr.txt", "w", encoding="utf-8") as f:
            f.write(sacct_result.stderr)
        with open(snapshots_dir / f"{file_prefix}_returncode.txt", "w", encoding="utf-8") as f:
            f.write(str(sacct_result.returncode))
        command_map[sacct_result.command] = file_prefix

    # 5. Capture scontrol show node for a few nodes (if we have any from sinfo)
    print("  - Capturing scontrol node outputs...")
    if sinfo_cpu_result.stdout:
        # Extract first few node names from sinfo output
        node_names = []
        for line in sinfo_cpu_result.stdout.strip().split("\n")[:3]:  # Get first 3 nodes
            if line:
                parts = line.strip().split(",")
                if parts:
                    node_names.append(parts[0])

        for i, node_name in enumerate(node_names):
            print(f"    - Node: {node_name}")
            scontrol_result = slurm_usage.run_scontrol_show_node(node_name)
            with open(snapshots_dir / f"scontrol_node_{i}_output.txt", "w", encoding="utf-8") as f:
                f.write(scontrol_result.stdout)
            with open(snapshots_dir / f"scontrol_node_{i}_stderr.txt", "w", encoding="utf-8") as f:
                f.write(scontrol_result.stderr)
            with open(snapshots_dir / f"scontrol_node_{i}_returncode.txt", "w", encoding="utf-8") as f:
                f.write(str(scontrol_result.returncode))
            # Save node name for reference
            with open(snapshots_dir / f"scontrol_node_{i}_name.txt", "w", encoding="utf-8") as f:
                f.write(node_name)
            command_map[scontrol_result.command] = f"scontrol_node_{i}"

    # 6. Capture sacct version
    print("  - Capturing sacct version...")
    version_result = slurm_usage.run_sacct_version()
    with open(snapshots_dir / "sacct_version_output.txt", "w", encoding="utf-8") as f:
        f.write(version_result.stdout)
    with open(snapshots_dir / "sacct_version_stderr.txt", "w", encoding="utf-8") as f:
        f.write(version_result.stderr)
    with open(snapshots_dir / "sacct_version_returncode.txt", "w", encoding="utf-8") as f:
        f.write(str(version_result.returncode))
    command_map[version_result.command] = "sacct_version"

    # Create command mapping for mock system
    with open(snapshots_dir / "command_map.json", "w", encoding="utf-8") as f:
        json.dump(command_map, f, indent=2)

    # Save metadata about the capture
    captured_dates = []
    for i in range(days_to_capture):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        captured_dates.append(date_str)

    metadata = {
        "capture_date": datetime.now().isoformat(),
        "reference_date": today.strftime("%Y-%m-%d"),  # The "today" when snapshots were captured
        "fields": fields,
        "captured_dates": captured_dates,
        "days_captured": days_to_capture,
    }
    with open(snapshots_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Snapshots saved to {snapshots_dir.absolute()}")
    print("\nNext steps:")
    print("1. Review the captured data for sensitive information")
    print("2. Run the anonymization script to sanitize the data")
    print("3. Commit the anonymized snapshots to the repository")


if __name__ == "__main__":
    capture_snapshots()
