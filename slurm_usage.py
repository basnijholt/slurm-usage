#!/usr/bin/env python
"""Command to list the current cluster usage per user.

Part of the [slurm-usage](https://github.com/basnijholt/slurm-usage) library.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from getpass import getuser
from typing import Any, NamedTuple

from rich.console import Console
from rich.table import Table


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
    cmd = ["squeue", "-ro", "%u/%t/%D/%P/%C/%N/%h"]
    # Get the output and skip the header
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    output = result.stdout.split("\n")[1:]
    return [SlurmJob.from_line(line) for line in output]


def get_total_cores(node_name: str) -> int:
    """Get total number of cores for a given node."""
    cmd = ["scontrol", "show", "node", node_name]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
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
                counted_nodes[s.user].add(
                    s.node,
                )  # Mark this node as counted for this user
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


def main() -> None:
    """Run the main slurm-usage command."""
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
            table.add_row(
                user,
                *partition_stats,
                summarize_status(combine_statuses(_stats)),
                **kw,
            )
        console = Console()
        console.print(table, justify="center")


if __name__ == "__main__":
    main()
