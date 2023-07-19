#!/usr/bin/env python
"""Command to list the current cluster usage per user.
Part of the [slurm-usage](https://github.com/basnijholt/slurm-usage) library.
"""
from getpass import getuser
import re
import subprocess
from collections import defaultdict

from rich.console import Console
from rich.table import Table


def squeue_output():
    cmd = [f"squeue -ro '%u/%t/%D/%P'"]
    return subprocess.getoutput(cmd).split("\n")[1:]


def process_data(output, cores_or_nodes):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_partition = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for out in output:
        user, status, nnodes, partition = out.split("/")
        nnodes = int(nnodes)
        n = nnodes * (1 if cores_or_nodes == "nodes" else get_ncores(partition))
        data[user][partition][status] += n
        total_partition[partition][status] += n
        totals[status] += n
    return data, total_partition, totals


def summarize_status(d):
    return " / ".join([f"{status}={n}" for status, n in d.items()])


def combine_statuses(d):
    tot = defaultdict(int)
    for partition, dct in d.items():
        for status, n in dct.items():
            tot[status] += n
    return dict(tot)


def get_max_lengths(rows):
    max_lengths = [0] * len(rows[0])
    for row in rows:
        for i, entry in enumerate(row):
            max_lengths[i] = max(len(entry), max_lengths[i])
    return max_lengths


def get_ncores(partition):
    numbers = re.findall(r'\d+', partition)
    return int(numbers[0])


def main():
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
            kw = dict(style="bold italic") if user == me else {}
            partition_stats = [
                summarize_status(_stats[p]) if p in _stats else "-" for p in partitions
            ]
            table.add_row(
                user, *partition_stats, summarize_status(combine_statuses(_stats)), **kw
            )
        console = Console()
        console.print(table, justify="center")


if __name__ == "__main__":
    main()
