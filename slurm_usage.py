#!/usr/bin/env python
"""Command to list the current cluster usage per user.

Part of the [slurm-usage](https://github.com/basnijholt/slurm-usage) library.
"""
import subprocess
from collections import defaultdict


base = "\033[{}m"
bold_start, bold_end = base.format(1), base.format(0)
green_start, color_end = base.format(92), base.format("0;0;0")
blue_start = base.format(94)


def squeue_output():
    cmd = [f"squeue -o '%u/%t/%D/%P'"]
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
    name, prio = partition.split("-")
    return int("".join([x for x in name if x.isdigit()]))


def get_rows(data, total_partition, totals):
    partitions = sorted(total_partition.keys())
    headers = ["user", *partitions, "total"]
    rows = [headers]
    users = sorted(data.keys())
    for user in users:
        _data = data[user]
        row = [user]
        for partition in partitions:
            if partition in _data:
                __data = _data[partition]
                row.append(summarize_status(__data))
            else:
                row.append("-")
        row.append(summarize_status(combine_statuses(_data)))
        rows.append(row)
    total_row = (
        [f"{len(rows)} users"]
        + [summarize_status(total_partition[partition]) for partition in partitions]
        + [summarize_status(totals)]
    )
    rows.append(total_row)
    return rows


def format_rows(rows):
    new_rows = []
    max_lengths = get_max_lengths(rows)
    for row in rows:
        _row = [entry.ljust(max_length) for max_length, entry in zip(max_lengths, row)]
        new_rows.append(_row)

    seperators = [f"{bold_start}-{bold_end}" * max_length for max_length in max_lengths]

    new_rows.insert(1, seperators)
    new_rows.insert(-1, seperators)

    new_rows[0] = [f"{bold_start}{e}{bold_end}" for e in new_rows[0]]
    new_rows[-1] = [f"{blue_start}{e}{color_end}" for e in new_rows[-1]]

    for row in new_rows[2:-2]:
        row[0] = green_start + row[0] + color_end
        row[-1] = blue_start + row[-1] + color_end
    return new_rows


def main():
    output = squeue_output()
    for which in ["cores", "nodes"]:
        data, total_partition, totals = process_data(output, which)
        print(f"Total number of {blue_start}{bold_start}{which}{bold_end}{color_end}.")
        rows = get_rows(data, total_partition, totals)
        rows = format_rows(rows)

        for row in rows:
            print(" | ".join(row))
        print()

if __name__ == "__main__":
    main()
