#!/usr/bin/env python
"""Command to list the current cluster usage per user."""
import subprocess
from collections import defaultdict


base = "\033[{}m"
bold_start, bold_end = base.format(1), base.format(0)
green_start, color_end = base.format(92), base.format("0;0;0")
blue_start = base.format(94)


def squeue_output():
    cmd = [f"squeue -o '%u/%t/%D/%P'"]
    return subprocess.getoutput(cmd).split("\n")[1:]


def process_data(output):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_partition = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for out in output:
        user, status, n, partition = out.split("/")
        n = int(n)
        data[user][partition][status] += n
        total_partition[partition][status] += n
        totals[status] += n
    return data, total_partition, totals


def summarize_status(d, factor=1):
    return " / ".join([f"{status}={n*factor}" for status, n in d.items()])


def combine_statuses(d, cores_or_nodes="nodes"):
    tot = defaultdict(int)
    for partition, dct in d.items():
        for status, n in dct.items():
            if cores_or_nodes == "cores":
                n *= get_ncores(partition)
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


def get_rows(data, total_partition, totals, cores_or_nodes="nodes"):
    partitions = tuple(total_partition.keys())
    headers = ["user", *partitions, "total"]
    rows = [headers]
    for user, _data in data.items():
        row = [user]
        for partition in partitions:
            if partition in _data:
                __data = _data[partition]
                factor = get_ncores(partition) if cores_or_nodes == "cores" else 1
                row.append(summarize_status(__data, factor))
            else:
                row.append("-")
        row.append(summarize_status(combine_statuses(_data, cores_or_nodes)))
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

    seperators = ["-" * max_length for max_length in max_lengths]
    seperators[0] = bold_start + seperators[0]
    seperators[-1] = seperators[-1] + bold_end

    new_rows.insert(1, seperators)
    new_rows.insert(-1, seperators)

    total_row = new_rows[-1]
    total_row[0] = blue_start + total_row[0]
    total_row[-1] = total_row[-1] + color_end

    for row in new_rows[2:-2]:
        row[0] = green_start + row[0] + color_end
        row[-1] = blue_start + row[-1] + color_end
    return new_rows


def main():
    output = squeue_output()
    data, total_partition, totals = process_data(output)
    for which in ["cores", "nodes"]:
        print(f"Total number of {blue_start}{bold_start}{which}{bold_end}{color_end}")
        rows = get_rows(data, total_partition, totals, which)
        rows = format_rows(rows)

        for row in rows:
            print(" | ".join(row))
        print()

if __name__ == "__main__":
    main()
