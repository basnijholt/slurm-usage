#!/usr/bin/env python
"""Command to list the current cluster usage per user."""
import subprocess


def is_running(line):
    return line.split("_")[0] == "R"


def get_n(line):
    return int(float(line.split("_")[-1]))


def main():
    active_users = set(subprocess.getoutput("squeue -o %u").split("\n")[1:])

    usage = {}
    for user in active_users:
        cmd = [f"squeue -u {user} -o '%t_%C'"]
        output = subprocess.getoutput(cmd).split("\n")[1:]
        n_cores = sum(get_n(i) for i in output if is_running(i))
        if n_cores > 0:
            usage[user] = n_cores

    n_total = sum(usage.values())

    base = "\033[{}m"
    bold_start, bold_end = base.format(1), base.format(0)
    green_start, color_end = base.format(92), base.format("0;0;0")
    blue_start = base.format(94)
    dotted_line = f"{bold_start}{23 * '-'}{bold_end}"

    print(f"{bold_start}User{9 * ' '}# of cores{bold_end}")
    print(dotted_line)
    for user in sorted(usage, key=lambda user: usage[user], reverse=True):
        print(f"{green_start}{user}".ljust(16), color_end, usage[user])
    print(dotted_line)
    print(f"{blue_start}Total{7 * ' '}{color_end} {n_total}")


if __name__ == "__main__":
    main()
