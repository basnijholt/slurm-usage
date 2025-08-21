#!/usr/bin/env python3
"""Example code for post-processing collected data with Polars."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

# Load processed data for last 7 days
dfs = []
for i in range(7):
    date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
    file = Path(f"data/processed/{date}.parquet")
    if file.exists():
        dfs.append(pl.read_parquet(file))

if dfs:
    df = pl.concat(dfs)

    # Find users with worst CPU efficiency
    worst_users = df.filter(pl.col("state") == "COMPLETED").group_by("user").agg(pl.col("cpu_efficiency").mean()).sort("cpu_efficiency").head(5)

    print("## Users with Worst CPU Efficiency")
    print(worst_users)

    # Find most wasted resources by partition
    waste_by_partition = df.group_by("partition").agg(pl.col("cpu_hours_wasted").sum()).sort("cpu_hours_wasted", descending=True)

    print("\n## CPU Hours Wasted by Partition")
    print(waste_by_partition)
else:
    print("No data files found. Run `./slurm_usage.py collect` first.")
