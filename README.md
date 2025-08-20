<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [SLURM Job Efficiency Monitor](#slurm-job-efficiency-monitor)
  - [Purpose](#purpose)
  - [Key Features](#key-features)
  - [What It Collects](#what-it-collects)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [CLI Commands](#cli-commands)
    - [Command Options](#command-options)
      - [`collect` - Gather job data from SLURM](#collect---gather-job-data-from-slurm)
      - [`analyze` - Analyze collected data](#analyze---analyze-collected-data)
      - [`status` - Show system status](#status---show-system-status)
      - [`test` - Test system configuration](#test---test-system-configuration)
  - [Output Structure](#output-structure)
    - [Data Organization](#data-organization)
    - [Sample Analysis Output](#sample-analysis-output)
  - [Smart Re-collection](#smart-re-collection)
    - [Tracked Incomplete States](#tracked-incomplete-states)
  - [Group Configuration](#group-configuration)
  - [Automated Collection](#automated-collection)
    - [Using Cron](#using-cron)
    - [Using Systemd Timer](#using-systemd-timer)
  - [Data Schema](#data-schema)
    - [ProcessedJob Model](#processedjob-model)
  - [Performance Optimizations](#performance-optimizations)
  - [Important Notes](#important-notes)
  - [Post-Processing with Polars](#post-processing-with-polars)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# SLURM Job Efficiency Monitor

A high-performance monitoring system that collects and analyzes SLURM job efficiency metrics, optimized for large-scale HPC environments.

## Purpose

SLURM's accounting database purges detailed job metrics (CPU usage, memory usage) after 30 days. This tool captures and preserves that data in efficient Parquet format for long-term analysis of resource utilization patterns.

## Key Features

- 📊 **Captures comprehensive efficiency metrics** from all job states
- 💾 **Efficient Parquet storage** - columnar format optimized for analytics
- 🔄 **Smart incremental processing** - tracks completed dates to minimize re-processing
- 📈 **Rich visualizations** - bar charts for resource usage, efficiency, and node utilization
- 👥 **Group-based analytics** - track usage by research groups/teams
- 🖥️ **Node utilization tracking** - analyze per-node CPU and GPU usage
- ⚡ **Parallel collection** - multi-threaded data collection by default
- ⏰ **Cron-ready** - designed for automated daily collection
- 🎯 **Intelligent re-collection** - only re-fetches incomplete job states

## What It Collects

For each job:

- **Job metadata**: ID, user, name, partition, state, node list
- **Time info**: submit, start, end times, elapsed duration
- **Allocated resources**: CPUs, memory, GPUs, nodes
- **Actual usage**: CPU seconds used (TotalCPU), peak memory (MaxRSS)
- **Calculated metrics**:
  - CPU efficiency % (actual CPU time / allocated CPU time)
  - Memory efficiency % (peak memory / allocated memory)
  - CPU hours wasted
  - Memory GB-hours wasted
  - Total reserved resources (CPU/GPU/memory hours)

## Requirements

- **uv** - Python package and project manager (will auto-install dependencies)
- **SLURM** with accounting enabled
- **sacct** command access

That's it! The script uses `uv` inline script dependencies, so all Python packages are automatically installed when you run the script.

## Installation

```bash
# Clone the repository
git clone https://github.com/basnijholt/slurm-usage
cd slurm-usage

# Make script executable (if needed)
chmod +x slurm_usage.py

# That's it! Dependencies are auto-installed via uv when you run the script
```

## Usage

### CLI Commands

```bash
# Collect data (uses 4 parallel workers by default)
./slurm_usage.py collect

# Collect last 7 days of data
./slurm_usage.py collect --days 7

# Collect with more parallel workers
./slurm_usage.py collect --n-parallel 8

# Analyze collected data
./slurm_usage.py analyze --days 7

# Check system status
./slurm_usage.py status

# Test system configuration
./slurm_usage.py test
```

### Command Options

#### `collect` - Gather job data from SLURM
- `--days/-d`: Days to look back (default: 1)
- `--data-dir`: Data directory location (default: ./data)
- `--summary/--no-summary`: Show analysis after collection (default: True)
- `--n-parallel/-n`: Number of parallel workers (default: 4)

#### `analyze` - Analyze collected data
- `--days/-d`: Days to analyze (default: 7)
- `--data-dir`: Data directory location

#### `status` - Show system status
- `--data-dir`: Data directory location

#### `test` - Test system configuration

## Output Structure

### Data Organization
```
data/
├── raw/                          # Raw SLURM data (archived)
│   ├── 2025-08-19.parquet      # Daily raw records
│   ├── 2025-08-20.parquet
│   └── ...
├── processed/                   # Processed job metrics
│   ├── 2025-08-19.parquet      # Daily processed data
│   ├── 2025-08-20.parquet
│   └── ...
└── .date_completion_tracker.json  # Tracks fully processed dates
```

### Sample Analysis Output

```
═══ Resource Usage by User ═══

┌─────────────┬──────┬───────────┬──────────────┬───────────┬─────────┬──────────┐
│ User        │ Jobs │ CPU Hours │ Memory GB-hrs│ GPU Hours │ CPU Eff │ Mem Eff  │
├─────────────┼──────┼───────────┼──────────────┼───────────┼─────────┼──────────┤
│ alice       │  124 │   12,847  │    48,291    │    1,024  │  45.2%  │  23.7%   │
│ bob         │   87 │    8,234  │    31,456    │      512  │  38.1%  │  18.4%   │
└─────────────┴──────┴───────────┴──────────────┴───────────┴─────────┴──────────┘

═══ Node Usage Analysis ═══

┌────────────┬──────┬───────────┬───────────┬───────────┐
│ Node       │ Jobs │ CPU Hours │ GPU Hours │ CPU Util% │
├────────────┼──────┼───────────┼───────────┼───────────┤
│ cluster-1  │  234 │   45,678  │    2,048  │   74.3%   │
│ cluster-2  │  198 │   41,234  │    1,536  │   67.1%   │
└────────────┴──────┴───────────┴───────────┴───────────┘
```

## Smart Re-collection

The monitor intelligently handles job state transitions:

- **Complete dates**: Once all jobs for a date reach final states (COMPLETED, FAILED, CANCELLED, etc.), the date is marked complete and won't be re-queried
- **Incomplete jobs**: Jobs in states like RUNNING, PENDING, or SUSPENDED are automatically re-collected on subsequent runs
- **Efficient updates**: Only changed jobs are updated, minimizing processing time

### Tracked Incomplete States

The following job states indicate a job may change and will trigger re-collection:
- Active: `RUNNING`, `PENDING`, `SUSPENDED`
- Transitional: `COMPLETING`, `CONFIGURING`, `STAGE_OUT`, `SIGNALING`
- Requeue: `REQUEUED`, `REQUEUE_FED`, `REQUEUE_HOLD`
- Other: `RESIZING`, `REVOKED`, `SPECIAL_EXIT`

## Group Configuration

Edit the `GROUP_MAPPINGS` dictionary in `slurm_usage.py` to define your organization's research groups:

```python
GROUP_MAPPINGS = {
    "applications": ["user1", "user2"],
    "control": ["user3", "user4"],
    "physical": ["user5", "user6"],
    "qec": ["user7", "user8"],
    "software": ["user9", "user10"],
    "interns": ["intern1", "intern2"],
}
```

## Automated Collection

### Using Cron

```bash
# Add to crontab (runs daily at 2 AM)
crontab -e

# Add this line (collects last 2 days to catch any state changes):
0 2 * * * /path/to/slurm-usage/slurm_usage.py collect --days 2
```

### Using Systemd Timer

Create `/etc/systemd/system/slurm-usage.service`:
```ini
[Unit]
Description=SLURM Job Monitor Collection

[Service]
Type=oneshot
User=your-username
ExecStart=/path/to/slurm-usage/slurm_usage.py collect --days 2
```

Create `/etc/systemd/system/slurm-usage.timer`:
```ini
[Unit]
Description=Daily SLURM Job Monitor Collection

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

Enable the timer:
```bash
sudo systemctl enable --now slurm-usage.timer
```

## Data Schema

### ProcessedJob Model

| Field | Type | Description |
|-------|------|-------------|
| job_id | str | SLURM job ID |
| user | str | Username |
| job_name | str | Job name (max 50 chars) |
| partition | str | SLURM partition |
| state | str | Final job state |
| submit_time | str | ISO format submission time |
| start_time | str | ISO format start time |
| end_time | str | ISO format end time |
| node_list | str | Nodes where job ran |
| elapsed_seconds | int | Runtime in seconds |
| alloc_cpus | int | CPUs allocated |
| req_mem_mb | float | Memory requested (MB) |
| max_rss_mb | float | Peak memory used (MB) |
| total_cpu_seconds | float | Actual CPU time used |
| alloc_gpus | int | GPUs allocated |
| cpu_efficiency | float | CPU efficiency % (0-100) |
| memory_efficiency | float | Memory efficiency % (0-100) |
| cpu_hours_wasted | float | Wasted CPU hours |
| memory_gb_hours_wasted | float | Wasted memory GB-hours |
| cpu_hours_reserved | float | Total CPU hours reserved |
| memory_gb_hours_reserved | float | Total memory GB-hours reserved |
| gpu_hours_reserved | float | Total GPU hours reserved |
| is_complete | bool | Whether job has reached final state |

## Performance Optimizations

- **Date completion tracking**: Dates with only finished jobs are marked complete and skipped
- **Parallel collection**: Default 4 workers fetch different dates simultaneously
- **Smart merging**: Only updates changed jobs when re-collecting
- **Efficient storage**: Parquet format provides ~10x compression over CSV
- **Date-based partitioning**: Data organized by date for efficient queries

## Important Notes

1. **30-day window**: SLURM purges detailed metrics after 30 days. Run collection at least weekly to ensure no data is lost.

2. **Batch steps**: Actual usage metrics (TotalCPU, MaxRSS) are stored in the `.batch` step, not the parent job record.

3. **State normalization**: All CANCELLED variants are normalized to "CANCELLED" for consistency.

4. **GPU tracking**: GPU allocation is extracted from the AllocTRES field.

5. **Raw data archival**: Raw SLURM records are preserved in case reprocessing is needed.

## Post-Processing with Polars

```python
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path

# Load processed data for last 7 days
dfs = []
for i in range(7):
    date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
    file = Path(f"data/processed/{date}.parquet")
    if file.exists():
        dfs.append(pl.read_parquet(file))

df = pl.concat(dfs)

# Find users with worst CPU efficiency
worst_users = (
    df.filter(pl.col("state") == "COMPLETED")
    .group_by("user")
    .agg(pl.col("cpu_efficiency").mean())
    .sort("cpu_efficiency")
)

# Find most wasted resources by partition
waste_by_partition = (
    df.group_by("partition")
    .agg(pl.col("cpu_hours_wasted").sum())
    .sort("cpu_hours_wasted", descending=True)
)
```

## Troubleshooting

**No efficiency data?**
- Check if SLURM accounting is configured: `scontrol show config | grep JobAcct`
- Verify jobs have `.batch` steps: `sacct -j JOBID`

**Collection is slow?**
- Increase parallel workers: `./slurm_usage.py collect --n-parallel 8`
- The first run processes historical data and will be slower

**Missing user groups?**
- Update `GROUP_MAPPINGS` in the script to include all users
- Ungrouped users will appear as "ungrouped" in group statistics

**Script won't run?**
- Ensure `uv` is installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Check SLURM access: `./slurm_usage.py test`

## License

MIT
