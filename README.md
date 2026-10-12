# SLURM Usage Monitor

[![PyPI](https://img.shields.io/pypi/v/slurm-usage.svg)](https://pypi.python.org/pypi/slurm-usage)
[![Build Status](https://github.com/basnijholt/slurm-usage/actions/workflows/pytest-uv.yml/badge.svg)](https://github.com/basnijholt/slurm-usage/actions/workflows/pytest-uv.yml)
[![Documentation](https://img.shields.io/badge/docs-slurm--usage.nijho.lt-blue)](https://slurm-usage.nijho.lt)

<!-- SECTION:intro:START -->
<img src="https://raw.githubusercontent.com/basnijholt/slurm-usage/main/docs/assets/logo.svg" alt="slurm-usage Logo" align="right" style="width: 150px;" />

A high-performance monitoring system that collects and analyzes SLURM job efficiency metrics, optimized for large-scale HPC environments.

> [!TIP]
> SLURM purges detailed metrics after 30 days. Run `slurm-usage collect` to preserve your data!
<!-- SECTION:intro:END -->

<!-- SECTION:purpose:START -->
## Purpose

SLURM's accounting database purges detailed job metrics (CPU usage, memory usage) after 30 days. This tool captures and preserves that data in efficient Parquet format for long-term analysis of resource utilization patterns.
<!-- SECTION:purpose:END -->

<!-- SECTION:features:START -->
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
<!-- SECTION:features:END -->

## Table of Contents

<details><summary>Click to expand</summary>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [What It Collects](#what-it-collects)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Quick Start (no installation needed)](#quick-start-no-installation-needed)
  - [Install as a Tool](#install-as-a-tool)
  - [Run from Source](#run-from-source)
- [Usage](#usage)
  - [CLI Commands](#cli-commands)
    - [Example Commands](#example-commands)
  - [Command Options](#command-options)
    - [`collect` - Gather job data from SLURM](#collect---gather-job-data-from-slurm)
    - [`analyze` - Analyze collected data](#analyze---analyze-collected-data)
    - [`status` - Show system status](#status---show-system-status)
    - [`current` - Display current cluster usage](#current---display-current-cluster-usage)
    - [`nodes` - Display node information](#nodes---display-node-information)
    - [`test` - Test system configuration](#test---test-system-configuration)
- [Output Structure](#output-structure)
  - [Data Organization](#data-organization)
  - [Sample Analysis Output](#sample-analysis-output)
- [Smart Re-collection](#smart-re-collection)
  - [Tracked Incomplete States](#tracked-incomplete-states)
- [Group Configuration](#group-configuration)
  - [Data Directory](#data-directory)
- [Automated Collection](#automated-collection)
  - [Using Cron](#using-cron)
- [Data Schema](#data-schema)
  - [ProcessedJob Model](#processedjob-model)
- [Performance Optimizations](#performance-optimizations)
- [Important Notes](#important-notes)
- [Post-Processing with Polars](#post-processing-with-polars)
- [Troubleshooting](#troubleshooting)
- [Support](#support)
- [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

</details>

<!-- SECTION:what-it-collects:START -->
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
<!-- SECTION:what-it-collects:END -->

<!-- SECTION:requirements:START -->
## Requirements

- **uv** - Python package and project manager (will auto-install dependencies)
- **SLURM** with accounting enabled
- **sacct** command access

That's it! The script uses `uv` inline script dependencies, so all Python packages are automatically installed when you run the script.
<!-- SECTION:requirements:END -->

<!-- SECTION:installation:START -->
## Installation

### Quick Start (no installation needed)

```bash
# Run directly with uvx (uv tool run)
uvx slurm-usage --help

# Or for a specific command
uvx slurm-usage collect --days 7
```

### Install as a Tool

```bash
# Install globally with uv
uv tool install slurm-usage

# Or with pip
pip install slurm-usage

# Then use directly
slurm-usage --help
```

### Run from Source

```bash
# Clone the repository
git clone https://github.com/basnijholt/slurm-usage
cd slurm-usage

# Run the script directly (dependencies auto-installed by uv)
./slurm_usage.py --help

# Or with Python
python slurm_usage.py --help
```
<!-- SECTION:installation:END -->

<!-- SECTION:usage:START -->
## Usage

### CLI Commands

The following commands are available:

<!-- CODE:BASH:START -->
<!-- echo '```bash' -->
<!-- ./slurm_usage.py --help 2>/dev/null || echo "slurm_usage.py [commands]" -->
<!-- echo '```' -->
<!-- CODE:END -->

<!-- OUTPUT:START -->
<!-- ⚠️ This content is auto-generated by `markdown-code-runner`. -->
```bash

 Usage: slurm_usage.py [OPTIONS] COMMAND [ARGS]...

 SLURM Job Monitor - Collect and analyze job efficiency metrics


╭─ Options ──────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                │
│ --show-completion             Show completion for the current shell, to copy it or     │
│                               customize the installation.                              │
│ --help                        Show this message and exit.                              │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────╮
│ collect   Collect job data from SLURM using parallel date-based queries.               │
│ analyze   Analyze collected job data.                                                  │
│ status    Show monitoring system status.                                               │
│ current   Display current cluster usage statistics from squeue.                        │
│ nodes     Display node information from SLURM.                                         │
│ test      Run a quick test of the system.                                              │
╰────────────────────────────────────────────────────────────────────────────────────────╯

```

<!-- OUTPUT:END -->

#### Example Commands

```bash
# Collect data (uses 4 parallel workers by default)
slurm-usage collect

# Collect last 7 days of data
slurm-usage collect --days 7

# Collect with more parallel workers
slurm-usage collect --n-parallel 8

# Analyze collected data
slurm-usage analyze --days 7

# Display current cluster usage
slurm-usage current

# Display node information
slurm-usage nodes

# Check system status
slurm-usage status

# Test system configuration
slurm-usage test
```

Note: If running from source, use `./slurm_usage.py` instead of `slurm-usage`.

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

#### `current` - Display current cluster usage

Shows real-time cluster utilization from `squeue`, broken down by user and partition.

#### `nodes` - Display node information

Shows information about cluster nodes including CPU and GPU counts.

#### `test` - Test system configuration

<!-- SECTION:usage:END -->

<!-- SECTION:output-structure:START -->
## Output Structure

### Data Organization

```
data/
├── raw/                        # Raw SLURM data (archived)
│   ├── 2025-08-19.parquet      # Daily raw records
│   ├── 2025-08-20.parquet
│   └── ...
├── processed/                  # Processed job metrics
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
<!-- SECTION:output-structure:END -->

<!-- SECTION:smart-recollection:START -->
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
<!-- SECTION:smart-recollection:END -->

<!-- SECTION:configuration:START -->
## Group Configuration

Create a configuration file to define your organization's research groups and optionally specify the data directory. The configuration file is searched in the following locations:

1. `$XDG_CONFIG_HOME/slurm-usage/config.yaml`
2. `~/.config/slurm-usage/config.yaml`
3. `/etc/slurm-usage/config.yaml`

### Data Directory

The data directory for storing collected metrics can be configured in three ways (in order of priority):

1. **Command line**: Use `--data-dir /path/to/data` with any command (highest priority)

2. **Configuration file**: Set `data_dir: /path/to/data` in the config file

3. **Default**: If not specified, data is stored in `./data` (current working directory)

This allows flexible deployment:
- **Default installation**: Data stored in `./data` subdirectory
- **System-wide deployment**: Set `data_dir: /var/lib/slurm-usage` in `/etc/slurm-usage/config.yaml`
- **Shared installations**: Use a network storage path in the config
- **Per-run override**: Use `--data-dir` flag to override for specific commands

Example `config.yaml`:

<!-- CODE:BASH:START -->
<!-- echo '```yaml' -->
<!-- cat config.example.yaml -->
<!-- echo '' -->
<!-- echo '```' -->
<!-- CODE:END -->

<!-- OUTPUT:START -->
<!-- ⚠️ This content is auto-generated by `markdown-code-runner`. -->
```yaml
# Example configuration file for slurm-usage
# Copy this file to one of the following locations:
#   - $XDG_CONFIG_HOME/slurm-usage/config.yaml
#   - ~/.config/slurm-usage/config.yaml
#   - /etc/slurm-usage/config.yaml (for system-wide configuration)

# Group configuration - organize users into research groups
groups:
  physics:
    - alice
    - bob
    - charlie
  chemistry:
    - david
    - eve
    - frank
  biology:
    - grace
    - henry
    - irene

# Data directory configuration (optional)
# - If not specified or set to null, defaults to ./data (current working directory)
# - Set to an explicit path to use a custom location
# - Useful for shared installations where data should be stored centrally
#
# Examples:
# data_dir: null                    # Use default ./data directory
# data_dir: /var/lib/slurm-usage    # System-wide data directory
# data_dir: /shared/slurm-data      # Shared network location

```

<!-- OUTPUT:END -->
<!-- SECTION:configuration:END -->

<!-- SECTION:automated-collection:START -->
## Automated Collection

### Using Cron

```bash
# Add to crontab (runs daily at 2 AM)
crontab -e

# If installed with uv tool or pip:
0 2 * * * /path/to/slurm-usage collect --days 2

# Or if running from source:
0 2 * * * /path/to/slurm-usage/slurm_usage.py collect --days 2
```
<!-- SECTION:automated-collection:END -->

## Data Schema

### ProcessedJob Model

<!-- CODE:START -->
<!-- import sys -->
<!-- sys.path.insert(0, '.') -->
<!-- import slurm_usage -->
<!-- from pydantic import BaseModel -->
<!-- -->
<!-- # Get the ProcessedJob model fields -->
<!-- fields = [] -->
<!-- for field_name, field_info in slurm_usage.ProcessedJob.model_fields.items(): -->
<!--     if field_name == 'processed_date': -->
<!--         continue  # Skip internal field -->
<!--     field_type = field_info.annotation -->
<!--     # Handle type annotations -->
<!--     if hasattr(field_type, '__name__'): -->
<!--         type_str = field_type.__name__ -->
<!--     elif hasattr(field_type, '__origin__'):  # Handle Optional, Union, etc -->
<!--         type_str = str(field_type).replace('typing.', '').replace('|', ' or ') -->
<!--     else: -->
<!--         type_str = str(field_type) -->
<!--     -->
<!--     # Get field description from docstring or field metadata -->
<!--     descriptions = { -->
<!--         'job_id': 'SLURM job ID', -->
<!--         'user': 'Username', -->
<!--         'job_name': 'Job name (max 50 chars)', -->
<!--         'partition': 'SLURM partition', -->
<!--         'state': 'Final job state', -->
<!--         'submit_time': 'ISO format submission time', -->
<!--         'start_time': 'ISO format start time', -->
<!--         'end_time': 'ISO format end time', -->
<!--         'node_list': 'Nodes where job ran', -->
<!--         'elapsed_seconds': 'Runtime in seconds', -->
<!--         'alloc_cpus': 'CPUs allocated', -->
<!--         'req_mem_mb': 'Memory requested (MB)', -->
<!--         'max_rss_mb': 'Peak memory used (MB)', -->
<!--         'total_cpu_seconds': 'Actual CPU time used', -->
<!--         'alloc_gpus': 'GPUs allocated', -->
<!--         'cpu_efficiency': 'CPU efficiency % (0-100)', -->
<!--         'memory_efficiency': 'Memory efficiency % (0-100)', -->
<!--         'cpu_hours_wasted': 'Wasted CPU hours', -->
<!--         'memory_gb_hours_wasted': 'Wasted memory GB-hours', -->
<!--         'cpu_hours_reserved': 'Total CPU hours reserved', -->
<!--         'memory_gb_hours_reserved': 'Total memory GB-hours reserved', -->
<!--         'gpu_hours_reserved': 'Total GPU hours reserved', -->
<!--         'is_complete': 'Whether job has reached final state', -->
<!--     } -->
<!--     desc = descriptions.get(field_name, '') -->
<!--     fields.append((field_name, type_str, desc)) -->
<!-- -->
<!-- # Print as markdown table -->
<!-- print('| Field | Type | Description |') -->
<!-- print('|-------|------|-------------|') -->
<!-- for field_name, type_str, desc in fields: -->
<!--     print(f'| {field_name} | {type_str} | {desc} |') -->
<!-- CODE:END -->
<!-- OUTPUT:START -->
<!-- ⚠️ This content is auto-generated by `markdown-code-runner`. -->
| Field | Type | Description |
|-------|------|-------------|
| job_id | str | SLURM job ID |
| user | str | Username |
| job_name | str | Job name (max 50 chars) |
| partition | str | SLURM partition |
| state | str | Final job state |
| submit_time | datetime.datetime | None | ISO format submission time |
| start_time | datetime.datetime | None | ISO format start time |
| end_time | datetime.datetime | None | ISO format end time |
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

<!-- OUTPUT:END -->

<!-- SECTION:performance-optimizations:START -->
## Performance Optimizations

- **Date completion tracking**: Dates with only finished jobs are marked complete and skipped
- **Parallel collection**: Default 4 workers fetch different dates simultaneously
- **Smart merging**: Only updates changed jobs when re-collecting
- **Efficient storage**: Parquet format provides ~10x compression over CSV
- **Date-based partitioning**: Data organized by date for efficient queries
<!-- SECTION:performance-optimizations:END -->

<!-- SECTION:important-notes:START -->
## Important Notes

1. **30-day window**: SLURM purges detailed metrics after 30 days. Run collection at least weekly to ensure no data is lost.

2. **Batch steps**: Actual usage metrics (TotalCPU, MaxRSS) are stored in the `.batch` step, not the parent job record.

3. **State normalization**: All CANCELLED variants are normalized to "CANCELLED" for consistency.

4. **GPU tracking**: GPU allocation is extracted from the AllocTRES field.

5. **Raw data archival**: Raw SLURM records are preserved in case reprocessing is needed.
<!-- SECTION:important-notes:END -->

## Post-Processing with Polars

You can use Polars to analyze the collected data. Here's an example:

<!-- CODE:BASH:START -->
<!-- echo '```python' -->
<!-- tail -n +3 tests/snippets/polars_example.py -->
<!-- echo '```' -->
<!-- CODE:END -->

<!-- OUTPUT:START -->
<!-- ⚠️ This content is auto-generated by `markdown-code-runner`. -->
```python
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
```

<!-- OUTPUT:END -->

<!-- SECTION:troubleshooting:START -->
## Troubleshooting

**No efficiency data?**

- Check if SLURM accounting is configured: `scontrol show config | grep JobAcct`
- Verify jobs have `.batch` steps: `sacct -j JOBID`

**Collection is slow?**

- Increase parallel workers: `slurm-usage collect --n-parallel 8`
- The first run processes historical data and will be slower

**Missing user groups?**

- Create or update the configuration file in `~/.config/slurm-usage/config.yaml`
- Ungrouped users will appear as "ungrouped" in group statistics

**Script won't run?**

- Ensure `uv` is installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Check SLURM access: `slurm-usage test` (or `./slurm_usage.py test` if running from source)
<!-- SECTION:troubleshooting:END -->

<!-- SECTION:support:START -->
## Support

We appreciate your feedback and contributions! If you encounter any issues or have suggestions for improvements, please file an issue on the [GitHub repository](https://github.com/basnijholt/slurm-usage/issues). We also welcome pull requests for bug fixes or new features.
<!-- SECTION:support:END -->

## License

MIT

---

Give `slurm-usage` a try today and start preserving your HPC job efficiency data!
