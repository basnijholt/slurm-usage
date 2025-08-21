"""Shared test fixtures and configuration."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Enable mock data mode for all tests
os.environ["SLURM_USE_MOCK_DATA"] = "1"

# Load the reference date from metadata - fail if not found
metadata_path = Path(__file__).parent / "snapshots" / "metadata.json"
if not metadata_path.exists():
    msg = f"Mock data snapshots not found at {metadata_path}. Run 'python tests/capture_snapshots.py' to generate them."
    raise FileNotFoundError(msg)

with metadata_path.open() as f:
    metadata = json.load(f)
    # Get the reference date - the "today" when snapshots were captured
    MOCK_DATA_REFERENCE_DATE = datetime.strptime(metadata["reference_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)


@pytest.fixture
def mock_datetime_now() -> Generator[MagicMock, None, None]:
    """Fixture that mocks datetime.now() to return the reference date."""
    with patch("slurm_usage.datetime") as mock_dt:
        # Make datetime.now() return our reference date
        mock_dt.now.return_value = MOCK_DATA_REFERENCE_DATE
        # Keep other datetime functionality
        mock_dt.fromisoformat = datetime.fromisoformat
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)  # noqa: DTZ001
        yield mock_dt


@pytest.fixture
def test_dates() -> dict[str, str]:
    """Provide consistent test dates based on mock data reference date."""
    # Use the reference date (the "today" when snapshots were captured)
    base_date = MOCK_DATA_REFERENCE_DATE.date()

    return {
        "today": base_date.isoformat(),
        "yesterday": (base_date - timedelta(days=1)).isoformat(),
        "week_ago": (base_date - timedelta(days=7)).isoformat(),
        "tomorrow": (base_date + timedelta(days=1)).isoformat(),
        "two_days_ago": (base_date - timedelta(days=2)).isoformat(),
        "today_iso": f"{base_date.isoformat()}T10:00:00",
        "yesterday_iso": f"{(base_date - timedelta(days=1)).isoformat()}T10:00:00",
    }


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    return data_dir
