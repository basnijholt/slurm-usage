"""Test configuration data directory determination logic."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable mock data mode for all tests
os.environ["SLURM_USE_MOCK_DATA"] = "1"

import slurm_usage


class TestConfigDataDir:
    """Test data directory configuration and auto-determination."""

    def test_config_data_dir_none_default(self) -> None:
        """Test that data_dir defaults to ./data when not specified."""
        with patch("slurm_usage._load_config_file") as mock_load:
            mock_load.return_value = ({}, None)  # No config file
            config = slurm_usage.Config.create()
            # Should use default ./data directory
            assert config.data_dir == Path("data")

    def test_config_explicit_data_dir(self) -> None:
        """Test explicit data_dir specification."""
        custom_dir = Path("/custom/data/dir")
        config = slurm_usage.Config.create(data_dir=custom_dir)
        assert config.data_dir == custom_dir

    def test_config_default_data_dir(self) -> None:
        """Test that default data_dir is ./data when no config file."""
        with patch("slurm_usage._load_config_file") as mock_load:
            mock_load.return_value = ({}, None)
            config = slurm_usage.Config.create()
            # Should use default ./data directory
            assert config.data_dir == Path("data")

    def test_config_data_dir_from_file(self, tmp_path: Path) -> None:
        """Test loading data_dir from config file."""
        config_dir = tmp_path / ".config" / "slurm-usage"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Write config with explicit data_dir
        config_data = {"groups": {"team1": ["alice", "bob"]}, "data_dir": "/custom/shared/data"}
        config_file.write_text(yaml.dump(config_data))

        # Mock the config loading to use our test file
        with patch("slurm_usage._load_config_file") as mock_load:
            mock_load.return_value = (config_data, config_file)

            config = slurm_usage.Config.create()
            assert config.data_dir == Path("/custom/shared/data")

    def test_config_data_dir_null_in_file(self, tmp_path: Path) -> None:
        """Test that null/None data_dir in config file uses default."""
        config_dir = tmp_path / ".config" / "slurm-usage"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Write config with null data_dir
        config_data = {
            "groups": {"team1": ["alice", "bob"]},
            "data_dir": None,  # Explicitly null
        }
        config_file.write_text(yaml.dump(config_data))

        # Mock the config loading to use our test file
        with patch("slurm_usage._load_config_file") as mock_load:
            mock_load.return_value = (config_data, config_file)
            config = slurm_usage.Config.create()
            # Should use default ./data directory
            assert config.data_dir == Path("data")

    def test_config_no_data_dir_uses_default(self) -> None:
        """Test that when config exists but no data_dir specified, use default."""
        config_path = Path("/etc/slurm-usage/config.yaml")

        with patch("slurm_usage._load_config_file") as mock_load:
            # Config file found, but no data_dir specified
            mock_load.return_value = ({"groups": {"team1": ["alice"]}}, config_path)
            config = slurm_usage.Config.create()
            # Should use default ./data directory
            assert config.data_dir == Path("data")

    def test_cli_commands_accept_data_dir(self) -> None:
        """Test that CLI commands properly handle data_dir parameter."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test with custom data_dir
        custom_dir = "/tmp/custom_data"  # noqa: S108
        result = runner.invoke(slurm_usage.app, ["status", "--data-dir", custom_dir])
        # Should not error
        assert result.exit_code == 0

        # Test without data_dir (should use auto-determination)
        result = runner.invoke(slurm_usage.app, ["status"])
        assert result.exit_code == 0
