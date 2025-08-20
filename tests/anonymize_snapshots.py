#!/usr/bin/env python3
"""Anonymize Slurm snapshots by replacing sensitive information.

This script should NOT be committed to the repository.
It contains logic to anonymize real data from your Slurm cluster.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

# Add parent directory to path to import slurm_usage
sys.path.insert(0, str(Path(__file__).parent.parent))

import slurm_usage

TEST_FOLDER = Path(__file__).parent


# Get field names from RawJobRecord to know the indices
_SACCT_FIELDS = slurm_usage.RawJobRecord.get_field_names()
_LEN_SACCT_FIELDS = len(_SACCT_FIELDS)

# Field indices we need to anonymize
_JOB_NAME_IDX = _SACCT_FIELDS.index("JobName")
_USER_IDX = _SACCT_FIELDS.index("User")
_GROUP_IDX = _SACCT_FIELDS.index("Group")
_ACCOUNT_IDX = _SACCT_FIELDS.index("Account")
_NODELIST_IDX = _SACCT_FIELDS.index("NodeList")
_WORKDIR_IDX = _SACCT_FIELDS.index("WorkDir")
_CLUSTER_IDX = _SACCT_FIELDS.index("Cluster")
_PARTITION_IDX = _SACCT_FIELDS.index("Partition")
_SUBMIT_LINE_IDX = _SACCT_FIELDS.index("SubmitLine")
_MAX_RSS_NODE_IDX = _SACCT_FIELDS.index("MaxRSSNode")
_MAX_VM_SIZE_NODE_IDX = _SACCT_FIELDS.index("MaxVMSizeNode")


class SimpleAnonymizer:
    """Simple anonymizer for Slurm outputs."""

    def __init__(self) -> None:
        """Initialize with replacement mappings."""
        self.user_map: dict[str, str] = {}
        self.job_name_map: dict[str, str] = {}
        self.node_map: dict[str, str] = {}
        self.cluster_map: dict[str, str] = {}
        self.partition_map: dict[str, str] = {}

        # Predefined test usernames to use
        self.test_users = [
            "alice",
            "bob",
            "charlie",
            "diana",
            "eve",
            "frank",
            "grace",
            "henry",
            "iris",
            "jack",
            "kate",
            "liam",
            "maya",
            "noah",
            "olivia",
            "peter",
            "quinn",
            "rachel",
            "sam",
            "tina",
            "uma",
            "victor",
            "wendy",
            "xavier",
            "yara",
            "zoe",
        ]
        self.user_counter = 0
        self.job_counter = 0
        self.node_counter = 0
        self.cluster_counter = 0
        self.partition_counter = 0

    def get_user(self, original: str) -> str:
        """Get or create anonymous username."""
        if not original or original in ["", "None", "N/A"]:
            return original

        if original not in self.user_map:
            if self.user_counter < len(self.test_users):
                self.user_map[original] = self.test_users[self.user_counter]
            else:
                self.user_map[original] = f"user{self.user_counter:03d}"
            self.user_counter += 1
        return self.user_map[original]

    def get_job_name(self, original: str) -> str:
        """Get or create anonymous job name."""
        if not original or original in ["", "None", "N/A"]:
            return original

        if original not in self.job_name_map:
            # Simple job naming
            self.job_name_map[original] = f"job_{self.job_counter:04d}"
            self.job_counter += 1
        return self.job_name_map[original]

    def get_node(self, original: str) -> str:
        """Get or create anonymous node name."""
        if not original or original in ["", "None", "N/A"]:
            return original

        if original not in self.node_map:
            # Create anonymous node name
            self.node_map[original] = f"node-{self.node_counter:03d}"
            self.node_counter += 1
        return self.node_map[original]

    def get_cluster(self, original: str) -> str:
        """Get or create anonymous cluster name."""
        if not original or original in ["", "None", "N/A"]:
            return original

        if original not in self.cluster_map:
            # Create anonymous cluster name
            self.cluster_map[original] = f"cluster-{self.cluster_counter:02d}"
            self.cluster_counter += 1
        return self.cluster_map[original]

    def get_partition(self, original: str) -> str:
        """Get or create anonymous partition name."""
        if not original or original in ["", "None", "N/A"]:
            return original

        if original not in self.partition_map:
            # Create anonymous partition name
            self.partition_map[original] = f"partition-{self.partition_counter:02d}"
            self.partition_counter += 1
        return self.partition_map[original]

    def anonymize_node_list(self, node_list: str) -> str:
        """Anonymize a node list expression like 'node[001-003]'."""
        if not node_list or node_list in ["", "None", "N/A"]:
            return node_list

        # Parse node list using slurm_usage's parser
        nodes = slurm_usage.parse_node_list(node_list)
        if not nodes:
            return node_list

        # Anonymize each node
        anon_nodes = [self.get_node(node) for node in nodes]

        # For simplicity, return as comma-separated list
        # In real SLURM, this could be compressed back to ranges
        if len(anon_nodes) == 1:
            return anon_nodes[0]
        # Try to compress back to range format if sequential
        return ",".join(anon_nodes)

    def anonymize_content(self, content: str, file_name: str) -> str:  # noqa: C901, PLR0912, PLR0915
        """Anonymize content based on file type."""
        # No longer doing global replacement - will handle via mappings

        lines = content.split("\n")
        anonymized_lines = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                anonymized_lines.append(line)
                continue

            # Handle sacct output (pipe-separated)
            if "sacct" in file_name and "|" in line:
                parts = line.split("|")
                assert len(parts) == _LEN_SACCT_FIELDS
                # JobName
                if parts[_JOB_NAME_IDX]:
                    parts[_JOB_NAME_IDX] = self.get_job_name(parts[_JOB_NAME_IDX])

                # User
                if parts[_USER_IDX]:
                    parts[_USER_IDX] = self.get_user(parts[_USER_IDX])

                # Group
                if parts[_GROUP_IDX]:
                    parts[_GROUP_IDX] = self.get_user(parts[_GROUP_IDX])

                # Account (might be empty)
                if parts[_ACCOUNT_IDX]:
                    parts[_ACCOUNT_IDX] = self.get_user(parts[_ACCOUNT_IDX])

                # NodeList - anonymize node names
                if parts[_NODELIST_IDX]:
                    parts[_NODELIST_IDX] = self.anonymize_node_list(parts[_NODELIST_IDX])
                # WorkDir - might contain sensitive paths
                if parts[_WORKDIR_IDX]:
                    # Anonymize any user-specific paths in WorkDir
                    workdir = parts[_WORKDIR_IDX]
                    if "/home/" in workdir or "/users/" in workdir.lower():
                        parts[_WORKDIR_IDX] = "/work/project"

                # Cluster - anonymize cluster name
                if parts[_CLUSTER_IDX]:
                    parts[_CLUSTER_IDX] = self.get_cluster(parts[_CLUSTER_IDX])

                # SubmitLine - anonymize the submit command
                if parts[_SUBMIT_LINE_IDX]:
                    # Replace any paths and sensitive info in submit line
                    submit_line = parts[_SUBMIT_LINE_IDX]
                    # Anonymize script paths
                    if ".sh" in submit_line or ".py" in submit_line or ".slurm" in submit_line:
                        parts[_SUBMIT_LINE_IDX] = "sbatch script.sh"
                    elif submit_line and submit_line != "":
                        # Keep sbatch but anonymize the rest
                        parts[_SUBMIT_LINE_IDX] = "sbatch job_script"

                # Partition - anonymize partition names
                if parts[_PARTITION_IDX]:
                    parts[_PARTITION_IDX] = self.get_partition(parts[_PARTITION_IDX])

                # MaxRSSNode - anonymize node names in this field
                if parts[_MAX_RSS_NODE_IDX] and parts[_MAX_RSS_NODE_IDX] not in ["", "Unknown"]:
                    parts[_MAX_RSS_NODE_IDX] = self.get_node(parts[_MAX_RSS_NODE_IDX])

                # MaxVMSizeNode - anonymize node names in this field
                if parts[_MAX_VM_SIZE_NODE_IDX] and parts[_MAX_VM_SIZE_NODE_IDX] not in ["", "Unknown"]:
                    parts[_MAX_VM_SIZE_NODE_IDX] = self.get_node(parts[_MAX_VM_SIZE_NODE_IDX])

                anonymized_lines.append("|".join(parts))

            # Handle squeue output (slash-separated)
            elif "squeue" in file_name and "/" in line and not line.startswith("USER"):
                parts = line.split("/")
                # User is first field (index 0)

                parts[0] = self.get_user(parts[0])
                # Partition is fourth field (index 3)
                # Anonymize all partition names
                parts[3] = self.get_partition(parts[3])
                # NodeList is sixth field (index 5)
                # Check if it's a node name (not empty)
                node_name = parts[5]
                if node_name and node_name not in ["", "(null)"]:
                    parts[5] = self.get_node(node_name)
                anonymized_lines.append("/".join(parts))

            # Handle scontrol output including the name files
            elif "scontrol" in file_name:
                # Special handling for scontrol_node_*_name.txt files
                if "scontrol_node" in file_name and "_name.txt" in file_name:
                    # These files contain just the node name
                    anonymized_lines.append(self.get_node(line.strip()))
                else:
                    # Regular scontrol output
                    line_anon = line
                    # Look for NodeName= pattern
                    if "NodeName=" in line:
                        match = re.search(r"NodeName=(\S+)", line)
                        if match:
                            orig_node = match.group(1)
                            anon_node = self.get_node(orig_node)
                            line_anon = line_anon.replace(f"NodeName={orig_node}", f"NodeName={anon_node}")
                    # Handle NodeHostName separately
                    if "NodeHostName=" in line_anon:
                        match = re.search(r"NodeHostName=(\S+)", line_anon)
                        if match:
                            orig_host = match.group(1)
                            anon_host = self.get_node(orig_host)
                            line_anon = line_anon.replace(f"NodeHostName={orig_host}", f"NodeHostName={anon_host}")
                    # Handle NodeAddr separately
                    if "NodeAddr=" in line_anon:
                        match = re.search(r"NodeAddr=(\S+)", line_anon)
                        if match:
                            orig_addr = match.group(1)
                            anon_addr = self.get_node(orig_addr)
                            line_anon = line_anon.replace(f"NodeAddr={orig_addr}", f"NodeAddr={anon_addr}")
                    # Handle Partitions field which may contain node names
                    if "Partitions=" in line_anon:
                        match = re.search(r"Partitions=(\S+)", line_anon)
                        if match:
                            partitions = match.group(1)
                            # Split by comma if multiple partitions
                            partition_list = partitions.split(",")
                            anon_partitions = [self.get_partition(p) for p in partition_list]
                            anon_partition_str = ",".join(anon_partitions)
                            line_anon = line_anon.replace(f"Partitions={partitions}", f"Partitions={anon_partition_str}")
                    anonymized_lines.append(line_anon)

            # Handle sinfo output
            elif "sinfo" in file_name:
                # Anonymize node names in sinfo output (comma-separated)
                if "," in line:
                    parts = line.split(",")
                    if len(parts) >= 1:
                        # First field is node name
                        parts[0] = self.get_node(parts[0])
                    anonymized_lines.append(",".join(parts))
                else:
                    anonymized_lines.append(line)

            else:
                # For any other content, just use the line as-is
                anonymized_lines.append(line)

        return "\n".join(anonymized_lines)

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """Process a single file."""
        content = input_path.read_text()

        # Skip certain files that don't need anonymization
        if any(x in input_path.name for x in ["returncode", "stderr", "version"]):
            output_path.write_text(content)
            return

        anonymized = self.anonymize_content(content, input_path.name)
        output_path.write_text(anonymized)

    def save_mappings(self, output_dir: Path) -> None:
        """Save the anonymization mappings (DO NOT COMMIT THIS)."""
        mappings = {
            "users": self.user_map,
            "job_names": self.job_name_map,
            "nodes": self.node_map,
            "clusters": self.cluster_map,
        }

        mapping_file = output_dir / "anonymization_mappings.json"
        with mapping_file.open("w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2, sort_keys=True)

        print(f"Mappings saved to {mapping_file}")
        print("WARNING: Do NOT commit this file - it contains the mapping of real to fake data!")


def main() -> None:
    """Main anonymization process."""
    input_dir = TEST_FOLDER / "snapshots_raw"
    output_dir = TEST_FOLDER / "snapshots"

    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        print("Please run capture_snapshots.py first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    anonymizer = SimpleAnonymizer()

    print("Anonymizing snapshots...")

    # Process all files
    for input_file in input_dir.glob("*"):
        if input_file.is_file():
            output_file = output_dir / input_file.name
            print(f"  - Processing {input_file.name}")

            if input_file.name == "command_map.json":
                # Anonymize the command_map.json file
                with open(input_file) as f:
                    command_map = json.load(f)

                # Anonymize node names in the commands
                anonymized_map = {}
                for cmd, file_prefix in command_map.items():
                    # Replace node names in scontrol commands
                    if "scontrol show node" in cmd:
                        # Extract the node name
                        parts = cmd.split()
                        if len(parts) == 4:  # scontrol show node <nodename>  # noqa: PLR2004
                            node_name = parts[3]
                            anon_node = anonymizer.get_node(node_name)
                            anon_cmd = f"scontrol show node {anon_node}"
                            anonymized_map[anon_cmd] = file_prefix
                        else:
                            anonymized_map[cmd] = file_prefix
                    else:
                        anonymized_map[cmd] = file_prefix

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(anonymized_map, f, indent=2)
            elif input_file.suffix == ".json":
                # Copy other JSON files as-is (like metadata.json)
                shutil.copy2(input_file, output_file)
            else:
                anonymizer.process_file(input_file, output_file)

    # Save mappings for reference (DO NOT COMMIT)
    anonymizer.save_mappings(output_dir)

    print(f"\nAnonymized snapshots saved to {output_dir}")
    print("\nIMPORTANT:")
    print("1. Review the anonymized files to ensure no sensitive data remains")
    print("2. Delete or .gitignore the anonymization_mappings.json file")
    print("3. Only commit the anonymized snapshots, not the originals")


if __name__ == "__main__":
    main()
