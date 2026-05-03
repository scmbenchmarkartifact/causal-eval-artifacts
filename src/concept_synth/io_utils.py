"""
io_utils.py - YAML I/O utilities for concept_synth

Provides:
- load_from_yaml: Load data from a YAML file with locking
- save_to_yaml: Save data to a YAML file atomically with optional key ordering
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.reader import ReaderError


def clean_yaml_file(filename: Union[str, Path], backup_suffix: str = ".backup") -> str:
    """
    Clean a YAML file by removing invalid control characters.
    Returns the path to the cleaned file.
    """
    filename_str = str(filename)

    # Create backup
    backup_filename = filename_str + backup_suffix
    with open(filename_str, "r", encoding="utf-8") as f:
        original_content = f.read()

    with open(backup_filename, "w", encoding="utf-8") as f:
        f.write(original_content)

    # Remove non-printable control characters (except newline, tab, carriage return)
    cleaned_content = "".join(
        char for char in original_content if char.isprintable() or char in "\n\t\r"
    )

    # Write cleaned content back
    with open(filename_str, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    return filename_str


def _try_clean_then_reload(fname: Path):
    """Helper to clean a YAML file and return a file handle for reloading."""
    print(f"Warning: YAML file {fname} contains invalid characters. Trying to clean…")
    clean_yaml_file(fname)
    return open(fname, "r", encoding="utf-8")


def load_from_yaml(filename: str) -> Any:
    """
    Load data from a YAML file with shared locking to prevent reading
    while another process is writing.

    Args:
        filename: Path to the YAML file

    Returns:
        Parsed YAML data (dict, list, or other YAML-compatible type)
    """
    yaml_loader = YAML()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
            try:
                data = yaml_loader.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (ReaderError, UnicodeDecodeError):
        # Fallback: clean invalid chars and reload
        with _try_clean_then_reload(Path(filename)) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                data = yaml_loader.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    return data


def save_to_yaml(data: Any, filename: str, key_order: Optional[List[str]] = None) -> None:
    """
    Save a dictionary or list of dictionaries to a YAML file with controlled key order.

    Uses atomic write (write to temp file, then rename) to prevent race conditions
    where readers might see a partially written or empty file.

    Args:
        data: The dictionary or list of dictionaries to save
        filename: The name of the file to save the YAML to
        key_order: Optional list of keys specifying the order for dictionary keys
    """
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 4096  # Avoid unnecessary wrapping in YAML files
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True

    def represent_multiline_string(dumper, value):
        if "\n" in value:
            return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
        elif "\\n" in value:
            return dumper.represent_scalar("tag:yaml.org,2002:str", value)
        return dumper.represent_scalar("tag:yaml.org,2002:str", value)

    yaml.representer.add_representer(str, represent_multiline_string)
    yaml.representer.add_representer(
        OrderedDict, lambda dumper, d: dumper.represent_mapping("tag:yaml.org,2002:map", d)
    )

    def reorder_dict(d, key_order):
        """Recursive reordering function for nested dictionaries."""
        if isinstance(d, dict):
            reordered = OrderedDict(
                (key, reorder_dict(d[key], key_order)) for key in key_order if key in d
            )
            # Add any remaining keys not in key_order
            reordered.update(
                (key, reorder_dict(value, key_order))
                for key, value in d.items()
                if key not in key_order
            )
            return reordered
        elif isinstance(d, CommentedSeq):
            # Preserve CommentedSeq (keeps flow-style metadata for compact predicates)
            return d
        elif isinstance(d, list):
            return [
                reorder_dict(item, key_order) if isinstance(item, (dict, list)) else item
                for item in d
            ]
        return d

    if key_order:
        data = reorder_dict(data, key_order)

    # Atomic write: write to temp file, then rename
    dir_name = os.path.dirname(filename) or "."
    os.makedirs(dir_name, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
        # Atomic rename (on POSIX systems)
        os.replace(temp_path, filename)
    except:
        # Clean up temp file if something goes wrong
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
