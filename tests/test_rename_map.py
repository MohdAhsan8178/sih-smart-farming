import re
from pathlib import Path
import pandas as pd
import pytest

RENAME_MAP_PATH = Path("data/packaged_min/splits_packaged/rename_map.csv")

@pytest.fixture(scope="module")
def rename_df():
    assert RENAME_MAP_PATH.exists(), f"Rename map not found at {RENAME_MAP_PATH}"
    df = pd.read_csv(RENAME_MAP_PATH)
    return df

def test_rename_map_row_count(rename_df):
    assert len(rename_df) == 41583, f"Expected exactly 41583 entries, got {len(rename_df)}"

def test_destination_paths_sanitized(rename_df):
    for dst in rename_df["new_relative_path"]:
        dst_path = Path(dst)
        fname = dst_path.name
        assert not re.search(r"[\s\(\)\[\]]", fname), f"Filename contains whitespace or brackets: {fname}"
        assert re.match(r"^[a-zA-Z0-9._-]+$", fname), f"Filename does not match strict regex: {fname}"
        for part in dst_path.parts:
            assert not re.search(r"[\s\(\)\[\]]", part), f"Path part contains whitespace or brackets: {part} in {dst}"
            assert re.match(r"^[a-zA-Z0-9._-]+$", part), f"Path part does not match strict regex: {part} in {dst}"

def test_destination_paths_unique(rename_df):
    dst_list = list(rename_df["new_relative_path"])
    assert len(dst_list) == len(set(dst_list)), "Duplicate destination paths found in rename_map!"

def test_newly_ingested_files_exist(rename_df):
    new_entries = rename_df.tail(6021)
    for dst in new_entries["new_relative_path"]:
        raw_path = Path("data/raw") / dst
        assert raw_path.exists(), f"File missing in data/raw: {raw_path}"
        assert raw_path.is_file(), f"Not a file: {raw_path}"
        assert raw_path.stat().st_size > 0, f"File empty: {raw_path}"

def test_newly_ingested_packaged_files_exist(rename_df):
    new_entries = rename_df.tail(6021)
    for dst in new_entries["new_relative_path"]:
        pkg_path = Path("data/packaged") / dst
        assert pkg_path.exists(), f"File missing in data/packaged: {pkg_path}"
        assert pkg_path.is_file(), f"Not a file: {pkg_path}"
        assert pkg_path.stat().st_size > 0, f"File empty: {pkg_path}"
