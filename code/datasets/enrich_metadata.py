"""
Temporary enrichment script — fills sql_command and target_table for rows
where those columns are NaN, using regex over the query text.

Only touches rows where the column is currently NaN (does not overwrite rbsqli
values). Writes to a temp file first; replaces the original only on success.
"""

import os
import re
from pathlib import Path

import pandas as pd

DATASET_PATH = Path("code/datasets/compiled/compiled_dataset.csv")
TEMP_PATH    = DATASET_PATH.with_name("compiled_dataset_tmp.csv")
CHUNK_SIZE   = 200_000

SQL_COMMANDS = [
    "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "EXEC", "EXECUTE", "CALL", "TRUNCATE", "REPLACE", "MERGE", "WITH",
    "DESCRIBE", "SHOW",
]
# Command must appear at the very start of the query (optional whitespace)
CMD_PATTERN   = r"^\s*(" + "|".join(SQL_COMMANDS) + r")\b"
# First word after FROM anywhere in the query
TABLE_PATTERN = r"\bFROM\s+(\w+)"

total_rows   = 0
cmd_filled   = 0
table_filled = 0
first_chunk  = True
LOG_EVERY    = 10  # print a progress line every N chunks

print(f"Source : {DATASET_PATH}")
print(f"Temp   : {TEMP_PATH}")
print()

for i, chunk in enumerate(pd.read_csv(
    DATASET_PATH, chunksize=CHUNK_SIZE, low_memory=False,
    dtype={"sql_command": "object", "target_table": "object"},
)):
    total_rows += len(chunk)

    # --- sql_command ---
    mask_cmd = chunk["sql_command"].isna() & chunk["query"].notna()
    if mask_cmd.any():
        extracted_cmd = (
            chunk.loc[mask_cmd, "query"]
            .astype(str)
            .str.extract(CMD_PATTERN, flags=re.IGNORECASE, expand=False)
            .str.upper()
        )
        n = int(extracted_cmd.notna().sum())
        cmd_filled += n
        chunk.loc[mask_cmd, "sql_command"] = extracted_cmd

    # --- target_table ---
    mask_tbl = chunk["target_table"].isna() & chunk["query"].notna()
    if mask_tbl.any():
        extracted_tbl = (
            chunk.loc[mask_tbl, "query"]
            .astype(str)
            .str.extract(TABLE_PATTERN, flags=re.IGNORECASE, expand=False)
        )
        n = int(extracted_tbl.notna().sum())
        table_filled += n
        chunk.loc[mask_tbl, "target_table"] = extracted_tbl

    chunk.to_csv(TEMP_PATH, mode="a" if not first_chunk else "w", header=first_chunk, index=False)
    first_chunk = False

    if (i + 1) % LOG_EVERY == 0:
        print(f"  {total_rows:>12,} rows  |  sql_command +{cmd_filled:,}  |  target_table +{table_filled:,}")

# Final progress line
print(f"  {total_rows:>12,} rows  |  sql_command +{cmd_filled:,}  |  target_table +{table_filled:,}")

print("\nReplacing original with enriched file ...")
os.replace(TEMP_PATH, DATASET_PATH)

print(f"\nDone.")
print(f"  Total rows processed : {total_rows:,}")
print(f"  sql_command filled   : {cmd_filled:,}")
print(f"  target_table filled  : {table_filled:,}")
