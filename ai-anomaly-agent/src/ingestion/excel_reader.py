"""FR-01: Excel Upload.

Reads a .xlsx file into a pandas DataFrame. Kept intentionally dumb -
all validation happens in src/validation/validator.py so this module
has a single responsibility: get bytes off disk into a DataFrame.
"""
from __future__ import annotations

import os
import pandas as pd

ALLOWED_EXTENSIONS = {".xlsx"}
MAX_FILE_SIZE_MB = 25  # NFR: restrict processing to expected formats/sizes (Sec. 31)


class ExcelReadError(Exception):
    """Raised when the uploaded file cannot be read at all."""


def read_excel_file(file_path: str, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read an Excel file and return a raw (unvalidated) DataFrame.

    Raises ExcelReadError for missing files, wrong extensions, oversized
    files, or files openpyxl can't parse. Downstream validation should
    still be run - this only guarantees "we could open the file".
    """
    if not os.path.exists(file_path):
        raise ExcelReadError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ExcelReadError(f"Unsupported file type '{ext}'. Only .xlsx is accepted.")

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ExcelReadError(f"File too large ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB limit).")

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001 - surface a clean, user-facing error
        raise ExcelReadError(f"Could not parse Excel file: {exc}") from exc

    if df.empty:
        raise ExcelReadError("The uploaded Excel file contains no rows.")

    return df
