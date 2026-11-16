"""
NEUROLECT AI - File Management Utility
Handles file saving, safe naming, temporary file extraction, and automatic cleanup.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
import config

def save_uploaded_file(uploaded_file, destination_dir: Optional[Path] = None) -> Path:
    """Saves an uploaded Streamlit file to a secure local path and returns the Path."""
    target_dir = destination_dir or config.UPLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename to avoid collision while preserving original extension
    suffix = Path(uploaded_file.name).suffix.lower()
    clean_stem = "".join(c for c in Path(uploaded_file.name).stem if c.isalnum() or c in ("-", "_"))[:30]
    unique_name = f"{clean_stem}_{uuid.uuid4().hex[:8]}{suffix}"
    destination_path = target_dir / unique_name
    
    with open(destination_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return destination_path

def cleanup_file(file_path: Optional[Path]) -> bool:
    """Safely removes a file if it exists."""
    if file_path is None:
        return False
    try:
        p = Path(file_path)
        if p.exists() and p.is_file():
            p.unlink()
            return True
    except Exception as e:
        print(f"[Warning] Failed to delete temporary file {file_path}: {e}")
    return False

def clear_directory(directory_path: Path):
    """Safely clears old files in a temporary directory."""
    if not directory_path.exists():
        return
    for item in directory_path.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception:
            pass
