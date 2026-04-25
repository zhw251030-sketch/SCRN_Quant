"""Project-local paths used by the SCRN/BRECQ integration code."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRN_ROOT = PROJECT_ROOT / "SCRN-main"
BRECQ_ROOT = PROJECT_ROOT / "BRECQ-main"
APP_ROOT = PROJECT_ROOT / "SCRN_BRECQ_app"

DEFAULT_SCRN_CHECKPOINT = APP_ROOT / "checkpoints" / "scrn_repro.pth"


def add_to_import_path(path: Path) -> None:
    """Make a source tree importable without editing the original repository."""
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def require_existing_path(path: Path, description: str) -> Path:
    """Return path when it exists, otherwise raise a clear setup error."""
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path
