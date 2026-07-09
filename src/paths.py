"""src/paths.py
Canonical path definitions for the Perch-Hoplite pipeline at MBARI.

All permanent data lives under /mnt/PAM_Analysis/perch-hoplite/ (on spark)
or /Volumes/PAM_Analysis/perch-hoplite/ (on Mac). The old duane_scratch
location is kept as a read fallback with a warning so we can find and migrate
any remaining files.

Usage:
    from src.paths import Paths
    db_dir   = Paths.db("MARS_20180401_20180430_32kHz")
    model    = Paths.model("orca_v9_clean.pt")
    results  = Paths.results("MARS_20180401_20180430_v9_clean_detections.csv")
"""
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Root detection — works on spark (Linux) and Mac
# ---------------------------------------------------------------------------

def _find_pam_root() -> Path:
    """Return the PAM_Analysis mount point for this machine."""
    candidates = [
        Path("/mnt/PAM_Analysis"),        # spark-ae0e, spark-0626
        Path("/Volumes/PAM_Analysis"),    # ICEFISH (Mac)
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise RuntimeError(
        "Cannot find PAM_Analysis mount. "
        "Check NFS/SMB mount: thalassa.shore.mbari.org:/PAM_Analysis"
    )


PAM_ROOT    = _find_pam_root()
PERCH_ROOT  = PAM_ROOT / "perch-hoplite"       # permanent location
WEIGHTS_DIR = PAM_ROOT / "perch_weights"        # Perch V2 weights (ONNX-extracted)

# Legacy scratch location — read-only fallback
_SCRATCH_ROOT = PAM_ROOT / "duane_scratch" / "perch_hoplite"

# Sub-directories under PERCH_ROOT
_SUBDIRS = ("db", "models", "results", "logs", "provenance",
            "provenance/labels", "provenance/training")


def ensure_dirs():
    """Create all sub-directories under PERCH_ROOT if they don't exist."""
    for sub in _SUBDIRS:
        (PERCH_ROOT / sub).mkdir(parents=True, exist_ok=True)


def _resolve(subdir: str, name: str, create_parent: bool = False) -> Path:
    """Return canonical path, falling back to duane_scratch with a warning."""
    canonical = PERCH_ROOT / subdir / name
    if canonical.exists():
        return canonical
    # Check new root without subdir (e.g. model files stored flat)
    flat = PERCH_ROOT / name
    if flat.exists():
        return flat
    # Fall back to scratch
    scratch = _SCRATCH_ROOT / subdir / name
    if scratch.exists():
        log.warning(
            "PATH FALLBACK: %s not found at %s — using legacy duane_scratch path %s. "
            "Please migrate: cp %s %s",
            name, canonical, scratch, scratch, canonical,
        )
        return scratch
    # Neither exists — return canonical (caller will create or raise)
    if create_parent:
        canonical.parent.mkdir(parents=True, exist_ok=True)
    return canonical


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Paths:
    """Namespace for canonical pipeline paths."""

    # ── Roots ────────────────────────────────────────────────────────────────
    root    = PERCH_ROOT
    weights = WEIGHTS_DIR
    scratch = _SCRATCH_ROOT   # read-only fallback

    # ── Sub-directories ──────────────────────────────────────────────────────
    @staticmethod
    def db_dir(name: str = "") -> Path:
        """Return path to a DB directory, e.g. Paths.db_dir('MARS_20180401_20180430_32kHz')."""
        if not name:
            return PERCH_ROOT / "db"
        return _resolve("db", name, create_parent=False)

    @staticmethod
    def model(name: str) -> Path:
        """Return path to a model file, e.g. Paths.model('orca_v9_clean.pt')."""
        return _resolve("models", name, create_parent=True)

    @staticmethod
    def results(name: str = "") -> Path:
        """Return path to a results file or the results directory."""
        if not name:
            return PERCH_ROOT / "results"
        return _resolve("results", name, create_parent=True)

    @staticmethod
    def log(name: str) -> Path:
        """Return path to a log file."""
        return _resolve("logs", name, create_parent=True)

    @staticmethod
    def provenance(subdir: str, name: str) -> Path:
        """Return path to a provenance file, e.g. Paths.provenance('labels', 'session.json')."""
        return _resolve(f"provenance/{subdir}", name, create_parent=True)

    @staticmethod
    def audio(year: int, month: int) -> Path:
        """Return path to resampled 32kHz audio for a given year/month."""
        return (PAM_ROOT / "GoogleMultiSpeciesWhaleModel2" /
                "resampled_32kHz" / str(year) / f"{month:02d}")

    @staticmethod
    def perch_weights() -> Path:
        """Return path to Perch V2 weights directory."""
        return WEIGHTS_DIR

    @staticmethod
    def exact_mel() -> Path | None:
        """Return path to the exact mel reference array, or None if not found."""
        p = PAM_ROOT / "perch_weights" / "const__pad1_output_0.npy"
        if p.exists():
            return p
        # Check pytorch repo
        pytorch_dir = Path.home() / "perch-pytorch"
        p2 = pytorch_dir / "const__pad1_output_0.npy"
        if p2.exists():
            return p2
        return None

    # ── Migration helper ─────────────────────────────────────────────────────
    @staticmethod
    def migration_report() -> dict:
        """Report files that exist in duane_scratch but not in PERCH_ROOT."""
        missing = {}
        for sub in ("db", "models", "results"):
            scratch_sub = _SCRATCH_ROOT / sub
            perch_sub   = PERCH_ROOT / sub
            if not scratch_sub.is_dir():
                continue
            for item in sorted(scratch_sub.iterdir()):
                canonical = perch_sub / item.name
                if not canonical.exists():
                    missing.setdefault(sub, []).append(str(item))
        return missing


# ---------------------------------------------------------------------------
# CLI — run as script to show migration report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"PAM_ROOT    : {PAM_ROOT}")
    print(f"PERCH_ROOT  : {PERCH_ROOT}")
    print(f"WEIGHTS_DIR : {WEIGHTS_DIR}")
    print(f"SCRATCH     : {_SCRATCH_ROOT}")
    print()

    ensure_dirs()
    print("Sub-directories ensured.")
    print()

    report = Paths.migration_report()
    if not report:
        print("✅ No migration needed — all scratch files are present in PERCH_ROOT.")
    else:
        print("⚠  Files in duane_scratch not yet migrated to PERCH_ROOT:")
        for sub, items in report.items():
            print(f"\n  {sub}/")
            for item in items:
                canonical = PERCH_ROOT / sub / Path(item).name
                print(f"    cp -r {item}")
                print(f"         {canonical}")
