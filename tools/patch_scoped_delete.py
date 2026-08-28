#!/usr/bin/env python3
"""tools/patch_scoped_delete.py — one-time surgical fix to phase2_classify.py

PROBLEM: the Gradio review autosave path deletes ANY existing annotation for a window
before inserting, regardless of which annotator made it. A second annotator therefore
DESTROYS the first annotator's label rather than adding an opinion. Combined with the
review UI being blind by default (radio is hardcoded value="unlabeled", never reads the
DB), the second annotator overwrites labels they never saw.

FIX: scope the DELETE to the current annotator's provenance, so changing your own mind
still replaces your own label, but another annotator's label is preserved.

Run from the repo root. Idempotent: refuses to double-apply. Writes a .bak.
"""
import shutil
import sys
from pathlib import Path

TARGET = Path("phase2_classify.py")

OLD = '''            # DELETE existing annotation for this window first (any label),
            # then INSERT fresh.
            con.execute("""
                DELETE FROM annotations
                WHERE recording_id=? AND offsets=?
            """, (rec_id, off_enc))'''

NEW = '''            # DELETE only THIS ANNOTATOR's existing annotation for this window,
            # then INSERT fresh. Scoped by provenance so a second annotator ADDS an
            # opinion rather than destroying the first annotator's label. (Fixed
            # Aug 28 2026 -- the unscoped form silently overwrote other annotators,
            # which matters because the review UI is blind by default: the radio is
            # hardcoded value="unlabeled" and never reads existing labels from the DB,
            # so an annotator overwrote labels they could not even see.)
            con.execute("""
                DELETE FROM annotations
                WHERE recording_id=? AND offsets=? AND provenance=?
            """, (rec_id, off_enc, prov))'''


def main():
    if not TARGET.exists():
        sys.exit(f"ERROR: {TARGET} not found — run this from the repo root.")

    src = TARGET.read_text()

    if "AND provenance=?" in src and "Fixed\n            # Aug 28 2026" in src:
        print("Already patched — nothing to do.")
        return
    if src.count(OLD) == 0:
        sys.exit("ERROR: anchor text not found. The file has changed; patch by hand.\n"
                 "Look for: DELETE FROM annotations WHERE recording_id=? AND offsets=?")
    if src.count(OLD) > 1:
        sys.exit(f"ERROR: anchor found {src.count(OLD)} times; expected 1. Patch by hand.")

    bak = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(src.replace(OLD, NEW, 1))

    # verify
    import ast
    try:
        ast.parse(TARGET.read_text())
    except SyntaxError as e:
        shutil.copy2(bak, TARGET)
        sys.exit(f"ERROR: patch produced a syntax error ({e}); reverted from {bak}.")

    print(f"Patched {TARGET}  (backup: {bak})")
    print("Syntax OK. Verify with:")
    print('  grep -n "AND provenance=?" phase2_classify.py')


if __name__ == "__main__":
    main()
