#!/usr/bin/env python3
"""Guard the three annotation properties `wtc-validate` reports but never fails on.

`wtc-validate --strict` promotes only hash-mismatch warnings to errors, so a lab
can lose coverage, carry no `important` lines, or repeat one annotation across
unrelated code and CI stays green. This script gates those three:

  coverage   Share of *code* lines carrying an annotation, per lab, against a
             checked-in floor. Blank and comment-only lines are excluded, since
             counting them just dilutes the number.
  important  Count of `important` annotations per lab, against a floor and a
             shared ceiling. Too few leaves a lab with no headline; too many
             means nothing stands out.
  duplicates One annotation text on two lines whose code differs. That is
             usually a declaration and its use sharing a sentence, in which
             case the sentence is wrong on one of them.

Usage:
    python walk-the-code/check_annotations.py
    python walk-the-code/check_annotations.py --update-baseline
"""

import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMENTS = ROOT / "walk-the-code" / "comments"
BASELINE = ROOT / "walk-the-code" / "annotation_baseline.json"
LABS = ROOT / "labs"

IMPORTANT_CEILING = 30  # above this, "important" stops marking anything out
IMPORTANT_TARGET = 4  # below this a lab has no headline, reported but not failed


def code_lines(src):
    """Line numbers worth annotating: everything that is not blank or comment-only.

    A docstring line is left in. It is prose the tutorial may well want to
    narrate, and excluding it would need a real parse to find reliably.
    """
    return {i for i, line in enumerate(src, start=1) if line.strip() and not line.strip().startswith("#")}


def measure():
    """Per-lab annotation stats, keyed by lab id."""
    out = {}
    for json_path in sorted(COMMENTS.glob("*/*.json")):
        lab = json_path.parent.name
        source = LABS / lab / f"{json_path.stem}.py"
        if not source.exists():
            print(f"  ERROR {json_path.relative_to(ROOT)}: no source file at {source.relative_to(ROOT)}")
            sys.exit(1)
        src = source.read_text().split("\n")
        ann = json.loads(json_path.read_text())
        code = code_lines(src)
        anchored = {int(k) for k in ann}

        by_text = defaultdict(list)
        for key, entry in ann.items():
            by_text[entry["text"]].append(int(key))

        collisions = []
        for text, keys in by_text.items():
            if len(keys) < 2:
                continue
            # Repeating a sentence across identical lines (five `super().__init__()`
            # calls, say) is fine. Repeating it across lines that differ is not.
            if len({src[k - 1].strip() for k in keys}) > 1:
                collisions.append((sorted(keys), text))

        out[lab] = {
            "file": json_path.stem,
            "annotations": len(ann),
            "code_lines": len(code),
            "covered": len(anchored & code),
            "coverage_pct": round(len(anchored & code) / len(code) * 100, 1),
            "important": sum(1 for entry in ann.values() if entry.get("important")),
            "collisions": collisions,
        }
    return out


def check(stats, baseline):
    failures = []
    for lab, s in sorted(stats.items()):
        base = baseline.get(lab)
        if base is None:
            failures.append(f"{lab}: no baseline entry (run --update-baseline to add it)")
            continue
        if s["coverage_pct"] < base["coverage_pct"]:
            failures.append(
                f"{lab}: code-line coverage fell to {s['coverage_pct']}% "
                f"from a floor of {base['coverage_pct']}% "
                f"({s['covered']}/{s['code_lines']} lines). Annotate the new code, "
                f"or raise the floor deliberately with --update-baseline."
            )
        if s["important"] < base["important"]:
            failures.append(f"{lab}: {s['important']} important annotation(s), floor is {base['important']}")
        if s["important"] > IMPORTANT_CEILING:
            failures.append(
                f"{lab}: {s['important']} important annotations, ceiling is {IMPORTANT_CEILING}. "
                f"Demote the ones that are not headlines."
            )
        for keys, text in s["collisions"]:
            failures.append(
                f"{lab}: one annotation on {len(keys)} lines that differ "
                f"({', '.join(str(k) for k in keys)}): {text[:70]!r}"
            )

    failures.extend(f"{lab}: in the baseline but has no annotation file" for lab in sorted(set(baseline) - set(stats)))
    return failures


def thin(stats):
    """Labs carrying almost no `important` annotations.

    Reported on every run rather than failed, because the floor that gates is
    the one in the baseline. A lab listed here has a gap someone should fill,
    and once filled the baseline stops it slipping back.
    """
    return sorted(lab for lab, s in stats.items() if s["important"] < IMPORTANT_TARGET)


def report(stats):
    width = max(len(lab) for lab in stats)
    print(f"  {'lab':<{width}}  {'cov':>6}  {'covered':>13}  {'ann':>5}  {'imp':>4}")
    for lab, s in sorted(stats.items(), key=lambda kv: kv[1]["coverage_pct"]):
        covered = f"{s['covered']}/{s['code_lines']}"
        print(
            f"  {lab:<{width}}  {s['coverage_pct']:>5.1f}%  {covered:>13}  {s['annotations']:>5}  {s['important']:>4}"
        )
    total_cov = sum(s["covered"] for s in stats.values())
    total_code = sum(s["code_lines"] for s in stats.values())
    print(
        f"\n  overall: {total_cov}/{total_code} code lines ({total_cov / total_code * 100:.1f}%), "
        f"{sum(s['annotations'] for s in stats.values())} annotations, "
        f"{sum(s['important'] for s in stats.values())} important"
    )


def main():
    updating = "--update-baseline" in sys.argv[1:]
    stats = measure()
    print(f"walk-the-code annotations: {len(stats)} labs\n")
    report(stats)

    if updating:
        old = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        new = {
            lab: {"coverage_pct": s["coverage_pct"], "important": s["important"]} for lab, s in sorted(stats.items())
        }
        lowered = [
            f"{lab}: {old[lab]['coverage_pct']}% -> {new[lab]['coverage_pct']}%"
            for lab in new
            if lab in old and new[lab]["coverage_pct"] < old[lab]["coverage_pct"]
        ]
        BASELINE.write_text(json.dumps(new, indent=2) + "\n")
        print(f"\n  wrote {BASELINE.relative_to(ROOT)}")
        if lowered:
            print("  LOWERED floors, check this is what you meant:")
            for line in lowered:
                print(f"    {line}")
        return 0

    if not BASELINE.exists():
        print(f"\n  no baseline at {BASELINE.relative_to(ROOT)}, create it with --update-baseline")
        return 1

    for lab in thin(stats):
        print(
            f"\n  NOTE {lab}: {stats[lab]['important']} important annotation(s), "
            f"below the target of {IMPORTANT_TARGET}. Nothing in this lab is marked "
            f"as a headline for the reader."
        )

    failures = check(stats, json.loads(BASELINE.read_text()))
    if failures:
        print(f"\n  FAIL: {len(failures)} problem(s)")
        for line in failures:
            print(f"    {line}")
        return 1
    print(f"\n  PASS: {len(stats)} labs at or above their floors, no duplicated annotations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
