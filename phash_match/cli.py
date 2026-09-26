"""phash-match CLI: hash, compare, or find near-duplicate images."""
import argparse
import sys
from pathlib import Path

from .core import HASHES, compare, find_duplicates, similarity


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="phash-match", description="Find visually similar images with perceptual hashes.")
    p.add_argument("--method", choices=sorted(HASHES), default="phash")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash", help="print the hash of each image")
    h.add_argument("images", nargs="+", type=Path)
    c = sub.add_parser("compare", help="distance between two images (0 = identical, 64 = opposite)")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--threshold", type=int, default=10)
    s = sub.add_parser("scan", help="list near-duplicate pairs in a folder")
    s.add_argument("folder", type=Path)
    s.add_argument("--threshold", type=int, default=10)
    a = p.parse_args(argv)

    try:
        if a.cmd == "hash":
            for img in a.images:
                print(f"{HASHES[a.method](img)}  {img}")
        elif a.cmd == "compare":
            d = compare(a.a, a.b, a.method)
            match = d <= a.threshold
            print(f"distance {d}/64  similarity {similarity(d):.0%}  {'MATCH' if match else 'DIFFERENT'}")
            return 0 if match else 1
        else:
            matches = find_duplicates(a.folder, a.threshold, a.method)
            for m in matches:
                print(f"{m.distance:>2}  {m.similarity:.0%}  {m.a}  {m.b}")
            print(f"{len(matches)} near-duplicate pair(s)")
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
