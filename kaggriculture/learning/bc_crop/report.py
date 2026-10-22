"""CLI for the immutable predeclared analysis function.

The frozen script wrote both reports, then failed its final print because its
json import was missing. Supply that standard-library name at the caller,
preserving every byte of the frozen calculations and identity checks.
"""
import argparse
import json
from pathlib import Path
from . import analyze


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    analyze.json=json
    analyze.run(args.out.resolve())
