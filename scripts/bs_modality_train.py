"""Ablation: selectively drop auxiliary modalities while training BS.

Only augmentation changes; checkpoint structure and inference are identical.
The --chdrop probability and remaining arguments go to firemon.bs.train.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs import train


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--drop-modality', choices=['sar', 'static', 'scl'], required=True)
    a, rest = ap.parse_known_args()
    train.CHANNEL_GROUPS = {'sar': ((64, 73),), 'static': ((73, 83),), 'scl': ((44, 64),)}[a.drop_modality]
    sys.argv = [sys.argv[0], *rest]
    train.main()


if __name__ == '__main__':
    main()
