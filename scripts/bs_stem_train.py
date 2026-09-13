"""Experiment: initialize the pretrained stem from physical spectral channels.

The default SMP stem repeats RGB weights across all feature channels. These
experiments instead put the RGB kernels on the corresponding pre/post bands
and start other feature weights at zero. Architecture/inference are unchanged.
Pass ordinary firemon.bs.train arguments after --stem-mode.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs import train


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--stem-mode', choices=['rgb_pair', 'swir_pair'], required=True)
    a, rest = ap.parse_known_args()
    original = train.build_model

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        net = model.net if kwargs.get('coord') else model
        stem = next(m for m in net.encoder.modules() if isinstance(m, torch.nn.Conv2d))
        if stem.in_channels < 27 or stem.groups != 1:
            raise ValueError(f'Unexpected input stem: {stem}')
        # timm/SMP adapt_input_conv repeats three RGB kernels and multiplies
        # by 3 / in_channels; recover the pretrained kernels before mapping.
        with torch.no_grad():
            rgb = stem.weight[:, :3].clone() * (stem.in_channels / 3.0)
            stem.weight.zero_()
            bands = (2, 1, 0) if a.stem_mode == 'rgb_pair' else (8, 7, 6)
            for colour, band in enumerate(bands):
                stem.weight[:, band] = rgb[:, colour] * .5
                stem.weight[:, 9 + band] = rgb[:, colour] * .5
        print('stem initialization:', a.stem_mode, flush=True)
        return model

    train.build_model = build
    sys.argv = [sys.argv[0], *rest]
    train.main()


if __name__ == '__main__':
    main()
