#!/usr/bin/env python3
"""Figures for this deck are pre-rendered PNGs committed under assets/ (ported
from the original Marp deck); nothing to regenerate."""

import pathlib


def main():
    pathlib.Path("assets").mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
