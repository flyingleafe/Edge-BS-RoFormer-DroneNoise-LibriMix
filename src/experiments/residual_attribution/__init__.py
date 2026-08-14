"""Attribute the VK broadband residual to individual rotors using the array.

Contract-fenced research sandbox (``src/experiments``): may import anything,
nothing imports it.

The question: ``scripts/vk_decompose.py`` splits a drone recording into
per-(rotor, k) harmonic tracks plus a **per-microphone** broadband residual
(67.8 % of the energy on DREGON free-flight). Is that residual attributable to
individual rotors, so the generator's noise branch can be supervised with
per-rotor source PSDs — or is it only ever a per-microphone floor?

Method (see the module docstrings): fit
``R(f) = sum_r P_r(f) g_r g_r^H + diag(D(f))`` to the measured array CSD, with
``g_r`` the free-field steering vector from the known, TDOA-validated geometry.
The information lives in the **cross-microphone** block only — the wind-channel
lesson (`docs/experiments/wind-channel-likelihood.md`) restated as linear
algebra.
"""

from __future__ import annotations

from . import csd, data, design, fit, power, steering, synth

__all__ = ["csd", "data", "design", "fit", "power", "steering", "synth"]
