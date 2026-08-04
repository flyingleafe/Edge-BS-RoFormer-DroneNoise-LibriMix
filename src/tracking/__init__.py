"""Rotor-speed tracking stack: VK order tracking, refinement, blind seeding.

Pure array code (numpy/scipy/torch). This package must not import
``data_processing``, ``models``, or ``training`` — see ``AGENTS.md``.
"""

from tracking.joint_beam_tracker import build_objective, joint_beam_track
from tracking.phase_increment_tracker import pi_kalman_refine
from tracking.rotor_dp import greedy_peel, viterbi_path
from tracking.rps_refinement import (
    RefineConfig,
    comb_confidence,
    compute_logmag,
    estimate_clock_offset,
    refine_coherent,
    refine_trajectories,
)
from tracking.vk_blind_seeding import SeedConfig, blind_seed, stage_guard, whitened_logmag
from tracking.vk_tracking import (
    VKConfig,
    demodulate,
    vk_envelopes,
    vk_reconstruct,
    vk_track,
)
from tracking.warp_refinement import iter_warp_refine

__all__ = [
    "RefineConfig",
    "SeedConfig",
    "VKConfig",
    "blind_seed",
    "build_objective",
    "comb_confidence",
    "compute_logmag",
    "demodulate",
    "estimate_clock_offset",
    "greedy_peel",
    "iter_warp_refine",
    "joint_beam_track",
    "pi_kalman_refine",
    "refine_coherent",
    "refine_trajectories",
    "stage_guard",
    "viterbi_path",
    "vk_envelopes",
    "vk_reconstruct",
    "vk_track",
    "whitened_logmag",
]
