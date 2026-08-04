"""Rotor-speed tracking stack: VK order tracking, refinement, blind seeding.

Pure array code (numpy/scipy/torch). This package must not import
``data_processing``, ``models``, or ``training`` — see ``AGENTS.md``.
"""

from tracking.joint_beam_tracker import build_objective, joint_beam_track
from tracking.phase_increment_tracker import pi_kalman_refine
from tracking.pipelines import (
    CAPTURE_CFG,
    MIDBAND_CFG,
    MIDBAND_CFGS,
    REFINE_CFG,
    SEED_CFG,
    TRACK_CFG,
    vit2dsp_pipeline,
    vit2dsp_stage,
)
from tracking.rotor_dp import greedy_peel, viterbi_path
from tracking.rps_refinement import (
    RefineConfig,
    comb_confidence,
    compute_logmag,
    estimate_clock_offset,
    refine_coherent,
    refine_trajectories,
)
from tracking.stages import (
    DEFAULT_HOP_S,
    Stage,
    blind_seed_stage,
    get_audio,
    get_rps,
    guarded,
    pi_kalman_stage,
    pipeline,
    refine_coherent_stage,
    tracking_frame,
    vk_stage,
    warp_stage,
    with_rps,
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
    "CAPTURE_CFG",
    "DEFAULT_HOP_S",
    "MIDBAND_CFG",
    "MIDBAND_CFGS",
    "REFINE_CFG",
    "SEED_CFG",
    "TRACK_CFG",
    "RefineConfig",
    "SeedConfig",
    "Stage",
    "VKConfig",
    "blind_seed",
    "blind_seed_stage",
    "build_objective",
    "comb_confidence",
    "compute_logmag",
    "demodulate",
    "estimate_clock_offset",
    "get_audio",
    "get_rps",
    "greedy_peel",
    "guarded",
    "iter_warp_refine",
    "joint_beam_track",
    "pi_kalman_refine",
    "pi_kalman_stage",
    "pipeline",
    "refine_coherent",
    "refine_coherent_stage",
    "refine_trajectories",
    "stage_guard",
    "tracking_frame",
    "vit2dsp_pipeline",
    "vit2dsp_stage",
    "viterbi_path",
    "vk_envelopes",
    "vk_reconstruct",
    "vk_stage",
    "vk_track",
    "warp_stage",
    "whitened_logmag",
    "with_rps",
]
