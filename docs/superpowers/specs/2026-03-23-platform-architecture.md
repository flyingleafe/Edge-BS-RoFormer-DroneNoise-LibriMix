# Postdoc Platform — Architecture Overview

**Date:** 2026-03-23
**Status:** Living document — updated as layers are designed and built

## Vision

An agentic ML experimentation platform for PhD research on low-SNR harmonic noise suppression. The platform enables rapid iteration on model architectures, training setups, and datasets through 5 layers, each with human-facing tools and agent-facing interfaces. Agents are Claude Code sessions orchestrated programmatically via zeroclaw.

## Research Context

**Problem:** Speech enhancement at very low SNR (-20 dB and below) in the presence of harmonic noise from rotating machinery (drone propellers, engines, fans).

**Key insight:** Harmonic noise from rotating parts has structure tied to motor RPM. Providing models with RPM information improves denoising. But labeled RPM data is scarce.

**Approach:** Self-supervised learning where models predict both denoised speech and motor RPM:
- Small labeled subset (DREGON): supervised RPM loss, ground-truth RPM fed to denoising module
- Large unlabeled pool (DN-LM, AeroSonicDB, MIMII, DADS, etc.): predicted RPM fed to denoising module, regularization losses (smoothness, etc.) on RPM predictions, end-to-end training

This combines the benefit of explicit noise modeling (via RPM) with scale (virtually infinite noisy mixtures from abundant speech + noise data).

## Layer Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        IDEAS LAYER                               │
│  Idea collection, literature search, experiment suggestions      │
├─────────────────────────────────────────────────────────────────┤
│                      REPORTING LAYER                             │
│  Experiment summaries, plots, presentations, paper drafts        │
├──────────────────────┬──────────────────────────────────────────┤
│    LITERATURE LAYER  │            JOB LAYER                      │
│  Paper DB + RAG,     │  Experiment definition, submission,       │
│  crawler agent       │  scheduling, failure handling, results    │
├──────────────────────┴──────────────────────────────────────────┤
│                         DATA LAYER                               │
│  Dataset versioning, storage, unified format, discovery          │
└─────────────────────────────────────────────────────────────────┘
```

**Build order:** Job (v0.1 local) → Data → Job (full cloud) → Literature → Reporting → Ideas

**Why this order:** Compute scheduling is the #1 pain (Job). Data preparation is #2 (Data). Literature/Reporting/Ideas don't block experiments.

---

## Layer 1: Job Layer

**Status:** Spec complete (v0.1 local + full cloud design)
**Specs:** `2026-03-23-job-layer-v0.1-local.md`, `2026-03-21-job-layer-design.md`

Experiment definition via YAML, job submission/scheduling, lifecycle management (TRAINING → EVAL → DONE/FAILED), results storage, failure handling. Abstract interfaces (`StorageBackend`, `Scheduler`, `JobTracker`) allow local and cloud backends to be swapped via config.

**v0.1 (immediate):** Local 2-GPU scheduler on existing vast.ai box.
**Full (later):** SkyPilot + GCP (phase 1), vast.ai direct API (phase 2). Automated failure triage, local repro, one-retry policy.

**CLI:** `postdoc job submit/list/status/logs/cancel`, `postdoc results show/compare`

---

## Layer 2: Data Layer

**Status:** High-level design captured here. Detailed spec to follow.

### Purpose

Manage the full lifecycle of datasets: discovery, adoption, versioning, storage, and serving to training jobs.

### Key Components

**Unified dataset format:**
- HuggingFace datasets format for streaming/lazy-loading and ecosystem compatibility
- All audio resampled to 16kHz mono
- Schema per sample:
  - `noisy_audio`: mixed signal (speech + noise)
  - `clean_audio`: target clean speech
  - `noise_audio`: noise-only (when available)
  - `rpm`: motor speed time series (when available, else null)
  - `has_rpm`: boolean flag
  - `metadata`: SNR, noise source type, duration, source dataset, etc.
- Mixtures created on-the-fly or pre-mixed, configurable per dataset

**Storage:**
- Cloudflare R2 as primary storage (free egress)
- DVC for versioning dataset creation pipelines and tracking dataset versions
- HuggingFace datasets library for loading (can stream from S3-compatible storage, which R2 is)

**Known datasets to onboard:**

| Dataset | Type | Has RPM | Size |
|---------|------|---------|------|
| DREGON | Drone noise + RPM | Yes | Small |
| DN-LM (existing) | Drone + speech mixtures | No | 2 hours |
| LibriSpeech | Clean speech | N/A | 6GB+ |
| MIMII | Industrial machine noise (fans, pumps, valves) | No | ~100GB |
| AeroSonicDB | Aircraft/propeller audio | No | 8.87 hours |
| DADS | Drone audio | No | Large |
| HornBase | Vehicle audio | No | 1080 samples |
| DroneAudioset | Multi-channel drone noise | No | 23.5 hours |

**Agents:**

1. **Dataset exploration agent** — searches the internet for relevant datasets (rotating machinery noise, speech corpora). Evaluates relevance, estimates size, checks licensing. Adds findings to a dataset registry.

2. **Dataset adoption agent** — given a dataset from the registry, prepares:
   - Download script (handles authentication, mirrors, checksums)
   - Pre-processing pipeline (resample, normalize, convert to unified format)
   - DVC pipeline definition
   - Validation checks (sample count, duration, format correctness)

**Integration with Job Layer:**
- `postdoc data pull <dataset-name>` resolves dataset name to R2 path, downloads/caches locally
- Experiment YAML `dataset.name` field references dataset names from the registry
- `dataset.name` can be a string or list (multiple datasets)

### Open Questions (for detailed spec)

- Mixture creation: on-the-fly during training (more variety, slower) vs. pre-mixed datasets (faster, less variety, larger storage)?
- How to handle datasets with different sample rates at the boundary (resample during adoption vs. during loading)?
- RPM alignment: when RPM data exists, how is it time-aligned with audio? (DREGON-specific format needs investigation)

---

## Layer 3: Literature Layer

**Status:** High-level design captured here. Detailed spec to follow after Data layer.

### Purpose

Maintain a searchable database of relevant papers with full-text RAG, so the researcher and agents can quickly find related work, compare approaches, and ground experiment ideas in literature.

### Key Components

**Paper database:**
- Papers stored as PDFs + extracted text + structured metadata (title, authors, year, venue, abstract, key findings, methods)
- Vector embeddings for semantic search (RAG)
- Citation graph tracking (who cites whom)
- Storage: local SQLite for metadata + R2 for PDFs + vector store (likely local FAISS or similar, lightweight)

**Paper crawler agent:**
- Periodically searches for new relevant papers on arXiv, Semantic Scholar, Google Scholar
- Search queries derived from: research keywords (low-SNR speech enhancement, harmonic noise, drone denoising, motor speed estimation, self-supervised audio), citation tracking (papers citing the user's published work), and references from papers already in the DB
- Adds candidates to the DB with a relevance score
- Human reviews and approves/rejects additions

**RAG interface:**
- `postdoc lit search "query"` — semantic search across paper database
- `postdoc lit summarize <paper-id>` — generate a summary of key findings and methods
- `postdoc lit related <paper-id>` — find papers related to a given paper
- Agent-callable: other agents (experiment design, ideas) can query literature to ground their suggestions

### Open Questions (for detailed spec)

- Which vector store? FAISS (local, simple) vs. something hosted?
- How to handle papers behind paywalls? (arXiv covers most ML papers, but some audio/signal processing venues are paywalled)
- How granular should the RAG chunks be? (per-section, per-paragraph, per-page?)

---

## Layer 4: Reporting Layer

**Status:** High-level design captured here. Detailed spec to follow.

### Purpose

Automatically generate experiment reports, comparison plots, and paper draft sections from structured results data.

### Key Components

**Reporter agent:**
- Triggered after experiments complete (or on-demand)
- Pulls metrics from `postdoc results` (via JobTracker/StorageBackend)
- Generates:
  - Metrics comparison tables (across experiments, across SNR levels)
  - Training curves (loss, SI-SDR over epochs)
  - Spectrogram comparisons (noisy vs. enhanced vs. clean)
  - Audio samples for listening tests
  - Summary narrative: what was tried, what worked, what didn't
- Output format: Markdown report saved to `reports/`, optionally rendered as HTML or slides

**Paper draft agent:**
- Combines multiple reports into running paper sections
- Maintains a structured paper outline (introduction, related work, method, experiments, results)
- Updates experiment tables and figures as new results come in
- Pulls related work summaries from the Literature layer
- Does NOT write the final paper — produces structured drafts that the human refines

**Integration with Job Layer:**
- Reporter agent reads from `postdoc results compare` and `postdoc results export`
- Can be triggered automatically when a job reaches DONE state (future: via notification hook)

### Open Questions (for detailed spec)

- Report template format? (Markdown with Jinja-like templating, or fully generated?)
- How to handle incremental updates? (Regenerate full report, or patch existing?)
- Presentation format? (The existing `generate_comparison.py` script already does some of this — reuse or replace?)

---

## Layer 5: Ideas Layer

**Status:** High-level design captured here. Detailed spec to follow.

### Purpose

Collect, organize, critique, and develop research ideas. Bridge the gap between reading papers, having hunches, and running well-designed experiments.

### Key Components

**Idea collection:**
- `postdoc idea add "description"` — capture an idea with context (what prompted it, expected impact)
- Ideas stored with: description, source (paper, experiment result, conversation), status (raw, critiqued, planned, tested, discarded), links to related papers and experiments
- Tags and categories for organization

**Idea critic agent:**
- When a new idea is added, the agent:
  - Searches the Literature layer for similar or related work
  - Identifies potential issues (has this been tried? does it conflict with known results?)
  - Rates novelty and expected impact
  - Suggests refinements or variations
- Human reviews critique and decides whether to proceed

**Experiment suggestion agent:**
- Given an idea that's been approved, proposes:
  - Which model architecture to use as a starting point
  - What dataset configuration is needed
  - What metrics would demonstrate success
  - A concrete experiment YAML (or set of YAMLs for an ablation study)
- Links back to the original idea so results can be traced to hypotheses

**Exploration agent:**
- Proactively suggests ideas based on:
  - Recent experiment results (e.g., "model X performed surprisingly well at -25dB, maybe try extending to -30dB")
  - New papers in the Literature layer
  - Gaps in the experiment history (e.g., "you've tested DCUNet and DPTNet but not HTDemucs with RPM")

### Open Questions (for detailed spec)

- How structured should ideas be? Free-form text vs. structured fields?
- How does the idea → experiment pipeline work concretely? (Idea → experiment suggestion → human approval → experiment YAML → job submission)
- How to prevent idea explosion? (Prioritization scheme?)

---

## Cross-Cutting Concerns

### Agent Orchestration

All agents are Claude Code sessions triggered programmatically via zeroclaw. Agents interact with the platform through the `postdoc` CLI and Python interfaces.

**Guardrails model:** Agents act within defined boundaries (see Job Layer spec Section 7). Key rules:
- Agents cannot: force-push, delete data, modify main branch, exceed retry limits, override budget without human approval
- All agent actions are logged with timestamps and session IDs
- Escalation to human on repeated failures or ambiguous situations

### Data Flow Between Layers

```
Ideas Layer → generates experiment suggestions
    ↓
Job Layer → runs experiments, produces results
    ↓
Reporting Layer → summarizes results into reports
    ↓
Ideas Layer → new ideas from results
    ↑
Literature Layer → grounds ideas and reports in prior work
    ↑
Data Layer → provides datasets to Job Layer
```

### Shared Infrastructure

- **`postdoc` CLI:** single entry point for all layers (`postdoc job`, `postdoc data`, `postdoc lit`, `postdoc report`, `postdoc idea`)
- **`postdoc.yaml`:** global configuration, one section per layer
- **SQLite DB:** shared job tracker and results store (JobTracker from Job Layer), extended with tables for papers, ideas, reports as other layers are built
- **R2 storage:** shared bucket with prefixed paths (`jobs/`, `datasets/`, `papers/`, `reports/`)
- **Abstract interfaces:** StorageBackend used across layers (not just jobs)

### Test Strategy

- **Non-GPU test mode:** experiments can be submitted with `--test` flag that runs on CPU with tiny data for 2-3 steps. Validates the full pipeline (config parsing → job submission → training → eval → metrics extraction) without burning GPU time.
- **Per-layer unit tests:** each layer has its own test suite testing the interfaces and implementations independently.
