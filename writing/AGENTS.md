# Writing — Typst-based Reports & Slides

> **Papers** remain in LaTeX (`writing/papers/`).  
> **Reports** and **slides** use Typst with templates.  
> All artifacts produce PDF. Visual checking via `pdftoppm` + image review.

---

## Directory Layout

```
writing/
├── AGENTS.md
├── papers/               # LaTeX papers (copy of old papers/ — untouched)
│   └── rps-from-drone-sound/
├── reports/              # Typst reports, one dir per report
│   └── <date>_<title>/
│       ├── report.typ          # main Typst source
│       ├── Makefile            # python prep → typst compile → check
│       ├── prepare.py          # generate tables, figures (each report has its own)
│       └── assets/             # static figures from prepare.py
├── slides/               # Typst slides (Touying), one dir per presentation
│   └── <date>_<title>/
│       ├── slides.typ
│       ├── Makefile
│       ├── prepare.py
│       └── assets/
└── templates/
    └── typst/             # Typst template files
        ├── report.typ     # wraps @preview/starter-journal-article
        └── slides.typ     # wraps Touying / simple theme
```

### Naming convention

Every report and slide directory is `<date>_<title>` where `<date>` is `YYYY-MM-DD`
and `<title>` is a kebab-case slug, e.g.:

```
2026-04-14_rps-prediction-study/
2026-06-02_rps-progress/
```

---

## Templates

### Report template (`writing/templates/typst/report.typ`)

- Base: `@preview/starter-journal-article` (Typst Universe)
- Customizations (to be applied on top):
  - **No title page** — content starts on first page
  - **Abstract** — included
  - **Table of contents** — auto-generated
  - **Standard sections** — Introduction, Methods, Results, Discussion
  - **Figure/table environments** — use the base template's; may customise later
  - **Header** — project name + report title
  - **Footer** — reserved for footnotes (page numbers via template default)
  - **Tone** — informal academic (less ceremony than a paper)

### Slide template (`writing/templates/typst/slides.typ`)

- Framework: [Touying](https://touying-typ.github.io/)
- Theme: [simple](https://touying-typ.github.io/docs/themes/simple)
- Aspect ratio: 16:9 (Touying default)
- Slide content: per-slide Typst markup with Touying slide breaks

### Shared base template

Use the universal templates in `writing/templates/typst/`. Individual report or slide directories MUST NOT contain their own `template.typ` or `slides_template.typ`; import them with a root-absolute path, e.g. `#import "/writing/templates/typst/report.typ": report, author-meta`.

The per-artifact `Makefile` must invoke Typst with `--root $(shell git rev-parse --show-toplevel)` so root-absolute paths resolve inside the repository.

### Fonts

Inherit from the respective base templates.

---

## Existing Figures

`eval.py` (the unified evaluation entry point) + `src/plots` comparison plots
(SI-SDR, STOI, PESQ plots, waveform/spectrogram comparisons; absorbs the former
`generate_comparison.py`/`plot_per_snr.py`) produce the figures. Their output
PNGs/PDFs are used as assets in both reports and slides — simply
`image("path/to/fig.png")` in Typst.

---

## Build & Visual Check

### Per-artifact `Makefile`

Each report/slide directory gets a `Makefile` with targets:

```makefile
TITLE := $(notdir $(CURDIR))
ROOT := $(shell git rev-parse --show-toplevel)

all: figures report.pdf   # or slides.pdf

figures:
	python prepare.py

report.pdf: report.typ figures
	typst compile --root $(ROOT) report.typ

watch:
	typst watch --root $(ROOT) report.typ

check: report.pdf
	pdftoppm -png -r 150 report.pdf $(basename $<)-page

.PHONY: all figures watch check
```

### Visual checking

1. `make check` → `pdftoppm` renders each page as a PNG.
2. Read the PNG images to review layout, alignment, figure placement.
3. Iterate.

`typst watch --root $(ROOT) report.typ` — recompiles on save.

---

## Migration Plan

1. Implement the two templates (`report.typ`, `slides.typ`).
2. Recreate **all old reports** in Typst (from the backup), validating against the
   original LaTeX PDFs.
3. Recreate **all old slides** in Typst (from the backup), comparing visual output.
4. Once validated, the old LaTeX reports and Slidev slides remain in the backup
   tarball only — the `writing/` tree becomes the single source of truth.

---

## Papers (LaTeX)

- Live in `writing/papers/` — a copy of the current `papers/` directory.
- Untouched by this system.
- Figures for papers are produced by each paper's own scripts (not shared).

---

## Skills

| Skill | When to use |
|-------|-------------|
| `create-typst-report` | Scaffold a new `writing/reports/<date>_<title>/` directory with Typst source, Makefile, and `prepare.py` stub. |
| `create-typst-slides` | Scaffold a new `writing/slides/<date>_<title>/` directory with Typst/Touying source, Makefile, and `prepare.py` stub. |

## Principles

1. **Self-contained artifacts.** Every report/slide owns its `prepare.py` — no
   cross-artifact script dependencies. Duplication is acceptable over coupling.
2. **PDF is the deliverable.** Everything compiles to PDF for review and distribution.
3. **Visually verified.** No artifact is "done" until page images have been inspected.
4. **Papers stay in LaTeX.** Only reports and slides migrate to Typst.
