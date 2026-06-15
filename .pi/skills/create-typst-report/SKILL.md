---
name: create-typst-report
description: Create a new Typst report in writing/reports/ from the project template. Use when the user wants a new internal report, experiment summary, or any PDF document that is not a paper.
---

# Create Typst Report

Scaffolds a new report directory under `writing/reports/<date>_<title>/` with the
project Typst template, a `Makefile`, and a `prepare.py` stub.

## Prerequisites

- `typst` installed (0.14.2+)
- `pdftoppm` available (for visual checking)

## Workflow

### 1. Scaffold the directory

```bash
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
mkdir -p "writing/reports/<date>_<title>"
cd "writing/reports/<date>_<title>"
```

### 2. Create the report source

```typst
// report.typ
#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [Report Title],
  authors: (
    "Author Name": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [Abstract text here.],
  keywords: ("keyword1", "keyword2"),
)

= Introduction

Content here.

= Methods

Methods here.

= Results

Results here.

= Discussion

Discussion here.
```

### 3. Create `prepare.py` (self-contained)

```python
#!/usr/bin/env python3
"""Generate figures and tables for this report."""
import pathlib

def main():
    assets = pathlib.Path("assets")
    assets.mkdir(exist_ok=True)
    # Generate figures and tables here.
    # Write them to assets/ as PNG or PDF.

if __name__ == "__main__":
    main()
```

### 4. Create `Makefile`

```makefile
TITLE := $(notdir $(CURDIR))
ROOT := $(shell git rev-parse --show-toplevel)

all: figures report.pdf

figures:
	python3 prepare.py

report.pdf: report.typ figures
	typst compile --root $(ROOT) report.typ

watch:
	typst watch --root $(ROOT) report.typ

check: report.pdf
	mkdir -p check
	pdftoppm -png -r 150 $< check/page

.PHONY: all figures watch check
```

### 5. Build and visually check

```bash
make all
make check
```

Then read the `check/page-*.png` images to verify layout.

## Template features

- Base: `@preview/starter-journal-article:0.5.1`
- No title page (content starts immediately)
- Abstract + auto-generated TOC
- Header with project name + report title on page 2+
- Informal academic tone
