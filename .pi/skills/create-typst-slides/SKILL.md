---
name: create-typst-slides
description: Create a new Typst slide deck in writing/slides/ from the project Touying template. Use when the user wants a new presentation or slide deck.
---

# Create Typst Slides

Scaffolds a new slide deck directory under `writing/slides/<date>_<title>/` with the
project Touying template, a `Makefile`, and a `prepare.py` stub.

## Prerequisites

- `typst` installed (0.14.2+)
- `pdftoppm` available (for visual checking)

## Workflow

### 1. Scaffold the directory

```bash
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
mkdir -p "writing/slides/<date>_<title>"
cd "writing/slides/<date>_<title>"
```

### 2. Create the slide source

```typst
// slides.typ
#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [Slide Deck Title],
  subtitle: [Subtitle],
  author: [Author Name],
  date: [2026-06-13],
)

= First Slide

Bullet points here.

= Second Slide

More content.

= Third Slide

Conclusion.
```

### 3. Create `prepare.py` (self-contained)

```python
#!/usr/bin/env python3
"""Generate figures and tables for this slide deck."""
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

all: figures slides.pdf

figures:
	python3 prepare.py

slides.pdf: slides.typ figures
	typst compile --root $(ROOT) slides.typ

watch:
	typst watch --root $(ROOT) slides.typ

check: slides.pdf
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

- Framework: Touying (https://touying-typ.github.io/)
- Theme: simple (https://touying-typ.github.io/docs/themes/simple)
- Aspect ratio: 16:9
- Title slide with title, subtitle, author, date
