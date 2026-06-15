/**
 * Prompt templates and instruction text for the report/slides extension.
 */

import type { ReportSession } from "./index.js";

// ---------------------------------------------------------------------------
// Scaffolding templates
// ---------------------------------------------------------------------------

export const REPORT_TEMPLATE = `#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [TITLE_PLACEHOLDER],
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
`;

export const SLIDES_TEMPLATE = `#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [TITLE_PLACEHOLDER],
  subtitle: [Subtitle],
  author: [Author Name],
  date: [DATE_PLACEHOLDER],
)

= First Slide

Content here.

= Second Slide

More content here.
`;

export const PREPARE_PY_TEMPLATE = `#!/usr/bin/env python3
"""Generate figures and tables for this TYPST_KIND."""
import pathlib

def main():
    assets = pathlib.Path("assets")
    assets.mkdir(exist_ok=True)
    # Generate figures and tables here.
    # Write them to assets/ as PNG or PDF.

if __name__ == "__main__":
    main()
`;

export function makeMakefile(mainFile: string, pdfFile: string): string {
  return `TITLE := $(notdir $(CURDIR))
ROOT := $(shell git rev-parse --show-toplevel)

all: figures ${pdfFile}

figures:
\tpython3 prepare.py

${pdfFile}: ${mainFile} figures
\ttypst compile --root $(ROOT) ${mainFile}

watch:
\ttypst watch --root $(ROOT) ${mainFile}

check: ${mainFile}
\tmkdir -p check
\ttypst compile --root $(ROOT) --format png ${mainFile} check/page-{p}.png

.PHONY: all figures watch check
`;
}

// ---------------------------------------------------------------------------
// Phase-specific instructions (appended to system prompt)
// ---------------------------------------------------------------------------

export function getPhaseInstructions(session: ReportSession): string {
  const kind = session.type;
  const dirName = session.reportPath.split("/").pop()!;
  const mainFile = session.mainFile;

  if (session.phase === "context_collection") {
    return contextCollectionInstructions(kind, dirName, session.reportPath);
  }

  if (session.phase === "iteration") {
    return iterationInstructions(kind, dirName, session.reportPath, mainFile);
  }

  return "";
}

function contextCollectionInstructions(
  kind: "report" | "slides",
  dirName: string,
  reportPath: string,
): string {
  const subdir = kind === "report" ? "writing/reports/" : "writing/slides/";
  return `
## Active Workflow: Creating ${kind === "report" ? "Report" : "Slides"}

You are in the **context collection** phase for ${kind} "${dirName}".
Directory: ${reportPath}

Your task:
1. Determine at which date the last ${kind} was created (check ${subdir} directories).
2. Discover what experimental work has been done since that date. Check:
   - git history (commits, branches merged)
   - recently modified files in results/ directory
   - Weights & Biases experiments (use \`wandb\` CLI or read experiment configs)
   - Session history in the current folder
3. Draft a plan for how to present the results. The plan should include:
   - Which metrics to report (SI-SDR, STOI, PESQ, etc.)
   - Which models/experiments/configurations to compare
   - What visualizations to include (metric plots, spectrograms, waveforms)
   - Structure of the ${kind} (sections or slides)

Present this plan to the user. Iterate until the user approves.

When the user approves, they will say something like "The plan is approved" or "approved". The system will automatically transition you into the iteration phase with specialized editing tools.
`;
}

function iterationInstructions(
  kind: "report" | "slides",
  dirName: string,
  reportPath: string,
  mainFile: string,
): string {
  return `
## Active Workflow: Writing ${kind === "report" ? "Report" : "Slides"}

You are in the **iteration** phase for ${kind} "${dirName}".
Directory: ${reportPath}

Available tools are restricted to the report workflow set:
- **read** — inspect files (templates, generated assets, etc.).
- **edit_typst** / **write_typst** — modify or overwrite \`${mainFile}\`.
- **edit_prepare_py** / **write_prepare_py** — modify or overwrite \`prepare.py\`.
- **prepare_figures** — Run \`python3 prepare.py\` in the report directory and read all generated/changed asset files. Use this after editing prepare.py.
- **read_pages** — Compile the Typst document to per-page PNGs and read them for visual inspection. Use this after editing ${mainFile} to verify layout, figure placement, and text fitting. Call with no arguments to check all pages, or with specific page numbers.
- **finish_report** — Final compilation to PDF + full page render. Call this when the ${kind} looks good: no rendering errors, data and narrative are self-consistent, no figure or table goes out of page limits.

Do NOT use bash, grep, find, or generic edit/write tools during this phase. Use the restricted workflow tools above.

Workflow:
1. Edit \`prepare.py\` (with **edit_prepare_py** or **write_prepare_py**) to generate figures and tables. Then call **prepare_figures** to run it and inspect results.
2. Edit \`${mainFile}\` (with **edit_typst** or **write_typst**) to write content. Then call **read_pages** to visually verify layout.
3. Iterate between prepare.py → prepare_figures and ${mainFile} → read_pages.
4. When everything looks correct, call **finish_report** to finalize.

IMPORTANT: After every edit to ${mainFile}, call read_pages to check the rendered output. After every edit to prepare.py, call prepare_figures to see the results.
`;
}

// ---------------------------------------------------------------------------
// Command handler messages (sent via pi.sendUserMessage)
// ---------------------------------------------------------------------------

export function newReportMessage(
  kind: "report" | "slides",
  dirPath: string,
  mainFile: string,
  extraContext?: string,
): string {
  const subdir = kind === "report" ? "writing/reports/" : "writing/slides/";
  let msg = `[${kind.toUpperCase()} WORKFLOW: New ${kind}]
Path: ${dirPath}
Main file: ${mainFile}

I've scaffolded the directory with:
- ${mainFile} (Typst source with template)
- prepare.py (figure/table generation stub)
- Makefile
- assets/ directory

## Phase: Initial Context Collection

1. Determine at which date the last ${kind} was created (look at ${subdir} directories).
2. Discover what experimental work has been done since that date:
   - Check git history (commits, branches merged)
   - Look at recently modified files in results/
   - Check Weights & Biases experiments
   - Review session history
3. Draft a plan for how to present the results:
   - Which metrics to report
   - Which models/experiments to compare
   - What visualizations to include
   - Structure outline`;

  if (extraContext) {
    msg += `\n\n## Extra context from user:\n${extraContext}`;
  }

  msg += `\n\nPresent this plan to me. We'll iterate until I approve it.`;
  return msg;
}

export function existingReportMessage(
  kind: "report" | "slides",
  dirPath: string,
  mainFile: string,
  extraContext?: string,
): string {
  let msg = `[${kind.toUpperCase()} WORKFLOW: Editing existing ${kind}]
Path: ${dirPath}

This ${kind} already exists. Let's review it and make changes.

1. Read the current ${mainFile} to understand what it covers.
2. Read prepare.py to see what figures/tables are generated.
3. Determine what has changed since this ${kind} was last updated (check git history, new experiments, new results).
4. Propose changes: what to add, update, or restructure.`;

  if (extraContext) {
    msg += `\n\n## Extra context from user:\n${extraContext}`;
  }

  msg += `\n\n5. Present the plan. Iterate with me until I approve.

After I approve, we'll enter the editing phase with specialized tools (edit_typst, edit_prepare_py, write_typst, write_prepare_py, prepare_figures, read_pages, finish_report).`;
  return msg;
}

// ---------------------------------------------------------------------------
// User approval detection
// ---------------------------------------------------------------------------

/** Phrases that indicate the user approved the plan. */
const APPROVAL_PATTERNS = [
  /^the plan is approved/i,
  /^approved[.!]?$/i,
  /^i approve/i,
  /^looks good/i,
  /^let'?s (go|proceed|do it)/i,
  /^proceed/i,
  /^go ahead/i,
  /^sounds good/i,
  /^lgtm/i,
  /^ok,? let'?s/i,
];

/**
 * Check whether a user message indicates plan approval.
 * Only matches when phrases appear at the start of the message
 * to avoid false positives from discussing approval.
 */
export function isUserApproval(message: string): boolean {
  if (!message || message.trim().length === 0) return false;
  const trimmed = message.trim();
  for (const pattern of APPROVAL_PATTERNS) {
    if (pattern.test(trimmed)) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Transition notification (sent when phase changes)
// ---------------------------------------------------------------------------

export const TRANSITION_MESSAGE =
  "Transitioning to iteration phase. Only the report workflow tools (read, edit_typst, edit_prepare_py, write_typst, write_prepare_py, prepare_figures, read_pages, finish_report) are now active.";
