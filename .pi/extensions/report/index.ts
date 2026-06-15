/**
 * Report & Slides Extension
 *
 * Provides /report and /slides commands that orchestrate a multi-phase
 * workflow for creating and iterating on Typst reports and slide decks.
 *
 * Workflow:
 *   /report [name] ["extra context"] → scaffold or resume, enter context-collection phase
 *   /slides [name] ["extra context"] → same for slides
 *
 * Specialized tools for the iteration loop:
 *   - prepare_figures  — run prepare.py, read changed asset files
 *   - read_pages       — compile Typst to per-page PNGs for visual inspection
 *   - finish_report    — final compilation (PDF + PNGs), mark complete
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";
import {
  REPORT_TEMPLATE,
  SLIDES_TEMPLATE,
  PREPARE_PY_TEMPLATE,
  makeMakefile,
  getPhaseInstructions,
  newReportMessage,
  existingReportMessage,
  isUserApproval,
  TRANSITION_MESSAGE,
} from "./prompts.js";
import { registerTools, setSessionStateGetter, setOnSessionDone } from "./tools.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ReportSession {
  reportPath: string; // absolute path to report/slides directory
  type: "report" | "slides";
  phase: "context_collection" | "iteration" | "done";
  mainFile: string; // "report.typ" | "slides.typ"
  startedAt: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getProjectRoot(): string {
  return execSync("git rev-parse --show-toplevel", { encoding: "utf8" }).trim();
}

function todayISO(): string {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Resolve a report name (first word of args) to a directory path. */
function resolveReportPath(
  name: string | undefined,
  kind: "report" | "slides",
): { dirPath: string; dirName: string; exists: boolean } {
  const root = getProjectRoot();
  const subdir = kind === "report" ? "writing/reports" : "writing/slides";
  const base = path.join(root, subdir);

  if (!name || name.trim() === "") {
    const dirName = todayISO();
    return { dirPath: path.join(base, dirName), dirName, exists: false };
  }

  const trimmed = name.trim();

  // Try exact match first
  const exactPath = path.join(base, trimmed);
  if (fs.existsSync(exactPath)) {
    return { dirPath: exactPath, dirName: trimmed, exists: true };
  }

  // Try as plain name (prefix with today's date)
  const slug = slugify(trimmed);
  const datedName = `${todayISO()}_${slug}`;
  const datedPath = path.join(base, datedName);
  if (fs.existsSync(datedPath)) {
    return { dirPath: datedPath, dirName: datedName, exists: true };
  }

  // Doesn't exist — create with date prefix
  return { dirPath: datedPath, dirName: datedName, exists: false };
}

/** Parse command args into (name, extraContext). */
function parseCommandArgs(
  rawArgs: string,
): { name: string | undefined; extraContext: string | undefined } {
  const trimmed = rawArgs.trim();
  if (!trimmed) return { name: undefined, extraContext: undefined };

  // If there's a quoted string at the end, treat it as extra context
  // e.g. my-report "focus on model X"
  const quotedMatch = trimmed.match(/^(.+?)\s+"(.+)"$/);
  if (quotedMatch) {
    return { name: quotedMatch[1].trim(), extraContext: quotedMatch[2].trim() };
  }

  // Otherwise split on first space: first word = name, rest = context
  const spaceIdx = trimmed.indexOf(" ");
  if (spaceIdx === -1) {
    return { name: trimmed, extraContext: undefined };
  }
  return {
    name: trimmed.slice(0, spaceIdx).trim(),
    extraContext: trimmed.slice(spaceIdx + 1).trim(),
  };
}

/** List existing report/slides directories for autocompletion. */
function listExisting(kind: "report" | "slides"): string[] {
  const root = getProjectRoot();
  const subdir = kind === "report" ? "writing/reports" : "writing/slides";
  const base = path.join(root, subdir);
  if (!fs.existsSync(base)) return [];
  return fs
    .readdirSync(base, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name);
}

// ---------------------------------------------------------------------------
// Scaffolding
// ---------------------------------------------------------------------------

function scaffoldDirectory(
  dirPath: string,
  kind: "report" | "slides",
): { mainFile: string; pdfFile: string } {
  fs.mkdirSync(dirPath, { recursive: true });
  fs.mkdirSync(path.join(dirPath, "assets"), { recursive: true });

  const mainFile = kind === "report" ? "report.typ" : "slides.typ";
  const pdfFile = kind === "report" ? "report.pdf" : "slides.pdf";

  // Main Typst file
  let template = kind === "report" ? REPORT_TEMPLATE : SLIDES_TEMPLATE;
  template = template.replace("TITLE_PLACEHOLDER", path.basename(dirPath));
  template = template.replace("DATE_PLACEHOLDER", todayISO());
  fs.writeFileSync(path.join(dirPath, mainFile), template);

  // prepare.py
  const prepTemplate = PREPARE_PY_TEMPLATE.replace(
    "TYPST_KIND",
    kind === "report" ? "report" : "slide deck",
  );
  fs.writeFileSync(path.join(dirPath, "prepare.py"), prepTemplate);
  fs.chmodSync(path.join(dirPath, "prepare.py"), 0o755);

  // Makefile
  fs.writeFileSync(path.join(dirPath, "Makefile"), makeMakefile(mainFile, pdfFile));

  return { mainFile, pdfFile };
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

function makeCommandHandler(
  pi: ExtensionAPI,
  kind: "report" | "slides",
): (args: string, ctx: ExtensionContext) => Promise<void> {
  return async (args: string, ctx: ExtensionContext) => {
    if (!ctx.isIdle()) {
      ctx.ui.notify("Agent is busy. Wait for it to finish.", "warning");
      return;
    }

    const { name, extraContext } = parseCommandArgs(args);
    const { dirPath, dirName, exists } = resolveReportPath(name, kind);

    if (exists) {
      const mainFile = kind === "report" ? "report.typ" : "slides.typ";
      const session: ReportSession = {
        reportPath: dirPath,
        type: kind,
        phase: "context_collection",
        mainFile,
        startedAt: Date.now(),
      };

      currentSession = session;
      pi.appendEntry("report-session", session);
      pi.setSessionName(`${kind}: ${dirName}`);

      pi.sendUserMessage(existingReportMessage(kind, dirPath, mainFile, extraContext));
    } else {
      const { mainFile } = scaffoldDirectory(dirPath, kind);

      const session: ReportSession = {
        reportPath: dirPath,
        type: kind,
        phase: "context_collection",
        mainFile,
        startedAt: Date.now(),
      };

      currentSession = session;
      pi.appendEntry("report-session", session);
      pi.setSessionName(`${kind}: ${dirName}`);

      ctx.ui.notify(
        `Created ${kind} directory: ${path.relative(getProjectRoot(), dirPath)}`,
        "info",
      );

      pi.sendUserMessage(newReportMessage(kind, dirPath, mainFile, extraContext));
    }
  };
}

// ---------------------------------------------------------------------------
// Iteration tool set
// ---------------------------------------------------------------------------

const ITERATION_TOOLS = [
  "read",
  "edit_typst",
  "edit_prepare_py",
  "write_typst",
  "write_prepare_py",
  "prepare_figures",
  "read_pages",
  "finish_report",
];

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

let currentSession: ReportSession | null = null;
let defaultActiveTools: string[] | null = null;
let toolsRestrictedByExtension = false;

function restrictToIterationTools(pi: ExtensionAPI) {
  pi.setActiveTools(ITERATION_TOOLS);
  toolsRestrictedByExtension = true;
}

function restoreDefaultTools(pi: ExtensionAPI) {
  if (defaultActiveTools) {
    pi.setActiveTools(defaultActiveTools);
  }
  toolsRestrictedByExtension = false;
}

function transitionToIteration(pi: ExtensionAPI) {
  if (!currentSession || currentSession.phase !== "context_collection") return;

  currentSession.phase = "iteration";
  pi.appendEntry("report-session", currentSession);
  restrictToIterationTools(pi);
  pi.sendMessage({
    customType: "report-session",
    content: TRANSITION_MESSAGE,
    display: true,
  });
}

export default function (pi: ExtensionAPI) {
  // Wire up session state access for tools
  setSessionStateGetter(() => currentSession);
  setOnSessionDone(() => {
    if (currentSession) {
      pi.appendEntry("report-session", currentSession);
      restoreDefaultTools(pi);
    }
  });

  // ---- State restoration on session start ----
  pi.on("session_start", async (_event, ctx) => {
    if (defaultActiveTools === null) {
      defaultActiveTools = pi.getActiveTools().map((t) => t.name);
    }

    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "custom" && entry.customType === "report-session") {
        currentSession = entry.data as ReportSession;
        break;
      }
    }

    if (currentSession?.phase === "iteration") {
      restrictToIterationTools(pi);
    }
  });

  // ---- Inject phase-specific instructions and enforce tool set ----
  pi.on("before_agent_start", async (event, _ctx) => {
    if (!currentSession) return;

    // --- Approval detection: check user's message, not assistant's ---
    if (
      currentSession.phase === "context_collection" &&
      isUserApproval(event.prompt)
    ) {
      transitionToIteration(pi);
    }

    // --- Enforce tool restrictions based on phase ---
    if (currentSession.phase === "iteration") {
      restrictToIterationTools(pi);
    } else if (toolsRestrictedByExtension) {
      restoreDefaultTools(pi);
    }

    // --- Inject phase-specific system prompt instructions ---
    const instructions = getPhaseInstructions(currentSession);
    if (!instructions) return;

    return {
      systemPrompt: event.systemPrompt + "\n\n" + instructions,
    };
  });

  // ---- Persist session state on agent end ----
  pi.on("agent_end", async (_event, _ctx) => {
    if (!currentSession) return;
    // Persist to session so finish_report's phase change survives
    pi.appendEntry("report-session", currentSession);
  });

  // ---- Register tools ----
  registerTools(pi);

  // ---- Register commands ----
  pi.registerCommand("report", {
    description: "Create or edit a Typst report in writing/reports/",
    getArgumentCompletions: (prefix: string) => {
      const existing = listExisting("report");
      const filtered = existing.filter((d) => d.startsWith(prefix));
      if (filtered.length === 0) return null;
      return filtered.map((d) => ({ value: d, label: d }));
    },
    handler: makeCommandHandler(pi, "report"),
  });

  pi.registerCommand("slides", {
    description: "Create or edit Typst slides in writing/slides/",
    getArgumentCompletions: (prefix: string) => {
      const existing = listExisting("slides");
      const filtered = existing.filter((d) => d.startsWith(prefix));
      if (filtered.length === 0) return null;
      return filtered.map((d) => ({ value: d, label: d }));
    },
    handler: makeCommandHandler(pi, "slides"),
  });

  // ---- Clean up on session shutdown ----
  pi.on("session_shutdown", async (_event, _ctx) => {
    if (currentSession) {
      pi.appendEntry("report-session", currentSession);
    }
    restoreDefaultTools(pi);
  });
}
