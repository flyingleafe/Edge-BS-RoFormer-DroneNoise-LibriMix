/**
 * Tool implementations for the report/slides extension.
 *
 * Registers: edit_typst, edit_prepare_py, write_typst, write_prepare_py,
 *            prepare_figures, read_pages, finish_report
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { withFileMutationQueue } from "@earendil-works/pi-coding-agent";
import type { ReportSession } from "./index.js";
import { Type } from "typebox";
import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";

// ---------------------------------------------------------------------------
// Callbacks wired by index.ts
// ---------------------------------------------------------------------------

let getSessionState: () => ReportSession | null = () => null;
let onSessionDone: (() => void) | null = null;

export function setSessionStateGetter(fn: () => ReportSession | null) {
  getSessionState = fn;
}

export function setOnSessionDone(fn: () => void) {
  onSessionDone = fn;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getProjectRoot(): string {
  return execSync("git rev-parse --show-toplevel", { encoding: "utf8" }).trim();
}

function imageContent(
  data: Buffer,
  mimeType: string,
): { type: "image"; data: string; mimeType: string } {
  return { type: "image", data: data.toString("base64"), mimeType };
}

function runInDir(
  cmd: string,
  cwd: string,
  timeoutMs = 60_000,
): { stdout: string; stderr: string; code: number } {
  try {
    const stdout = execSync(cmd, {
      cwd,
      timeout: timeoutMs,
      encoding: "utf8",
      maxBuffer: 50 * 1024 * 1024,
      stdio: ["ignore", "pipe", "pipe"],
    });
    return { stdout, stderr: "", code: 0 };
  } catch (err: any) {
    return {
      stdout: err.stdout?.toString() || "",
      stderr: err.stderr?.toString() || err.message || "",
      code: err.status ?? 1,
    };
  }
}

function compileToPngPages(session: ReportSession): {
  pages: string[];
  exitCode: number;
  stderr: string;
} {
  const root = getProjectRoot();
  const checkDir = path.join(session.reportPath, "check");
  fs.mkdirSync(checkDir, { recursive: true });

  // Clean old page PNGs
  for (const f of fs.readdirSync(checkDir)) {
    if (f.match(/^page-?\d+\.png$/i)) {
      fs.unlinkSync(path.join(checkDir, f));
    }
  }

  const mainPath = path.join(session.reportPath, session.mainFile);
  const cmd = `typst compile --root '${root}' --format png '${mainPath}' 'check/page-{p}.png'`;
  const result = runInDir(cmd, session.reportPath);

  const pages = fs
    .readdirSync(checkDir)
    .filter((f) => f.match(/^page-?\d+\.png$/i))
    .sort((a, b) => {
      const na = parseInt(a.match(/(\d+)/)?.[1] || "0");
      const nb = parseInt(b.match(/(\d+)/)?.[1] || "0");
      return na - nb;
    });

  return { pages, exitCode: result.code, stderr: result.stderr };
}

function readPageImages(
  checkDir: string,
  pageFiles: string[],
): Array<
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string }
> {
  const content: Array<
    | { type: "text"; text: string }
    | { type: "image"; data: string; mimeType: string }
  > = [];

  for (const pf of pageFiles) {
    const fp = path.join(checkDir, pf);
    try {
      const data = fs.readFileSync(fp);
      content.push({ type: "text", text: `\n--- ${pf} ---` });
      content.push(imageContent(data, "image/png"));
    } catch {
      content.push({ type: "text", text: `\n⚠️ Could not read ${pf}` });
    }
  }

  return content;
}

// ---------------------------------------------------------------------------
// Tool registration
// ---------------------------------------------------------------------------

export function registerTools(pi: ExtensionAPI) {
  // ---- edit_typst ----
  pi.registerTool({
    name: "edit_typst",
    label: "Edit Typst",
    description:
      "Edit the current report/slides Typst source file using exact text replacement. Only available during the report iteration phase.",
    promptSnippet: "Edit the current report/slides .typ file",
    promptGuidelines: [
      "Use edit_typst to modify the current report/slides Typst source file during the iteration phase.",
      "Do not use generic edit during the iteration phase; use edit_typst instead.",
    ],
    parameters: Type.Object({
      edits: Type.Array(
        Type.Object({
          oldText: Type.String({ description: "Exact existing text to replace" }),
          newText: Type.String({ description: "Replacement text" }),
        }),
        { description: "One or more replacement blocks, applied in order" },
      ),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const session = getSessionState();
      if (!session) {
        return {
          content: [{ type: "text", text: "No active report/slides session. Start one with /report or /slides." }],
          details: {},
        };
      }
      const targetPath = path.join(session.reportPath, session.mainFile);
      return withFileMutationQueue(targetPath, async () => {
        if (!fs.existsSync(targetPath)) {
          throw new Error(`${session.mainFile} does not exist at ${targetPath}`);
        }
        let current = fs.readFileSync(targetPath, "utf8");
        for (const edit of params.edits) {
          if (!current.includes(edit.oldText)) {
            throw new Error(`oldText not found in ${session.mainFile}`);
          }
          current = current.replace(edit.oldText, edit.newText);
        }
        fs.writeFileSync(targetPath, current, "utf8");
        return {
          content: [{ type: "text", text: `Updated ${session.mainFile}` }],
          details: {},
        };
      });
    },
  });

  // ---- edit_prepare_py ----
  pi.registerTool({
    name: "edit_prepare_py",
    label: "Edit prepare.py",
    description:
      "Edit the current report/slides prepare.py file using exact text replacement. Only available during the report iteration phase.",
    promptSnippet: "Edit the current report/slides prepare.py file",
    promptGuidelines: [
      "Use edit_prepare_py to modify the current report/slides prepare.py file during the iteration phase.",
      "Do not use generic edit during the iteration phase; use edit_prepare_py instead.",
    ],
    parameters: Type.Object({
      edits: Type.Array(
        Type.Object({
          oldText: Type.String({ description: "Exact existing text to replace" }),
          newText: Type.String({ description: "Replacement text" }),
        }),
        { description: "One or more replacement blocks, applied in order" },
      ),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const session = getSessionState();
      if (!session) {
        return {
          content: [{ type: "text", text: "No active report/slides session. Start one with /report or /slides." }],
          details: {},
        };
      }
      const targetPath = path.join(session.reportPath, "prepare.py");
      return withFileMutationQueue(targetPath, async () => {
        if (!fs.existsSync(targetPath)) {
          throw new Error(`prepare.py does not exist at ${targetPath}`);
        }
        let current = fs.readFileSync(targetPath, "utf8");
        for (const edit of params.edits) {
          if (!current.includes(edit.oldText)) {
            throw new Error(`oldText not found in prepare.py`);
          }
          current = current.replace(edit.oldText, edit.newText);
        }
        fs.writeFileSync(targetPath, current, "utf8");
        return {
          content: [{ type: "text", text: `Updated prepare.py` }],
          details: {},
        };
      });
    },
  });

  // ---- write_typst ----
  pi.registerTool({
    name: "write_typst",
    label: "Write Typst",
    description:
      "Overwrite the current report/slides Typst source file. Only available during the report iteration phase.",
    promptSnippet: "Overwrite the current report/slides .typ file",
    promptGuidelines: [
      "Use write_typst to write the current report/slides Typst source file during the iteration phase.",
      "Do not use generic write during the iteration phase; use write_typst instead.",
    ],
    parameters: Type.Object({
      content: Type.String({ description: "Full new contents of the Typst file" }),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const session = getSessionState();
      if (!session) {
        return {
          content: [{ type: "text", text: "No active report/slides session. Start one with /report or /slides." }],
          details: {},
        };
      }
      const targetPath = path.join(session.reportPath, session.mainFile);
      return withFileMutationQueue(targetPath, async () => {
        fs.mkdirSync(path.dirname(targetPath), { recursive: true });
        fs.writeFileSync(targetPath, params.content, "utf8");
        return {
          content: [{ type: "text", text: `Wrote ${session.mainFile}` }],
          details: {},
        };
      });
    },
  });

  // ---- write_prepare_py ----
  pi.registerTool({
    name: "write_prepare_py",
    label: "Write prepare.py",
    description:
      "Overwrite the current report/slides prepare.py file. Only available during the report iteration phase.",
    promptSnippet: "Overwrite the current report/slides prepare.py file",
    promptGuidelines: [
      "Use write_prepare_py to write the current report/slides prepare.py file during the iteration phase.",
      "Do not use generic write during the iteration phase; use write_prepare_py instead.",
    ],
    parameters: Type.Object({
      content: Type.String({ description: "Full new contents of prepare.py" }),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const session = getSessionState();
      if (!session) {
        return {
          content: [{ type: "text", text: "No active report/slides session. Start one with /report or /slides." }],
          details: {},
        };
      }
      const targetPath = path.join(session.reportPath, "prepare.py");
      return withFileMutationQueue(targetPath, async () => {
        fs.mkdirSync(path.dirname(targetPath), { recursive: true });
        fs.writeFileSync(targetPath, params.content, "utf8");
        return {
          content: [{ type: "text", text: `Wrote prepare.py` }],
          details: {},
        };
      });
    },
  });

  // ---- prepare_figures ----
  pi.registerTool({
    name: "prepare_figures",
    label: "Prepare Figures",
    description:
      "Run prepare.py in the current report/slides directory and read all generated or changed asset files (images, data files). Use this after editing prepare.py to see what figures and tables were produced.",
    promptSnippet: "Run prepare.py and inspect generated figures/tables",
    promptGuidelines: [
      "Use prepare_figures after editing prepare.py to run it and visually inspect the generated figures and tables.",
      "prepare_figures will list all asset files that were created or modified, and read image files so you can see them.",
    ],
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, signal, onUpdate, _ctx) {
      const session = getSessionState();
      if (!session) {
        return {
          content: [{ type: "text", text: "No active report/slides session. Start one with /report or /slides." }],
          details: {},
        };
      }

      onUpdate?.({ content: [{ type: "text", text: "Running prepare.py..." }] });

      const assetsDir = path.join(session.reportPath, "assets");
      const prepPy = path.join(session.reportPath, "prepare.py");

      if (!fs.existsSync(prepPy)) {
        return {
          content: [{ type: "text", text: `prepare.py not found at ${prepPy}` }],
          details: {},
        };
      }

      // Snapshot assets before running
      const beforeFiles = new Set<string>();
      if (fs.existsSync(assetsDir)) {
        for (const f of fs.readdirSync(assetsDir)) {
          const fp = path.join(assetsDir, f);
          if (fs.statSync(fp).isFile()) beforeFiles.add(f);
        }
      }

      const result = runInDir(`python3 '${prepPy}'`, session.reportPath, 120_000);
      const { stdout, stderr, code: exitCode } = result;

      // Find new/changed files
      const afterFiles: string[] = [];
      if (fs.existsSync(assetsDir)) {
        for (const f of fs.readdirSync(assetsDir)) {
          const fp = path.join(assetsDir, f);
          if (fs.statSync(fp).isFile()) {
            afterFiles.push(f);
          }
        }
      }

      const newFiles = afterFiles.filter((f) => !beforeFiles.has(f));
      const existingFiles = afterFiles.filter((f) => beforeFiles.has(f));

      const content: Array<
        | { type: "text"; text: string }
        | { type: "image"; data: string; mimeType: string }
      > = [];

      let summary = "";
      if (exitCode !== 0) {
        summary += `⚠️ prepare.py exited with code ${exitCode}\n\n`;
        if (stderr) summary += `Stderr:\n${stderr}\n\n`;
        if (stdout) summary += `Stdout:\n${stdout}\n`;
      } else {
        summary += `✅ prepare.py ran successfully.\n\n`;
        if (newFiles.length > 0) {
          summary += `New files: ${newFiles.join(", ")}\n`;
        }
        if (existingFiles.length > 0) {
          summary += `Updated files: ${existingFiles.join(", ")}\n`;
        }
        if (stdout.trim()) {
          summary += `\nOutput:\n${stdout}\n`;
        }
      }

      content.push({ type: "text", text: summary });

      // Read image files
      const imageExts = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"]);
      const mediaTypes: Record<string, string> = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
      };

      const allChanged = [...new Set([...newFiles, ...existingFiles])];
      for (const fname of allChanged) {
        const ext = path.extname(fname).toLowerCase();
        if (imageExts.has(ext)) {
          const fp = path.join(assetsDir, fname);
          try {
            const data = fs.readFileSync(fp);
            content.push({ type: "text", text: `\n--- ${fname} ---` });
            content.push(imageContent(data, mediaTypes[ext] || "image/png"));
          } catch {
            content.push({ type: "text", text: `\n⚠️ Could not read ${fname}` });
          }
        }
      }

      return { content, details: { exitCode, newFiles, existingFiles } };
    },
  });

  // ---- read_pages ----
  pi.registerTool({
    name: "read_pages",
    label: "Read Pages",
    description:
      "Compile the Typst document to per-page PNGs and read them for visual inspection. Call with no arguments to render all pages, or specify page numbers to focus on specific pages (all pages are rendered but only specified pages are shown).",
    promptSnippet: "Compile Typst to per-page PNGs for visual inspection",
    promptGuidelines: [
      "Use read_pages after every edit to the .typ file to visually verify layout, figure placement, text fitting, and overall appearance.",
      "Use read_pages without arguments to check all pages at once.",
      "Use read_pages with specific page numbers (e.g., pages: [1, 3]) to focus on pages affected by recent edits — all pages are compiled but only requested ones are shown.",
    ],
    parameters: Type.Object({
      pages: Type.Optional(Type.Array(Type.Number())),
    }),
    async execute(_toolCallId, params, signal, onUpdate, _ctx) {
      const session = getSessionState();
      if (!session) {
        return {
          content: [{ type: "text", text: "No active report/slides session. Start one with /report or /slides." }],
          details: {},
        };
      }

      const mainPath = path.join(session.reportPath, session.mainFile);
      if (!fs.existsSync(mainPath)) {
        return {
          content: [{ type: "text", text: `${session.mainFile} not found at ${mainPath}` }],
          details: {},
        };
      }

      onUpdate?.({ content: [{ type: "text", text: "Compiling to per-page PNGs..." }] });
      const { pages: pageFiles, exitCode, stderr } = compileToPngPages(session);

      if (exitCode !== 0 || pageFiles.length === 0) {
        let errText = `❌ Compilation failed (exit code ${exitCode})\n\n`;
        if (stderr) errText += `Stderr:\n${stderr}\n`;
        return {
          content: [{ type: "text", text: errText }],
          details: { exitCode, stderr },
        };
      }

      const specifiedPages = params.pages as number[] | undefined;
      let pagesToShow = pageFiles;
      if (specifiedPages && specifiedPages.length > 0) {
        const pageSet = new Set(specifiedPages);
        pagesToShow = pageFiles.filter((f) => {
          const m = f.match(/(\d+)/);
          return m && pageSet.has(parseInt(m[1]));
        });
        if (pagesToShow.length === 0) {
          pagesToShow = pageFiles;
        }
      }

      const content: Array<
        | { type: "text"; text: string }
        | { type: "image"; data: string; mimeType: string }
      > = [];
      content.push({
        type: "text",
        text: `✅ Compilation succeeded. ${pageFiles.length} page(s) total, showing ${pagesToShow.length}.\n`,
      });

      const checkDir = path.join(session.reportPath, "check");
      content.push(...readPageImages(checkDir, pagesToShow));

      return { content, details: { pageCount: pageFiles.length, shownPages: pagesToShow } };
    },
  });

  // ---- finish_report ----
  pi.registerTool({
    name: "finish_report",
    label: "Finish Report",
    description:
      "Finalize the report/slides: run prepare.py, compile to PDF, render all pages as PNGs for final review, and mark as complete. Use only when the document is ready — no rendering errors, data and narrative are self-consistent, no figure or table overflows page limits.",
    promptSnippet: "Finalize and mark the report as complete",
    promptGuidelines: [
      "Use finish_report ONLY when you are confident the document is ready: no rendering errors, all content is self-consistent, no figures or tables overflow page boundaries.",
      "finish_report runs prepare.py, compiles to PDF, renders all pages as PNGs for your final verification, then marks the session as done.",
    ],
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, signal, onUpdate, _ctx) {
      const session = getSessionState();
      if (!session) {
        return {
          content: [{ type: "text", text: "No active report/slides session." }],
          details: {},
        };
      }

      const root = getProjectRoot();
      const mainFile = session.mainFile;
      const mainPath = path.join(session.reportPath, mainFile);
      const pdfFile = session.type === "report" ? "report.pdf" : "slides.pdf";
      const pdfPath = path.join(session.reportPath, pdfFile);

      // 1. Run prepare.py if it has actual content
      const prepPy = path.join(session.reportPath, "prepare.py");
      if (fs.existsSync(prepPy)) {
        const prepContent = fs.readFileSync(prepPy, "utf8");
        if (
          !prepContent.includes("# Generate figures and tables here.") ||
          prepContent.split("\n").length > 15
        ) {
          onUpdate?.({ content: [{ type: "text", text: "Running prepare.py..." }] });
          runInDir(`python3 '${prepPy}'`, session.reportPath, 120_000);
        }
      }

      // 2. Compile to PDF
      onUpdate?.({ content: [{ type: "text", text: "Compiling to PDF..." }] });
      const pdfResult = runInDir(
        `typst compile --root '${root}' '${mainPath}' '${pdfPath}'`,
        session.reportPath,
      );
      const compileErr = pdfResult.stderr;
      const compileCode = pdfResult.code;

      if (compileCode !== 0 || !fs.existsSync(pdfPath)) {
        return {
          content: [
            {
              type: "text",
              text: `❌ Final PDF compilation failed (exit code ${compileCode})\n\n${compileErr}`,
            },
          ],
          details: {},
        };
      }

      // 3. Render all pages as PNGs
      onUpdate?.({ content: [{ type: "text", text: "Rendering pages for final check..." }] });
      const { pages: pageFiles, exitCode: pngCode, stderr: pngErr } = compileToPngPages(session);

      // 4. Mark as done — callback persists state and restores tools
      session.phase = "done";
      onSessionDone?.();

      const content: Array<
        | { type: "text"; text: string }
        | { type: "image"; data: string; mimeType: string }
      > = [];

      let header = `✅ Report finalized!\n`;
      header += `PDF: ${pdfPath}\n`;
      header += `Pages: ${pageFiles.length} rendered as PNGs in check/\n`;
      header += `\nUse \`make watch\` in the directory for live preview, or \`make check\` to re-render pages.\n`;

      if (pngCode !== 0) {
        header += `\n⚠️ PNG rendering had issues: ${pngErr}\n`;
      }

      content.push({ type: "text", text: header });

      const checkDir = path.join(session.reportPath, "check");
      content.push(...readPageImages(checkDir, pageFiles));

      return {
        content,
        details: { pageCount: pageFiles.length, pdfPath, done: true },
        terminate: true,
      };
    },
  });
}
