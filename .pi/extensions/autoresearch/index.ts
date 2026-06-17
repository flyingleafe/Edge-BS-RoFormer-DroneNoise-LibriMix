import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { registerSlurmTools, SLURM_TOOL_NAMES } from "./tools.js";
import {
  DATASETS_ROOT,
  RESULTS_ROOT,
  experimentsTemplate,
  ideasTemplate,
  kickoffPrompt,
  leaderboardTemplate,
  loopInstructions,
} from "./prompts.js";

export interface AutoresearchSession {
  id: string;
  safeId: string;
  dataset: string;
  metrics: string;
  baseline: string;
  trainingArgs: string;
  initialIdeas: string;
  artifactDir: string; // repo-relative, committed to git
  artifactAbsDir: string;
  ideasPath: string;
  experimentsPath: string;
  leaderboardPath: string;
  sessionJsonPath: string;
  startedAt: string;
}

let currentSession: AutoresearchSession | null = null;
let defaultActiveTools: string[] | null = null;

function getProjectRoot(): string {
  return execSync("git rev-parse --show-toplevel", { encoding: "utf8" }).trim();
}

function slugify(input: string): string {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
}

function nowStamp(): string {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const min = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${yyyy}${mm}${dd}-${hh}${min}${ss}`;
}

function activeToolNames(pi: ExtensionAPI): string[] {
  return (pi.getActiveTools() as any[])
    .map((tool) => (typeof tool === "string" ? tool : tool?.name))
    .filter((name): name is string => typeof name === "string" && name.length > 0);
}

function ensureSlurmToolsActive(pi: ExtensionAPI) {
  const active = activeToolNames(pi);
  const merged = [...new Set([...active, ...SLURM_TOOL_NAMES])];
  pi.setActiveTools(merged);
}

function restoreDefaultTools(pi: ExtensionAPI) {
  if (defaultActiveTools) pi.setActiveTools(defaultActiveTools);
}

function listDatasets(): string[] {
  try {
    if (!fs.existsSync(DATASETS_ROOT)) return [];
    return fs
      .readdirSync(DATASETS_ROOT, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name)
      .sort();
  } catch {
    return [];
  }
}

async function chooseDataset(ctx: ExtensionContext): Promise<string | undefined> {
  const datasets = listDatasets();
  if (datasets.length > 0) {
    const choice = await ctx.ui.select("Dataset under /gpfs/scratch/acw592/datasets", [
      ...datasets.slice(0, 50),
      "Other / type manually...",
    ]);
    if (!choice) return undefined;
    if (choice !== "Other / type manually...") return choice;
  }
  return ctx.ui.input("Dataset name under /gpfs/scratch/acw592/datasets:", "DREGON-LM-V4");
}

function makeSession(input: {
  dataset: string;
  metrics: string;
  baseline: string;
  trainingArgs: string;
  initialIdeas: string;
}): AutoresearchSession {
  const root = getProjectRoot();
  const stamp = nowStamp();
  const safeDataset = slugify(input.dataset) || "dataset";
  const safeBaseline = slugify(input.baseline) || "baseline";
  const safeId = `${stamp}-${safeDataset}-${safeBaseline}`;
  const id = safeId;
  const artifactDir = path.join("autoresearch", id);
  const artifactAbsDir = path.join(root, artifactDir);
  return {
    id,
    safeId,
    dataset: input.dataset,
    metrics: input.metrics,
    baseline: input.baseline,
    trainingArgs: input.trainingArgs,
    initialIdeas: input.initialIdeas,
    artifactDir,
    artifactAbsDir,
    ideasPath: path.join(artifactDir, "ideas.md"),
    experimentsPath: path.join(artifactDir, "experiments.md"),
    leaderboardPath: path.join(artifactDir, "leaderboard.md"),
    sessionJsonPath: path.join(artifactDir, "session.json"),
    startedAt: new Date().toISOString(),
  };
}

function scaffoldArtifacts(session: AutoresearchSession) {
  fs.mkdirSync(session.artifactAbsDir, { recursive: true });
  const abs = (rel: string) => path.join(getProjectRoot(), rel);
  fs.writeFileSync(abs(session.ideasPath), ideasTemplate(session), "utf8");
  fs.writeFileSync(abs(session.experimentsPath), experimentsTemplate(session), "utf8");
  fs.writeFileSync(abs(session.leaderboardPath), leaderboardTemplate(session), "utf8");
  fs.writeFileSync(
    abs(session.sessionJsonPath),
    JSON.stringify(
      {
        ...session,
        resultsRoot: `${RESULTS_ROOT}/${session.id}`,
        datasetPath: `${DATASETS_ROOT}/${session.dataset}`,
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
}

function restoreSessionFromBranch(ctx: ExtensionContext): AutoresearchSession | null {
  const branch = ctx.sessionManager.getBranch();
  for (let i = branch.length - 1; i >= 0; i--) {
    const entry = branch[i];
    if (entry.type === "custom" && entry.customType === "autoresearch-session") {
      return entry.data as AutoresearchSession;
    }
  }
  return null;
}

function listSavedSessions(): Array<{ label: string; sessionJson: string; mtimeMs: number }> {
  const root = getProjectRoot();
  const dir = path.join(root, "autoresearch");
  if (!fs.existsSync(dir)) return [];

  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => {
      const sessionJson = path.join(dir, d.name, "session.json");
      if (!fs.existsSync(sessionJson)) return null;
      const raw = JSON.parse(fs.readFileSync(sessionJson, "utf8"));
      const stat = fs.statSync(sessionJson);
      return {
        sessionJson,
        mtimeMs: stat.mtimeMs,
        label: `${raw.id ?? d.name} — ${raw.dataset ?? "?"} / ${raw.baseline ?? "?"}`,
      };
    })
    .filter((x): x is { label: string; sessionJson: string; mtimeMs: number } => x !== null)
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
}

function loadSessionFromJson(sessionJson: string): AutoresearchSession {
  const root = getProjectRoot();
  const absSessionJson = path.resolve(sessionJson);
  const raw = JSON.parse(fs.readFileSync(absSessionJson, "utf8"));
  const artifactAbsDir = path.dirname(absSessionJson);
  const artifactDir = path.relative(root, artifactAbsDir);
  const id = String(raw.id ?? path.basename(artifactAbsDir));

  return {
    id,
    safeId: String(raw.safeId ?? slugify(id) ?? id),
    dataset: String(raw.dataset ?? "DREGON-LM-V4"),
    metrics: String(raw.metrics ?? "pit_mse:min, mae_frame:min, mae_clip:min, r2:max"),
    baseline: String(raw.baseline ?? "simple_conv_v2"),
    trainingArgs: String(raw.trainingArgs ?? ""),
    initialIdeas: String(raw.initialIdeas ?? ""),
    artifactDir,
    artifactAbsDir,
    ideasPath: String(raw.ideasPath ?? path.join(artifactDir, "ideas.md")),
    experimentsPath: String(raw.experimentsPath ?? path.join(artifactDir, "experiments.md")),
    leaderboardPath: String(raw.leaderboardPath ?? path.join(artifactDir, "leaderboard.md")),
    sessionJsonPath: String(raw.sessionJsonPath ?? path.join(artifactDir, "session.json")),
    startedAt: String(raw.startedAt ?? new Date(fs.statSync(absSessionJson).mtimeMs).toISOString()),
  };
}

async function resumeAutoresearch(pi: ExtensionAPI, args: string, ctx: ExtensionContext) {
  if (!ctx.hasUI) {
    ctx.ui.notify("/autoresearch-resume requires an interactive UI.", "warning");
    return;
  }
  if (!ctx.isIdle()) {
    ctx.ui.notify("Agent is busy. Wait before resuming autoresearch.", "warning");
    return;
  }

  let sessionJson = args.trim();
  if (sessionJson) {
    if (fs.existsSync(sessionJson) && fs.statSync(sessionJson).isDirectory()) {
      sessionJson = path.join(sessionJson, "session.json");
    }
  } else {
    const sessions = listSavedSessions();
    if (sessions.length === 0) {
      ctx.ui.notify("No autoresearch/*/session.json files found.", "warning");
      return;
    }
    const choice = await ctx.ui.select("Resume autoresearch session", sessions.map((s) => s.label));
    if (!choice) return;
    const selected = sessions.find((s) => s.label === choice);
    if (!selected) return;
    sessionJson = selected.sessionJson;
  }

  if (!fs.existsSync(sessionJson)) {
    ctx.ui.notify(`Missing session file: ${sessionJson}`, "error");
    return;
  }

  const session = loadSessionFromJson(sessionJson);
  currentSession = session;
  pi.appendEntry("autoresearch-session", session);
  pi.setSessionName(`autoresearch: ${session.baseline} on ${session.dataset}`);
  ensureSlurmToolsActive(pi);
  ctx.ui.setStatus("autoresearch", session.id);

  pi.sendUserMessage(`[AUTORESEARCH RESUME]

Resume the existing autoresearch session:

- Session: ${session.id}
- Artifacts: ${session.artifactDir}
- Dataset: ${session.dataset}
- Baseline: ${session.baseline}
- Results root: ${RESULTS_ROOT}/${session.id}

Do not create a new session. First re-read:

1. ${session.sessionJsonPath}
2. ${session.ideasPath}
3. ${session.experimentsPath}
4. ${session.leaderboardPath}

Then check git status and reconstruct the current state. If jobs are listed as running/pending/unknown, use slurm_status and slurm_logs before editing code or submitting new jobs. Continue from the next safe step in the loop.`);
}

async function startAutoresearch(pi: ExtensionAPI, _args: string, ctx: ExtensionContext) {
  if (!ctx.hasUI) {
    ctx.ui.notify("/autoresearch requires an interactive UI so it can gather initial context.", "warning");
    return;
  }
  if (!ctx.isIdle()) {
    ctx.ui.notify("Agent is busy. Wait for it to finish before starting /autoresearch.", "warning");
    return;
  }

  const dataset = (await chooseDataset(ctx))?.trim();
  if (!dataset) return;

  const metrics =
    (await ctx.ui.input(
      "Target validation metrics (include direction if useful):",
      "pit_mse:min, mae_frame:min, mae_clip:min, r2:max",
    ))?.trim() || "pit_mse:min, mae_frame:min, mae_clip:min, r2:max";

  const baseline =
    (await ctx.ui.input("Baseline model key:", "simple_conv_v2"))?.trim() || "simple_conv_v2";

  const trainingArgs =
    (await ctx.ui.editor(
      "Fixed extra training args for baseline and all candidates:",
      "--batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress",
    )) ?? "";

  const initialIdeas =
    (await ctx.ui.editor(
      "Initial architecture ideas / constraints (optional):",
      "Start near simple_conv_v2. Prefer small, comparable variants that can smoke-test with one forward pass and train within gpushort. Keep the first batch hypothesis-diverse rather than many hyperparameter-only tweaks.",
    )) ?? "";

  const session = makeSession({ dataset, metrics, baseline, trainingArgs, initialIdeas });
  const summary = `Dataset: ${dataset}\nMetrics: ${metrics}\nBaseline: ${baseline}\nArtifacts: ${session.artifactDir}\nResults: ${RESULTS_ROOT}/${session.id}\n\nStart autoresearch?`;
  const ok = await ctx.ui.confirm("Start autoresearch session", summary);
  if (!ok) return;

  scaffoldArtifacts(session);
  currentSession = session;
  pi.appendEntry("autoresearch-session", session);
  pi.setSessionName(`autoresearch: ${baseline} on ${dataset}`);
  ensureSlurmToolsActive(pi);

  ctx.ui.notify(`Scaffolded ${session.artifactDir}`, "info");
  pi.sendUserMessage(kickoffPrompt(session));
}

export default function autoresearchExtension(pi: ExtensionAPI) {
  registerSlurmTools(pi);

  pi.on("session_start", async (_event, ctx) => {
    if (defaultActiveTools === null) {
      defaultActiveTools = activeToolNames(pi);
    }
    currentSession = restoreSessionFromBranch(ctx);
    ensureSlurmToolsActive(pi);
    if (currentSession) {
      ctx.ui.setStatus("autoresearch", currentSession.id);
    }
  });

  pi.on("before_agent_start", async (event, _ctx) => {
    if (!currentSession) return;
    return { systemPrompt: event.systemPrompt + "\n\n" + loopInstructions(currentSession) };
  });

  pi.on("agent_end", async (_event, _ctx) => {
    if (currentSession) pi.appendEntry("autoresearch-session", currentSession);
  });

  pi.on("session_shutdown", async (_event, ctx) => {
    if (currentSession) pi.appendEntry("autoresearch-session", currentSession);
    ctx.ui.setStatus("autoresearch", undefined);
    restoreDefaultTools(pi);
  });

  pi.registerCommand("autoresearch", {
    description: "Start the architectural autoresearch loop for RPS models on gpushort Slurm.",
    handler: async (args, ctx) => startAutoresearch(pi, args, ctx),
  });

  pi.registerCommand("autoresearch-resume", {
    description: "Resume an existing autoresearch session from autoresearch/*/session.json.",
    handler: async (args, ctx) => resumeAutoresearch(pi, args, ctx),
  });
}
