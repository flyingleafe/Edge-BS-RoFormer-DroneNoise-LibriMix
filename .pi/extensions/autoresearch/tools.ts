import * as fs from "node:fs";
import * as path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  formatSize,
  truncateTail,
} from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";

const SCRATCH = "/gpfs/scratch/acw592";
const LOG_DIR = path.join(SCRATCH, "logs");
const DEFAULT_TIMEOUT = 20_000;

function text(content: string) {
  return [{ type: "text" as const, text: content }];
}

async function run(
  pi: ExtensionAPI,
  command: string,
  args: string[],
  signal: AbortSignal | undefined,
  timeout = DEFAULT_TIMEOUT,
): Promise<{ code: number; stdout: string; stderr: string }> {
  const result = await pi.exec(command, args, { signal, timeout });
  return {
    code: result.code ?? 0,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

function truncateForTool(output: string): { content: string; details: Record<string, unknown> } {
  const truncation = truncateTail(output, {
    maxLines: DEFAULT_MAX_LINES,
    maxBytes: DEFAULT_MAX_BYTES,
  });
  let content = truncation.content;
  const details: Record<string, unknown> = { truncation };
  if (truncation.truncated) {
    content += `\n\n[Output truncated: showing last ${truncation.outputLines} of ${truncation.totalLines} lines`;
    content += ` (${formatSize(truncation.outputBytes)} of ${formatSize(truncation.totalBytes)}).]`;
  }
  return { content, details };
}

function parseSubmittedJobId(stdout: string): string | undefined {
  const match = stdout.match(/Submitted batch job\s+(\d+)/);
  return match?.[1];
}

async function querySqueue(
  pi: ExtensionAPI,
  signal: AbortSignal | undefined,
  opts: { jobId?: string; jobName?: string },
): Promise<string> {
  const args = ["--noheader", "-o", "%i|%j|%T|%M|%R"];
  if (opts.jobId) args.push("-j", opts.jobId);
  if (opts.jobName) args.push("-n", opts.jobName);
  const result = await run(pi, "squeue", args, signal);
  if (result.code !== 0) return `squeue failed:\n${result.stderr || result.stdout}`;
  return result.stdout.trim();
}

async function querySacct(
  pi: ExtensionAPI,
  signal: AbortSignal | undefined,
  jobId: string,
): Promise<string> {
  const result = await run(
    pi,
    "sacct",
    ["-j", jobId, "--format=JobID,JobName,State,Elapsed,ExitCode", "-P", "-n"],
    signal,
  );
  if (result.code !== 0) return `sacct unavailable/failed:\n${result.stderr || result.stdout}`;
  return result.stdout.trim();
}

function latestMtime(file: string): number {
  try {
    return fs.statSync(file).mtimeMs;
  } catch {
    return 0;
  }
}

function findLogFiles(jobId?: string, jobName?: string): string[] {
  if (!fs.existsSync(LOG_DIR)) return [];
  const files = fs.readdirSync(LOG_DIR);
  const matches = files.filter((file) => {
    if (jobId && file.endsWith(`.o${jobId}`)) return true;
    if (jobName && file.startsWith(`${jobName}.o`)) return true;
    return false;
  });
  return matches
    .map((file) => path.join(LOG_DIR, file))
    .sort((a, b) => latestMtime(b) - latestMtime(a));
}

function tailFile(file: string, lines: number): string {
  const content = fs.readFileSync(file, "utf8");
  const split = content.split(/\r?\n/);
  return split.slice(Math.max(0, split.length - lines)).join("\n");
}

type SlurmPartition = "gpushort" | "sae";

const SHORT_PARTITION: SlurmPartition = "gpushort";
const LONG_PARTITION: SlurmPartition = "sae";
const SHORT_DEFAULT_TIME = "1:00:00";
const LONG_DEFAULT_TIME = "4:00:00";

export const SLURM_TOOL_NAMES = [
  "slurm_submit_short",
  "slurm_submit_long",
  "slurm_submit",
  "slurm_status",
  "slurm_logs",
];

const SUBMIT_BASE_PARAMETERS = {
  command: Type.String({
    description:
      "Shell command to run inside the Slurm job, e.g. python train_rps_predictor.py --model simple_conv_v2 ...",
  }),
  jobName: Type.Optional(Type.String({ description: "Slurm job name (-J)." })),
  time: Type.Optional(
    Type.String({
      description:
        "Slurm time limit. gpushort jobs must be <= 1:00:00; sae jobs may be longer (cluster max: 10-00:00:00).",
    }),
  ),
  slurmArgs: Type.Optional(
    Type.Array(Type.String(), {
      description:
        "Additional sbatch arguments before --, e.g. ['--cpus-per-gpu=4']. Do not include --partition or --time; use the dedicated parameters/tool instead.",
    }),
  ),
};

function submitParameters(includePartition = false) {
  const params: any = { ...SUBMIT_BASE_PARAMETERS };
  if (includePartition) {
    params.partition = Type.Optional(
      Type.String({
        description:
          "Slurm partition: 'gpushort' for short <=1h jobs or 'sae' for longer jobs. Default: gpushort.",
      }),
    );
  }
  return Type.Object(params);
}

function normalizePartition(value: string | undefined, fallback: SlurmPartition): SlurmPartition {
  if (!value) return fallback;
  if (value === SHORT_PARTITION || value === LONG_PARTITION) return value;
  throw new Error(`Unsupported Slurm partition '${value}'; expected '${SHORT_PARTITION}' or '${LONG_PARTITION}'.`);
}

function defaultTimeForPartition(partition: SlurmPartition): string {
  return partition === LONG_PARTITION ? LONG_DEFAULT_TIME : SHORT_DEFAULT_TIME;
}

function assertNoControlledSlurmArgs(slurmArgs: string[] | undefined) {
  for (const arg of slurmArgs ?? []) {
    if (arg === "-p" || arg === "--partition" || arg.startsWith("-p") || arg.startsWith("--partition=")) {
      throw new Error("Do not pass partition in slurmArgs; use slurm_submit_short/slurm_submit_long or the partition parameter.");
    }
    if (arg === "-t" || arg === "--time" || arg.startsWith("-t") || arg.startsWith("--time=")) {
      throw new Error("Do not pass time in slurmArgs; use the time parameter.");
    }
  }
}

async function executeSubmit(
  pi: ExtensionAPI,
  params: {
    command: string;
    jobName?: string;
    time?: string;
    slurmArgs?: string[];
    partition?: string;
  },
  signal: AbortSignal | undefined,
  ctx: { cwd: string },
  fallbackPartition: SlurmPartition,
) {
  const wrapper = path.join(ctx.cwd, "sbatch.sh");
  if (!fs.existsSync(wrapper)) {
    throw new Error(`Missing ${wrapper}; create/update the Slurm wrapper before submitting jobs.`);
  }

  const partition = normalizePartition(params.partition, fallbackPartition);
  const time = params.time ?? defaultTimeForPartition(partition);
  assertNoControlledSlurmArgs(params.slurmArgs);

  const args: string[] = ["--partition", partition, "--time", time];
  if (params.jobName) args.push("-J", params.jobName);
  if (params.slurmArgs) args.push(...params.slurmArgs);
  args.push("--", "bash", "-lc", params.command);

  const result = await run(pi, "./sbatch.sh", args, signal, 30_000);
  const combined = `${result.stdout}${result.stderr ? `\n${result.stderr}` : ""}`.trim();
  if (result.code !== 0) {
    throw new Error(`sbatch failed (exit ${result.code}):\n${combined}`);
  }

  const jobId = parseSubmittedJobId(result.stdout);
  const status = jobId ? await querySqueue(pi, signal, { jobId }) : "";
  const files = findLogFiles(jobId, params.jobName);
  const summary = [
    combined,
    jobId ? `Job ID: ${jobId}` : "Job ID: unknown",
    params.jobName ? `Job name: ${params.jobName}` : undefined,
    `Partition: ${partition}`,
    `Time limit: ${time}`,
    status ? `Immediate squeue status:\n${status}` : undefined,
    files.length ? `Log file(s):\n${files.join("\n")}` : `Logs will appear under ${LOG_DIR}/%x.o%j`,
  ]
    .filter(Boolean)
    .join("\n\n");

  return {
    content: text(summary),
    details: { jobId, jobName: params.jobName, command: params.command, partition, time, args, status, logFiles: files },
  };
}

function registerPartitionSubmitTool(
  pi: ExtensionAPI,
  opts: {
    name: string;
    label: string;
    partition: SlurmPartition;
    description: string;
    promptSnippet: string;
    promptGuidelines: string[];
  },
) {
  pi.registerTool({
    name: opts.name,
    label: opts.label,
    description: opts.description,
    promptSnippet: opts.promptSnippet,
    promptGuidelines: opts.promptGuidelines,
    parameters: submitParameters(false),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      return executeSubmit(pi, params as any, signal, ctx, opts.partition);
    },
  });
}

export function registerSlurmTools(pi: ExtensionAPI) {
  registerPartitionSubmitTool(pi, {
    name: "slurm_submit_short",
    label: "Submit Short Slurm Job",
    partition: SHORT_PARTITION,
    description:
      "Submit a short GPU job (<= 1 hour) to the gpushort Slurm partition through ./sbatch.sh. Use for smoke tests and quick training jobs; never run GPU training directly on the login node.",
    promptSnippet: "Submit a short <=1h gpushort Slurm job through ./sbatch.sh",
    promptGuidelines: [
      "Use slurm_submit_short for jobs expected to finish within 1 hour on the gpushort partition.",
      "slurm_submit_short commands should put datasets and results under /gpfs/scratch/acw592.",
    ],
  });

  registerPartitionSubmitTool(pi, {
    name: "slurm_submit_long",
    label: "Submit Long Slurm Job",
    partition: LONG_PARTITION,
    description:
      "Submit a longer GPU job to the sae Slurm partition through ./sbatch.sh. Use for training jobs expected to exceed 1 hour; default time is 4:00:00 unless overridden.",
    promptSnippet: "Submit a longer sae Slurm training job through ./sbatch.sh",
    promptGuidelines: [
      "Use slurm_submit_long for jobs expected to exceed 1 hour or needing the better GPUs on the sae partition.",
      "Set an explicit time limit when you know the expected runtime; sae supports longer jobs than gpushort.",
      "slurm_submit_long commands should put datasets and results under /gpfs/scratch/acw592.",
    ],
  });

  pi.registerTool({
    name: "slurm_submit",
    label: "Submit Slurm Job",
    description:
      "Submit a Slurm GPU job through ./sbatch.sh. Prefer slurm_submit_short for <=1h gpushort jobs and slurm_submit_long for longer sae jobs; this compatibility tool defaults to gpushort.",
    promptSnippet: "Submit a Slurm training job through ./sbatch.sh",
    promptGuidelines: [
      "Prefer slurm_submit_short for <=1h jobs and slurm_submit_long for longer sae jobs.",
      "Use the partition parameter only when you need one generic submission call.",
      "slurm_submit commands should put datasets and results under /gpfs/scratch/acw592.",
    ],
    parameters: submitParameters(true),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      return executeSubmit(pi, params as any, signal, ctx, SHORT_PARTITION);
    },
  });

  pi.registerTool({
    name: "slurm_status",
    label: "Slurm Job Status",
    description:
      "Check Slurm job status via squeue and sacct. Provide jobId when possible; jobName is supported for queued/running jobs.",
    promptSnippet: "Check status of Slurm jobs with squeue/sacct",
    promptGuidelines: [
      "Use slurm_status after slurm_submit_short/slurm_submit_long/slurm_submit to decide whether a job is RUNNING or PENDING; stop submitting new autoresearch jobs once the first job is pending/queued.",
    ],
    parameters: Type.Object({
      jobId: Type.Optional(Type.String({ description: "Slurm job id." })),
      jobName: Type.Optional(Type.String({ description: "Slurm job name." })),
    }),
    async execute(_toolCallId, params, signal) {
      if (!params.jobId && !params.jobName) {
        throw new Error("Provide jobId or jobName.");
      }
      const squeue = await querySqueue(pi, signal, params);
      const sacct = params.jobId ? await querySacct(pi, signal, params.jobId) : "sacct skipped: no jobId provided";
      const output = ["squeue:", squeue || "<no queued/running match>", "", "sacct:", sacct || "<no accounting rows>"].join("\n");
      const truncated = truncateForTool(output);
      return {
        content: text(truncated.content),
        details: { ...truncated.details, squeue, sacct },
      };
    },
  });

  pi.registerTool({
    name: "slurm_logs",
    label: "Slurm Job Logs",
    description:
      "Read/tail Slurm logs from /gpfs/scratch/acw592/logs. Output is truncated to the standard Pi tool limit if necessary.",
    promptSnippet: "Read Slurm logs from /gpfs/scratch/acw592/logs",
    promptGuidelines: [
      "Use slurm_logs to diagnose failed training jobs before editing code or restarting the job.",
    ],
    parameters: Type.Object({
      jobId: Type.Optional(Type.String({ description: "Slurm job id." })),
      jobName: Type.Optional(Type.String({ description: "Slurm job name." })),
      lines: Type.Optional(Type.Number({ description: "Number of tail lines per log file. Default: 200." })),
    }),
    async execute(_toolCallId, params) {
      if (!params.jobId && !params.jobName) {
        throw new Error("Provide jobId or jobName.");
      }
      const lines = Math.max(1, Math.min(5000, Math.floor(params.lines ?? 200)));
      const files = findLogFiles(params.jobId, params.jobName);
      if (files.length === 0) {
        return {
          content: text(`No matching log files found in ${LOG_DIR}. Logs usually appear as <job-name>.o<job-id>.`),
          details: { logDir: LOG_DIR, files: [] },
        };
      }

      const output = files
        .map((file) => `===== ${file} (last ${lines} lines) =====\n${tailFile(file, lines)}`)
        .join("\n\n");
      const truncated = truncateForTool(output);
      return {
        content: text(truncated.content),
        details: { ...truncated.details, logDir: LOG_DIR, files, lines },
      };
    },
    renderCall(args, theme) {
      const target = args.jobId ? `job ${args.jobId}` : args.jobName ? `name ${args.jobName}` : "job";
      return new Text(
        `${theme.fg("toolTitle", theme.bold("slurm_logs "))}${theme.fg("accent", target)}`,
        0,
        0,
      );
    },
  });
}
