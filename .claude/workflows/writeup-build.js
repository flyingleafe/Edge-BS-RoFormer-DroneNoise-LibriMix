export const meta = {
  name: 'writeup-build',
  description:
    'Build one Typst deck or report from an approved narrative: jailed creator agent + adversarial read-only critic rounds. Requires /writeup phases 1-4 done first (narrative written, .claude/writeup/target.json armed).',
  phases: ['preflight', 'build', 'critique', 'collect'],
}

// args: { dir, kind, rounds?, prevArtifacts? }
//   dir           repo-relative artifact dir, e.g. "writing/slides/2026-07-13_foo"
//   kind          "slides" | "report"
//   rounds        max critic rounds (default 3)
//   prevArtifacts repo-relative dirs of the 2-3 most recent artifacts of the same kind
const _args = typeof args === 'string' ? JSON.parse(args) : args
if (!_args || !_args.dir || !_args.kind) {
  throw new Error(
    'writeup-build needs args {dir, kind, rounds?, prevArtifacts?}. Run /writeup phases 1-4 first (inventory -> narrative -> user checkpoint -> arm target.json), then launch this workflow.',
  )
}
const dir = _args.dir
const kind = _args.kind
const maxRounds = _args.rounds || 3
const prev = (_args.prevArtifacts || []).join(', ') || '(none listed)'
const doc = kind === 'slides' ? 'slides.typ' : 'report.typ'

// ---- preflight: the jail must be armed for exactly this dir ----------------
const pre = await agent(
  `Check the /writeup arming state. Read .claude/writeup/target.json and report:
- armed: does it exist and does its "dir" field equal "${dir}"?
- missing: which of these files do NOT exist: ${dir}/workflow/narrative.md, ${dir}/workflow/inventory.md, ${dir}/workflow/baseline-status.txt.
Do not fix anything; just report.`,
  {
    label: 'preflight',
    phase: 'preflight',
    effort: 'low',
    schema: {
      type: 'object',
      required: ['armed', 'missing'],
      properties: {
        armed: { type: 'boolean' },
        missing: { type: 'array', items: { type: 'string' } },
      },
    },
  },
)
if (!pre || !pre.armed || pre.missing.length > 0) {
  throw new Error(
    `writeup-build preflight failed: armed=${pre && pre.armed}, missing=[${pre ? pre.missing.join(', ') : 'preflight agent died'}]. Arm the workflow via /writeup phase 4 first.`,
  )
}

// ---- shared prompt fragments ----------------------------------------------
const contract = `Target: the ${kind} at ${dir}.
Narrative (your contract): ${dir}/workflow/narrative.md
Inventory (pointers to sources): ${dir}/workflow/inventory.md
Previous artifacts for voice/context: ${prev}
Keep ${dir}/workflow/creator-log.md up to date as you go (guard denials, friction, workarounds).`

const verifySpec = {
  type: 'object',
  required: ['clean', 'violations'],
  properties: {
    clean: { type: 'boolean' },
    violations: { type: 'array', items: { type: 'string' } },
  },
}

async function verifyTree(round) {
  return await agent(
    `Run "git status --porcelain" and compare its output with the baseline snapshot in ${dir}/workflow/baseline-status.txt.
Report violations: every path that changed relative to the baseline and is NOT under ${dir} and NOT under .claude/writeup/. Do not fix or restore anything — report only.`,
    { label: `verify r${round}`, phase: 'critique', effort: 'low', schema: verifySpec },
  )
}

// ---- round 1: build ---------------------------------------------------------
const build = await agent(
  `Build a ${kind} at ${dir}.
${contract}
Follow your build procedure: scaffold per the project template skill, prepare.py for assets, then make -C ${dir} check and visually inspect every rendered page in ${dir}/check/ before reporting. Your final reply: what you built, confirmation of visual inspection, remaining [TODO verify] markers, full file list.`,
  { label: 'creator r1', phase: 'build', agentType: 'writeup-creator', model: 'sonnet', effort: 'low' },
)
if (!build) {
  throw new Error('creator round 1 died; nothing to review. Re-run the workflow.')
}

// ---- critique loop ----------------------------------------------------------
const criticSpec = {
  type: 'object',
  required: ['verdict', 'issues'],
  properties: {
    verdict: { type: 'string', enum: ['APPROVE', 'REVISE'] },
    issues: {
      type: 'array',
      maxItems: 8,
      items: {
        type: 'object',
        required: ['category', 'where', 'defect', 'fix'],
        properties: {
          category: { type: 'string', enum: ['layout', 'figures', 'clarity', 'tone', 'narrative'] },
          where: { type: 'string' },
          defect: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const rounds = []
let approved = false

for (let round = 1; round <= maxRounds; round++) {
  const critique = await agent(
    `Review the ${kind} at ${dir}, round ${round}.
Previous artifacts the reader has seen: ${prev}
Rendered pages: ${dir}/check/page-*.png
Follow your procedure exactly: look at every rendered page first, then sources, then check against ${dir}/workflow/narrative.md.`,
    { label: `critic r${round}`, phase: 'critique', agentType: 'writeup-critic', model: 'opus', schema: criticSpec },
  )
  if (!critique) {
    rounds.push({ round, verdict: 'CRITIC-DIED', issues: [], verify: await verifyTree(round) })
    continue
  }

  const verify = await verifyTree(round)
  rounds.push({ round, verdict: critique.verdict, issues: critique.issues, verify })

  if (critique.verdict === 'APPROVE') {
    approved = true
    break
  }
  if (round === maxRounds) break

  const issueList = critique.issues
    .map((i, k) => `${k + 1}. [${i.category}] (${i.where}) ${i.defect} — fix: ${i.fix}`)
    .join('\n')
  const violationNote =
    verify && !verify.clean
      ? `\nALSO: you (or a previous instance) modified files outside your directory: ${verify.violations.join(', ')}. This is a protocol violation — do not do it again; the orchestrator will restore them.`
      : ''

  const revise = await agent(
    `A previous creator instance built the ${kind} at ${dir}; all state is on disk. You are CONTINUING its work, not restarting.
${contract}
First read the narrative, the creator log, and the current ${dir}/${doc}. Save the critique below verbatim to ${dir}/workflow/critique-round-${round}.md. Then address every numbered issue (fix it, or rebut at most 1-2 with a one-line reason), rebuild with make -C ${dir} check, visually re-inspect the changed pages, and append friction to the creator log.${violationNote}
Critic round ${round} said REVISE:
${issueList}
Your final reply: a numbered list mirroring the critique — what changed or why rejected.`,
    { label: `creator r${round + 1}`, phase: 'build', agentType: 'writeup-creator', model: 'sonnet', effort: 'low' },
  )
  if (!revise) {
    rounds.push({ round: round + 1, verdict: 'CREATOR-DIED', issues: [], verify: null })
    break
  }
}

// ---- collect: friction report + final state --------------------------------
const collect = await agent(
  `Read-only collection pass for ${dir}:
1. Return the full verbatim content of ${dir}/workflow/creator-log.md (empty string if absent).
2. List the files under ${dir} (top level + workflow/, not check/ pages individually).
3. Grep the ${doc} for "[TODO verify]" occurrences and list them.
4. Confirm ${dir}/check/ contains rendered page PNGs newer than ${doc} (stale = false).`,
  {
    label: 'collect',
    phase: 'collect',
    effort: 'low',
    schema: {
      type: 'object',
      required: ['frictionLog', 'files', 'todos', 'renderFresh'],
      properties: {
        frictionLog: { type: 'string' },
        files: { type: 'array', items: { type: 'string' } },
        todos: { type: 'array', items: { type: 'string' } },
        renderFresh: { type: 'boolean' },
      },
    },
  },
)

return {
  dir,
  kind,
  approved,
  roundsRun: rounds.length,
  rounds,
  frictionLog: collect ? collect.frictionLog : '(collect agent died)',
  files: collect ? collect.files : [],
  todos: collect ? collect.todos : [],
  renderFresh: collect ? collect.renderFresh : false,
  next: 'Orchestrator: restore any reported violations, persist critiques to workflow/, run user review (/writeup phase 7), then disarm target.json.',
}
