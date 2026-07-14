export const meta = {
  name: 'writeup-probe',
  description: 'Probe: verify the workflow runtime is available and can spawn one agent',
}

const res = await agent(
  'Reply with a one-line confirmation containing the word PROBE-OK and the current git branch of the repository you are in (run: git branch --show-current).',
  { label: 'probe' },
)

return res
