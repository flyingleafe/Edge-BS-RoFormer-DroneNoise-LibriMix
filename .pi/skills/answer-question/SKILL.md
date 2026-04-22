---
name: answer-question
description: Information-retrieval tasks — explaining how something works, locating code or data, summarizing domain knowledge. Use whenever the current task is retrieval-and-explanation rather than creation or modification.
---

# Answer a Question

For information queries that don't require creating or modifying anything.

> The Bootstrap and `record-and-remember` closing are global invariants — see root AGENTS.md. This skill only adds task-specific substance.

## Steps

1. **Locate the answer.** Check in this order:
   - Relevant subdirectory `AGENTS.md`
   - Root `AGENTS.md` key facts section
   - Code comments and docstrings in the relevant source files
   - Source code itself
   - Existing docs in `docs/`

2. **Answer concisely.** Cite the source — file path, line, or section.

3. **Assess documentation quality.** Was the answer easy to find? If not, that's a gap worth recording.

## Principles

- **Don't guess.** If unsure, read the source.
- **Point to the file.** Exact paths let the user learn more on their own.
- **Documentation gaps are problems.** If you struggled, that's a `record-and-remember` event — fix the gap, don't just answer.
