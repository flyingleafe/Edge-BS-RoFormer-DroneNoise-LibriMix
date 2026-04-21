---
name: examine-presentation-slides
description: Start Slidev and visually inspect presentation slides using browser MCP (cursor-ide-browser or user-playwright). Use when the user wants to run a Slidev presentation and see how it looks, check slide rendering, or capture slides as images.
---

# Visually Examine Presentation Slides

Start a Slidev dev server and use browser MCP to navigate, screenshot, and step through slides so the user (or agent) can verify layout, content, and styling.

## Prerequisites

- Slidev entry exists (e.g. `slides/slides.md`)
- `npx slidev` available (default port **3030**)

## Workflow

### 1. Start Slidev so it stays running

Slidev/Vite often exits when run without a TTY (e.g. in background or headless). Use a pseudo-TTY so the dev server keeps running:

```bash
cd slides && script -q -c "npx slidev slides.md --open=false" /dev/null &
```

- Run from the project root or from the directory that contains `slides.md`.
- `--open=false` avoids opening a system browser.
- Wait **~10–14 seconds** after starting before using the URL.

**Verify**: `curl -sI http://127.0.0.1:3030/` or `ss -tlnp | grep 3030`.

### 2. Inspect with browser MCP

Prefer **cursor-ide-browser** (uses the IDE’s embedded browser; no extra install). If the user wants **user-playwright**, ensure Chromium is installed: `npx playwright install chrome`.

| Step | cursor-ide-browser | user-playwright |
|------|--------------------|-----------------|
| Open deck | `browser_navigate` → `http://localhost:3030/` | `browser_navigate` → `http://localhost:3030/` |
| Let slide render | `browser_wait_for` ~1–2 s | Same |
| Capture view | `browser_take_screenshot` (optional `filename`, `fullPage`) | `browser_take_screenshot` |
| Next slide | `browser_press_key` → `ArrowRight` (or `Space`) | Same |
| Previous slide | `browser_press_key` → `ArrowLeft` | Same |

**Suggested sequence**:

1. `browser_navigate` to `http://localhost:3030/` (use `newTab: true` if you want a fresh tab).
2. `browser_wait_for` 1–2 seconds so the first slide finishes rendering.
3. `browser_take_screenshot` for slide 1.
4. To review more slides: `browser_press_key` with `ArrowRight`, then optional short wait + `browser_take_screenshot`; repeat as needed.

### 3. Report what you see

Summarize for the user:

- **Title** of the presentation (from the page title or first slide).
- **Slide count** (e.g. from footer “1 / N”).
- **Content of captured slides**: titles, bullets, layout, theme (e.g. dark/light).
- **Screenshot paths** if you passed `filename` (e.g. under `/tmp/cursor/screenshots/`).

## Browser MCP details

- **cursor-ide-browser**: Use `browser_tabs` with `action: "list"` to get `viewId` for `viewId` in other calls. Screenshots can be saved with `filename`; omit to use default naming.
- **user-playwright**: Requires `npx playwright install chrome` if you get “Chromium distribution 'chrome' is not found”.

## Notes

- Slidev URLs: `/` = slide 1, `/1` = slide 1, `/2` = slide 2, etc. Navigating to `http://localhost:3030/` or `http://localhost:3030/1` both show the first slide.
- To examine a specific slide by URL, use `http://localhost:3030/<slide_index>` (1-based).
- If the server fails to bind, re-run the start command and wait again; avoid starting multiple Slidev instances on 3030.
