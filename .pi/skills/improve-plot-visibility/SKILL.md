---
name: improve-plot-visibility
description: How to look at the generated plots, assess their quality, and improve them.
---

# Improve Plot Visibility

When a figure's key features are hard to see (e.g., faint lines, low contrast, vague structures), systematically test visualization parameters and verify the result.

## Workflow

1. **Locate the plotting code** — find the script/function that generates the figure.
2. **Render the current version** — save as PNG/image and inspect to confirm the problem.
3. **Identify the visual channel** — what makes the feature hard to see?
   - Dynamic range too wide? → try percentile clipping (`vmin`, `vmax`)
   - Colormap doesn't contrast the feature? → test alternatives (`hot`, `inferno`, `afmhot`, `bone_r`, etc.)
   - Feature buried in background? → try background subtraction (median per row/bin)
   - Wrong scaling? → try log/linear/dB conversions
4. **Build a diagnostic matrix** — create a grid figure testing 6-12 (colormap, clipping, scaling) combinations. Use the actual figure size/dpi if possible.
5. **Select the best** — choose the combination that makes the target feature most visible while keeping the figure publication-ready.
6. **Apply and regenerate** — edit the production plotting code, regenerate the figure, and render to image for final verification.
7. **Clean up** — remove temporary diagnostic scripts.

## Key parameters to try for spectrogram-like data

- **Colormaps**: `hot`, `afmhot`, `inferno`, `magma`, `bone_r`, `gist_heat`
- **Clipping**: `vmin=np.percentile(data, 1-5)`, `vmax=np.percentile(data, 99-99.5)`
- **Scaling**: `np.log1p`, `20*np.log10`, median-subtracted
- **Per-sample clipping**: compute percentiles per subplot, not globally, when samples have very different SNRs

## Verification

Always render the final figure to PNG at ≥150 dpi and visually inspect. Do not rely on PDF alone — PDF viewers vary in rendering quality.
