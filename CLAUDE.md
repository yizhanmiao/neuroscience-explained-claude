# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A collection of [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebooks explaining methods commonly used in neuroscience, authored by Claude and curated by a human reviewer.

## Folder Structure

- `generated/` — notebooks produced directly by Claude; may contain errors or imprecisions
- `reviewed/` — notebooks that have been human-proofread and are considered reliable

New notebooks always go into `generated/`. A notebook is moved to `reviewed/` only after a human has verified its content.

## Working with Pluto Notebooks

Install the Pluto:
```bash
julia -e 'import Pkg; Pkg.add("Pluto");'
```

Launch the Pluto server (Julia 1.12+):

```bash
julia -e 'import Pluto; Pluto.run()'
```

Open a specific notebook directly:

```bash
julia -e 'import Pluto; Pluto.run(notebook="generated/my_notebook.jl")'
```

Pluto notebooks are plain `.jl` files with cell metadata encoded in comments — edit them in the Pluto browser UI, not a text editor, to avoid corrupting cell order.

## Notebook Conventions

- Each notebook should be self-contained: install its own dependencies via `Pkg` inside a `@bind`-free setup cell.
- Target Julia 1.12+ (the installed version).
- Notebook filenames should be lowercase, hyphen-separated, and descriptive of the method covered (e.g., `spike-sorting-kilosort.jl`, `calcium-imaging-deconvolution.jl`).
