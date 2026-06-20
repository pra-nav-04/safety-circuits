# Midterm slides

Beamer deck for the MA-INF 4330 midterm (10-min talk + 5-min Q&A).

## Deliverables
- **`main.pdf`** — the compiled presentation (18 slides: 13 talk + 1 divider + 4 backup for Q&A).
- **`main.tex`** — its source.
- **`talking_points.md`** — word-for-word read-aloud script, one section per slide + a Q&A-prep section.

## Build
Requires **XeLaTeX** (the UniBonn theme loads the bundled Exo 2 font via `fontspec`):

```bash
latexmk -xelatex main.tex
```

## What's here
- `beamer*UniBonn.sty` — the [UniBonn beamer theme](https://github.com/fseiffarth/LatexBeamerThemeUniBonnStyle),
  vendored. Local patches: Calibri → bundled Exo 2 (no OS-font dependency), title-page logo sized by
  width, footline uses the short title, and the subitem bullet is a plain dash (the theme's
  `\contour` marker emits PostScript specials XeLaTeX's driver can't read).
- `Exo_2/` — the theme's font family.
- `logos/logo.png` — University of Bonn logo (title page).
- `figures/` — the five result figures, copied from `../paper/figures/`.
