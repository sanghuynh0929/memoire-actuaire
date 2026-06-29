# ISFA mémoire — Quarto template

A Quarto **book** project that reproduces the ISFA actuarial *mémoire* layout
(title page with logos and jury signature boxes, bilingual abstracts, synthesis
notes, numbered chapters, Harvard/`agsm` bibliography, appendices) — the same
structure as the original Dutang–Guibert LaTeX template, but written in
`.qmd` so you can mix prose, code, and results.

## Requirements

- [Quarto](https://quarto.org) ≥ 1.4
- A LaTeX distribution. The easiest is Quarto's bundled TinyTeX:
  ```bash
  quarto install tinytex
  ```

## Render

```bash
quarto render            # builds both PDF and HTML into _output/
quarto render --to pdf   # PDF only -> _output/Huynh_memoire_ISFA.pdf
```

## What goes where

| File | Role |
|------|------|
| `_quarto.yml` | Project config + **all the cover fields** (candidate, title, company, tutor, confidentiality…) and the chapter list. **Edit metadata here, not in the .tex.** |
| `tex/before-body.tex` | The ISFA **title page** + the **English & French abstracts**. Pulls `$candidate$`, `$memoir-title$`, etc. from `_quarto.yml`. |
| `tex/in-header.tex` | Extra LaTeX packages (babel fr/en, tikz, multirow…) and your custom macros (`\red`, `\sigle`, `\code`, `\R`, `\II`…). |
| `index.qmd` | Introduction (unnumbered). Also flips page numbering from roman to arabic for the body. |
| `note-synthese-fr.qmd`, `executive-summary.qmd`, `remerciements.qmd` | Unnumbered front-matter chapters. |
| `chapters/*.qmd` | The four numbered chapters + conclusion. |
| `appendices/*.qmd` | Listed under `book: appendices:` → rendered after `\appendix`. |
| `references.qmd` | Bibliography heading. |
| `references.bib` | Your BibTeX database (replace the stub entries). |
| `logo/` | Put `logo_ida.png`, `ucbl.jpg`, `logoISFAlong.jpg` here. The cover detects them automatically; it still compiles if they're missing. |
| `img/` | Figures (e.g. `effort.png` used in chapter 1). |

## Editing the cover

Open `_quarto.yml` and change the top block:

```yaml
candidate: "Sang HUYNH"
memoir-title: "An Optimal Transport Framework for ..."
defense-date: "15 janvier 2026"
company: "Lya Protect"
company-tutor: "Antoine PAULET"
confidential: false        # true ticks "OUI"
```

## Writing content

Quarto passes raw LaTeX straight through to PDF, so you can keep what you know:

- Headings: `#` = chapter, `##` = section, `###` = subsection.
- Maths: `$...$` inline, `$$...$$` display. Add `{#eq-name}` to a display block
  and reference it with `@eq-name`.
- Citations: `@key` → *Author (year)*, `[@key]` → *(Author, year)*,
  `[@a; @b]` for several. Defined in `references.bib`.
- Tables / figures: write normal Markdown, **or** drop a raw LaTeX
  `\begin{table}…\end{table}` block inside a ` ```{=latex} ` fence (see
  `chapters/02-optimal-transport.qmd`). Your macros like `\red{}` work there.

## Notes & options

- **Bibliography style.** This template uses `cite-method: natbib` with
  `biblio-style: agsm` (Harvard), matching the original. With natbib the
  reference list is emitted by LaTeX at the bibliography location. If you'd
  rather have one engine that also works for HTML, switch to Quarto's default
  citation processor by removing `cite-method`/`biblio-style` and adding a
  `csl:` Harvard style file.
- **Ordering.** The two abstracts sit *before* the table of contents (they live
  in `before-body.tex`). The synthesis notes and acknowledgements are authored
  as `.qmd` and appear *after* the TOC. To force them before the TOC instead,
  move that text into `before-body.tex`.
- **Document class.** Set to `report` (twoside, openright, 11pt, a4) to mirror
  the original. Swap to `scrbook` in `_quarto.yml` for KOMA-Script styling.
- Run with `keep-tex: true` (already on) and inspect the generated `.tex` in
  `_output/` whenever you need to fine-tune the LaTeX.
