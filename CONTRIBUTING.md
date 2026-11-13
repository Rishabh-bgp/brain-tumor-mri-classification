# Contributing

Thank you for your interest in this project. It is a research and education project maintained by Rishabh Aryan;
issues and pull requests are welcome.

## Reporting problems
- **Bugs** (a script fails, a number does not match, the demo misbehaves): open an issue with the
  [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include the command, the full error, your OS, Python and
  package versions, and the browser for demo issues.
- **Ideas** (new analysis, documentation, demo features): use the
  [feature request template](.github/ISSUE_TEMPLATE/feature_request.md). See [future work](documentation/future_work.md)
  for directions that are already planned.
- **Data issues** (a duplicate the audit missed, a wrong label): name the files (e.g. `Testing/glioma/Te-gl_68.jpg`) and,
  if possible, the Hamming distance and correlation from
  [`candidate_pairs_hamming_le10.csv.gz`](v2/results/review/candidate_pairs_hamming_le10.csv.gz).

## Pull requests
1. Fork the repository and create a branch from `main`.
2. Work in `v2/` (current version). **`v1/` is frozen**: it mirrors the `v1.0` tag and only receives link fixes.
3. **Never edit generated files by hand.** Change the generator and re-run it:
   | Output | Generator (run from `v2/`) |
   |---|---|
   | `v2/README.md` | `scripts/README_template.md` + `python scripts/fill_readme.py` |
   | `README.md` (root) and `documentation/*.md` | `python scripts/build_documentation.py` |
   | `v2/paper/numbers.tex`, `v2/paper/tables/*.tex` | `make_paper_numbers.py`, `make_report.py`, `make_v1_v2.py` |
   | `v2/notebooks/*.ipynb` | `python scripts/build_notebooks.py`, then execute with nbconvert |
   | `v2/results/**`, `docs/stats.json`, `docs/presets.json` | the pipeline scripts ([reproducibility guide](REPRODUCIBILITY.md)) |
4. **Every number must come from a results file.** Do not type metrics into prose; read them in a generator.
5. Do not commit the dataset or large new binaries.
6. Run the checks from the repository root before opening the PR:
   ```bash
   pytest -q tests
   node tests/test_logic.js
   cd v2 && python scripts/check_samples.py
   ```
7. Describe what changed and why, and fill in the [pull request template](.github/pull_request_template.md).

## Style
- Python: PEP 8, lines up to 120 characters, type hints where helpful, a docstring at the top of each script that states
  its inputs and outputs.
- Prose: plain, specific and accurate British English; state limitations; avoid clinical claims.
- Commits: one topic per commit, imperative summary line.

## Conduct
By participating you agree to follow the [code of conduct](CODE_OF_CONDUCT.md).
