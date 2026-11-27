# Working together on VigiLense

- **One branch per feature**, named `feature/<what>` (for example `feature/tracking`). Do not commit straight to `main`.
- **Open a pull request into `main`** and ask at least one teammate to look at it. Changes to `backend/app.py` need a
  review from the backend owner, because every feature touches that file.
- **Start each session with `git pull`** (or `git fetch` and `git merge origin/main` on your branch) so you do not build
  on old code.
- **Tell the owner before changing someone else's area** (detection: Yasir, backend and tracking: Khurram, dashboard
  and ops: Hassnain). Say what you changed in the pull request.
- **Keep commits small and named by what they do**, for example "Add --smoke-conf option".
- **Do not commit** videos, datasets, `.venv`, `evidence/`, passwords, API keys or camera URLs with credentials
  (`.env` is ignored). Trained models go in `weights/` only when they are the final ones the app uses (about 5 MB each).
- **Run the checks before a pull request:** `pytest tests`, and for model changes the false-alarm tests listed in
  [RUN_COMMANDS.md](RUN_COMMANDS.md) section 8. Record new numbers in [MODEL_CARD.md](MODEL_CARD.md).
- **Scripts** live in `tools/` (`data`, `eval`, `inspect`, `tracking`) and are run from the project root, for example
  `python tools/eval/compare_false_alarms.py`.
