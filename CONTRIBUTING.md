Contributing
============

Thanks for contributing! Short guide to run tests and validate changes locally.

Setup
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml_spam_svm\requirements.txt
```

Run tests
```
pytest -q
```

Run Streamlit app
```
streamlit run ml_spam_svm\streamlit_app.py
```

Create PR checklist
- Follow `openspec` workflow: create proposal, tasks, and spec deltas under `openspec/changes/`.
- Ensure tests pass and update `openspec/changes/<change>/tasks.md` (check boxes) before requesting review.
- Do not commit `models/` or large binary artifacts.

OpenSpec validation
- If you have the OpenSpec CLI, run: `openspec validate <change-id> --strict` before creating the PR.
