# Contributing

Open an issue describing the behavior, seed cohort, command contract, dependency
versions, and whether the result is development, benchmark replay, or fresh held-out.
Keep simulator truth out of policy inputs and do not relax physical success checks.
Include tests for changes to data alignment, split guards, and execution transforms.
Run `pytest -q` and `python tools/audit_release.py` before submitting a pull request.
Do not commit recordings, checkpoints, credentials, personal paths, or experiment logs.
Contributions are submitted under Apache-2.0; retain third-party notices.

