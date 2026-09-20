"""Fail on accidentally staged data or common secret/private-path signatures.

This is a narrow release hygiene check, not a complete security audit.
Never print matched values. Report only paths, line numbers and rule names.
"""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private-home-path': re.compile(r'/(?:Users|home)/[A-Za-z][A-Za-z0-9_.-]+/'),
    'training-host': re.compile(r'connect\.[a-z0-9]+\.seetacloud\.com'),
    'private-key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'hub-token': re.compile(r'\bhf_[A-Za-z0-9]{25,}\b'),
    'github-token': re.compile(r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}\b'),
    'credential-assignment': re.compile(r'(?i)(?:password|passwd|SSHPASS|api_key|access_token)\s*[:=]\s*[\x22\x27][^\x22\x27\n]{6,}[\x22\x27]'),
}
FORBIDDEN = {'.pt', '.pth', '.ckpt', '.npz', '.h5', '.hdf5', '.mp4', '.log'}


def main():
    proc = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT, capture_output=True)
    paths = ([ROOT / n for n in proc.stdout.decode().split('\0') if n]
             if proc.returncode == 0 else [p for p in ROOT.rglob('*') if p.is_file()])
    findings = []
    checked = 0
    for path in paths:
        rel = path.relative_to(ROOT)
        if any(v in rel.parts for v in ('.git', '__pycache__', '.pytest_cache')):
            continue
        if path.is_symlink():
            findings.append({'path': str(rel), 'rule': 'symlink'})
            continue
        if path.suffix.lower() in FORBIDDEN or path.name == '.env':
            findings.append({'path': str(rel), 'rule': 'private-data-or-runtime-file'})
        if path.suffix.lower() not in {'.py', '.json', '.md', '.toml', '.yml', '.yaml', '.cff', '.xml'}:
            continue
        checked += 1
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            for name, pattern in RULES.items():
                if pattern.search(line):
                    findings.append({'path': str(rel), 'line': line_number, 'rule': name})
    print(json.dumps({'checked_text_files': checked, 'findings': findings}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == '__main__':
    main()
