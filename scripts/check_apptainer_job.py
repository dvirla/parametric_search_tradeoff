r"""
Syntax-check the in-container payload of an apptainer SLURM job.

WHY THIS EXISTS: `bash -n <job file>` does NOT validate the script that actually runs inside the
container. The payload is a double-quoted argument to `/bin/bash -c "..."`, so to the outer parser
it is just a string -- any syntax error inside it is invisible until the job runs, at which point
bash refuses to execute the WHOLE payload (it parses before running) and the job exits clean having
burned its allocation doing nothing. That happened once here: an `echo '... \'tools\' ...'` used
escaped single quotes inside a single-quoted string, which terminates the quote.

This extracts the payload, undoes the outer shell's double-quote unescaping (\\ \" \$ \`), and runs
`bash -n` on the result.

Usage:
    uv run python scripts/check_apptainer_job.py scripts/athena_*.job
Exit status is non-zero if any job's payload fails to parse.
"""

import re
import sys
import subprocess


def check(path: str) -> int:
    src = open(path).read()
    m = re.search(r'/bin/bash -c "(.*)"\s*\n', src, re.S)
    if not m:
        print(f"{path}: no `/bin/bash -c \"...\"` payload found -- nothing to check")
        return 0
    raw = m.group(1)
    out, i = [], 0
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw) and raw[i + 1] in '\\"$`\n':
            out.append(raw[i + 1])
            i += 2
        else:
            out.append(raw[i])
            i += 1
    payload = "".join(out)
    r = subprocess.run(["bash", "-n"], input=payload, text=True, capture_output=True)
    status = "OK" if r.returncode == 0 else "FAIL"
    print(f"{path}: payload {len(payload)} chars -> {status}")
    if r.stderr.strip():
        print(r.stderr.strip()[:2000])
    return r.returncode


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    return max(check(p) for p in sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
