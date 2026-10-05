"""Common stdout-only filters auto-approve as reads; their write forms still prompt."""

from __future__ import annotations

import pytest

from kiro_crew.security.readonly_bash import is_read_only_bash, unsafe_bash_reason

READ_ONLY_LEADS = [
    "jq . package.json",
    "jq -r '.name' package.json",
    "tr a-z A-Z",
    "nl -ba src/app.py",
    "rev notes.txt",
    "comm -12 a.txt b.txt",
    "od -c data.bin",
    "xxd data.bin",
    "xxd -l 64 data.bin",
    "xxd -s -16 data.bin",
    "xxd -c16 -g 1 data.bin",
    "column -t table.txt",
]

READ_ONLY_PIPES = [
    "cat package.json | jq .",
    "cat f | jq -r '.items[]'",
    "cat f | cut -d: -f1",
    "cat f | tr -d '\\r'",
    "cat f | nl",
    "cat f | rev",
    "cat f | od -An -tx1",
    "cat f | xxd",
    "cat f | xxd -l 32",
    "cat f | column -t",
    "ls | comm -23 - other.txt",
]


@pytest.mark.parametrize("cmd", READ_ONLY_LEADS + READ_ONLY_PIPES)
def test_common_filter_is_read_only(cmd: str) -> None:
    assert unsafe_bash_reason(cmd) == ""


@pytest.mark.parametrize(
    "cmd",
    [
        # rg stays off: `--pre` runs a program on every file it searches.
        "rg --pre ./payload pattern",
        "rg pattern",
        "cat f | rg --pre ./payload x",
        # cut is a pipe target only.
        "cut -d: -f1 /etc/passwd",
        # A real-file redirect still trips the shell check.
        "tr a-z A-Z > out.txt",
        "jq . f > out.json",
        "cat f | tr -d x > out",
        # xxd writes its second operand, however that word is spelled.
        "xxd in out",
        "xxd -r dump.hex out.bin",
        "xxd -l 16 in out",
        "xxd in -o",
        "xxd - out",
        "xxd -- in out",
        "xxd -cols 4 in out",
        "cat f | xxd - out",
        "xxd in* ",
        # printf stays off the allowlist: `-v` assigns a shell variable.
        "printf -v PATH /tmp/x && ls",
        "printf '%s' x",
    ],
)
def test_write_or_exec_form_still_prompts(cmd: str) -> None:
    assert not is_read_only_bash(cmd)
