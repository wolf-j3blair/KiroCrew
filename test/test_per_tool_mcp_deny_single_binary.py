"""Per-tool MCP deny on the two single-binary harnesses, opencode and goose.

Each harness carries a switched-off tool in its own vocabulary, and a narrowed server
stays mounted:

* opencode: a ``deny`` rule in the ``permission`` config Crew seeds on
  ``OPENCODE_CONFIG_CONTENT``, keyed on the harness's own tool id. The routing
  read-back evaluates the resolved rules the way the harness does (last match wins),
  and a rule a lower config source outranks withholds its server.
* goose: a ``(server, tool)`` pair the client refuses at the permission request, by
  the identity goose puts on the ``tool_call`` frame. A tool goose's own
  ``permission.yaml`` pre-approves never asks, so its server is withheld.

The unit half drives the projections and the client's read-back. The live half drives
the real harnesses against a local fake model, so no credential and no remote model
are involved: a denied tool must be blocked while its sibling on the same server
still runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from real_adapter_gate import (
    MEASURED_GOOSE_VERSION,
    MEASURED_OPENCODE_VERSION,
    require_real_adapter,
)

from kiro_crew import agent as agent_mod
from kiro_crew import platform_compat
from kiro_crew.acp import client as client_mod
from kiro_crew.acp import session_mcp
from kiro_crew.acp.client import AcpClient
from kiro_crew.acp.types import AcpEvent, JsonRpcMessage
from kiro_crew.acp_backends import ACP_BACKEND_GOOSE, ACP_BACKEND_OPENCODE
from kiro_crew.providers.mirrors import goose as goose_mod
from kiro_crew.providers.mirrors import opencode as opencode_mod
from kiro_crew.providers.mirrors.goose import (
    goose_always_allowed,
    goose_permission_files,
    goose_projection,
    goose_unhonoured_servers,
)

_CORE = {"command": "/opt/kirocrew", "args": ["mcp-core"]}
_CRON = {"command": "/opt/kirocrew", "args": ["mcp-cron"]}


@pytest.fixture
def agents_dir(tmp_path, monkeypatch):
    """Point the agent-spec resolver at a temp agents directory (same seam as the
    opencode and codex session-MCP tests), and keep the operator's goose config out."""
    d = tmp_path / "agents"
    d.mkdir()
    monkeypatch.setattr(agent_mod, "KIRO_AGENTS_DIR", d)
    monkeypatch.setattr(agent_mod, "_KIRO_MCP_JSON", tmp_path / "settings-mcp.json")
    monkeypatch.setattr(session_mcp, "ensure_agent_materialized", lambda _a: True)
    managed = {"kirocrew-core": dict(_CORE), "kirocrew-cron": dict(_CRON)}
    monkeypatch.setattr(
        session_mcp,
        "managed_mcp_spec_entry",
        lambda name: dict(managed[name]) if name in managed else None,
    )
    monkeypatch.setattr(session_mcp, "_mcp_registry_mode", lambda: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("GOOSE_PATH_ROOT", raising=False)
    # The explicit envs below spell POSIX locations; the Windows case is its own test.
    monkeypatch.setattr(platform_compat, "IS_WINDOWS", False)
    return d


def _write_spec(agents_dir: Path, servers: dict, tools: list) -> None:
    spec = {"name": "kirocrew", "mcpServers": servers, "tools": tools}
    (agents_dir / "kirocrew.json").write_text(json.dumps(spec), encoding="utf-8")


def _names(projection) -> list[str]:
    return [str(e.get("name")) for e in projection.params["mcpServers"]]


def _write_permission_yaml(path: Path, always_allow: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["user:", "  always_allow:"] + [f"    - {e}" for e in always_allow]
    lines += ["  ask_before: []", "  never_allow: []", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── goose: per call, and the permission.yaml hazard ─────────────────────────


class TestGooseKeepsANarrowedServerMounted:
    def test_both_a_third_party_server_and_the_control_plane_stay_mounted(self, agents_dir):
        _write_spec(
            agents_dir,
            {
                "narrowed": {"command": "/bin/x", "disabledTools": ["danger"]},
                "kirocrew-core": {"command": "/x", "disabledTools": ["spawn_run"]},
            },
            ["@narrowed", "@kirocrew-core"],
        )
        projection = goose_projection("kirocrew")
        assert set(_names(projection)) == {"narrowed", "kirocrew-core"}
        assert projection.denied_tools == {("narrowed", "danger"), ("kirocrew-core", "spawn_run")}
        assert projection.unhonoured_servers == frozenset()

    def test_a_server_goose_would_rename_is_withheld_whole(self, agents_dir):
        """goose 1.50.1 reports ``Probe_X.y`` as ``probe_x_y``, so a refusal keyed on
        the spec's spelling would never match it."""
        _write_spec(
            agents_dir,
            {"Probe_X.y": {"command": "/bin/x", "disabledTools": ["danger"]}},
            ["@Probe_X.y"],
        )
        projection = goose_projection("kirocrew")
        assert "Probe_X.y" not in _names(projection)
        assert projection.unhonoured_servers == frozenset({"Probe_X.y"})


class TestGoosePermissionYaml:
    def test_the_files_follow_goose_path_root_and_xdg(self, tmp_path):
        env = {"GOOSE_PATH_ROOT": str(tmp_path / "root"), "XDG_CONFIG_HOME": str(tmp_path / "x")}
        assert goose_permission_files(env, windows=False) == (
            tmp_path / "root" / "config" / "permission.yaml",
            tmp_path / "x" / "goose" / "permission.yaml",
        )
        assert goose_permission_files({"HOME": str(tmp_path)}, windows=False) == (
            tmp_path / ".config" / "goose" / "permission.yaml",
        )

    def test_always_allow_is_read_from_every_section(self, tmp_path):
        path = tmp_path / "goose" / "permission.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            "user:\n  always_allow: [Probe__Secret]\nsmart_approve:\n  always_allow: [a__b]\n",
            encoding="utf-8",
        )
        assert goose_always_allowed({"XDG_CONFIG_HOME": str(tmp_path)}, windows=False) == {
            "probe__secret",
            "a__b",
        }

    def test_windows_reads_the_appdata_location(self, tmp_path):
        """goose's Windows config dir is %APPDATA%\\Block\\goose\\config."""
        path = tmp_path / "Block" / "goose" / "config" / "permission.yaml"
        _write_permission_yaml(path, ["probe__secret"])
        env = {"APPDATA": str(tmp_path)}
        assert goose_permission_files(env, windows=True) == (path,)
        assert goose_always_allowed(env, windows=True) == {"probe__secret"}

    @pytest.mark.parametrize(
        ("env", "windows"),
        [
            ({}, False),
            ({}, True),
            # Windows never reads the XDG path, so it cannot stand in for APPDATA.
            ({"XDG_CONFIG_HOME": "/x", "HOME": "/h"}, True),
        ],
    )
    def test_an_underivable_location_is_unknown_not_empty(self, env, windows):
        """No candidate for the platform's own location is "allows everything"."""
        assert goose_permission_files(env, windows=windows) == ()
        assert goose_always_allowed(env, windows=windows) is None

    def test_an_underivable_location_withholds_the_narrowed_server(self, agents_dir):
        _write_spec(
            agents_dir,
            {
                "narrowed": {"command": "/bin/x", "disabledTools": ["danger"]},
                "plain": {"command": "/bin/y"},
            },
            ["@narrowed", "@plain"],
        )
        projection = goose_projection("kirocrew", harness_env={})
        assert _names(projection) == ["plain"]
        assert projection.unhonoured_servers == frozenset({"narrowed"})

    def test_a_windows_session_without_appdata_withholds(self, agents_dir, monkeypatch):
        _write_spec(
            agents_dir,
            {"narrowed": {"command": "/bin/x", "disabledTools": ["danger"]}},
            ["@narrowed"],
        )
        monkeypatch.setattr(platform_compat, "IS_WINDOWS", True)
        projection = goose_projection("kirocrew", harness_env={"XDG_CONFIG_HOME": "/x"})
        assert "narrowed" not in _names(projection)

    def test_a_missing_file_allows_nothing(self, tmp_path):
        assert (
            goose_always_allowed({"XDG_CONFIG_HOME": str(tmp_path)}, windows=False) == frozenset()
        )

    @pytest.mark.parametrize(
        "text", ["user: [", "- just\n- a list\n", "user:\n  always_allow: x\n"]
    )
    def test_an_unparseable_file_counts_as_allowing_everything(self, tmp_path, text):
        path = tmp_path / "goose" / "permission.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="utf-8")
        assert goose_always_allowed({"XDG_CONFIG_HOME": str(tmp_path)}, windows=False) is None
        assert goose_unhonoured_servers([("s", "t")], None) == frozenset({"s"})

    def test_a_pre_approved_denied_tool_withholds_its_server(self, agents_dir, tmp_path):
        _write_spec(
            agents_dir,
            {
                "narrowed": {"command": "/bin/x", "disabledTools": ["danger"]},
                "other": {"command": "/bin/y", "disabledTools": ["risky"]},
            },
            ["@narrowed", "@other"],
        )
        xdg = tmp_path / "xdg2"
        _write_permission_yaml(
            xdg / "goose" / "permission.yaml", ["narrowed__danger", "other__safe"]
        )
        projection = goose_projection("kirocrew", harness_env={"XDG_CONFIG_HOME": str(xdg)})
        assert _names(projection) == ["other"]
        assert projection.unhonoured_servers == frozenset({"narrowed"})
        assert projection.restricted_servers == frozenset({"narrowed"})

    def test_the_mirror_face_reads_the_harness_env(self, agents_dir, tmp_path):
        _write_spec(
            agents_dir,
            {"narrowed": {"command": "/bin/x", "disabledTools": ["danger"]}},
            ["@narrowed"],
        )
        xdg = tmp_path / "xdg3"
        _write_permission_yaml(xdg / "goose" / "permission.yaml", ["narrowed__danger"])
        mirror = goose_mod.GooseMirror()
        assert "narrowed" in _names(mirror.session_projection("kirocrew"))
        withheld = mirror.session_projection("kirocrew", harness_env={"XDG_CONFIG_HOME": str(xdg)})
        assert "narrowed" not in _names(withheld)


# ── opencode: the read-back's evaluation of the resolved rules ──────────────


class TestOpencodeRuleEvaluation:
    def test_the_wildcard_is_the_harnesss(self):
        match = client_mod._opencode_wildcard_match
        assert match("probe_secret", "*")
        assert match("probe_secret", "probe_*")
        assert match("probe_secret", "probe_secre?")
        assert not match("probe_secret", "probe")
        assert not match("probe.secret", "probe_secret")
        assert match("git", "git *") and match("git push", "git *")

    def test_a_trailing_deny_is_in_force(self):
        resolved = {"permission": {"bash": "allow", "*": "ask", "probe_secret": "deny"}}
        assert (
            client_mod._opencode_unenforced_denies(resolved, "permission", ["probe_secret"])
            == frozenset()
        )

    def test_a_deny_a_lower_source_put_before_the_star_is_not_in_force(self):
        """Measured on opencode 1.18.30: a global ``"probe_secret": "allow"`` keeps
        its place, the seed overrides only its value, and the seed's ``"*"`` lands
        after it, so the deny reads as ask."""
        resolved = {"permission": {"probe_secret": "deny", "*": "ask"}}
        assert client_mod._opencode_unenforced_denies(
            resolved, "permission", ["probe_secret"]
        ) == frozenset({"probe_secret"})

    def test_an_agent_override_is_appended_and_can_outrank(self):
        resolved = {
            "permission": {"*": "ask", "probe_secret": "deny", "a_b": "deny"},
            "agent": {"build": {"permission": {"*": "ask"}}, "plan": {"model": "x"}},
        }
        assert client_mod._opencode_unenforced_denies(
            resolved, "permission", ["probe_secret", "a_b"]
        ) == frozenset({"probe_secret", "a_b"})

    def test_a_later_pattern_rule_on_the_same_tool_unhides_it(self):
        """The harness hides a tool only when the last rule naming it denies EVERY
        pattern."""
        resolved = {"permission": {"*": "ask", "probe_secret": "deny", "probe_*": {"x": "allow"}}}
        assert client_mod._opencode_unenforced_denies(
            resolved, "permission", ["probe_secret"]
        ) == frozenset({"probe_secret"})

    def test_the_seeded_rules_are_read_from_the_seed(self):
        seed = json.dumps({"permission": {"*": "ask", "a_b": "deny", "c_d": "deny"}})
        assert client_mod._opencode_seeded_deny_rules(seed, "permission") == ("a_b", "c_d")
        assert client_mod._opencode_seeded_deny_rules('{"permission": "ask"}', "permission") == ()


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout, "")


class TestOpencodeReadBack:
    @pytest.fixture
    def client(self, tmp_path):
        c = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
        c._session_harness_deny_rules = ("probe_secret",)
        return c

    def test_the_seed_puts_each_deny_after_the_star(self, client, monkeypatch):
        monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
        seed = json.loads(client._opencode_routing_config())
        assert list(seed["permission"].items()) == [("*", "ask"), ("probe_secret", "deny")]

    def test_no_rules_keeps_the_plain_ask(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
        c = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
        assert json.loads(c._opencode_routing_config())["permission"] == "ask"

    def _read_back(self, client, monkeypatch, resolved: dict) -> tuple[str, str]:
        monkeypatch.setattr(
            client_mod.subprocess_mod, "run", lambda *a, **k: _completed(json.dumps(resolved))
        )
        seed = json.dumps({"permission": {"*": "ask", "probe_secret": "deny"}})
        return client._verify_opencode_routing(["opencode", "debug", "config"], seed)

    def test_the_seeded_shape_passes_with_every_deny_in_force(self, client, monkeypatch):
        issue = self._read_back(
            client, monkeypatch, {"permission": {"*": "ask", "probe_secret": "deny"}}
        )
        assert issue == ("", "")
        assert client._opencode_denies_unenforced == frozenset()

    def test_an_outranked_deny_is_recorded_not_refused(self, client, monkeypatch):
        issue = self._read_back(
            client, monkeypatch, {"permission": {"probe_secret": "deny", "*": "ask"}}
        )
        assert issue == ("", "")
        assert client._opencode_denies_unenforced == frozenset({"probe_secret"})

    def test_an_operators_own_outranked_deny_is_still_refused(self, client, monkeypatch):
        issue, _remedy = self._read_back(
            client,
            monkeypatch,
            {"permission": {"bash": {"pwd": "deny"}, "*": "ask", "probe_secret": "deny"}},
        )
        assert issue and "deny" in issue


class TestOpencodeDeniesInForce:
    """Which deny rules are in force comes from the seed, never from intent."""

    @pytest.fixture
    def client(self, tmp_path):
        c = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
        c._session_harness_deny_rules = ("probe_secret",)
        return c

    def test_a_seed_that_carries_the_rule_keeps_it(self, client):
        client._opencode_config_content = json.dumps(
            {"permission": {"*": "ask", "probe_secret": "deny"}}
        )
        assert client._settle_opencode_denies() == frozenset()
        assert client._opencode_denies_in_force == {"probe_secret"}

    def test_a_seed_that_left_the_rule_out_loses_it(self, client):
        """The routing value is not a rule map, so no deny was written."""
        client._opencode_config_content = json.dumps({"permission": "ask"})
        assert client._settle_opencode_denies() == {"probe_secret"}
        assert client._opencode_denies_in_force == frozenset()

    def test_an_outranked_rule_is_lost_too(self, client):
        client._opencode_config_content = json.dumps(
            {"permission": {"*": "ask", "probe_secret": "deny"}}
        )
        client._opencode_denies_unenforced = frozenset({"probe_secret"})
        assert client._settle_opencode_denies() == {"probe_secret"}

    def test_a_lost_rule_withholds_its_server(self, agents_dir, tmp_path):
        _write_spec(
            agents_dir,
            {"narrowed": {"command": "/bin/x", "disabledTools": ["danger"]}},
            ["@narrowed"],
        )
        c = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
        assert "narrowed" in [e["name"] for e in c._resolve_session_mcp_servers()]
        c._opencode_config_content = json.dumps({"permission": "ask"})
        assert c._settle_opencode_denies() == {"narrowed_danger"}
        assert "narrowed" not in [e["name"] for e in c._resolve_session_mcp_servers()]

    def test_the_spawn_arm_settles_after_the_read_back(self):
        """The arm calls the settle step, and nothing else sets the in-force set."""
        import inspect

        source = inspect.getsource(AcpClient)
        assert "lost = self._settle_opencode_denies()" in source
        assert source.count("self._opencode_denies_in_force = ") == 2


class TestTheClientWiring:
    def test_a_rule_not_in_force_withholds_its_server_on_a_reprojection(self, agents_dir, tmp_path):
        _write_spec(
            agents_dir,
            {"narrowed": {"command": "/bin/x", "disabledTools": ["danger"]}},
            ["@narrowed"],
        )
        c = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
        mounted = c._resolve_session_mcp_servers()
        assert "narrowed" in [e["name"] for e in mounted]
        assert c._session_harness_deny_rules == ("narrowed_danger",)
        c._opencode_denies_in_force = frozenset()
        withheld = c._resolve_session_mcp_servers()
        assert "narrowed" not in [e["name"] for e in withheld]
        assert c._session_mcp_unhonoured == frozenset({"narrowed"})

    def test_goose_hands_the_client_its_pairs(self, agents_dir, tmp_path):
        _write_spec(
            agents_dir,
            {"narrowed": {"command": "/bin/x", "disabledTools": ["danger"]}},
            ["@narrowed"],
        )
        c = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_GOOSE)
        assert "narrowed" in [e["name"] for e in c._resolve_session_mcp_servers()]
        assert c._spec_denied_tools == {("narrowed", "danger")}

    def test_goose_reads_permission_yaml_from_the_session_env(self, agents_dir, tmp_path):
        _write_spec(
            agents_dir,
            {"narrowed": {"command": "/bin/x", "disabledTools": ["danger"]}},
            ["@narrowed"],
        )
        xdg = tmp_path / "session-xdg"
        _write_permission_yaml(xdg / "goose" / "permission.yaml", ["narrowed__danger"])
        c = AcpClient(
            work_dir=tmp_path,
            acp_backend=ACP_BACKEND_GOOSE,
            extra_env={"XDG_CONFIG_HOME": str(xdg)},
        )
        assert "narrowed" not in [e["name"] for e in c._resolve_session_mcp_servers()]

    def test_a_member_mount_never_re_adds_an_unhonoured_server(self, tmp_path):
        c = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_GOOSE)
        c._session_mcp_unhonoured = frozenset({"kirocrew-dashboard"})
        assert c._member_mount_withheld(
            "kirocrew-dashboard", "dispatch", frozenset({"kirocrew-dashboard"}), frozenset()
        )

    def test_the_tripwire_reads_the_meta_identity_a_goose_call_carried(self, tmp_path):
        c = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_GOOSE)
        c._spec_denied_tools = frozenset({("probe", "secret")})
        c._tool_call_params["call_1"] = {"": "{}"}
        c._tool_call_mcp_server["call_1"] = "probe"
        c._tool_call_tool_name["call_1"] = "secret"
        audited: list = []
        c._audit_spec_restriction = lambda **kw: audited.append(kw)  # type: ignore[method-assign]
        c._tripwire_spec_disabled_tool(
            AcpEvent(kind="tool_result", tool_call_id="call_1", tool_final=True)
        )
        assert audited and audited[0]["outcome"] == "ran_despite_spec_disable"


# ── live: the real harnesses, a fake model, one probe server ────────────────

_PROBE_MCP = r"""
import json, os, sys
LOG = os.environ["CALL_LOG"]
TOOLS = [{"name": n, "description": "Probe tool " + n,
          "inputSchema": {"type": "object", "properties": {}}} for n in ("secret", "public")]
for t in TOOLS:
    if t["name"] in (os.environ.get("READ_ONLY") or "").split(","):
        t["annotations"] = {"readOnlyHint": True}
def send(m):
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    try:
        req = json.loads(line)
    except ValueError:
        continue
    m, rid = req.get("method"), req.get("id")
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": req["params"].get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}}, "serverInfo": {"name": "probe", "version": "0"}}})
    elif m == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
    elif m == "tools/call":
        with open(LOG, "a") as f:
            f.write(req["params"]["name"] + "\n")
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text", "text": "word-from-" + req["params"]["name"]}]}})
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
"""


class _FakeModel:
    """An OpenAI-compatible chat endpoint that calls *calls* in order, when offered.

    Each request records the tool names it offered. The model asks for the next tool
    in *calls* that the request offers, one per turn, then answers ``done``.
    """

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.offered: list[list[str]] = []
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                self._json({"object": "list", "data": [{"id": "m", "object": "model"}]})

            def _json(self, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:  # noqa: N802
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                tools = [t.get("function", {}).get("name") for t in req.get("tools") or []]
                if tools:
                    model.offered.append(tools)
                done = sum(1 for m in req.get("messages", []) if m.get("role") == "tool")
                pending = [t for t in model.calls if t in tools][done:]
                if pending:
                    msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{done}",
                                "type": "function",
                                "function": {"name": pending[0], "arguments": "{}"},
                            }
                        ],
                    }
                    finish = "tool_calls"
                else:
                    msg, finish = {"role": "assistant", "content": "done"}, "stop"
                if req.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    delta = {"role": "assistant"}
                    if msg.get("content"):
                        delta["content"] = msg["content"]
                    if msg.get("tool_calls"):
                        delta["tool_calls"] = [dict(tc, index=0) for tc in msg["tool_calls"]]
                    base = {
                        "id": "x",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "m",
                    }
                    chunks = [
                        dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
                        dict(
                            base,
                            choices=[{"index": 0, "delta": {}, "finish_reason": finish}],
                            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                        ),
                    ]
                    for c in chunks:
                        self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                self._json(
                    {
                        "id": "x",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "m",
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def _isolated_home(root: Path) -> dict[str, str]:
    home = root / "home"
    (home / ".config").mkdir(parents=True)
    # The harnesses unpack native modules into the temp dir; keep them in tmp_path.
    tmp = root / "tmp"
    tmp.mkdir()
    return {
        "TMPDIR": str(tmp),
        "TEMP": str(tmp),
        "TMP": str(tmp),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
    }


def _drive_acp(
    argv: list[str],
    env: dict[str, str],
    cwd: Path,
    servers: list[dict],
    decide=None,
    *,
    on_update=None,
    answer=None,
) -> list[dict]:
    """One ``session/new`` with *servers* and one prompt; returns every frame.

    A permission request is answered by *answer(msg)*, which returns the whole
    ``result`` to send, when given; otherwise *decide(update_by_call, tool_call)*
    picks the option kind, ``allow_once`` or ``reject_once``. *on_update(msg)* sees
    every ``session/update`` frame before anything else does.
    """
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        start_new_session=platform_compat.IS_POSIX,
        creationflags=platform_compat.CREATE_NEW_PROCESS_GROUP,
    )
    assert proc.stdin is not None and proc.stdout is not None
    frames: list[dict] = []
    calls: dict[str, dict] = {}
    waiting: dict[int, list] = {}
    lock = threading.Lock()

    def write(msg: dict) -> None:
        with lock:
            proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]

    def reader() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            frames.append(msg)
            if msg.get("method") == "session/update":
                if on_update is not None:
                    on_update(msg)
                update = msg["params"]["update"]
                if update.get("sessionUpdate") == "tool_call":
                    calls[update.get("toolCallId")] = update
            elif msg.get("method") == "session/request_permission" and answer is not None:
                write({"jsonrpc": "2.0", "id": msg["id"], "result": answer(msg)})
            elif msg.get("method") == "session/request_permission":
                tool_call = msg["params"].get("toolCall") or {}
                want = decide(calls.get(tool_call.get("toolCallId")), tool_call)
                options = msg["params"]["options"]
                pick = next((o for o in options if o["kind"] == want), None)
                assert pick is not None, options
                write(
                    {
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "result": {
                            "outcome": {"outcome": "selected", "optionId": pick["optionId"]}
                        },
                    }
                )
            elif "id" in msg and "method" in msg:
                write(
                    {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "no"}}
                )
            elif msg.get("id") in waiting:
                waiting[msg["id"]][1] = msg
                waiting[msg["id"]][0].set()

    def call(rid: int, method: str, params: dict, timeout: float) -> dict:
        waiting[rid] = [threading.Event(), None]
        write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        assert waiting[rid][0].wait(timeout), f"{method} timed out"
        return waiting[rid][1]

    threading.Thread(target=reader, daemon=True).start()
    try:
        call(1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}}, 120)
        new = call(2, "session/new", {"cwd": str(cwd), "mcpServers": servers}, 120)
        assert "result" in new, new
        prompt = [{"type": "text", "text": "Call the probe tools, then say done."}]
        done = call(
            3, "session/prompt", {"sessionId": new["result"]["sessionId"], "prompt": prompt}, 300
        )
        assert "result" in done, done
    finally:
        # The whole tree: the harness spawns the MCP child, which a plain kill of the
        # harness would leave running. Cross-platform, and the wait always runs.
        try:
            platform_compat.kill_process_tree(proc.pid, platform_compat.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    return frames


def _probe_spec(agents_dir: Path, probe: Path, log: Path) -> None:
    _write_spec(
        agents_dir,
        {
            "probe": {
                "command": sys.executable,
                "args": [str(probe)],
                "env": {"CALL_LOG": str(log)},
                "disabledTools": ["secret"],
            }
        },
        ["@probe"],
    )


def _self_served_bin(backend: str) -> str | None:
    resolved, _search = client_mod._resolve_self_served_bin(backend)
    return resolved or None


@pytest.mark.real_adapter
def test_real_opencode_hides_a_denied_tool_and_keeps_its_sibling(agents_dir, tmp_path, monkeypatch):
    """The opencode projection, end to end through Crew's own seed and read-back.

    The spec switches off ``secret`` on ``probe``. Crew's projection mounts ``probe``
    and seeds ``probe_secret: deny`` after ``"*": "ask"``; the real read-back finds it
    in force; and in the live session the model is never offered ``probe_secret``,
    while ``probe_public`` is offered, asks, and runs.
    """
    binary = _self_served_bin(ACP_BACKEND_OPENCODE)
    require_real_adapter(
        binary, what="opencode", install=f"npm i -g opencode-ai@{MEASURED_OPENCODE_VERSION}"
    )
    assert binary is not None
    probe = tmp_path / "probe_mcp.py"
    probe.write_text(_PROBE_MCP, encoding="utf-8")
    log = tmp_path / "calls.log"
    _probe_spec(agents_dir, probe, log)
    model = _FakeModel(["probe_secret", "probe_public"])
    try:
        provider = {
            "model": "fake/m",
            "provider": {
                "fake": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {"baseURL": model.url + "/v1", "apiKey": "x"},
                    "models": {"m": {"tool_call": True}},
                }
            },
        }
        monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(provider))
        isolated = _isolated_home(tmp_path)
        work = tmp_path / "work"
        work.mkdir()
        client = AcpClient(work_dir=work, acp_backend=ACP_BACKEND_OPENCODE, extra_env=isolated)
        servers = client._resolve_session_mcp_servers()
        assert [e["name"] for e in servers] == ["probe"]
        seed = client._opencode_routing_config()
        assert client._verify_opencode_routing([binary, "debug", "config"], seed) == ("", "")
        assert client._opencode_denies_unenforced == frozenset()

        env = {**os.environ, **isolated, "OPENCODE_CONFIG_CONTENT": seed}
        asked: list[str] = []

        def decide(_update, tool_call):
            asked.append(str(tool_call.get("title")))
            return "allow_once"

        _drive_acp([binary, "acp"], env, work, servers, decide)
    finally:
        model.close()
    offered = {name for request in model.offered for name in request}
    assert "probe_public" in offered, model.offered
    assert "probe_secret" not in offered, model.offered
    assert log.read_text(encoding="utf-8").split() == ["public"]
    assert asked == ["probe_public"], "the sibling tool must still ask per call"


@pytest.mark.real_adapter
def test_real_opencode_an_outranking_global_key_is_detected(agents_dir, tmp_path):
    """The last-match-wins hazard, against the real config resolution.

    A global ``"probe_secret": "allow"`` keeps its earlier place when the seed is
    merged, so the seed's ``"*": "ask"`` lands after it and outranks the deny. The
    read-back must find that rule not in force rather than report the session as
    narrowed.
    """
    binary = _self_served_bin(ACP_BACKEND_OPENCODE)
    require_real_adapter(
        binary, what="opencode", install=f"npm i -g opencode-ai@{MEASURED_OPENCODE_VERSION}"
    )
    assert binary is not None
    isolated = _isolated_home(tmp_path)
    (Path(isolated["XDG_CONFIG_HOME"]) / "opencode").mkdir()
    (Path(isolated["XDG_CONFIG_HOME"]) / "opencode" / "opencode.json").write_text(
        json.dumps({"permission": {"probe_secret": "allow"}}), encoding="utf-8"
    )
    work = tmp_path / "work"
    work.mkdir()
    client = AcpClient(work_dir=work, acp_backend=ACP_BACKEND_OPENCODE, extra_env=isolated)
    client._session_harness_deny_rules = ("probe_secret",)
    seed = client._opencode_routing_config()
    assert client._verify_opencode_routing([binary, "debug", "config"], seed) == ("", "")
    assert client._opencode_denies_unenforced == frozenset({"probe_secret"})


def _goose_env(tmp_path: Path, model: _FakeModel) -> dict[str, str]:
    return {
        **os.environ,
        **_isolated_home(tmp_path),
        "GOOSE_DISABLE_KEYRING": "1",
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": "m",
        "OPENAI_HOST": model.url,
        "OPENAI_API_KEY": "fake",
        "GOOSE_MODE": "approve",
    }


def _goose_identity(update: dict | None) -> tuple[str, str] | None:
    """The pair Crew reads off a goose ``tool_call`` frame (``acp._dispatch``)."""
    from kiro_crew.acp._dispatch import _kiro_mcp_server_name, _kiro_tool_name

    if not update:
        return None
    return _kiro_mcp_server_name(update), _kiro_tool_name(update)


def _require_goose() -> str:
    """The goose binary a real session would spawn, through the real-adapter gate."""
    binary = _self_served_bin(ACP_BACKEND_GOOSE)
    require_real_adapter(
        binary,
        what="goose",
        install=f"the goose {MEASURED_GOOSE_VERSION} release (test/real_adapters/goose.json)",
    )
    assert binary is not None
    return binary


@pytest.mark.real_adapter
def test_real_goose_refuses_a_denied_tool_and_runs_its_sibling(agents_dir, tmp_path):
    """The goose projection, end to end, through the client's own refusal.

    Three goose behaviours this projection stands on, all asserted against the live
    harness:

    * under ``GOOSE_MODE=approve`` goose ASKS for every MCP call, a tool annotated
      ``readOnlyHint`` included -- the switched-off ``secret`` carries that hint here;
    * the ``tool_call`` frame names the pair as ``_meta.goose.toolCall``, which the
      client's own frame reader turns into a trusted identity;
    * goose honours the ``reject_once`` the client answers with: ``secret`` never runs.

    Every frame goes through ``AcpClient``'s real code: ``_extract_tool_event`` caches
    the identity, ``_build_permission_event`` builds the request, and
    ``_deny_spec_disabled_tool`` -> ``reject_tool`` writes the refusal. Only the wire
    write is redirected to the harness's stdin.
    """
    binary = _require_goose()
    probe = tmp_path / "probe_mcp.py"
    probe.write_text(_PROBE_MCP, encoding="utf-8")
    log = tmp_path / "calls.log"
    _probe_spec(agents_dir, probe, log)
    model = _FakeModel(["probe__secret", "probe__public"])
    try:
        env = {**_goose_env(tmp_path, model), "READ_ONLY": "secret"}
        work = tmp_path / "work"
        work.mkdir()
        client = AcpClient(work_dir=work, acp_backend=ACP_BACKEND_GOOSE, extra_env=env)
        servers = client._resolve_session_mcp_servers()
        assert [e["name"] for e in servers] == ["probe"]
        assert client._spec_denied_tools == {("probe", "secret")}
        client._session_id = "goose-live"
        audited: list[dict] = []
        client._audit_spec_restriction = lambda **kw: audited.append(kw)  # type: ignore[method-assign]
        asked: list[tuple] = []

        def on_update(msg: dict) -> None:
            client._extract_tool_event(
                JsonRpcMessage(method=msg.get("method"), params=msg.get("params"))
            )

        def answer(msg: dict) -> dict:
            event = client._build_permission_event(
                JsonRpcMessage(id=msg["id"], method=msg["method"], params=msg["params"])
            )
            assert event is not None
            asked.append((event.mcp_server_name, event.tool_name, event.mcp_identity_trusted))
            sent: list[dict] = []

            async def capture(_request_id, result):
                sent.append(result)

            client._send_response = capture  # type: ignore[method-assign]
            if asyncio.run(client._deny_spec_disabled_tool(event)):
                assert len(sent) == 1
                return sent[0]
            allow = next(o for o in msg["params"]["options"] if o["kind"] == "allow_once")
            return {"outcome": {"outcome": "selected", "optionId": allow["optionId"]}}

        _drive_acp([binary, "acp"], env, work, servers, on_update=on_update, answer=answer)
    finally:
        model.close()
    assert asked == [
        ("probe", "secret", True),
        ("probe", "public", True),
    ], "goose must ask for both calls, the readOnlyHint one included, and name each pair"
    assert audited and audited[0]["reason"] == "spec_disabled_tool"
    assert log.read_text(encoding="utf-8").split() == ["public"]


@pytest.mark.real_adapter
def test_real_goose_runs_an_always_allowed_tool_without_asking(agents_dir, tmp_path):
    """ANTI-DRIFT GUARD for the permission.yaml hazard the projection withholds on.

    A tool under ``always_allow`` runs with no permission request at all, so a
    per-call refusal never sees it. The projection reads that file and withholds the
    server; this drives goose with the server mounted anyway to show the hazard is
    real. If goose starts asking here, the withhold can be relaxed.
    """
    binary = _require_goose()
    probe = tmp_path / "probe_mcp.py"
    probe.write_text(_PROBE_MCP, encoding="utf-8")
    log = tmp_path / "calls.log"
    _probe_spec(agents_dir, probe, log)
    model = _FakeModel(["probe__secret"])
    try:
        env = _goose_env(tmp_path, model)
        _write_permission_yaml(
            Path(env["XDG_CONFIG_HOME"]) / "goose" / "permission.yaml", ["probe__secret"]
        )
        work = tmp_path / "work"
        work.mkdir()
        client = AcpClient(work_dir=work, acp_backend=ACP_BACKEND_GOOSE, extra_env=env)
        assert client._resolve_session_mcp_servers() == []
        element = {
            "name": "probe",
            "command": sys.executable,
            "args": [str(probe)],
            "env": [{"name": "CALL_LOG", "value": str(log)}],
        }
        asked: list = []
        _drive_acp(
            [binary, "acp"], env, work, [element], lambda u, t: asked.append(t) or "reject_once"
        )
    finally:
        model.close()
    assert asked == []
    assert log.read_text(encoding="utf-8").split() == ["secret"]


def test_the_live_probe_is_valid_python():
    """The live tests skip where a harness is absent; keep their probe parseable."""
    import ast

    ast.parse(_PROBE_MCP)
    assert opencode_mod.opencode_tool_id("probe", "secret") == "probe_secret"
