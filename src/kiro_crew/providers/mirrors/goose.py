"""Projects the agent spec onto ``goose acp``.

The second single-binary spec harness to carry a session ``mcpServers`` array, and
the array's MECHANICS are the same ones :mod:`kiro_crew.providers.mirrors.opencode`
already owns: the same shared translation
(:func:`kiro_crew.acp.session_mcp.session_mcp_servers`, no new translator), the same
``tools`` allowlist, the same registry filter, the same control-plane re-derivation,
and the same one-owner rule for the pooled broker stubs. So this module DELEGATES the
array to that one's :func:`~kiro_crew.providers.mirrors.opencode.place_single_binary_array`
rather than restating it, and carries what is actually goose's: how a switched-off
tool is honoured, the rulings prose, and the wire facts that differ.

What differs:

* The tool-name grammar is ``<server>__<tool>`` -- two underscores, and no ``mcp__``
  prefix -- where opencode fuses with one. goose also carries the pair separately as
  ``_meta.goose.toolCall.toolName`` and ``extensionName``, which is what lets the
  client refuse a switched-off tool per call (:func:`goose_projection`).
* An element whose command cannot start is DROPPED rather than failing
  ``session/new`` whole.

The duplication that remains between the two harnesses is the mirror CLASS, not the
mechanics: a shared base for "single binary, spec dialect, session array" is a
refactor of both, which is not this change's to make.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Collection, Mapping

import yaml  # type: ignore[import-untyped]

from kiro_crew import platform_compat
from kiro_crew.acp.session_mcp import session_mcp_projection
from kiro_crew.agent_sdk.backends import ACP_BACKEND_GOOSE
from kiro_crew.providers.mirrors.base import AgentConfigMirror, Concern
from kiro_crew.providers.mirrors.base import Disposition as _D
from kiro_crew.providers.mirrors.base import Ruling, SessionProjection
from kiro_crew.providers.mirrors.opencode import place_single_binary_array

__all__ = ["GooseMirror", "goose_always_allowed", "goose_projection"]

logger = logging.getLogger(__name__)


#: A server name goose reports back UNCHANGED as ``extensionName``. Measured on goose
#: 1.50.1: ``probe-core`` comes back as ``probe-core``, while ``Probe_X.y`` comes back
#: as ``probe_x_y`` (lower-cased, other characters folded to ``_``). The per-call
#: refusal compares that reported name with the spec's, so a name outside this class
#: would never match and its deny would silently never fire. Such a server is withheld
#: whole instead. Tool names are reported as given, so they need no such check.
_GOOSE_PLAIN_SERVER_NAME = re.compile(r"[a-z0-9_-]+")

#: The file goose reads per-tool permission levels from, under its config directory.
_GOOSE_PERMISSION_FILE = "permission.yaml"


def goose_permission_files(
    env: Mapping[str, str], *, windows: bool | None = None
) -> tuple[Path, ...]:
    """Every ``permission.yaml`` a goose child started with *env* may read.

    goose 1.50.1 puts its config directory at ``$GOOSE_PATH_ROOT/config`` when that is
    set. Otherwise it is ``%APPDATA%\\Block\\goose\\config`` on Windows, and
    ``$XDG_CONFIG_HOME/goose`` (``~/.config/goose`` without it) everywhere else; the
    POSIX paths were measured. Every candidate the environment names is returned, on
    every platform: reading one file more than the harness does can only withhold,
    never mount.

    EMPTY when the platform's own location cannot be derived -- no ``GOOSE_PATH_ROOT``
    and no ``APPDATA`` on Windows, or no ``XDG_CONFIG_HOME`` and no ``HOME`` elsewhere.
    The caller reads that as "unknown", never as "allows nothing". *windows* defaults
    to the running platform.
    """
    is_windows = platform_compat.IS_WINDOWS if windows is None else windows
    paths: list[Path] = []
    root = (env.get("GOOSE_PATH_ROOT") or "").strip()
    if root:
        paths.append(Path(root).expanduser() / "config" / _GOOSE_PERMISSION_FILE)
    appdata = (env.get("APPDATA") or "").strip()
    if appdata:
        paths.append(Path(appdata) / "Block" / "goose" / "config" / _GOOSE_PERMISSION_FILE)
    xdg = (env.get("XDG_CONFIG_HOME") or "").strip()
    home = (env.get("HOME") or "").strip()
    if xdg:
        paths.append(Path(xdg).expanduser() / "goose" / _GOOSE_PERMISSION_FILE)
    elif home:
        paths.append(Path(home) / ".config" / "goose" / _GOOSE_PERMISSION_FILE)
    native = bool(root) or (bool(appdata) if is_windows else bool(xdg or home))
    return tuple(paths) if native else ()


def goose_always_allowed(
    env: Mapping[str, str], *, windows: bool | None = None
) -> frozenset[str] | None:
    """Tool names goose runs WITHOUT asking, from its ``permission.yaml``, lower-cased.

    goose lists a tool there as ``<extension>__<tool>`` under ``always_allow``, and a
    tool listed there is run with no ``session/request_permission`` at all -- measured
    on goose 1.50.1 even with ``GOOSE_MODE=approve``. That skips Crew's per-call
    refusal, so a switched-off tool listed there would run. Every section's
    ``always_allow`` list is read (``user`` and ``smart_approve`` alike), because
    reading more can only withhold more.

    ``None`` when a file exists but cannot be read or parsed, or when goose's own
    config location cannot be derived from *env* at all: what it allows is unknown,
    and the caller treats unknown as "allows everything". A missing file allows
    nothing.
    """
    paths = goose_permission_files(env, windows=windows)
    if not paths:
        return None
    found: set[str] = set()
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError):
            return None
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return None
        if data is None:
            continue
        if not isinstance(data, dict):
            return None
        for section in data.values():
            if not isinstance(section, dict):
                continue
            allowed = section.get("always_allow")
            if allowed is None:
                continue
            if not isinstance(allowed, list):
                return None
            found.update(str(entry).casefold() for entry in allowed)
    return frozenset(found)


def goose_unhonoured_servers(
    disabled_tools: Collection[tuple[str, str]], always_allowed: frozenset[str] | None
) -> frozenset[str]:
    """Narrowed servers whose per-call refusal cannot be trusted to fire on goose.

    Two reasons, each measured: goose reports the server under a different name than
    the spec's (:data:`_GOOSE_PLAIN_SERVER_NAME`), so no refusal could match; or goose
    runs the switched-off tool without asking, because ``permission.yaml`` lists it
    under ``always_allow`` (or that file could not be read, *always_allowed* ``None``).
    Free of I/O.
    """
    out: set[str] = set()
    for server, tool in disabled_tools:
        if not _GOOSE_PLAIN_SERVER_NAME.fullmatch(server):
            out.add(server)
        elif always_allowed is None or f"{server}__{tool}".casefold() in always_allowed:
            out.add(server)
    return frozenset(out)


def goose_projection(
    agent: str | None,
    *,
    stub_server_names: Collection[str] = (),
    stub_elements: Collection[Mapping[str, Any]] = (),
    work_dir: object = None,
    session_key: str = "",
    channel_id: str = "",
    session_token: str = "",
    harness_env: Mapping[str, str] | None = None,
) -> SessionProjection:
    """The whole goose array -- spec translation AND pooled stubs -- plus the deny set.

    The array is placed by the same function opencode's is
    (:func:`~kiro_crew.providers.mirrors.opencode.place_single_binary_array`). What
    differs is how a switched-off tool is honoured: per CALL, the way codex does it.
    goose asks ``session/request_permission`` for every MCP call under the mode Crew
    pins (``GOOSE_MODE=approve``), including a tool annotated ``readOnlyHint`` --
    measured on goose 1.50.1 -- and the ``tool_call`` frame before it names the pair
    as ``_meta.goose.toolCall.{extensionName,toolName}``. So ``denied_tools`` carries
    every switched-off pair, and the client answers a call to one with goose's
    ``reject_once`` before it runs (``AcpClient._deny_spec_disabled_tool``). That
    channel is complete for a third-party server too, unlike codex's, so a narrowed
    server stays MOUNTED, Crew's own control plane included.

    The exception is :func:`goose_unhonoured_servers`: a server goose renames, or
    whose switched-off tool its ``permission.yaml`` pre-approves. Those are withheld
    whole, with a warning. *harness_env* is the environment the goose child starts
    with, which decides where that file is; ``None`` reads the gateway's own.

    Blocking (parses the agent spec once and reads ``permission.yaml``), so callers
    run it off the event loop.
    """
    projection = session_mcp_projection(
        agent,
        stub_server_names=stub_server_names,
        work_dir=work_dir,  # type: ignore[arg-type]
    )
    always_allowed: frozenset[str] | None = frozenset()
    if projection.disabled_tools:
        always_allowed = goose_always_allowed(os.environ if harness_env is None else harness_env)
    unhonoured = goose_unhonoured_servers(projection.disabled_tools, always_allowed)
    for server in sorted(unhonoured):
        logger.warning(
            "goose session MCP: a switched-off tool on %r cannot be refused per call on "
            "this session -- %s",
            server,
            (
                "goose reports this server under a different name, so a refusal could "
                "never match it; rename the server to lower-case letters, digits, _ and -"
                if not _GOOSE_PLAIN_SERVER_NAME.fullmatch(server)
                else (
                    "goose's permission.yaml could not be read or located, so it may "
                    "pre-approve it"
                    if always_allowed is None
                    else "goose's permission.yaml lists it under always_allow, so goose runs it "
                    "without asking; remove that entry to restore per-tool deny"
                )
            ),
        )
    out = place_single_binary_array(
        projection,
        label="goose",
        unhonoured=unhonoured,
        stub_elements=stub_elements,
        session_key=session_key,
        channel_id=channel_id,
        session_token=session_token,
    )
    return SessionProjection(
        params={"mcpServers": out},
        # Every pair, withheld servers included: a name something else re-adds stays
        # refusable at the permission request.
        denied_tools=frozenset(projection.disabled_tools),
        disabled_servers=projection.disabled_servers,
        restricted_servers=unhonoured,
        unhonoured_servers=unhonoured,
        derived_spec_snapshot=projection.derived_spec_snapshot,
    )


class GooseMirror(AgentConfigMirror):
    """Projects the agent spec onto ``goose acp``."""

    backend = ACP_BACKEND_GOOSE

    def rulings(self) -> Mapping[Concern, Ruling]:
        return {
            Concern.MCP_SERVERS: Ruling(
                _D.DELIVERED,
                "the session/new + session/load mcpServers array, translated by "
                "acp.session_mcp.session_mcp_servers with NO new translator and then "
                "narrowed the way the sibling single-binary harness narrows it. The "
                "channel is measured as a ROUND TRIP rather than as an accepted "
                "element, which is the distinction this folder exists to draw: driven "
                "against goose 1.50.1, the element Crew already emits is accepted, the "
                "named stdio child is asked initialize, notifications/initialized, "
                "tools/list AND tools/call by goose itself, and the tool's own result "
                "comes back on tool_call_update. Its initialize advertises "
                "mcpCapabilities of http and sse with no stdio flag, which is not a "
                "refusal: ACP's McpCapabilities schema has only those two boolean "
                "fields, so a conforming agent cannot advertise stdio and that answer "
                "is what full support looks like",
            ),
            Concern.TOOL_ALLOWLIST: Ruling(
                _D.TRANSLATED,
                "into the allowlist that decides which servers enter the array -- the "
                "shared translation's own rule, not a second one here",
            ),
            Concern.DENIED_TOOLS: Ruling(
                _D.TRANSLATED,
                "per call, into the deny set the client refuses a permission request "
                "by, and declared as per_tool_deny=per-call on the projection record. "
                "goose asks session/request_permission for every MCP call under "
                "GOOSE_MODE=approve, a readOnlyHint tool included, and the tool_call "
                "frame before it names the pair as _meta.goose.toolCall.extensionName "
                "and toolName, which the identity table in acp._dispatch reads -- all "
                "measured on goose 1.50.1, where answering reject_once means the "
                "tool never runs. That channel is complete for a third-party server "
                "as well as for Crew's own control plane, so a narrowed server stays "
                "MOUNTED. Two measured gaps withhold a server whole instead: goose "
                "reports a name outside lower-case letters, digits, _ and - under a "
                "folded spelling no refusal could match, and a tool listed under "
                "always_allow in goose's permission.yaml runs without asking at all. "
                "The projection reads that file before the session starts "
                "(goose_always_allowed); an unreadable one counts as allowing "
                "everything",
            ),
            Concern.MODEL: Ruling(
                _D.DELIVERED,
                "as a session config option, from the provider and model selects the "
                "harness advertises on session/new -- not from this array",
            ),
            Concern.MODEL_ALLOWLIST: Ruling(
                _D.WITHHELD,
                "the harness advertises its own provider and model vocabulary on "
                "session/new, so a Crew-side list would be a second answer to a "
                "question the session already answers",
            ),
            Concern.AUTO_APPROVE: Ruling(
                _D.WITHHELD,
                "this harness's approval granularity is a session MODE, not a per-tool "
                "value, and the only mode Crew puts on the wire is the one that asks. "
                "An auto-approve projection would have to name auto, which "
                "session/set_mode accepts and which suppresses the permission frames "
                "the host gate is reached through",
            ),
            Concern.PERMISSION_MODE: Ruling(
                _D.WITHHELD,
                "not projected as spec data. The mode Crew requires travels as GOOSE_MODE "
                "in the child's environment and is read back off the session response; "
                "putting it here as well would give one guarantee two owners",
            ),
            Concern.PROMPT: Ruling(
                _D.WITHHELD,
                "no definition is projected in any shape, so there is nothing for a "
                "prompt to ride on; the session's instructions arrive as the turn's own "
                "text",
            ),
            Concern.RESOURCES: Ruling(
                _D.WITHHELD,
                "no channel is advertised for them, and the array carries servers rather "
                "than resources",
            ),
            Concern.HOOKS: Ruling(
                _D.TRANSLATED,
                "the ACP session/new element set has no hooks field, so the harness never "
                "receives the spec's hooks block, and Crew's turn loop runs it instead "
                "(agent_sdk/spec_hooks.py, ACP_BACKENDS_CREW_FIRES_SPEC_HOOKS), as it does "
                "for KAS. That is sound here because every tool call arrives as a "
                "permission request, which is where a PreToolUse hook runs and can block. "
                "A tool matcher is written in kiro-cli's names; the tool name goose "
                "states in _meta.goose.toolCall is mapped back to them "
                "(acp/harness_tool_names.py), so execute_bash meets goose's shell. "
                "This covers the chat, subagent and task-runner turn loops; a "
                "channel-agent turn runs no script hooks on any backend",
            ),
        }

    def session_params(
        self,
        agent: str | None,
        *,
        stub_server_names: Collection[str] = (),
        work_dir: object = None,
        session_key: str = "",
        channel_id: str = "",
        session_token: str = "",
        **kwargs: object,
    ) -> dict[str, object]:
        """The wire face: the ``mcpServers`` array for this goose session.

        ``permission_surface_owned`` is accepted and IGNORED (it arrives in
        ``kwargs``), which is the documented behaviour for a mirror outside claude's
        class. That flag exists because claude's permission surface is a file Crew may
        not own, so a pre-approved tool there never sends
        ``session/request_permission``. goose has no such file in play: its routing is
        one environment variable, read back off the session response before the first
        prompt, so a session that reaches this point is one whose every tool call asks.

        ``stub_elements`` are the shared gateway's broker stubs for this session, which
        the caller holds and this mirror places, so the withhold rule covers both halves
        of the array -- see :func:`goose_projection`.

        Blocking -- it reads the agent spec. The caller warms this on the goose spawn
        path and serves the shared ``session/new`` call site from that cache
        (harness-parity H13).
        """
        stubs = kwargs.get("stub_elements") or ()
        return self.session_projection(
            agent,
            stub_server_names=stub_server_names,
            stub_elements=stubs if isinstance(stubs, (list, tuple)) else (),
            work_dir=work_dir,
            session_key=session_key,
            channel_id=channel_id,
            session_token=session_token,
        ).params

    def session_projection(
        self,
        agent: str | None,
        *,
        stub_server_names: Collection[str] = (),
        stub_elements: Collection[Mapping[str, Any]] = (),
        work_dir: object = None,
        session_key: str = "",
        channel_id: str = "",
        session_token: str = "",
        **kwargs: object,
    ) -> SessionProjection:
        """The structured face: :func:`goose_projection`.

        ``harness_env`` arrives in ``kwargs``: the environment the goose child starts
        with, which decides where its ``permission.yaml`` is. Every other keyword in
        ``kwargs`` is ignored, as :meth:`session_params` documents."""
        env = kwargs.get("harness_env")
        return goose_projection(
            agent,
            stub_server_names=stub_server_names,
            stub_elements=stub_elements,
            work_dir=work_dir,
            session_key=session_key,
            channel_id=channel_id,
            session_token=session_token,
            harness_env=env if isinstance(env, Mapping) else None,
        )
