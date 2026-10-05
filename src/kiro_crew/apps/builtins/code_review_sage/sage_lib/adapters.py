#!/usr/bin/env python3
"""Source adapter — normalize a GitHub PR or Azure DevOps PR link into a single
``ReviewTarget``.

The brain only ever sees a ``ReviewTarget``. Adding a new platform is a new
adapter, not a brain change. This build ships two: the **GitHub PR** adapter and
the **Azure DevOps PR** adapter.

The network fetch itself is performed by the pipeline — GitHub via the ``gh``
CLI, Azure DevOps via the ``azure-devops`` MCP (which, unlike GitHub, has no
PR-diff endpoint, so the pipeline assembles per-file unified diffs from the two
blob sides; see ``pipeline.ADO_FETCH_SPEC``). This module is the deterministic,
token-free part: parsing the fetched payload into a ``ReviewTarget``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

from sage_lib import store

# Matches the PR path grammar shared by github.com and GitHub Enterprise Server:
# /<owner>/<repo>/pull/<number>[...]. Applied to the PARSED URL path only, AFTER
# the hostname allowlist check — never to the raw link.
_PR_PATH_RE = re.compile(r"^/([^/]+)/([^/]+)/pull/(\d+)")
# A change is a "fix" if its title/description signals a bug/revert/incident.
FIX_RE = re.compile(r"\b(fix(es|ed)?|revert(s|ed)?|bug|hotfix|regression|incident|patch)\b", re.I)
# GitHub-style issue reference (e.g. "#204") linked from the PR body.
GH_ISSUE_RE = re.compile(r"#(\d+)")

# --- Azure DevOps PR path grammar -------------------------------------------
# dev.azure.com (and Azure DevOps Server): /<org>/<project>/_git/<repo>/pullrequest/<id>
_ADO_PATH_DEVAZURE = re.compile(
    r"^/([^/]+)/([^/]+)/_git/([^/]+)/pullrequest/(\d+)", re.I)
# Legacy *.visualstudio.com: the ORG is the subdomain, so the path omits it:
# /<project>/_git/<repo>/pullrequest/<id>
_ADO_PATH_VSTS = re.compile(
    r"^/([^/]+)/_git/([^/]+)/pullrequest/(\d+)", re.I)
# Canonical public Azure DevOps host.
_ADO_HOST = "dev.azure.com"
# Azure DevOps work-item reference (e.g. "AB#1234" / "#1234") linked from a PR.
ADO_WORKITEM_RE = re.compile(r"(?:AB)?#(\d+)", re.I)


class AdapterError(ValueError):
    """Base class for adapter failures (fail-fast)."""


class UnsupportedPlatform(AdapterError):
    """The link's platform is not supported in this build."""


class AdapterParseError(AdapterError):
    """The fetched payload could not be normalized into a ReviewTarget."""


@dataclass
class ReviewTarget:
    """The single normalized shape the review brain consumes."""

    platform: str
    repo_identity: str          # host/org/repo — the learning key
    change_id: str
    url: str
    title: str = ""
    description: str = ""
    linked_issue: str = ""
    author: str = ""
    target_branch: str = ""
    revision: str = ""
    files: list[dict] = field(default_factory=list)        # [{path, diff}]
    existing_comments: list[dict] = field(default_factory=list)
    design_discussion: list[dict] = field(default_factory=list)
    is_fix: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

# Canonical public-GitHub hostnames. Named constants (rather than inline
# literals) keep membership tests on the parsed-host SET from reading as
# URL-substring checks — the sets these appear in hold bare hostnames, never
# URLs.
_GITHUB_HOST = "github.com"
_WWW_GITHUB_HOST = "www.github.com"


def canonical_host(host: str) -> str:
    """``www.github.com`` -> ``github.com``; otherwise the lowercased hostname.

    Persisted identities (``repo_identity``, change ids, reviewed keys) use the
    canonical form so the two spellings of github.com map to one record."""
    h = (host or "").strip().lower()
    return _GITHUB_HOST if h == _WWW_GITHUB_HOST else h


def _urlparse_host_path(text: str) -> tuple[str, str]:
    """Parse ``text`` into ``(hostname, path)``, reading a malformed URL as
    having neither.

    ``urlparse`` raises ``ValueError`` on some malformed input (an unmatched
    ``[`` is "Invalid IPv6 URL"), and ``.hostname`` can raise on a malformed
    netloc — both reachable from user-pasted link text, where one bad link must
    read as "not a link" (default-deny) rather than crash the whole request."""
    try:
        parsed = urlparse(text)
        return (parsed.hostname or "").lower(), parsed.path or ""
    except ValueError:
        return "", ""


def allowed_hosts(config: dict | None = None) -> frozenset[str]:
    """The exact set of hostnames accepted as GitHub-API-compatible.

    The single resolution point for "which hosts are acceptable". Sourced from
    ``config.json``'s ``github_hosts`` list — GitHub Enterprise Server hosts are
    opt-in, mirroring ``gh auth login --hostname`` — and defaults to github.com,
    so behaviour is unchanged when nothing is configured. ``www.github.com`` is
    accepted whenever ``github.com`` is.

    Membership tests against this set MUST use the PARSED URL hostname and
    exact equality — never a substring, suffix, or regex-on-raw-URL test — so
    ``notgithub.com``, ``github.com.evil.example``, and a permitted host that
    appears only in a URL's path are all refused."""
    cfg = config if config is not None else store.read_config_quiet()
    raw = cfg.get("github_hosts") if isinstance(cfg, dict) else None
    hosts: set[str] = set()
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            h = str(entry or "").strip().lower()
            if "://" in h:  # tolerate a pasted URL in config (malformed -> skipped)
                h = _urlparse_host_path(h)[0]
            h = h.strip("/").rstrip(".")
            if h:
                hosts.add(h)
    if not hosts:
        hosts = set(store.DEFAULT_GITHUB_HOSTS)
    # `hosts` holds bare hostnames (never URLs); this is exact set membership.
    # The two public-GitHub spellings imply each other: `www.github.com`
    # canonicalizes to `github.com` downstream, so a www-only config must also
    # accept the canonical form or accepted links would fail to round-trip.
    if hosts & {_GITHUB_HOST, _WWW_GITHUB_HOST}:
        hosts.add(_GITHUB_HOST)
        hosts.add(_WWW_GITHUB_HOST)
    return frozenset(hosts)


def detect_platform(link: str, *, config: dict | None = None) -> str:
    """Return ``github`` or ``ado`` for a recognized PR link, else raise
    UnsupportedPlatform.

    Azure DevOps is tried FIRST because its grammar is unambiguous
    (``/_git/.../pullrequest/<id>``) and cannot be confused with a GitHub
    ``/pull/<n>`` path. Both platforms validate the host by EXACT match of the
    PARSED URL hostname (not a substring of the raw link), so a URL where an
    allowed host merely appears in the path/query/userinfo (e.g.
    ``https://evil.example/github.com/x/pull/1``) or as a spoofable
    prefix/suffix (``notgithub.com``, ``github.com.evil.example``) is rejected,
    and a malformed URL reads as unsupported rather than raising ``ValueError``.
    Aligns with SSRF/allowlist guidance (parse to components, default-deny)."""
    if not link or not isinstance(link, str):
        raise UnsupportedPlatform("empty or non-string link")
    if detect_platform_ado(link, config=config):
        return "ado"
    host, path = _urlparse_host_path(link)
    if host in allowed_hosts(config) and "/pull/" in path:
        return "github"
    raise UnsupportedPlatform(
        f"unsupported link/platform: {link!r} "
        "(expected a GitHub PR URL or an Azure DevOps PR URL)")


def _sanitize_seg(s: str) -> str:
    """Make an owner/repo segment safe for use in a change-id (which names a
    result record on disk). Non ``[A-Za-z0-9.]`` chars — including ``-`` — become
    ``_``. Excluding ``-`` is deliberate: ``-`` is the segment delimiter in
    ``github_change_id`` (``GH-<owner>-<repo>-<n>``), so keeping it inside a
    segment would make different owner/repo pairs collide (e.g. ``a-b``/``c`` vs
    ``a``/``b-c``). Stripping it to ``_`` keeps ``-`` unambiguous as the delimiter."""
    return re.sub(r"[^A-Za-z0-9.]", "_", str(s or "")).strip("_") or "unknown"


def github_pr_ref(link: str, *, config: dict | None = None) -> tuple[str, str, str, str]:
    """Parse ``(host, owner, repo, number)`` from a PR URL on an allowed GitHub
    host. Fails fast.

    Host membership uses the same PARSED-hostname exact-match allowlist as
    ``detect_platform`` (never a substring of the raw link). The host comes
    back canonicalized (``www.github.com`` -> ``github.com``) so identities
    derived from it are stable. A scheme-less link (``github.com/o/r/pull/1``)
    is tolerated by retrying with ``https://``; a malformed link is rejected
    like any other non-PR link (never a ``ValueError`` out of ``urlparse``)."""
    if not link or not isinstance(link, str):
        raise AdapterParseError(f"not a GitHub PR link: {link!r}")
    text = link.strip()
    host, path = _urlparse_host_path(text)
    if not host and "://" not in text:
        host, path = _urlparse_host_path("https://" + text)
    if host not in allowed_hosts(config):
        raise AdapterParseError(f"not a GitHub PR link: {link!r}")
    m = _PR_PATH_RE.match(path)
    if not m:
        raise AdapterParseError(f"not a GitHub PR link: {link!r}")
    owner, repo, number = m.group(1), m.group(2), m.group(3)
    repo = re.sub(r"\.git$", "", repo)  # tolerate a trailing .git
    return canonical_host(host), owner, repo, number


def github_pr_parts(link: str) -> tuple[str, str, str]:
    """Parse ``(owner, repo, number)`` from a PR URL on an allowed GitHub host.
    Fails fast. Callers that need the host use ``github_pr_ref``."""
    _host, owner, repo, number = github_pr_ref(link)
    return owner, repo, number


def link_names_a_host(link: str) -> bool:
    """Whether ``link`` plausibly NAMES a network host — an explicit
    ``scheme://`` form (even with an unparseable host) or a leading
    domain-shaped segment (``ghe.corp/…``).

    Callers use this to distinguish a URL whose host failed validation (must
    FAIL CLOSED — routing it at a default host could cross GitHub instances)
    from a bare legacy change token (``CR-1``) that carries no host to cross
    to. Deliberately over-matches: a dot in the first segment reads as a
    domain, because refusing is the safe side."""
    if not link or not isinstance(link, str):
        return False
    text = link.strip()
    if _urlparse_host_path(text)[0]:
        return True
    if "://" in text:  # a scheme is present but the host is unparseable
        return True
    return "." in text.split("/", 1)[0]


def parse_repo_ref(link: str, *, config: dict | None = None) -> tuple[str, str, str]:
    """Parse ``(host, owner, repo)`` from a GitHub REPO URL (no ``/pull/``).

    Mirrors ``detect_platform``'s PARSED-hostname allowlist (default-deny,
    SSRF/allowlist guidance) but accepts a bare repo URL like
    ``https://github.com/<owner>/<repo>`` so a batch of that repo's open PRs can
    be enumerated. Raises ``UnsupportedPlatform`` for a host outside the
    allowlist (including a malformed URL, which parses to no host) and
    ``AdapterParseError`` when the owner/repo path segments are missing."""
    if not link or not isinstance(link, str):
        raise UnsupportedPlatform("empty or non-string repo link")
    host, path = _urlparse_host_path(link)
    hosts = allowed_hosts(config)
    if host not in hosts:
        raise UnsupportedPlatform(
            f"unsupported repo host: {link!r} "
            f"(expected a repo URL on one of: {', '.join(sorted(hosts))})")
    if "/pull/" in path:
        # A PR URL, not a repo URL — route the user to the paste flow so we don't
        # silently review the PR's whole repo.
        raise AdapterParseError(
            f"that's a PR URL, not a repo URL: {link!r} (paste it in the PR box)")
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise AdapterParseError(f"not a GitHub repo link: {link!r}")
    owner, repo = parts[0], re.sub(r"\.git$", "", parts[1])
    _seg = re.compile(r"^[A-Za-z0-9._-]+$")
    if owner in (".", "..") or repo in (".", "..") or not (_seg.match(owner) and _seg.match(repo)):
        raise AdapterParseError(f"invalid owner/repo in {link!r}")
    return canonical_host(host), owner, repo


def parse_repo_url(link: str) -> tuple[str, str]:
    """Parse ``(owner, repo)`` from a GitHub REPO URL (no ``/pull/``). Callers
    that need the host use ``parse_repo_ref``."""
    _host, owner, repo = parse_repo_ref(link)
    return owner, repo


def github_change_id(owner: str, repo: str, number: str | int,
                     host: str = "github.com") -> str:
    """Filesystem-safe, platform-namespaced change id: ``GH-<owner>-<repo>-<n>``.
    Unlike a raw URL, this is a valid filename.

    A non-github.com (GitHub Enterprise) host gets a leading sanitized host
    segment (``GH-<host>-<owner>-<repo>-<n>``) so the same owner/repo/number on
    two hosts cannot share one result file. github.com ids keep the host-less
    shape byte-identical so already-persisted records still resolve."""
    h = canonical_host(host)
    prefix = f"GH-{_sanitize_seg(h)}-" if h and h != "github.com" else "GH-"
    return f"{prefix}{_sanitize_seg(owner)}-{_sanitize_seg(repo)}-{number}"


def github_review_key(owner: str, repo: str, number: str | int,
                      host: str = "github.com") -> str:
    """Collision-free canonical identity for the durable reviewed-index key.

    Distinct from ``github_change_id``: that value ALSO names an on-disk result
    file, so it runs owner/repo through ``_sanitize_seg`` — which collapses ``-``
    to ``_`` to keep ``-`` unambiguous as its segment delimiter. That sanitization
    is lossy: ``acme/service-api`` and ``acme/service_api`` both become
    ``GH-acme-service_api-<n>``, so two DIFFERENT repos with the same PR number
    shared one ``reviewed.json`` key and clobbered each other's dedup record —
    silently skipping a requested review when their PR heads happened to share a
    commit SHA (as mirrored repos can).

    This key never names a file, so it keeps owner/repo verbatim and joins with
    ``/`` (a character GitHub owner/repo can never contain), giving a lossless,
    unambiguous identity. Owner/repo are lower-cased because GitHub treats them
    case-insensitively for identity. The key is HOST-qualified so the same
    owner/repo on two GitHub hosts cannot collide; the github.com default keeps
    already-persisted keys byte-identical."""
    h = canonical_host(host) or "github.com"
    return f"{h}/{str(owner).lower()}/{str(repo).lower()}#{number}"


# ---------------------------------------------------------------------------
# Azure DevOps adapter
# ---------------------------------------------------------------------------

def _is_vsts_host(host: str) -> bool:
    """``<org>.visualstudio.com`` — the legacy Azure DevOps host where the org
    is the subdomain. Matched by SUFFIX (never substring): a bare ``.`` prefix
    and a lookalike like ``notvisualstudio.com`` must not match."""
    h = (host or "").lower()
    return h.endswith(".visualstudio.com") and len(h) > len(".visualstudio.com")


def ado_allowed_hosts(config: dict | None = None) -> frozenset[str]:
    """Exact hostname allowlist for Azure DevOps, mirroring ``allowed_hosts``.

    Defaults to ``dev.azure.com``. On-prem Azure DevOps Server hosts are opt-in
    via ``config.json``'s ``ado_hosts`` list (the list REPLACES the default, so
    keep ``dev.azure.com`` to still review there). ``*.visualstudio.com`` is NOT
    in this set — it is matched separately by :func:`_is_vsts_host` because each
    org is its own subdomain, so an exact set cannot enumerate them. Membership
    tests MUST use the PARSED URL hostname and exact equality — never a
    substring — so ``dev.azure.com.evil.example`` is refused."""
    cfg = config if config is not None else store.read_config_quiet()
    raw = cfg.get("ado_hosts") if isinstance(cfg, dict) else None
    hosts: set[str] = set()
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            h = str(entry or "").strip().lower()
            if "://" in h:  # tolerate a pasted URL in config (malformed -> skipped)
                h = _urlparse_host_path(h)[0]
            h = h.strip("/").rstrip(".")
            if h:
                hosts.add(h)
    if not hosts:
        hosts = set(store.DEFAULT_ADO_HOSTS)
    return frozenset(hosts)


def detect_platform_ado(link: str, *, config: dict | None = None) -> bool:
    """True iff *link* is a recognized Azure DevOps PR URL on an allowed host.

    Returns a bool (not a platform string) so :func:`detect_platform` owns the
    canonical return values. Host check is exact-match on the PARSED hostname
    (default-deny), plus the documented ``*.visualstudio.com`` suffix. A
    malformed URL reads as False, never a raised ``ValueError``."""
    if not link or not isinstance(link, str):
        return False
    host, path = _urlparse_host_path(link if "://" in link else "https://" + link)
    if "/pullrequest/" not in path.lower():
        return False
    return host in ado_allowed_hosts(config) or _is_vsts_host(host)


def ado_pr_ref(link: str, *, config: dict | None = None) -> tuple[str, str, str, str, str]:
    """Parse ``(host, org, project, repo, pr_id)`` from an Azure DevOps PR URL.
    Fails fast.

    Handles both ``dev.azure.com`` (org in path) and ``*.visualstudio.com`` (org
    in the subdomain). A scheme-less link is tolerated by retrying with
    ``https://``; any other malformed link is rejected like a non-PR link
    (never a ``ValueError`` out of ``urlparse``)."""
    if not link or not isinstance(link, str):
        raise AdapterParseError(f"not an Azure DevOps PR link: {link!r}")
    text = link.strip()
    host, path = _urlparse_host_path(text)
    if not host and "://" not in text:
        host, path = _urlparse_host_path("https://" + text)

    if _is_vsts_host(host):
        org = host.split(".", 1)[0]            # org is the subdomain
        m = _ADO_PATH_VSTS.match(path)
        if not m:
            raise AdapterParseError(f"not an Azure DevOps PR link: {link!r}")
        project, repo, pr_id = m.group(1), m.group(2), m.group(3)
    else:
        if host not in ado_allowed_hosts(config):
            raise AdapterParseError(f"not an allowed Azure DevOps host: {link!r}")
        m = _ADO_PATH_DEVAZURE.match(path)
        if not m:
            raise AdapterParseError(f"not an Azure DevOps PR link: {link!r}")
        org, project, repo, pr_id = m.group(1), m.group(2), m.group(3), m.group(4)

    repo = re.sub(r"\.git$", "", repo)  # tolerate a trailing .git
    return host.lower(), org, project, repo, pr_id


def ado_change_id(org: str, project: str, repo: str, number: str | int) -> str:
    """Filesystem-safe, platform-namespaced change id: ``ADO-<org>-<project>-<repo>-<n>``.

    The distinct ``ADO-`` prefix guarantees an Azure DevOps record can never
    name the same on-disk file as a GitHub ``GH-`` record. Segments run through
    :func:`_sanitize_seg` (the same ``-``-collapsing used for GitHub) so the
    ``-`` delimiter stays unambiguous and different org/project/repo tuples
    cannot collide."""
    s = _sanitize_seg
    return f"ADO-{s(org)}-{s(project)}-{s(repo)}-{number}"


def ado_review_key(org: str, project: str, repo: str, number: str | int) -> str:
    """Collision-free durable reviewed-index key for an Azure DevOps PR — the
    ADO twin of :func:`github_review_key`. Never names a file, so it keeps
    org/project/repo verbatim (lower-cased for case-insensitive identity) and
    joins with ``/`` (a character these segments cannot contain)."""
    return (f"{str(org).lower()}/{str(project).lower()}/"
            f"{str(repo).lower()}#{number}")


def extract_linked_workitem(text: str) -> str:
    """Extract an Azure DevOps work-item reference (``AB#1234`` / ``#1234``) from
    the PR description. Empty when none is present."""
    m = ADO_WORKITEM_RE.search(text or "")
    return f"#{m.group(1)}" if m else ""


def parse_ado_payload(raw: dict | str, *, link: str | None = None,
                      config: dict | None = None) -> ReviewTarget:
    """Normalize an Azure DevOps PR payload into the SAME ``ReviewTarget`` shape
    the brain consumes for GitHub.

    The pipeline assembles this payload from the ``azure-devops`` MCP (see
    ``pipeline.ADO_FETCH_SPEC``): the ``repo_pull_request`` get object, a
    ``files`` array whose per-file ``diff`` the pipeline built from the two blob
    sides (ADO has no PR-diff endpoint), optional ``threads``, and a ``_sage``
    context block the fetcher stamps with the parsed host/org/project/repo.
    Tolerant of field-name variants; fails fast when there is no usable content.

    GitHub→ADO field map:
      head SHA     <- ``lastMergeSourceCommit.commitId``
      target_branch<- ``targetRefName`` (``refs/heads/main`` -> ``main``)
      author       <- ``createdBy.uniqueName`` (falls back to displayName)
      change_id    <- ``ADO-<org>-<project>-<repo>-<pullRequestId>``
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AdapterParseError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise AdapterParseError("payload must be a JSON object")

    sage = raw.get("_sage") if isinstance(raw.get("_sage"), dict) else {}
    host = str(sage.get("host") or "")
    org = str(sage.get("org") or "")
    project = str(sage.get("project") or "")
    repo = str(sage.get("repo") or "")
    number = str(raw.get("pullRequestId") or raw.get("number") or "")

    # Fill any missing part from the link.
    if not all([host, org, project, repo, number]) and link:
        try:
            lh, lo, lp, lr, lid = ado_pr_ref(link, config=config)
            host = host or lh
            org = org or lo
            project = project or lp
            repo = repo or lr
            number = number or lid
        except AdapterParseError:
            pass
    host = host or _ADO_HOST
    if not all([org, project, repo, number]):
        raise AdapterParseError(
            "could not determine Azure DevOps org/project/repo/id from payload or link")

    description = _first(raw, "description", "body", default="")
    title = _first(raw, "title", default="") or (
        description.splitlines()[0] if description else "")

    created_by = raw.get("createdBy") if isinstance(raw.get("createdBy"), dict) else {}
    author = _first(created_by, "uniqueName", "displayName", default="") if created_by else ""

    revision = ""
    lmsc = raw.get("lastMergeSourceCommit")
    if isinstance(lmsc, dict):
        revision = str(lmsc.get("commitId") or "")
    if not revision:
        revision = _first(raw, "sourceCommitId", "revision", default="")

    target_branch = re.sub(r"^refs/heads/", "",
                           str(_first(raw, "targetRefName", "target_branch", default="")))

    raw_files = raw.get("files") or raw.get("changes") or []
    files: list[dict] = []
    for d in raw_files:
        if not isinstance(d, dict):
            continue
        path = _first(d, "path", "filename", "name", default="")
        diff = _first(d, "diff", "patch", "unifiedDiff", default="")
        if path:
            # ADO item paths carry a leading slash; drop it so a finding's file
            # path reads like a repo-relative path (matches GitHub's shape).
            files.append({"path": path.lstrip("/"), "diff": diff})

    if not files and not description:
        raise AdapterParseError("payload has no files and no description")

    # ADO comment threads -> flat text list (the brain only needs the content).
    comments: list[dict] = []
    for t in raw.get("threads") or []:
        if not isinstance(t, dict):
            continue
        for c in t.get("comments") or []:
            if isinstance(c, dict) and c.get("content"):
                comments.append({"body": str(c.get("content"))})

    return ReviewTarget(
        platform="ado",
        repo_identity=f"{host}/{org}/{project}/{repo}",
        change_id=ado_change_id(org, project, repo, number),
        url=link or f"https://{host}/{org}/{project}/_git/{repo}/pullrequest/{number}",
        title=title,
        description=description,
        linked_issue=extract_linked_workitem(description),
        author=str(author) if author else "",
        target_branch=target_branch,
        revision=str(revision),
        files=files,
        existing_comments=comments,
        design_discussion=[],
        is_fix=detect_is_fix(title, description),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first(d: dict, *keys, default=""):
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return default


def _author_alias(raw: dict) -> str:
    a = _first(raw, "author", "authorAlias", "owner", default="")
    if isinstance(a, dict):
        return _first(a, "alias", "login", "name", default="")
    return str(a) if a else ""


def detect_is_fix(title: str, description: str) -> bool:
    return bool(FIX_RE.search(f"{title}\n{description}"))


def extract_linked_issue(text: str) -> str:
    """Extract a linked GitHub issue reference (``#123``) from the PR body."""
    m = GH_ISSUE_RE.search(text or "")
    return f"#{m.group(1)}" if m else ""


# ---------------------------------------------------------------------------
# GitHub adapter
# ---------------------------------------------------------------------------

def parse_github_payload(raw: dict | str, *, link: str | None = None) -> ReviewTarget:
    """Normalize a GitHub PR payload into a ReviewTarget. The worker assembles
    this payload from ``gh api``: the ``pulls/{n}`` object merged with a ``files``
    array (each carrying its per-file ``patch``) and optional ``comments``. Tolerant
    of field-name variants (``filename``/``path``, ``patch``/``diff``); fails fast
    when there is no usable content. ``owner``/``repo``/``number`` are taken from
    the payload (``base.repo.full_name`` + ``number``) and fall back to the link/
    ``html_url`` so the adapter works whether or not the caller echoes the URL."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AdapterParseError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise AdapterParseError("payload must be a JSON object")

    _base = raw.get("base")
    base: dict = _base if isinstance(_base, dict) else {}
    _head = raw.get("head")
    head: dict = _head if isinstance(_head, dict) else {}
    _base_repo = base.get("repo")
    base_repo: dict = _base_repo if isinstance(_base_repo, dict) else {}

    # NOTE: do NOT fall back to raw["id"] — on GitHub that is the internal
    # database id (e.g. 1847293847), NOT the PR number in the URL. Using it would
    # produce a change_id that mismatches _cid()'s URL-derived id (write/read would
    # hit different result files). The link/html_url fallback below supplies the
    # number when the payload omits it.
    number = _first(raw, "number", default="")
    owner = repo = host = ""
    full = _first(base_repo, "full_name", default="")
    if full and "/" in full:
        owner, repo = full.split("/", 1)

    # Fill any missing part (including the host) from the link, then from
    # html_url. Without a parseable URL the host defaults to github.com.
    html_url = _first(raw, "html_url", "url", default="")
    for candidate in (link, html_url):
        if owner and repo and number and host:
            break
        if not candidate:
            continue
        try:
            lh, lo, lr, ln = github_pr_ref(candidate)
        except AdapterParseError:
            continue
        host = host or lh
        owner = owner or lo
        repo = repo or lr
        number = number or ln
    host = host or "github.com"

    if not (owner and repo and number):
        raise AdapterParseError(
            "could not determine GitHub owner/repo/number from payload or link")

    description = _first(raw, "body", "description", default="")
    title = _first(raw, "title", default="") or (description.splitlines()[0] if description else "")

    raw_files = raw.get("files") or raw.get("diffs") or []
    files: list[dict] = []
    for d in raw_files:
        if not isinstance(d, dict):
            continue
        path = _first(d, "filename", "path", "name", default="")
        diff = _first(d, "patch", "diff", "unifiedDiff", default="")
        if path:
            files.append({"path": path, "diff": diff})

    # Fail fast: a PR with neither files nor a description is unusable.
    if not files and not description:
        raise AdapterParseError("payload has no files and no description")

    # GitHub author lives under user.login (fall back to the generic extractor).
    author = ""
    user = raw.get("user")
    if isinstance(user, dict):
        author = _first(user, "login", "name", default="")
    if not author:
        author = _author_alias(raw)

    revision = (_first(head, "sha", default="")
                or _first(raw, "head_sha", "sha", "revision", default=""))
    target_branch = (_first(base, "ref", default="")
                     or _first(raw, "base_ref", "targetBranch", default=""))

    comments = raw.get("comments") or raw.get("review_comments") or raw.get("allComments") or []
    if not isinstance(comments, list):
        comments = []

    return ReviewTarget(
        platform="github",
        repo_identity=f"{host}/{owner}/{repo}",
        change_id=github_change_id(owner, repo, number, host=host),
        url=html_url or f"https://{host}/{owner}/{repo}/pull/{number}",
        title=title,
        description=description,
        linked_issue=extract_linked_issue(description),
        author=str(author) if author else "",
        target_branch=target_branch,
        revision=str(revision),
        files=files,
        existing_comments=comments,
        design_discussion=[],
        is_fix=detect_is_fix(title, description),
    )


def normalize(link: str, raw_payload: dict | str, *, config: dict | None = None) -> ReviewTarget:
    """Top-level entry: detect platform, then parse. Fails fast on unsupported.

    ``config`` is threaded into ``detect_platform`` so a configured Azure DevOps
    Server host (``ado_hosts``) or GitHub Enterprise host (``github_hosts``) is
    recognized; it defaults to None (public github.com / dev.azure.com only),
    keeping the one-arg call sites unchanged."""
    platform = detect_platform(link, config=config)
    if platform == "ado":
        return parse_ado_payload(raw_payload, link=link, config=config)
    if platform == "github":
        return parse_github_payload(raw_payload, link=link)
    raise UnsupportedPlatform(f"unsupported platform: {platform!r}")


def validate_review_target(target: ReviewTarget) -> list[str]:
    """Non-fatal warnings about a normalized target — surfaces likely GitHub
    payload-mapping gaps before review. Empty == looks complete."""
    warns: list[str] = []
    if not target.files:
        warns.append("no files/diffs parsed — check the payload's `files` mapping")
    else:
        if not any(f.get("diff") for f in target.files):
            warns.append("files present but all diffs are empty — check the `patch` field name")
    if not target.title and not target.description:
        warns.append("no title or description — check the payload field names")
    if not target.target_branch:
        warns.append("no target branch parsed (branch-gate checks will be skipped)")
    if target.repo_identity.endswith("/unknown"):
        warns.append("repo could not be determined — learning key will be 'unknown'")
    return warns
