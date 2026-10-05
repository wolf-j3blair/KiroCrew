"""Unit tests for the Azure DevOps source adapter."""
import json
import unittest
from unittest import mock

from sage_lib import adapters as A  # noqa: N812

DEVAZURE = "https://dev.azure.com/myorg/myproj/_git/myrepo/pullrequest/1925"
VSTS = "https://myorg.visualstudio.com/myproj/_git/myrepo/pullrequest/1925"

# A realistic merged payload shaped like what the pipeline assembles from the
# azure-devops MCP (repo_pull_request get + blob-diff files + threads + _sage).
ADO_PAYLOAD = {
    "pullRequestId": 1925,
    "title": "feat(p5): artifact search, detail, and tag editing",
    "description": "Adds the read/search surface. Fixes AB#1234.",
    "createdBy": {"displayName": "Justin Blair", "uniqueName": "j3blair@cree.com"},
    "sourceRefName": "refs/heads/feat/p5-search",
    "targetRefName": "refs/heads/main",
    "lastMergeSourceCommit": {"commitId": "7f40548facd39779b1dc45b27769ec51a34e45d2"},
    "lastMergeTargetCommit": {"commitId": "8cfd58ab72c44499c46d3688692eb7b15276df48"},
    "_sage": {"host": "dev.azure.com", "org": "cree-mfg",
              "project": "sicarbonite", "repo": "sicarbonite"},
    "files": [
        {"path": "/sicapi/src/sicapi/routes/artifacts.py",
         "diff": "@@ -0,0 +1 @@\n+async def search():\n"},
        {"path": "/sicapi/pyproject.toml", "diff": "@@ -1 +1 @@\n-x\n+y\n"},
    ],
    "threads": [
        {"comments": [{"content": "please rename this"}]},
        {"comments": [{"content": "ok done"}]},
    ],
}


class TestAdoPlatformDetection(unittest.TestCase):
    def test_devazure(self):
        self.assertEqual(A.detect_platform(DEVAZURE), "ado")

    def test_visualstudio(self):
        self.assertEqual(A.detect_platform(VSTS), "ado")

    def test_github_still_github(self):
        self.assertEqual(A.detect_platform("https://github.com/o/r/pull/5"), "github")

    def test_ado_not_mistaken_for_github(self):
        # ADO grammar must win: a /_git/.../pullrequest/ path has no /pull/ so it
        # could never be GitHub, but detection order matters for clarity.
        self.assertEqual(A.detect_platform(DEVAZURE), "ado")

    def test_detect_platform_ado_bool(self):
        self.assertTrue(A.detect_platform_ado(DEVAZURE))
        self.assertTrue(A.detect_platform_ado(VSTS))
        self.assertFalse(A.detect_platform_ado("https://github.com/o/r/pull/5"))
        self.assertFalse(A.detect_platform_ado(""))
        self.assertFalse(A.detect_platform_ado(None))


class TestAdoAllowedHosts(unittest.TestCase):
    def test_default_is_devazure(self):
        self.assertEqual(A.ado_allowed_hosts({}), frozenset({"dev.azure.com"}))

    def test_configured_onprem_host_replaces_default(self):
        hosts = A.ado_allowed_hosts({"ado_hosts": ["tfs.corp.example"]})
        self.assertEqual(hosts, frozenset({"tfs.corp.example"}))

    def test_config_entries_normalized(self):
        hosts = A.ado_allowed_hosts({"ado_hosts": ["https://TFS.corp.example/", "", None]})
        self.assertEqual(hosts, frozenset({"tfs.corp.example"}))

    def test_non_list_falls_back_to_default(self):
        self.assertEqual(A.ado_allowed_hosts({"ado_hosts": "nope"}),
                         frozenset({"dev.azure.com"}))

    def test_refused_config_read_falls_back(self):
        with mock.patch.object(A.store, "read_config_quiet", return_value={}):
            self.assertEqual(A.ado_allowed_hosts(), frozenset({"dev.azure.com"}))


class TestAdoHostSpoofing(unittest.TestCase):
    def test_spoofable_hosts_rejected(self):
        for url in (
            "https://dev.azure.com.evil.example/o/p/_git/r/pullrequest/1",
            "https://notdev.azure.com/o/p/_git/r/pullrequest/1",
            "https://evil.example/dev.azure.com/o/p/_git/r/pullrequest/1",
            "https://notvisualstudio.com/p/_git/r/pullrequest/1",
            "https://visualstudio.com/p/_git/r/pullrequest/1",  # bare, no org subdomain
        ):
            self.assertFalse(A.detect_platform_ado(url), url)

    def test_malformed_url_reads_as_not_ado(self):
        for bad in ("https://[::1", "https://[dev.azure.com]/o/p/_git/r/pullrequest/1"):
            self.assertFalse(A.detect_platform_ado(bad), bad)


class TestAdoPrRef(unittest.TestCase):
    def test_devazure_parts(self):
        self.assertEqual(
            A.ado_pr_ref(DEVAZURE),
            ("dev.azure.com", "myorg", "myproj", "myrepo", "1925"))

    def test_vsts_org_from_subdomain(self):
        self.assertEqual(
            A.ado_pr_ref(VSTS),
            ("myorg.visualstudio.com", "myorg", "myproj", "myrepo", "1925"))

    def test_trailing_git_tolerated(self):
        link = "https://dev.azure.com/o/p/_git/r.git/pullrequest/7"
        self.assertEqual(A.ado_pr_ref(link)[3], "r")

    def test_schemeless_tolerated(self):
        self.assertEqual(
            A.ado_pr_ref("dev.azure.com/o/p/_git/r/pullrequest/9")[1:],
            ("o", "p", "r", "9"))

    def test_non_ado_rejected(self):
        with self.assertRaises(A.AdapterParseError):
            A.ado_pr_ref("https://github.com/o/r/pull/1")


class TestAdoIdentity(unittest.TestCase):
    def test_change_id_shape_and_prefix(self):
        cid = A.ado_change_id("myorg", "myproj", "myrepo", 1925)
        self.assertEqual(cid, "ADO-myorg-myproj-myrepo-1925")
        self.assertNotIn("/", cid)

    def test_change_id_never_collides_with_github(self):
        self.assertNotEqual(
            A.ado_change_id("o", "p", "r", 1),
            A.github_change_id("o", "r", 1))

    def test_change_id_delimiter_safe(self):
        # '-' inside a segment collapses so tuples cannot collide.
        a = A.ado_change_id("a-b", "c", "d", 1)
        b = A.ado_change_id("a", "b-c", "d", 1)
        self.assertNotEqual(a, b)

    def test_review_key_lossless_and_lowercased(self):
        self.assertEqual(
            A.ado_review_key("MyOrg", "MyProj", "Repo-API", 5),
            "myorg/myproj/repo-api#5")


class TestAdoParse(unittest.TestCase):
    def setUp(self):
        self.t = A.parse_ado_payload(ADO_PAYLOAD)

    def test_identity(self):
        self.assertEqual(self.t.platform, "ado")
        self.assertEqual(self.t.change_id, "ADO-cree_mfg-sicarbonite-sicarbonite-1925")
        self.assertEqual(self.t.repo_identity, "dev.azure.com/cree-mfg/sicarbonite/sicarbonite")

    def test_metadata(self):
        self.assertEqual(self.t.author, "j3blair@cree.com")
        self.assertEqual(self.t.target_branch, "main")       # refs/heads stripped
        self.assertEqual(self.t.revision, "7f40548facd39779b1dc45b27769ec51a34e45d2")
        self.assertEqual(self.t.linked_issue, "#1234")       # AB#1234 -> #1234

    def test_files_paths_stripped(self):
        self.assertEqual(len(self.t.files), 2)
        # Leading slash dropped so it reads repo-relative like GitHub.
        self.assertEqual(self.t.files[0]["path"], "sicapi/src/sicapi/routes/artifacts.py")
        self.assertIn("search", self.t.files[0]["diff"])

    def test_comments(self):
        self.assertEqual(len(self.t.existing_comments), 2)

    def test_parse_from_json_string(self):
        t = A.parse_ado_payload(json.dumps(ADO_PAYLOAD))
        self.assertEqual(t.change_id, "ADO-cree_mfg-sicarbonite-sicarbonite-1925")

    def test_fields_from_link_when_sage_absent(self):
        payload = {"pullRequestId": 42, "description": "x",
                   "files": [{"path": "/a", "diff": "d"}]}
        t = A.parse_ado_payload(payload, link=DEVAZURE)
        self.assertEqual(t.change_id, "ADO-myorg-myproj-myrepo-42")

    def test_fail_fast_no_files_no_desc(self):
        with self.assertRaises(A.AdapterParseError):
            A.parse_ado_payload({"pullRequestId": 1, "_sage":
                                 {"org": "o", "project": "p", "repo": "r"}})

    def test_fail_fast_no_identity(self):
        with self.assertRaises(A.AdapterParseError):
            A.parse_ado_payload({"description": "x", "files": [{"path": "a", "diff": "d"}]})

    def test_normalize_routes_to_ado(self):
        t = A.normalize(DEVAZURE, ADO_PAYLOAD)
        self.assertEqual(t.platform, "ado")
        self.assertEqual(t.change_id, "ADO-cree_mfg-sicarbonite-sicarbonite-1925")


if __name__ == "__main__":
    unittest.main()
