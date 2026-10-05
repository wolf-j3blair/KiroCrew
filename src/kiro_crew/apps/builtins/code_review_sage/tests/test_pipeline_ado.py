"""Unit tests for the Azure DevOps pipeline helpers (fetch/posting specs + builders)."""
import unittest

from sage_lib import pipeline


class TestAdoSpecs(unittest.TestCase):
    def test_fetch_spec_ado_uses_mcp_not_cli(self):
        spec = pipeline.fetch_spec("ado")
        self.assertIn("azure-devops", spec)
        self.assertIn("repo_pull_request", spec)
        self.assertIn("difflib", spec)          # blob-diff assembly is spelled out
        self.assertNotIn("--hostname", spec)    # no gh host interpolation for ADO

    def test_fetch_spec_github_unchanged(self):
        spec = pipeline.fetch_spec("github")
        self.assertIn("gh", spec)
        self.assertIn("--hostname github.com", spec)

    def test_posting_spec_ado_is_threads_no_vote(self):
        spec = pipeline.posting_spec("ado")
        self.assertIn("repo_pull_request_thread_write", spec["tool"])
        self.assertIn("NEVER set a vote", spec["tool"])
        self.assertIn("threadContext", spec["anchor"])

    def test_build_comment_payload_ado_is_draft(self):
        f = {"severity": "red", "file": "a.py", "line": 7,
             "observation": "x", "consequence": "y", "suggestion": "z"}
        p = pipeline.build_comment_payload(f, "ADO-o-p-r-1", "deadbeef", platform="ado")
        self.assertEqual(p["publish"], False)   # NON-NEGOTIABLE
        self.assertEqual(p["line"], 7)
        self.assertNotIn("side", p)             # ADO has no RIGHT/LEFT side


class TestAdoThreadPayloads(unittest.TestCase):
    def _record(self):
        return {
            "revision": "abc123",
            "pending_comments": [
                {"kind": "design", "key": "design", "body": "ship summary"},
                {"kind": "finding", "key": "finding:0", "file": "a.py", "line": 10,
                 "body": "finding one"},
                {"kind": "finding", "key": "finding:1", "file": "", "line": 0,
                 "body": "unanchored finding"},
            ],
        }

    def test_ship_thread_leads_and_is_pr_level(self):
        threads = pipeline.build_ado_thread_payloads(self._record())
        self.assertEqual(threads[0]["threadContext"], None)
        self.assertEqual(threads[0]["content"], "ship summary")

    def test_anchored_finding_has_threadcontext(self):
        threads = pipeline.build_ado_thread_payloads(self._record())
        anchored = [t for t in threads if t["threadContext"]]
        self.assertEqual(len(anchored), 1)
        tc = anchored[0]["threadContext"]
        self.assertEqual(tc["filePath"], "/a.py")       # leading slash restored
        self.assertEqual(tc["rightFileStart"]["line"], 10)

    def test_unanchored_finding_folds_to_pr_level(self):
        threads = pipeline.build_ado_thread_payloads(self._record())
        bodies = [t["content"] for t in threads if t["threadContext"] is None]
        self.assertIn("unanchored finding", bodies)

    def test_all_threads_are_draft_no_vote(self):
        threads = pipeline.build_ado_thread_payloads(self._record())
        self.assertTrue(threads)
        for t in threads:
            self.assertEqual(t["publish"], False)
            self.assertEqual(t["status"], "active")

    def test_unit_count_matches_thread_count(self):
        # expected_units in the driver = len(ado_thread_payloads): 1 ship + 2 findings.
        threads = pipeline.build_ado_thread_payloads(self._record())
        self.assertEqual(len(threads), 3)


if __name__ == "__main__":
    unittest.main()
