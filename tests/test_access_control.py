import tempfile
import unittest
from pathlib import Path

from rag_demo.access_control import AccessControlError, AccessPolicyStore


class AccessPolicyStoreTest(unittest.TestCase):
    def test_server_side_intersection_excludes_requested_protected_source(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AccessPolicyStore(Path(directory) / "access.json")
            store.set_policy("finance-policy", ["finance"])
            finance = store.resolve_principal("finance-demo")
            staff = store.resolve_principal("staff-demo")

            allowed, denied = store.allowed_source_ids(
                finance,
                ["public-guide", "finance-policy"],
                ["public-guide", "finance-policy"],
            )
            staff_allowed, staff_denied = store.allowed_source_ids(
                staff,
                ["public-guide", "finance-policy"],
                ["public-guide", "finance-policy"],
            )

        self.assertEqual(allowed, ["public-guide", "finance-policy"])
        self.assertEqual(denied, [])
        self.assertEqual(staff_allowed, ["public-guide"])
        self.assertEqual(staff_denied, ["finance-policy"])

    def test_omitted_source_list_is_also_filtered_and_admin_can_read_all(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AccessPolicyStore(Path(directory) / "access.json")
            store.set_policy("hr-only", ["hr"])
            staff = store.resolve_principal("staff-demo")
            admin = store.resolve_principal("admin-demo")

            allowed, denied = store.allowed_source_ids(staff, ["public", "hr-only"])
            admin_allowed, admin_denied = store.allowed_source_ids(admin, ["public", "hr-only"])

        self.assertEqual(allowed, ["public"])
        self.assertEqual(denied, ["hr-only"])
        self.assertEqual(admin_allowed, ["public", "hr-only"])
        self.assertEqual(admin_denied, [])

    def test_policy_is_persisted_and_requires_defined_role(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "access.json"
            store = AccessPolicyStore(path)
            saved = store.set_policy("it-runbook", ["it"])
            reloaded = AccessPolicyStore(path)

            with self.assertRaisesRegex(AccessControlError, "未定義"):
                reloaded.set_policy("it-runbook", ["does-not-exist"])

            restored = reloaded.policy_for_source("it-runbook")

        self.assertEqual(saved["allowed_roles"], ["it"])
        self.assertFalse(restored["inherited_default"])
        self.assertEqual(restored["allowed_roles"], ["it"])
