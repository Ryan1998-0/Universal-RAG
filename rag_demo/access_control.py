"""Local access-policy support for the RAG demo.

The policy decision is deliberately separate from the browser.  A caller may
ask to search a source, but the server intersects that request with the
sources that the active principal is allowed to read before retrieval starts.

This module supplies *demo identities* for the local workbench.  They are
useful for exercising role policies, but are not a replacement for OIDC/JWT
authentication in :mod:`rag_demo.production.auth`.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCESS_POLICY_PATH = Path(
    os.getenv("RAG_ACCESS_POLICY_PATH", str(PROJECT_ROOT / ".local" / "access_policy.json"))
).expanduser()
ROLE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
PRINCIPAL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

ALL_AUTHENTICATED_ROLE = "all"
ADMIN_ROLE = "admin"

DEFAULT_ROLES = (
    {"id": ADMIN_ROLE, "label": "系統管理員", "description": "可管理權限、文件與資料夾"},
    {"id": "hr", "label": "人資", "description": "可查詢獲授權的人資文件"},
    {"id": "finance", "label": "財務", "description": "可查詢獲授權的財務文件"},
    {"id": "it", "label": "資訊", "description": "可查詢獲授權的資訊文件"},
    {"id": "staff", "label": "一般同仁", "description": "可查詢全員公開文件"},
)
DEFAULT_PRINCIPALS = (
    {"id": "admin-demo", "label": "管理員（Demo）", "roles": [ADMIN_ROLE]},
    {"id": "hr-demo", "label": "人資（Demo）", "roles": ["hr", "staff"]},
    {"id": "finance-demo", "label": "財務（Demo）", "roles": ["finance", "staff"]},
    {"id": "it-demo", "label": "資訊（Demo）", "roles": ["it", "staff"]},
    {"id": "staff-demo", "label": "一般同仁（Demo）", "roles": ["staff"]},
)


class AccessControlError(ValueError):
    """Raised for a malformed or unauthorized local access-policy action."""


class AccessPolicyStore:
    """A small, atomically persisted RBAC policy registry for the local demo."""

    def __init__(self, path: Path = DEFAULT_ACCESS_POLICY_PATH):
        self.path = Path(path)
        self._lock = threading.RLock()

    def list_roles(self) -> list[dict]:
        return [dict(role) for role in self._load()["roles"]]

    def list_principals(self) -> list[dict]:
        return [dict(principal) for principal in self._load()["principals"]]

    def resolve_principal(self, principal_id: Optional[str]) -> dict:
        clean_id = str(principal_id or "admin-demo").strip() or "admin-demo"
        for principal in self._load()["principals"]:
            if principal["id"] == clean_id:
                return dict(principal)
        raise AccessControlError("找不到指定的存取身分。")

    def is_admin(self, principal: dict) -> bool:
        return ADMIN_ROLE in set(principal.get("roles") or ())

    def policy_for_source(self, source_id: str) -> dict:
        clean_source_id = _clean_source_id(source_id)
        policies = self._load()["source_policies"]
        roles = policies.get(clean_source_id, [ALL_AUTHENTICATED_ROLE])
        return {
            "source_id": clean_source_id,
            "allowed_roles": list(roles),
            "inherited_default": clean_source_id not in policies,
        }

    def set_policy(self, source_id: str, allowed_roles: Sequence[object]) -> dict:
        clean_source_id = _clean_source_id(source_id)
        payload = self._load()
        valid_roles = {role["id"] for role in payload["roles"]}
        clean_roles = _clean_allowed_roles(allowed_roles, valid_roles)
        payload["source_policies"][clean_source_id] = clean_roles
        self._save(payload)
        return {
            "source_id": clean_source_id,
            "allowed_roles": clean_roles,
            "inherited_default": False,
        }

    def allowed_source_ids(
        self,
        principal: dict,
        available_source_ids: Iterable[object],
        requested_source_ids: Optional[Sequence[object]] = None,
    ) -> tuple[list[str], list[str]]:
        """Return (allowed, denied) without trusting a client source list."""
        available = list(dict.fromkeys(
            _clean_source_id(source_id) for source_id in available_source_ids if str(source_id).strip()
        ))
        available_set = set(available)
        requested = None
        if requested_source_ids is not None:
            requested = list(dict.fromkeys(
                _clean_source_id(source_id)
                for source_id in requested_source_ids
                if str(source_id).strip()
            ))
        candidates = available if requested is None else [item for item in requested if item in available_set]
        policies = self._load()["source_policies"]
        principal_roles = set(principal.get("roles") or ())
        is_admin = ADMIN_ROLE in principal_roles
        allowed = []
        denied = []
        for source_id in candidates:
            policy_roles = set(policies.get(source_id, [ALL_AUTHENTICATED_ROLE]))
            if is_admin or ALL_AUTHENTICATED_ROLE in policy_roles or principal_roles.intersection(policy_roles):
                allowed.append(source_id)
            else:
                denied.append(source_id)
        return allowed, denied

    def _load(self) -> dict:
        with self._lock:
            if not self.path.is_file():
                return _default_policy_payload()
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AccessControlError("權限設定檔損壞，無法讀取。") from exc
            return _normalise_policy_payload(raw)

    def _save(self, payload: dict) -> None:
        with self._lock:
            normalized = _normalise_policy_payload(payload)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as temporary:
                json.dump(normalized, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            temporary_path.replace(self.path)


def _default_policy_payload() -> dict:
    return {
        "schema_version": 1,
        "roles": [dict(role) for role in DEFAULT_ROLES],
        "principals": [dict(principal) for principal in DEFAULT_PRINCIPALS],
        # Existing documents remain usable until an administrator explicitly
        # classifies them.  New protected documents must receive a policy.
        "source_policies": {},
    }


def _normalise_policy_payload(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise AccessControlError("權限設定檔格式錯誤。")
    raw_roles = raw.get("roles")
    raw_principals = raw.get("principals")
    raw_policies = raw.get("source_policies", {})
    if not isinstance(raw_roles, list) or not isinstance(raw_principals, list) or not isinstance(raw_policies, dict):
        raise AccessControlError("權限設定檔格式錯誤。")

    roles = []
    role_ids = set()
    for item in raw_roles:
        if not isinstance(item, dict):
            raise AccessControlError("權限角色格式錯誤。")
        role_id = _clean_role_id(item.get("id"))
        if role_id in role_ids:
            raise AccessControlError("權限角色不得重複。")
        role_ids.add(role_id)
        roles.append({
            "id": role_id,
            "label": _clean_label(item.get("label"), "角色"),
            "description": str(item.get("description") or "").strip()[:160],
        })
    if ADMIN_ROLE not in role_ids:
        raise AccessControlError("權限設定必須保留 admin 角色。")

    principals = []
    principal_ids = set()
    for item in raw_principals:
        if not isinstance(item, dict):
            raise AccessControlError("存取身分格式錯誤。")
        principal_id = _clean_principal_id(item.get("id"))
        if principal_id in principal_ids:
            raise AccessControlError("存取身分不得重複。")
        principal_ids.add(principal_id)
        assigned_roles = _clean_allowed_roles(item.get("roles") or [], role_ids, allow_all=False)
        if not assigned_roles:
            raise AccessControlError("每個存取身分至少要有一個角色。")
        principals.append({
            "id": principal_id,
            "label": _clean_label(item.get("label"), "使用者"),
            "roles": assigned_roles,
        })
    if not any(ADMIN_ROLE in item["roles"] for item in principals):
        raise AccessControlError("權限設定必須保留至少一個管理員。")

    policies = {}
    for source_id, assigned_roles in raw_policies.items():
        policies[_clean_source_id(source_id)] = _clean_allowed_roles(assigned_roles, role_ids)
    return {
        "schema_version": 1,
        "roles": roles,
        "principals": principals,
        "source_policies": policies,
    }


def _clean_role_id(value: object) -> str:
    role_id = str(value or "").strip()
    if not ROLE_ID_PATTERN.fullmatch(role_id):
        raise AccessControlError("角色 ID 格式錯誤。")
    return role_id


def _clean_principal_id(value: object) -> str:
    principal_id = str(value or "").strip()
    if not PRINCIPAL_ID_PATTERN.fullmatch(principal_id):
        raise AccessControlError("存取身分 ID 格式錯誤。")
    return principal_id


def _clean_source_id(value: object) -> str:
    source_id = str(value or "").strip()
    if not source_id or len(source_id) > 240:
        raise AccessControlError("文件來源 ID 格式錯誤。")
    return source_id


def _clean_label(value: object, fallback: str) -> str:
    label = " ".join(str(value or "").split())
    if not label:
        label = fallback
    return label[:80]


def _clean_allowed_roles(
    values: object,
    valid_roles: set[str],
    *,
    allow_all: bool = True,
) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        raise AccessControlError("允許角色必須是陣列。")
    result = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    allowed = set(valid_roles)
    if allow_all:
        allowed.add(ALL_AUTHENTICATED_ROLE)
    if not result or any(role not in allowed for role in result):
        raise AccessControlError("權限角色包含未定義項目，或未指定任何可存取角色。")
    return result
