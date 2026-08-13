"""RBAC 权限测试：用户/授权/越权拒绝（auth.py 单元测试）。"""
from __future__ import annotations

from services.kb.auth import AuthStore


def test_create_user_and_grant(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    uid = auth.create_user("alice")
    assert auth.get_user(uid)["name"] == "alice"
    assert auth.get_allowed_kbs(uid) == []  # 初始无权限

    auth.grant_access(uid, "default")
    auth.grant_access(uid, "product")
    assert auth.get_allowed_kbs(uid) == ["default", "product"]
    assert auth.can_access(uid, "default") is True
    assert auth.can_access(uid, "finance") is False  # 未授权 → 拒绝


def test_revoke_access(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    uid = auth.create_user("bob")
    auth.grant_access(uid, "default")
    assert auth.can_access(uid, "default") is True

    removed = auth.revoke_access(uid, "default")
    assert removed == 1
    assert auth.can_access(uid, "default") is False  # 撤销后拒绝


def test_list_users(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    uid = auth.create_user("carol")
    auth.grant_access(uid, "default")
    users = auth.list_users()
    assert len(users) == 1
    assert users[0]["user_id"] == uid
    assert users[0]["allowed_kbs"] == ["default"]


def test_get_user_not_exist(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    assert auth.get_user("nonexistent") is None


def test_grant_idempotent(tmp_path):
    """重复授权同一 kb 不报错（INSERT OR IGNORE）。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid = auth.create_user("dan")
    auth.grant_access(uid, "default")
    auth.grant_access(uid, "default")  # 重复
    assert auth.get_allowed_kbs(uid) == ["default"]  # 仍只一条
