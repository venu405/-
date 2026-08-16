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


def test_create_admin_and_is_admin(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    admin = auth.create_user("root", role="admin")
    member = auth.create_user("alice")  # 默认 member
    assert auth.is_admin(admin) is True
    assert auth.is_admin(member) is False
    assert auth.get_user(admin)["role"] == "admin"


def test_admin_can_access_all(tmp_path):
    """admin 全通：无需授权即可访问任意知识库。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    admin = auth.create_user("root", role="admin")
    member = auth.create_user("bob")
    auth.grant_access(member, "default")

    assert auth.can_access(admin, "任意库") is True  # admin 不需要授权
    assert auth.can_access(member, "default") is True
    assert auth.can_access(member, "其他库") is False  # member 未授权则拒


def test_set_role(tmp_path):
    """提升 member 为 admin / 降级。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid = auth.create_user("carol")  # member
    assert auth.is_admin(uid) is False

    assert auth.set_role(uid, "admin") is True
    assert auth.is_admin(uid) is True

    assert auth.set_role(uid, "member") is True
    assert auth.is_admin(uid) is False


def test_invalid_role_rejected(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    try:
        auth.create_user("eve", role="superuser")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    uid = auth.create_user("frank")
    try:
        auth.set_role(uid, "root")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass
