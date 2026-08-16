"""用户与权限管理（RBAC）——SQLite 存储（P3 §3.4 / v3 §6.1）。

数据模型：
  users(user_id PK, name, role, created_at)   role: member | admin
  kb_access(user_id, kb_id)                   -- member 可访问的知识库

权限模型（本次审查修复后）：
  - admin：全通（可访问任意知识库 + 管理用户/角色）
  - member：只能访问 kb_access 里被授权的知识库（读写均可）
  - 越权防护在入口校验（检索层之前拦截），绝不放检索后再补救

注：当前为"指定库 + 校验"最小版（kb_id 单库）。
    v3 §6.1 的"跨库检索 where $in"为增强项，需 BM25 改多库索引，留作后续。
    生产环境 user_id 应换 token 鉴权（当前 demo 级，请求直传 user_id）。
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuthStore:
    """用户与知识库访问权限（SQLite）。"""

    VALID_ROLES = {"member", "admin"}

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # P3：WAL 模式 + busy_timeout，避免并发写时 "database is locked"
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT 'member',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS kb_access (
                user_id TEXT NOT NULL,
                kb_id   TEXT NOT NULL,
                PRIMARY KEY (user_id, kb_id)
            );
            """
        )
        # 迁移：旧 users 表补 role 列（默认 member）
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(users)").fetchall()]
        if "role" not in cols:
            self._conn.execute(
                "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'member'"
            )
        self._conn.commit()

    # ---------- 用户 ----------
    def create_user(self, name: str, role: str = "member") -> str:
        """新建用户，返回 user_id。role: member | admin。"""
        if role not in self.VALID_ROLES:
            raise ValueError(f"非法角色: {role}")
        uid = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO users(user_id, name, role) VALUES(?, ?, ?)", (uid, name, role)
        )
        self._conn.commit()
        logger.info("新建用户 name=%s id=%s role=%s", name, uid, role)
        return uid

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT user_id, name, role FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "user_id": row[0],
            "name": row[1],
            "role": row[2],
            "allowed_kbs": self.get_allowed_kbs(user_id),
        }

    def list_users(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT user_id, name, role, created_at FROM users ORDER BY created_at"
        ).fetchall()
        result = []
        for uid, name, role, created in rows:
            result.append(
                {
                    "user_id": uid,
                    "name": name,
                    "role": role,
                    "created_at": created,
                    "allowed_kbs": self.get_allowed_kbs(uid),
                }
            )
        return result

    def is_admin(self, user_id: str) -> bool:
        """是否为管理员（admin 全通，可管理用户/角色）。"""
        row = self._conn.execute(
            "SELECT role FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        return row is not None and row[0] == "admin"

    def set_role(self, user_id: str, role: str) -> bool:
        """提升/降级用户角色。用户不存在返回 False。"""
        if role not in self.VALID_ROLES:
            raise ValueError(f"非法角色: {role}")
        cur = self._conn.execute(
            "UPDATE users SET role=? WHERE user_id=?", (role, user_id)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ---------- 权限 ----------
    def grant_access(self, user_id: str, kb_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO kb_access(user_id, kb_id) VALUES(?, ?)",
            (user_id, kb_id),
        )
        self._conn.commit()

    def revoke_access(self, user_id: str, kb_id: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM kb_access WHERE user_id=? AND kb_id=?", (user_id, kb_id)
        )
        self._conn.commit()
        return cur.rowcount

    def get_allowed_kbs(self, user_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT kb_id FROM kb_access WHERE user_id=?", (user_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def can_access(self, user_id: str, kb_id: str) -> bool:
        """校验用户是否能访问指定知识库。admin 全通。"""
        if self.is_admin(user_id):
            return True
        row = self._conn.execute(
            "SELECT 1 FROM kb_access WHERE user_id=? AND kb_id=? LIMIT 1",
            (user_id, kb_id),
        ).fetchone()
        return row is not None
