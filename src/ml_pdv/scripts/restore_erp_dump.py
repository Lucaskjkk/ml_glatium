"""Restore an ERP pg_dump into local Postgres (Docker or Homebrew)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from ml_pdv.config import get_settings
from ml_pdv.database.connection import reset_engines, resolve_erp_dump_path
from ml_pdv.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

COMPOSE_SERVICE = "postgres-erp"


def _parse_db_url(url: str) -> dict[str, str | int | None]:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.path:
        raise ValueError(f"Invalid ERP_DATABASE_URL: {url}")
    db = parsed.path.lstrip("/")
    return {
        "user": parsed.username or "erp",
        "password": parsed.password,
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "database": db,
    }


def _detect_dump_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".sql", ".txt"}:
        return "plain"
    if suffix in {".dump", ".backup", ".fc"}:
        return "custom"
    header = path.read_bytes()[:5]
    if header.startswith(b"PGDMP"):
        return "custom"
    return "plain"


def sanitize_plain_sql(sql: str, *, owner: str = "erp") -> str:
    """Make pg_dump 17/18 plain SQL restorable on local Postgres."""
    lines: list[str] = []
    for line in sql.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("\\restrict") or stripped.startswith("\\unrestrict"):
            continue
        if re.match(r"SET\s+transaction_timeout\s*=", stripped, flags=re.IGNORECASE):
            continue
        if re.match(r"CREATE SCHEMA\s+public\s*;", stripped, flags=re.IGNORECASE):
            continue
        line = re.sub(
            r"CREATE SCHEMA\s+(\w+)\s*;",
            r"CREATE SCHEMA IF NOT EXISTS \1;",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"OWNER TO\s+postgres\b",
            f"OWNER TO {owner}",
            line,
            flags=re.IGNORECASE,
        )
        lines.append(line)
    return "".join(lines)


def _run(cmd: list[str], *, input_text: str | None = None, env: dict[str, str] | None = None) -> None:
    logger.info("shell_cmd", cmd=" ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        input=input_text,
        env=merged,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _psql_available() -> bool:
    return shutil.which("psql") is not None


def _wait_for_postgres_docker(user: str, db: str, timeout: int = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                COMPOSE_SERVICE,
                "pg_isready",
                "-U",
                user,
                "-d",
                db,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(2)
    raise TimeoutError(f"Postgres service '{COMPOSE_SERVICE}' not ready after {timeout}s")


def _recreate_database_local(db: dict[str, str | int | None]) -> None:
    user = str(db["user"])
    database = str(db["database"])
    host = str(db["host"])
    port = str(db["port"])
    password = db["password"]
    env = {"PGPASSWORD": str(password)} if password else None

    admin_url_user = os.environ.get("USER") or user
    # Prefer connecting as current OS superuser to admin DB when available.
    for admin_user in (admin_url_user, "postgres", user):
        try:
            _run(
                [
                    "psql",
                    "-h",
                    host,
                    "-p",
                    port,
                    "-U",
                    admin_user,
                    "-d",
                    "postgres",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-c",
                    (
                        f"DO $$ BEGIN "
                        f"IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{user}') THEN "
                        f"CREATE ROLE {user} LOGIN PASSWORD '{password or user}' SUPERUSER; "
                        f"END IF; END $$;"
                    ),
                ],
                env=env if admin_user == user else None,
            )
            _run(
                [
                    "psql",
                    "-h",
                    host,
                    "-p",
                    port,
                    "-U",
                    admin_user,
                    "-d",
                    "postgres",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-c",
                    (
                        f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        f"WHERE datname = '{database}' AND pid <> pg_backend_pid(); "
                        f'DROP DATABASE IF EXISTS "{database}"; '
                        f'CREATE DATABASE "{database}" OWNER "{user}";'
                    ),
                ],
                env=env if admin_user == user else None,
            )
            return
        except RuntimeError:
            continue
    raise RuntimeError("Could not recreate database with local psql (tried local superusers).")


def restore_via_local_psql(dump_path: Path) -> None:
    settings = get_settings()
    db = _parse_db_url(settings.erp_database_url)
    user = str(db["user"])
    database = str(db["database"])
    host = str(db["host"])
    port = str(db["port"])
    password = db["password"]
    env = {"PGPASSWORD": str(password)} if password else None
    fmt = _detect_dump_format(dump_path)

    _recreate_database_local(db)

    if fmt == "custom":
        _run(
            [
                "pg_restore",
                "-h",
                host,
                "-p",
                port,
                "-U",
                user,
                "-d",
                database,
                "--no-owner",
                "--no-acl",
                "--verbose",
                str(dump_path),
            ],
            env=env,
        )
        return

    sanitized = sanitize_plain_sql(
        dump_path.read_text(encoding="utf-8", errors="replace"),
        owner=user,
    )
    _run(
        [
            "psql",
            "-h",
            host,
            "-p",
            port,
            "-U",
            user,
            "-d",
            database,
            "-v",
            "ON_ERROR_STOP=1",
        ],
        input_text=sanitized,
        env=env,
    )


def restore_via_docker(dump_path: Path) -> None:
    settings = get_settings()
    db = _parse_db_url(settings.erp_database_url)
    user = str(db["user"])
    database = str(db["database"])
    fmt = _detect_dump_format(dump_path)

    _run(["docker", "compose", "up", "-d", COMPOSE_SERVICE])
    _wait_for_postgres_docker(user=user, db="postgres")
    container_path = f"/dumps/{dump_path.name}"

    for sql in (
        (
            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{database}' AND pid <> pg_backend_pid();"
        ),
        f'DROP DATABASE IF EXISTS "{database}";',
        f'CREATE DATABASE "{database}" OWNER "{user}";',
    ):
        _run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                COMPOSE_SERVICE,
                "psql",
                "-U",
                user,
                "-d",
                "postgres",
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                sql,
            ]
        )

    if fmt == "custom":
        _run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                COMPOSE_SERVICE,
                "pg_restore",
                "-U",
                user,
                "-d",
                database,
                "--no-owner",
                "--no-acl",
                "--verbose",
                container_path,
            ]
        )
        return

    sanitized = sanitize_plain_sql(
        dump_path.read_text(encoding="utf-8", errors="replace"),
        owner=user,
    )
    _run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            COMPOSE_SERVICE,
            "psql",
            "-U",
            user,
            "-d",
            database,
            "-v",
            "ON_ERROR_STOP=1",
        ],
        input_text=sanitized,
    )


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    if settings.erp_source_mode != "dump":
        print(
            f"ERP_SOURCE_MODE={settings.erp_source_mode}. "
            "Set ERP_SOURCE_MODE=dump to restore from file.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    dump_path = resolve_erp_dump_path()
    if not dump_path.exists():
        print(
            f"Dump not found: {dump_path}\n"
            "Place your pg_dump file there (or set ERP_DUMP_PATH), then re-run:\n"
            "  uv run restore-erp-dump",
            file=sys.stderr,
        )
        raise SystemExit(1)

    logger.info(
        "restore_erp_dump_start",
        dump=str(dump_path),
        target=settings.erp_database_url,
        mode=settings.erp_source_mode,
    )
    try:
        if _docker_available():
            restore_via_docker(dump_path)
        elif _psql_available():
            logger.info("docker_missing_using_local_psql")
            restore_via_local_psql(dump_path)
        else:
            print(
                "Neither docker nor psql found. Install Docker Desktop or PostgreSQL client tools.",
                file=sys.stderr,
            )
            raise SystemExit(1)
    except Exception as exc:
        logger.error("restore_erp_dump_failed", error=str(exc))
        print(f"Restore failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    reset_engines()
    logger.info("restore_erp_dump_done", dump=str(dump_path))
    print(f"Restored {dump_path} → {settings.erp_database_url}")
    print("Next: uv run introspect-erp")


if __name__ == "__main__":
    main()
