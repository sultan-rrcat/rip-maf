import logging
import os
from contextlib import contextmanager

from dotenv import find_dotenv, load_dotenv
from psycopg2.pool import ThreadedConnectionPool

load_dotenv(find_dotenv(usecwd=True))

logger = logging.getLogger(__name__)

_pool: ThreadedConnectionPool | None = None


def _db_params() -> dict:
    """Resolve DB params from env with Settings fallback; normalize host."""
    from rip_maf.core.config import _normalize_db_host, settings

    host = os.getenv("DB_HOST", settings.db_host)
    port = os.getenv("DB_PORT", str(settings.db_port))
    name = os.getenv("DB_NAME", settings.db_name)
    user = os.getenv("DB_USER", settings.db_user)
    password = os.getenv("DB_PASSWORD", settings.db_password)
    try:
        timeout = int(os.getenv("DB_CONNECT_TIMEOUT_S", str(settings.db_connect_timeout_s)))
    except ValueError:
        timeout = settings.db_connect_timeout_s
    return {
        "host": _normalize_db_host(host or ""),
        "port": port,
        "dbname": name,
        "user": user,
        "password": password,
        "connect_timeout": timeout,
    }


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(minconn=4, maxconn=20, **_db_params())
    return _pool


@contextmanager
def pg_connection():
    conn = None
    pool = None
    try:
        pool = _get_pool()
        conn = pool.getconn()
        yield conn
        conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        params = _db_params()
        logger.error("Error connecting to database: %s", e)
        logger.error(
            "DB Credentials -> "
            f"Host: {params['host']}, "
            f"Port: {params['port']}, "
            f"Name: {params['dbname']}, "
            f"User: {params['user']}, "
            "Password: ***"
        )
        raise

    finally:
        if pool is not None and conn is not None:
            pool.putconn(conn)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            _pool.closeall()
        except Exception:
            logger.warning("DB pool close failed", exc_info=True)
        finally:
            _pool = None
