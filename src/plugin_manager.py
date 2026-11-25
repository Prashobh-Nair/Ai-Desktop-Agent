"""
On-Demand Persistent Plugin Manager for JARVIS.

- SQLite registry (`memory/plugins.db`) stores plugin metadata, methods, capabilities,
  auth/connection status, and config locations across restarts and updates.
- Windows DPAPI (`crypt32.CryptProtectData`) encrypts credentials/tokens separately
  in `memory/secrets/<plugin_id>.dpapi` — never plaintext in SQLite, logs, or LLM prompts.
- Supports REST APIs, dynamic stdio MCP servers, and custom Python plugins (`plugins/*.py`).
- Zero startup wizards or unprompted suggestions: plugins are created strictly on demand.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import getpass
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
DB_PATH = MEMORY_DIR / "plugins.db"
SECRETS_DIR = MEMORY_DIR / "secrets"
CUSTOM_PLUGINS_DIR = PROJECT_ROOT / "plugins"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or "custom_plugin"


# ---------------------------------------------------------------------------
# 1. Secure Credential Storage (Windows DPAPI — separate from SQLite)
# ---------------------------------------------------------------------------

class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


class SecretStore:
    """Per-user Windows DPAPI encrypted credential store (never plaintext in SQLite)."""

    def __init__(self, secrets_dir: Path = SECRETS_DIR):
        self.secrets_dir = secrets_dir
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, plugin_id: str) -> Path:
        return self.secrets_dir / f"{_slugify(plugin_id)}.dpapi"

    @staticmethod
    def _dpapi_protect(raw: bytes) -> bytes:
        if os.name != "nt":
            raise RuntimeError("Windows DPAPI is only available on Windows.")
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        in_buf = (ctypes.c_byte * len(raw)).from_buffer_copy(raw)
        in_blob = _DATA_BLOB(len(raw), in_buf)
        out_blob = _DATA_BLOB()
        if not crypt32.CryptProtectData(
            ctypes.byref(in_blob), "JarvisPluginSecret", None, None, None, 0, ctypes.byref(out_blob)
        ):
            raise OSError("CryptProtectData failed.")
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(out_blob.pbData)

    @staticmethod
    def _dpapi_unprotect(enc: bytes) -> bytes:
        if os.name != "nt":
            raise RuntimeError("Windows DPAPI is only available on Windows.")
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        in_buf = (ctypes.c_byte * len(enc)).from_buffer_copy(enc)
        in_blob = _DATA_BLOB(len(enc), in_buf)
        out_blob = _DATA_BLOB()
        if not crypt32.CryptUnprotectData(
            ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)
        ):
            raise OSError("CryptUnprotectData failed.")
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(out_blob.pbData)

    def set_secret(self, plugin_id: str, data: Dict[str, str]) -> Path:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        enc = self._dpapi_protect(raw)
        path = self._path(plugin_id)
        path.write_bytes(enc)
        return path

    def get_secret(self, plugin_id: str) -> Dict[str, str]:
        path = self._path(plugin_id)
        if not path.exists():
            return {}
        try:
            dec = self._dpapi_unprotect(path.read_bytes())
            val = json.loads(dec.decode("utf-8"))
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}

    def has_secret(self, plugin_id: str) -> bool:
        return self._path(plugin_id).exists()

    def delete_secret(self, plugin_id: str) -> None:
        path = self._path(plugin_id)
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# 2. Persistent SQLite Plugin Registry
# ---------------------------------------------------------------------------

class PluginRegistry:
    """SQLite-backed permanent plugin metadata registry (zero secrets stored here)."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plugins (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    method TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    auth_status TEXT NOT NULL,
                    config_location TEXT NOT NULL,
                    connection_status TEXT NOT NULL,
                    install_info TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "method": row["method"],
            "config": json.loads(row["config_json"]),
            "capabilities": json.loads(row["capabilities_json"]),
            "auth_status": row["auth_status"],
            "config_location": row["config_location"],
            "connection_status": row["connection_status"],
            "install_info": row["install_info"],
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert(
        self,
        plugin_id: str,
        name: str,
        method: str,
        config: Dict[str, Any],
        capabilities: List[str],
        auth_status: str = "connected",
        config_location: str = "",
        connection_status: str = "healthy",
        install_info: str = "on-demand",
        enabled: bool = True,
    ) -> Dict[str, Any]:
        pid = _slugify(plugin_id)
        now = _utc_now()
        existing = self.get(pid)
        created = existing["created_at"] if existing else now
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plugins (
                    id, name, method, config_json, capabilities_json,
                    auth_status, config_location, connection_status,
                    install_info, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    method=excluded.method,
                    config_json=excluded.config_json,
                    capabilities_json=excluded.capabilities_json,
                    auth_status=excluded.auth_status,
                    config_location=excluded.config_location,
                    connection_status=excluded.connection_status,
                    install_info=excluded.install_info,
                    enabled=excluded.enabled,
                    updated_at=excluded.updated_at
                """,
                (
                    pid, name, method,
                    json.dumps(config, ensure_ascii=False),
                    json.dumps(capabilities, ensure_ascii=False),
                    auth_status, config_location, connection_status,
                    install_info, 1 if enabled else 0, created, now,
                ),
            )
        return self.get(pid)  # type: ignore[return-value]

    def get(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        pid = _slugify(plugin_id)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM plugins WHERE id = ?", (pid,)).fetchone()
            return self._row_to_dict(row) if row else None

    def list_all(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM plugins" + (" WHERE enabled = 1" if enabled_only else "") + " ORDER BY name"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def set_enabled(self, plugin_id: str, enabled: bool) -> bool:
        pid = _slugify(plugin_id)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE plugins SET enabled = ?, connection_status = ?, updated_at = ? WHERE id = ?",
                (1 if enabled else 0, "healthy" if enabled else "disabled", _utc_now(), pid),
            )
            return cur.rowcount > 0

    def update_status(self, plugin_id: str, auth_status: str, connection_status: str) -> None:
        pid = _slugify(plugin_id)
        with self._connect() as conn:
            conn.execute(
                "UPDATE plugins SET auth_status = ?, connection_status = ?, updated_at = ? WHERE id = ?",
                (auth_status, connection_status, _utc_now(), pid),
            )

    def remove(self, plugin_id: str) -> bool:
        pid = _slugify(plugin_id)
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM plugins WHERE id = ?", (pid,))
            return cur.rowcount > 0


# ---------------------------------------------------------------------------
# 3. Integration Blueprints (Knowledge used ONLY when user asks to create)
# ---------------------------------------------------------------------------

INTEGRATION_BLUEPRINTS: Dict[str, Dict[str, Any]] = {
    "github": {
        "name": "GitHub",
        "method": "rest_api",
        "auth_prompt": "GitHub Personal Access Token (with repo/read:user scope)",
        "auth_guide": "Create a fine-grained or classic token at https://github.com/settings/tokens",
        "secret_key": "token",
        "config": {
            "base_url": "https://api.github.com",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
            "extra_headers": {"Accept": "application/vnd.github+json", "User-Agent": "Jarvis-Agent/1.0"},
            "test_endpoint": "/user",
        },
        "capabilities": ["check_profile", "latest_commits", "list_repos", "list_notifications", "create_issue"],
        "sensitive_actions": ["create_issue"],
    },
    "gmail": {
        "name": "Gmail",
        "method": "rest_api",
        "auth_prompt": "Gmail OAuth2 Access Token (readonly or modify scope)",
        "auth_guide": "Generate an OAuth token via Google OAuth Playground (https://developers.google.com/oauthplayground) with https://www.googleapis.com/auth/gmail.readonly",
        "secret_key": "token",
        "config": {
            "base_url": "https://gmail.googleapis.com/gmail/v1",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
            "extra_headers": {"Accept": "application/json"},
            "test_endpoint": "/users/me/profile",
        },
        "capabilities": ["list_emails", "summarize_emails", "search_emails", "send_email"],
        "sensitive_actions": ["send_email"],
    },
    "notion": {
        "name": "Notion",
        "method": "rest_api",
        "auth_prompt": "Notion Internal Integration Secret (starts with ntn_ or secret_)",
        "auth_guide": "Create an integration at https://www.notion.so/my-integrations and share target pages with it.",
        "secret_key": "token",
        "config": {
            "base_url": "https://api.notion.com/v1",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
            "extra_headers": {"Notion-Version": "2022-06-28", "Content-Type": "application/json"},
            "test_endpoint": "/users/me",
        },
        "capabilities": ["search_pages", "read_page", "create_page"],
        "sensitive_actions": ["create_page"],
    },
    "spotify": {
        "name": "Spotify",
        "method": "rest_api",
        "auth_prompt": "Spotify OAuth Access Token",
        "auth_guide": "Get a user token from https://developer.spotify.com with user-read-currently-playing and playlist-read-private.",
        "secret_key": "token",
        "config": {
            "base_url": "https://api.spotify.com/v1",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
            "extra_headers": {"Accept": "application/json"},
            "test_endpoint": "/me",
        },
        "capabilities": ["current_track", "recent_tracks", "list_playlists"],
        "sensitive_actions": [],
    },
}


# ---------------------------------------------------------------------------
# 4. Dynamic Runners (REST API, MCP Stdio, Custom Python)
# ---------------------------------------------------------------------------

def _http_json(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:250]
        return e.code, {"error": err_body}


class MCPStdioClient:
    """Lightweight JSON-RPC 2.0 stdio client for dynamic MCP server integrations."""

    def __init__(self, command: str, args: List[str], env_secrets: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args
        self.env_secrets = env_secrets or {}

    def _request(self, rpc_method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 12.0) -> Any:
        env = os.environ.copy()
        env.update(self.env_secrets)
        proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            init_msg = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "Jarvis", "version": "1.0"},
                },
            })
            notif_msg = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
            call_msg = json.dumps({"jsonrpc": "2.0", "id": 2, "method": rpc_method, "params": params or {}})
            payload = f"{init_msg}\n{notif_msg}\n{call_msg}\n"
            stdout, stderr = proc.communicate(input=payload, timeout=timeout)
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("id") == 2:
                        if "error" in msg:
                            raise RuntimeError(str(msg["error"]))
                        return msg.get("result")
                except json.JSONDecodeError:
                    continue
            raise RuntimeError(stderr.strip()[:200] or "No valid MCP response received.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def list_tools(self) -> List[str]:
        res = self._request("tools/list", {})
        tools = res.get("tools", []) if isinstance(res, dict) else []
        return [t.get("name", "") for t in tools if t.get("name")]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        res = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        if isinstance(res, dict) and "content" in res:
            parts = [c.get("text", "") for c in res["content"] if isinstance(c, dict)]
            return "\n".join(p for p in parts if p).strip() or json.dumps(res)
        return json.dumps(res)


# ---------------------------------------------------------------------------
# 5. PluginManager (On-Demand Creation, Discovery, Health, Execution)
# ---------------------------------------------------------------------------

class PluginManager:
    """Orchestrates on-demand plugin creation, encrypted secrets, and execution."""

    def __init__(self, db_path: Path = DB_PATH, secrets_dir: Path = SECRETS_DIR, plugins_dir: Path = CUSTOM_PLUGINS_DIR):
        self.registry = PluginRegistry(db_path=db_path)
        self.secrets = SecretStore(secrets_dir=secrets_dir)
        self.plugins_dir = plugins_dir
        self.discover_custom_plugins()

    def discover_custom_plugins(self) -> int:
        """Auto-discovers any custom Python plugins placed in `plugins/*.py`."""
        if not self.plugins_dir.exists():
            return 0
        discovered = 0
        for py_file in self.plugins_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            pid = _slugify(py_file.stem)
            if self.registry.get(pid):
                continue
            mod = self._load_python_module(py_file)
            if mod is None:
                continue
            name = getattr(mod, "PLUGIN_NAME", pid.title())
            caps = list(getattr(mod, "CAPABILITIES", ["run"]))
            self.registry.upsert(
                plugin_id=pid,
                name=name,
                method="custom_python",
                config={"module_path": str(py_file)},
                capabilities=caps,
                auth_status="connected" if not getattr(mod, "REQUIRES_AUTH", False) else "needs_auth",
                config_location=str(py_file),
                connection_status="healthy",
                install_info="auto-discovered",
            )
            discovered += 1
        return discovered

    @staticmethod
    def _load_python_module(path: Path):
        try:
            spec = importlib.util.spec_from_file_location(f"jarvis_plugin_{path.stem}", str(path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        except Exception:
            pass
        return None

    def find_plugin(self, name_or_text: str) -> Optional[Dict[str, Any]]:
        """Fast SQLite lookup matching a plugin id or display name."""
        low = (name_or_text or "").strip().lower()
        if not low:
            return None
        direct = self.registry.get(low)
        if direct:
            return direct
        for p in self.registry.list_all():
            if p["id"] in low or p["name"].lower() in low:
                return p
        return None

    def list_plugins_summary(self) -> str:
        plugins = self.registry.list_all()
        if not plugins:
            return "You don't have any plugins connected yet. Just say 'create a plugin for' followed by the app name whenever you want one."
        parts = []
        for p in plugins:
            state = "enabled" if p["enabled"] else "disabled"
            parts.append(f"{p['name']} ({p['method']}, {state}, auth: {p['auth_status']})")
        return "Connected plugins: " + "; ".join(parts) + "."

    def capabilities_summary(self, plugin_name: str) -> str:
        p = self.find_plugin(plugin_name)
        if not p:
            return f"I don't have a {plugin_name} integration set up yet. Want to configure it?"
        caps = ", ".join(p["capabilities"]) if p["capabilities"] else "general queries"
        return f"With {p['name']}, I can do: {caps}."

    def test_connection(self, plugin_id: str) -> Tuple[bool, str]:
        p = self.registry.get(plugin_id)
        if not p:
            return False, f"Plugin '{plugin_id}' is not registered."
        secrets = self.secrets.get_secret(plugin_id)
        method = p["method"]
        cfg = p["config"]

        if method == "rest_api":
            base = (cfg.get("base_url") or "").rstrip("/")
            test_ep = cfg.get("test_endpoint") or ""
            headers = dict(cfg.get("extra_headers") or {})
            token = secrets.get("token") or secrets.get("api_key") or ""
            if cfg.get("auth_header") and token:
                headers[cfg["auth_header"]] = f"{cfg.get('auth_prefix', '')}{token}"
            status, data = _http_json(f"{base}{test_ep}", "GET", headers=headers)
            if 200 <= status < 300:
                self.registry.update_status(plugin_id, "connected", "healthy")
                return True, f"{p['name']} connection verified."
            if status in (401, 403):
                self.registry.update_status(plugin_id, "expired", "error")
                return False, f"{p['name']} authentication failed or expired (HTTP {status})."
            self.registry.update_status(plugin_id, "error", "error")
            return False, f"{p['name']} returned HTTP {status}."

        if method == "mcp":
            try:
                client = MCPStdioClient(cfg["command"], cfg.get("args", []), env_secrets=secrets)
                tools = client.list_tools()
                self.registry.upsert(
                    plugin_id=p["id"],
                    name=p["name"],
                    method="mcp",
                    config=cfg,
                    capabilities=tools or p["capabilities"],
                    auth_status="connected",
                    config_location=p["config_location"],
                    connection_status="healthy",
                    install_info=p["install_info"],
                    enabled=p["enabled"],
                )
                return True, f"{p['name']} MCP server verified ({len(tools)} tools)."
            except Exception as e:
                self.registry.update_status(plugin_id, "error", "error")
                return False, f"{p['name']} MCP test failed: {e}"

        if method == "custom_python":
            mod = self._load_python_module(Path(cfg.get("module_path", "")))
            if mod is None:
                self.registry.update_status(plugin_id, "error", "error")
                return False, f"Could not load custom plugin file for {p['name']}."
            if hasattr(mod, "test_connection"):
                ok, msg = mod.test_connection(secrets, cfg)
                self.registry.update_status(plugin_id, "connected" if ok else "error", "healthy" if ok else "error")
                return bool(ok), str(msg)
            self.registry.update_status(plugin_id, "connected", "healthy")
            return True, f"{p['name']} custom plugin loaded."

        return False, f"Unknown plugin method: {method}"

    def interactive_setup(
        self,
        app_name: str,
        input_fn: Optional[Callable[[str], str]] = None,
        secret_fn: Optional[Callable[[str], str]] = None,
        speak_fn: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Conversational on-demand plugin setup.
        Never asks for passwords/tokens over voice — uses masked terminal input.
        """
        inp = input_fn or (lambda prompt: input(prompt).strip())
        sec = secret_fn or (lambda prompt: getpass.getpass(prompt).strip())
        say = speak_fn or (lambda msg: print(f">>> JARVIS : \"{msg}\"", flush=True))

        pid = _slugify(app_name)
        bp = INTEGRATION_BLUEPRINTS.get(pid)

        if bp:
            display_name = bp["name"]
            method = bp["method"]
            say(f"Sure, I can set up {display_name} using its official REST API. Want me to go ahead?")
            approval = inp("Approve setup (yes/no) > ").lower()
            if not any(w in approval for w in ("yes", "y", "sure", "ok", "okay", "go ahead")):
                return "No worries, skipped setup."

            print(f"\n[{display_name} Setup] {bp['auth_guide']}", flush=True)
            say(f"Please paste your {bp['auth_prompt']} in the terminal. It stays encrypted locally.")
            token = sec(f"Enter {bp['auth_prompt']} (hidden): ")
            if not token:
                return f"Cancelled {display_name} setup — no token entered."

            self.secrets.set_secret(pid, {bp["secret_key"]: token})
            self.registry.upsert(
                plugin_id=pid,
                name=display_name,
                method=method,
                config=bp["config"],
                capabilities=bp["capabilities"],
                auth_status="needs_auth",
                config_location=str(DB_PATH),
                connection_status="unverified",
                install_info=f"{display_name} official REST API",
                enabled=True,
            )
            ok, msg = self.test_connection(pid)
            if ok:
                return f"Done! {display_name} is connected, tested, and saved."
            return f"Saved {display_name}, but the connection test failed: {msg}"

        # Generic on-demand setup for ANY custom app
        display_name = app_name.strip().title()
        say(f"Sure, let's set up {display_name}. I can connect it via REST API, an MCP server, or a custom Python plugin.")
        approval = inp("Approve setup (yes/no) > ").lower()
        if not any(w in approval for w in ("yes", "y", "sure", "ok", "okay", "go ahead")):
            return "No worries, skipped setup."

        choice = inp("Choose method [rest / mcp / python] (default: rest): ").lower() or "rest"
        if choice == "mcp":
            cmd = inp("MCP command (e.g. npx or python): ")
            args_str = inp("MCP arguments (space-separated): ")
            env_name = inp("Secret env var name if needed (or leave blank): ")
            secrets_dict: Dict[str, str] = {}
            if env_name:
                val = sec(f"Enter {env_name} (hidden): ")
                if val:
                    secrets_dict[env_name] = val
            if secrets_dict:
                self.secrets.set_secret(pid, secrets_dict)
            self.registry.upsert(
                plugin_id=pid,
                name=display_name,
                method="mcp",
                config={"command": cmd, "args": args_str.split()},
                capabilities=["mcp_tools"],
                auth_status="connected",
                config_location=str(DB_PATH),
                connection_status="unverified",
                install_info=f"MCP: {cmd} {args_str}",
            )
            ok, msg = self.test_connection(pid)
            return f"Done! {msg}" if ok else f"Saved {display_name} MCP config, but test failed: {msg}"

        if choice == "python":
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
            py_path = self.plugins_dir / f"{pid}.py"
            if not py_path.exists():
                py_path.write_text(
                    f'"""Custom JARVIS plugin for {display_name}."""\n'
                    f'PLUGIN_NAME = "{display_name}"\n'
                    f'CAPABILITIES = ["run"]\n'
                    f'REQUIRES_AUTH = False\n\n'
                    f'def test_connection(secrets, config):\n'
                    f'    return True, "{display_name} custom plugin ready."\n\n'
                    f'def execute(query, secrets, config):\n'
                    f'    return f"{display_name} executed: {{query}}"\n',
                    encoding="utf-8",
                )
            self.registry.upsert(
                plugin_id=pid,
                name=display_name,
                method="custom_python",
                config={"module_path": str(py_path)},
                capabilities=["run"],
                auth_status="connected",
                config_location=str(py_path),
                connection_status="healthy",
                install_info=f"Custom Python: {py_path.name}",
            )
            ok, msg = self.test_connection(pid)
            return f"Done! Created and registered {py_path.name}." if ok else msg

        # Default: generic REST API
        base_url = inp("API base URL (e.g. https://api.example.com): ").rstrip("/")
        test_ep = inp("Test GET endpoint (e.g. /me or /health, optional): ")
        auth_header = inp("Auth header name (default Authorization): ") or "Authorization"
        auth_prefix = inp("Auth prefix (default 'Bearer '): ")
        if auth_prefix == "":
            auth_prefix = "Bearer "
        token = sec("Enter API key/token (hidden, leave blank if none): ")
        if token:
            self.secrets.set_secret(pid, {"token": token})
        self.registry.upsert(
            plugin_id=pid,
            name=display_name,
            method="rest_api",
            config={
                "base_url": base_url,
                "auth_header": auth_header,
                "auth_prefix": auth_prefix,
                "extra_headers": {"Accept": "application/json"},
                "test_endpoint": test_ep,
            },
            capabilities=["query"],
            auth_status="connected" if token else "needs_auth",
            config_location=str(DB_PATH),
            connection_status="unverified",
            install_info=f"REST API: {base_url}",
        )
        ok, msg = self.test_connection(pid)
        return f"Done! {msg}" if ok else f"Saved {display_name}, but test returned: {msg}"

    def execute_plugin_task(
        self,
        plugin_name: str,
        task_query: str,
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """Executes a user request against a registered plugin."""
        p = self.find_plugin(plugin_name)
        if not p:
            return f"I don't have a {plugin_name.strip().title()} integration set up yet. Want to configure it?"
        if not p["enabled"]:
            return f"Your {p['name']} plugin is currently disabled. Say 'enable {p['name']}' if you want to use it."

        pid = p["id"]
        secrets = self.secrets.get_secret(pid)
        cfg = p["config"]
        q_low = (task_query or "").lower()

        # Confirm sensitive write operations
        if any(w in q_low for w in ("send ", "create ", "delete ", "post ", "remove ")):
            if confirm_fn and not confirm_fn(f"Do you want me to run '{task_query}' on {p['name']}?"):
                return "No worries, cancelled."

        if p["method"] == "rest_api":
            headers = dict(cfg.get("extra_headers") or {})
            token = secrets.get("token") or secrets.get("api_key") or ""
            if cfg.get("auth_header") and token:
                headers[cfg["auth_header"]] = f"{cfg.get('auth_prefix', '')}{token}"
            base = (cfg.get("base_url") or "").rstrip("/")

            # Smart endpoint routing for registered services
            endpoint = cfg.get("test_endpoint") or ""
            if pid == "github":
                if "commit" in q_low or "activity" in q_low:
                    endpoint = "/user/repos?sort=pushed&per_page=3"
                elif "notif" in q_low:
                    endpoint = "/notifications?per_page=5"
                elif "repo" in q_low:
                    endpoint = "/user/repos?sort=updated&per_page=5"
            elif pid == "gmail":
                endpoint = "/users/me/messages?maxResults=5"
            elif pid == "spotify":
                if "recent" in q_low:
                    endpoint = "/me/player/recently-played?limit=5"
                elif "playlist" in q_low:
                    endpoint = "/me/playlists?limit=5"
                else:
                    endpoint = "/me/player/currently-playing"
            elif pid == "notion":
                status, data = _http_json(f"{base}/search", "POST", headers=headers, body={"page_size": 5})
                return self._format_rest_result(p, status, data, task_query)

            status, data = _http_json(f"{base}{endpoint}", "GET", headers=headers)
            return self._format_rest_result(p, status, data, task_query)

        if p["method"] == "mcp":
            try:
                client = MCPStdioClient(cfg["command"], cfg.get("args", []), env_secrets=secrets)
                tools = p["capabilities"] or client.list_tools()
                if not tools:
                    return f"{p['name']} MCP server has no tools available."
                return client.call_tool(tools[0], {"query": task_query})[:400]
            except Exception as e:
                self.registry.update_status(pid, "error", "error")
                return f"{p['name']} hit an error: {e}"

        if p["method"] == "custom_python":
            mod = self._load_python_module(Path(cfg.get("module_path", "")))
            if mod and hasattr(mod, "execute"):
                return str(mod.execute(task_query, secrets, cfg))
            return f"{p['name']} custom plugin has no execute() function."

        return f"Unsupported plugin method: {p['method']}."

    def _format_rest_result(self, p: Dict[str, Any], status: int, data: Any, query: str) -> str:
        pid = p["id"]
        if status in (401, 403):
            self.registry.update_status(pid, "expired", "error")
            return f"Your {p['name']} authentication expired. Say 'reconnect {p['name']}' when you want to renew it."
        if status < 200 or status >= 300:
            return f"{p['name']} returned an error (HTTP {status})."

        self.registry.update_status(pid, "connected", "healthy")
        if pid == "github" and isinstance(data, list):
            names = [f"{r.get('full_name', '?')} (pushed {r.get('pushed_at', '')[:10]})" for r in data[:3]]
            return "Latest GitHub activity: " + ", ".join(names) + "." if names else "No recent GitHub repos found."
        if pid == "gmail" and isinstance(data, dict):
            msgs = data.get("messages") or []
            return f"You have {len(msgs)} recent emails in Gmail." if msgs else "Your Gmail inbox looks empty."
        if pid == "notion" and isinstance(data, dict):
            res = data.get("results") or []
            return f"Found {len(res)} Notion items."
        if isinstance(data, dict):
            summary_keys = [f"{k}: {data[k]}" for k in ("login", "name", "emailAddress", "display_name") if k in data]
            if summary_keys:
                return f"{p['name']}: " + ", ".join(summary_keys) + "."
        return f"{p['name']} responded OK for '{query}'."

    def handle_action(
        self,
        op: str,
        target: str = "",
        query: str = "",
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """Dispatcher called by ActionExecutor for `plugin_op` steps."""
        op_l = (op or "list").strip().lower()
        t = (target or "").strip()

        if op_l == "list":
            return self.list_plugins_summary()
        if op_l == "capabilities":
            return self.capabilities_summary(t)
        if op_l == "enable":
            p = self.find_plugin(t)
            if not p:
                return f"I don't have a {t} plugin set up yet."
            self.registry.set_enabled(p["id"], True)
            return f"Enabled {p['name']}."
        if op_l == "disable":
            p = self.find_plugin(t)
            if not p:
                return f"I don't have a {t} plugin set up yet."
            self.registry.set_enabled(p["id"], False)
            return f"Disabled {p['name']}."
        if op_l == "remove":
            p = self.find_plugin(t)
            if not p:
                return f"I don't have a {t} plugin set up."
            if confirm_fn and not confirm_fn(f"Remove your {p['name']} plugin and its saved credentials?"):
                return "No worries, kept it."
            self.secrets.delete_secret(p["id"])
            self.registry.remove(p["id"])
            return f"Removed {p['name']} and deleted its saved credentials."
        if op_l == "health":
            p = self.find_plugin(t)
            if not p:
                return f"I don't have a {t} plugin set up yet."
            _, msg = self.test_connection(p["id"])
            return msg
        if op_l in ("create", "reconnect"):
            # Marker handled interactively by jarvis_hybrid.py
            return f"__SETUP_PLUGIN__:{t or query}"
        if op_l == "use":
            res = self.execute_plugin_task(t, query, confirm_fn=confirm_fn)
            return res
        return self.list_plugins_summary()
