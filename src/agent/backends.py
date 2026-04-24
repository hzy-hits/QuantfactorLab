from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _backend_mode() -> str:
    mode = os.environ.get("FACTOR_LAB_AGENT_BACKEND", "auto").strip().lower()
    return mode if mode in {"auto", "claude", "codex"} else "auto"


def _codex_model() -> str:
    return os.environ.get("FACTOR_LAB_CODEX_MODEL", "gpt-5.4").strip() or "gpt-5.4"


def _codex_reasoning_effort() -> str:
    effort = os.environ.get("FACTOR_LAB_CODEX_REASONING_EFFORT", "xhigh").strip().lower()
    return effort if effort in {"low", "medium", "high", "xhigh"} else "xhigh"


def _claude_sdk_available() -> bool:
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )


def _call_claude_sdk(prompt: str, *, model: str, max_tokens: int) -> str:
    if not _claude_sdk_available():
        raise RuntimeError("Anthropic SDK credentials not configured")

    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _call_claude_cli(prompt: str, *, timeout: int) -> str:
    env = {"CLAUDECODE": "", **os.environ}
    result = subprocess.run(
        ["claude", "-p", "--output-format", "text"],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        detail = stderr or stdout or "no output"
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): {detail[:240]}"
        )

    if not stdout:
        detail = stderr or "no output"
        raise RuntimeError(f"claude CLI returned empty output: {detail[:240]}")

    return stdout


def _call_codex_cli(
    prompt: str,
    *,
    repo_root: Path,
    timeout: int,
) -> str:
    codex = shutil.which("codex")
    if codex is None:
        raise RuntimeError("codex CLI not found in PATH")

    with tempfile.NamedTemporaryFile(
        prefix="factor_lab_codex_",
        suffix=".txt",
        delete=False,
    ) as tmp:
        output_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                codex,
                "exec",
                "-m",
                _codex_model(),
                "-c",
                f'model_reasoning_effort="{_codex_reasoning_effort()}"',
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--skip-git-repo-check",
                "-C",
                str(repo_root),
                "-o",
                str(output_path),
                "-",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=os.environ.copy(),
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            detail = stderr or stdout or "no output"
            raise RuntimeError(
                f"codex exec failed (exit {result.returncode}): {detail[:240]}"
            )

        response = (
            output_path.read_text(encoding="utf-8", errors="replace").strip()
            if output_path.exists()
            else ""
        )
        if not response:
            detail = stderr or stdout or "no output"
            raise RuntimeError(
                f"codex exec returned empty output: {detail[:240]}"
            )

        return response
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def call_agent(
    prompt: str,
    *,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 2000,
    repo_root: Path | None = None,
    claude_timeout: int = 120,
    codex_timeout: int = 300,
) -> str:
    repo_root = repo_root or REPO_ROOT
    mode = _backend_mode()
    errors: list[str] = []

    if mode in {"auto", "claude"}:
        try:
            return _call_claude_sdk(prompt, model=model, max_tokens=max_tokens)
        except Exception as exc:
            errors.append(f"Claude SDK: {exc}")

        try:
            return _call_claude_cli(prompt, timeout=claude_timeout)
        except Exception as exc:
            errors.append(f"Claude CLI: {exc}")

        if mode == "claude":
            raise RuntimeError("; ".join(errors))

    try:
        return _call_codex_cli(prompt, repo_root=repo_root, timeout=codex_timeout)
    except Exception as exc:
        errors.append(f"Codex: {exc}")
        raise RuntimeError("; ".join(errors))
