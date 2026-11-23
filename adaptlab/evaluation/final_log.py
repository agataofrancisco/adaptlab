"""Registo auditável das leituras finais do conjunto de teste (``runs/final_eval_log.jsonl``).

Regras:
- Cada ``--final`` escreve uma linha JSON ANTES de ler as predições (o acesso fica registado
  mesmo que a execução falhe a meio).
- A primeira leitura não precisa de justificação. Qualquer leitura seguinte (mesmo com o mesmo
  modelo e configuração) exige ``--allow-retest "<motivo>"``. O motivo fica gravado na linha.
- O registo é append-only: nada apaga nem reescreve linhas anteriores.
- O número de utilizações do teste é o número de linhas do registo.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG_FILENAME = "final_eval_log.jsonl"


class RetestRefusedError(RuntimeError):
    """Segunda (ou seguinte) leitura final sem ``--allow-retest``."""


def canonical_config_hash(*parts: Any) -> str:
    """SHA-256 da configuração (dicts/listas/escalares), com chaves ordenadas."""
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit(cwd: Path) -> str:
    """Commit HEAD do repositório (ou ``unknown`` se não houver git). Nunca lança."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: linha inválida no registo final ({exc.msg})") from exc
        if isinstance(row, dict):
            entries.append(row)
    return entries


def check_retest(prior_uses: int, allow_reason: str | None) -> str | None:
    """Devolve o motivo gravado (ou None na primeira leitura). Recusa sem motivo nas seguintes."""
    if prior_uses == 0:
        return None
    if allow_reason is None:
        raise RetestRefusedError(
            f"o conjunto de teste já foi lido {prior_uses} vez(es). Uma nova leitura final exige "
            '--allow-retest "<motivo>" (fica registado no log).'
        )
    reason = allow_reason.strip()
    if not reason:
        raise RetestRefusedError("--allow-retest exige um motivo não vazio")
    return reason


def append_entry(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def build_entry(
    *,
    git_root: Path,
    config_hash: str,
    model_id: str,
    approach: str,
    test_sha256: str,
    examples: int,
    use_index: int,
    retest_reason: str | None,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "use_index": use_index,
        "git_commit": git_commit(git_root),
        "config_hash": config_hash,
        "approach": approach,
        "model_id": model_id,
        "test_sha256": test_sha256,
        "examples": examples,
        "retest_reason": retest_reason,
    }
