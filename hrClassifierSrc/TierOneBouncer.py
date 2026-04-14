"""Tier 1 Rule Engine — Clasificador basado en reglas HR.

Optimizaciones:
- Un solo regex combinado para TODOS los lexicons (1 pasada vs N).
- re.search en vez de findall (early return al primer hit).
- Sin re.IGNORECASE — el input se espera en lowercase.
- Palabras pre-lowered al compilar.
- Regex rules también combinados donde comparten categoría.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RULES_PATH = _PROJECT_ROOT / "hr_rules.yaml"


class TierOneBouncer:
    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else DEFAULT_RULES_PATH
        with open(path, encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Almacena tuplas (category, compiled_regex)
        # Lexicons se fusionan en UN solo regex con grupos nombrados
        # para resolver la categoría sin iterar.
        self._build_rules(config["rules"])

    def _build_rules(self, rules: list[dict]) -> None:
        """Compila las reglas en la menor cantidad de regex posible."""

        # --- Paso 1: Agrupar lexicons por categoría ---
        # Cada categoría se convierte en un grupo nombrado dentro de
        # un solo regex maestro.
        lexicon_parts: list[str] = []
        self._group_to_category: dict[str, str] = {}
        group_idx = 0

        for rule in rules:
            if rule["type"] != "lexicon":
                continue

            # Nombre de grupo seguro para regex (solo alfanuméricos + _)
            group_name = f"lex{group_idx}"
            group_idx += 1

            category = rule["category"]
            self._group_to_category[group_name] = category

            # Pre-lower todas las palabras
            words = sorted(
                (w.lower() for w in rule["words"]),
                key=len, reverse=True,  # longest first para match correcto
            )
            joined = "|".join(map(re.escape, words))

            if rule.get("boundary", True):
                lexicon_parts.append(rf"(?P<{group_name}>\b(?:{joined})\b)")
            else:
                lexicon_parts.append(rf"(?P<{group_name}>(?:{joined}))")

        # Compilar UN solo regex para todos los lexicons
        self._lexicon_re: re.Pattern[str] | None = None
        if lexicon_parts:
            self._lexicon_re = re.compile("|".join(lexicon_parts))

        # --- Paso 2: Regex rules (se mantienen individuales) ---
        self._regex_rules: list[tuple[str, re.Pattern[str]]] = []
        for rule in rules:
            if rule["type"] == "regex":
                self._regex_rules.append(
                    (rule["category"], re.compile(rule["pattern"]))
                )

    def inspect(self, text: str) -> dict:
        """Evalúa el texto contra todas las reglas. Early return al primer hit."""

        # 1. Lexicon — una sola pasada con search (no findall)
        if self._lexicon_re is not None:
            m = self._lexicon_re.search(text)
            if m:
                group_name = m.lastgroup
                return {
                    "status": "FLAGGED",
                    "reason": self._group_to_category[group_name],
                    "trigger": m.group(),
                }

        # 2. Regex rules — search en vez de findall
        for category, regex in self._regex_rules:
            m = regex.search(text)
            if m:
                return {
                    "status": "FLAGGED",
                    "reason": category,
                    "trigger": m.group(),
                }

        return {"status": "PASS"}