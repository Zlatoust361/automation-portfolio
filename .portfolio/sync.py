#!/usr/bin/env python3
"""
Синхронизация портфолио с исходным (закрытым) репозиторием.

Портфолио — витрина: его файлы либо ЗЕРКАЛЯТ файлы источника 1:1, либо
принадлежат только портфолио (README, HONESTY.md, CI, лицензия). Этот
скрипт держит первую категорию честной: показывает, где витрина отстала
от источника, и копирует изменения — но только то, что перечислено в
манифесте, и только после проверки на запрещённые слова (цены, бренд,
заимствованные тексты).

Использование (из корня портфолио):

    python .portfolio/sync.py check              # дрейф: что изменилось в источнике
    python .portfolio/sync.py apply              # скопировать изменения
    python .portfolio/sync.py apply --prune      # ...и удалить то, чего в источнике больше нет
    python .portfolio/sync.py verify             # прогнать команды verify из манифеста

Путь к источнику — ключ `source` в манифесте или переменная окружения
PORTFOLIO_SOURCE. Манифест: .portfolio/manifest.yaml. Зависимости: pyyaml.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "manifest.yaml"


@dataclass
class Manifest:
    source: Path
    mirror: list  # str (одинаковый путь) или dict {портфолио: источник}
    patched: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    verify: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        src = os.environ.get("PORTFOLIO_SOURCE") or raw.get("source")
        if not src:
            sys.exit("Источник не задан: ключ `source` в манифесте или PORTFOLIO_SOURCE")
        source = (ROOT / src).resolve() if not Path(src).is_absolute() else Path(src)
        if not source.is_dir():
            sys.exit(f"Источник не найден: {source}")
        return cls(
            source=source,
            mirror=list(raw.get("mirror", [])),
            patched=list(raw.get("patched", [])),
            exclude=list(raw.get("exclude", [])) + ["**/__pycache__/**", "**/.pytest_cache/**",
                                                     "**/.ruff_cache/**", "**/*.pyc"],
            forbidden=list(raw.get("forbidden", [])),
            verify=list(raw.get("verify", [])),
        )


def _sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _excluded(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch("/" + rel, p) for p in patterns)


def _pairs(m: Manifest) -> dict[str, str]:
    """Маппинг `путь в портфолио -> путь в источнике` по записям манифеста.

    Строка — одинаковый путь в обоих репо (файл или каталог целиком; каталог
    разворачивается по объединению файлов источника и портфолио, чтобы
    увидеть и новое, и удалённое). Словарь `{портфолио: источник}` — файл с
    разными путями.
    """
    pairs: dict[str, str] = {}
    for entry in m.mirror:
        if isinstance(entry, dict):
            for port_rel, src_rel in entry.items():
                pairs[Path(port_rel).as_posix()] = Path(src_rel).as_posix()
            continue
        rels: set[str] = set()
        for root in (m.source, ROOT):
            p = root / entry
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(root).as_posix()
                        if not _excluded(rel, m.exclude):
                            rels.add(rel)
            elif p.is_file():
                rels.add(Path(entry).as_posix())
        for rel in rels:
            pairs[rel] = rel
    return pairs


@dataclass
class Drift:
    changed: list[str]
    added: list[str]      # есть в источнике, нет в портфолио
    removed: list[str]    # есть в портфолио, нет в источнике
    patched_diverged: list[str]
    patched_equal: list[str]
    pairs: dict[str, str]

    @property
    def clean(self) -> bool:
        return not (self.changed or self.added or self.removed)


def compute_drift(m: Manifest) -> Drift:
    pairs = _pairs(m)
    patched = set(m.patched)
    changed, added, removed, pd, pe = [], [], [], [], []
    for rel in sorted(pairs):
        s, p = m.source / pairs[rel], ROOT / rel
        if rel in patched:
            if not (s.exists() and p.exists()):
                changed.append(rel)
            elif _sha(s) != _sha(p):
                pd.append(rel)
            else:
                pe.append(rel)
            continue
        if s.exists() and not p.exists():
            added.append(rel)
        elif p.exists() and not s.exists():
            removed.append(rel)
        elif _sha(s) != _sha(p):
            changed.append(rel)
    return Drift(changed, added, removed, pd, pe, pairs)


def scan_forbidden(m: Manifest, rels: list[str], pairs: dict[str, str]) -> list[tuple[str, int, str]]:
    """Запрещённые слова в файлах, которые собираемся опубликовать."""
    if not m.forbidden:
        return []
    patterns = [re.compile(p, re.IGNORECASE) for p in m.forbidden]
    hits = []
    for rel in rels:
        try:
            text = (m.source / pairs[rel]).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for pat in patterns:
                if pat.search(line):
                    hits.append((rel, n, pat.pattern))
    return hits


def cmd_check(m: Manifest) -> int:
    d = compute_drift(m)
    print(f"Источник: {m.source}")
    print(f"Зеркалируется: {len(d.pairs)} файлов")
    for title, items in (("ИЗМЕНЕНО в источнике", d.changed),
                         ("НОВОЕ в источнике (нет в портфолио)", d.added),
                         ("УДАЛЕНО в источнике (есть только в портфолио)", d.removed),
                         ("PATCHED — локальная правка, источник отличается (норма, но лучше вернуть в источник)", d.patched_diverged),
                         ("PATCHED — уже совпадает с источником, запись в манифесте можно убрать", d.patched_equal)):
        if items:
            print(f"\n{title}: {len(items)}")
            for rel in items:
                print(f"  {rel}")
    if d.clean:
        print("\nДрейфа нет: зеркало совпадает с источником.")
        return 0
    print("\nЕсть дрейф. `apply` скопирует изменения; `apply --prune` ещё и удалит лишнее.")
    return 1


def cmd_apply(m: Manifest, prune: bool) -> int:
    d = compute_drift(m)
    to_copy = d.changed + d.added
    hits = scan_forbidden(m, to_copy, d.pairs)
    if hits:
        print("СТОП: запрещённые слова в файлах, которые собираемся опубликовать:")
        for rel, n, pat in hits:
            print(f"  {rel}:{n}  /{pat}/")
        print("Уберите их в источнике или исключите файл из манифеста. Ничего не скопировано.")
        return 2
    for rel in to_copy:
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(m.source / d.pairs[rel], dst)
        print(f"  copied   {rel}")
    if prune:
        for rel in d.removed:
            (ROOT / rel).unlink()
            print(f"  removed  {rel}")
    elif d.removed:
        print(f"\nВ источнике больше нет {len(d.removed)} файлов — оставлены (используйте --prune, чтобы удалить).")
    for rel in d.patched_diverged:
        print(f"  skipped  {rel}  (patched)")
    if not to_copy and not (prune and d.removed):
        print("Нечего применять.")
    else:
        print("\nДальше: `python .portfolio/sync.py verify`, затем обновить README/HONESTY под то, что изменилось.")
    return 0


def cmd_verify(m: Manifest) -> int:
    if not m.verify:
        print("В манифесте нет команд verify.")
        return 0
    failed = 0
    for cmd in m.verify:
        print(f"\n$ {cmd}")
        rc = subprocess.call(cmd, shell=True, cwd=ROOT)
        if rc != 0:
            failed += 1
            print(f"  -> код возврата {rc}")
    print(f"\nПроверок: {len(m.verify)}, упало: {failed}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--prune", action="store_true")
    sub.add_parser("verify")
    args = ap.parse_args()
    m = Manifest.load(MANIFEST)
    if args.cmd == "check":
        return cmd_check(m)
    if args.cmd == "apply":
        return cmd_apply(m, args.prune)
    return cmd_verify(m)


if __name__ == "__main__":
    sys.exit(main())
