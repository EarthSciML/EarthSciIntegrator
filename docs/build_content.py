#!/usr/bin/env python3
"""Build the generated part of docs/content from the files the repository
already keeps: docs/*.md, docs/rfcs/ and spec/. Nothing is copied into git, so
the site cannot drift from the specification and the GitHub links keep working.

Run from the repository root before the docs bundle is built (docs/site.toml
`prebuild`). Output (gitignored): docs/content/{architecture,decisions,survey}.md,
docs/content/rfcs/, docs/content/spec/ and docs/static/files/ (the JSON
schemas, tableaus and error list the spec links to).

Relative links between these files become site paths, links to other
repository files become GitHub links, and each page gets front matter from its
first heading.
"""
from __future__ import annotations

import posixpath
import re
import shutil
import sys
from pathlib import Path

REPO = "https://github.com/EarthSciML/EarthSciIntegrator"
ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "docs" / "content"
STATIC_FILES = ROOT / "docs" / "static" / "files"

# source (repo-relative) -> page path under content/ (no extension)
PAGES: dict[str, str] = {
    "docs/architecture.md": "architecture",
    "docs/decisions.md": "decisions",
    "docs/survey.md": "survey",
    "docs/rfcs/README.md": "rfcs/_index",
    "spec/README.md": "spec/_index",
}
for p in sorted((ROOT / "docs" / "rfcs").glob("U*.md")):
    PAGES[f"docs/rfcs/{p.name}"] = f"rfcs/{p.stem}"
for p in sorted((ROOT / "spec").glob("*.md")):
    PAGES.setdefault(f"spec/{p.name}", f"spec/{p.stem}")

WEIGHTS = {"architecture": 10, "decisions": 20, "survey": 30, "rfcs/_index": 40, "spec/_index": 5}
DATA_DIRS = ("spec/schemas", "spec/tableaus")
DATA_FILES = ("spec/planner-errors.json",)

LINK = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^)\s]+)\)")


def site_path(page: str) -> str:
    return "/" + page.removesuffix("_index").rstrip("/") + "/" if page != "_index" else "/"


def data_targets() -> set[str]:
    out = set(DATA_FILES)
    for d in DATA_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and p.suffix == ".json":
                out.add(p.relative_to(ROOT).as_posix())
    return out


def rewrite_target(src: str, target: str, data: set[str]) -> str:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
        return target
    path, _, frag = target.partition("#")
    frag = f"#{frag}" if frag else ""
    if not path:
        return target
    rel = posixpath.normpath(posixpath.join(posixpath.dirname(src), path))
    if rel in PAGES:
        return site_path(PAGES[rel]) + frag
    if rel in data:
        return f"/files/{rel}{frag}"
    if rel.startswith(".."):
        return target
    kind = "tree" if (ROOT / rel).is_dir() else "blob"
    return f"{REPO}/{kind}/main/{rel}{frag}"


def rewrite_links(src: str, text: str, data: set[str]) -> str:
    out, fenced = [], None
    for line in text.split("\n"):
        m = re.match(r"^\s*(`{3,}|~{3,})", line)
        if m:
            fence = m.group(1)
            if fenced is None:
                fenced = fence
            elif fence[0] == fenced[0] and len(fence) >= len(fenced):
                fenced = None
            out.append(line)
            continue
        if fenced is None:
            line = LINK.sub(
                lambda mm: f"[{mm.group(1)}]({rewrite_target(src, mm.group(2), data)})", line)
        out.append(line)
    return "\n".join(out)


def yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def convert(src: str, page: str, data: set[str]) -> str:
    text = (ROOT / src).read_text(encoding="utf-8")
    if text.startswith("---\n"):  # already has front matter (the RFC drafts)
        return rewrite_links(src, text, data)
    lines = text.split("\n")
    title = page.rsplit("/", 1)[-1]
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            del lines[i]
            break
    fm = ["---", f"title: {yaml_str(title)}"]
    if page in WEIGHTS:
        fm.append(f"weight: {WEIGHTS[page]}")
    fm.append("---")
    return "\n".join(fm) + "\n" + rewrite_links(src, "\n".join(lines), data)


def main() -> int:
    for sub in ("architecture.md", "decisions.md", "survey.md", "rfcs", "spec"):
        t = CONTENT / sub
        shutil.rmtree(t) if t.is_dir() else t.unlink(missing_ok=True)
    shutil.rmtree(STATIC_FILES, ignore_errors=True)
    data = data_targets()
    for src, page in PAGES.items():
        dest = CONTENT / f"{page}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(convert(src, page, data), encoding="utf-8")
    for rel in sorted(data):
        dest = STATIC_FILES / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    print(f"build_content: {len(PAGES)} pages, {len(data)} data files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
