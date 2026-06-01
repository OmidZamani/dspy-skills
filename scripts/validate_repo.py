#!/usr/bin/env python3
"""Validate plugin manifests, skill metadata, and local Markdown links."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "skills"
PLUGIN_PATH = ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE_PATH = ROOT / ".claude-plugin" / "marketplace.json"
STABLE_DSPY_VERSION = "3.2.1"
EXTERNAL_SKILLS = {"dspy-optimize-anything"}
STABLE_GEPA_VERSION = "0.1.1"
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def error(message: str) -> None:
    errors.append(message)


def parse_frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        error(f"{path.relative_to(ROOT)}: missing YAML frontmatter")
        return {}

    try:
        end = lines.index("---", 1)
    except ValueError:
        error(f"{path.relative_to(ROOT)}: unterminated YAML frontmatter")
        return {}

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        match = re.match(r"^([a-zA-Z0-9_-]+):\s*(.*)$", line)
        if not match:
            continue
        key, value = match.groups()
        metadata[key] = value.strip().strip('"').strip("'")
    return metadata


def validate_manifests() -> None:
    plugin = json.loads(PLUGIN_PATH.read_text(encoding="utf-8"))
    marketplace = json.loads(MARKETPLACE_PATH.read_text(encoding="utf-8"))

    if plugin.get("name") != "dspy-skills":
        error(".claude-plugin/plugin.json: expected name 'dspy-skills'")
    if not plugin.get("version"):
        error(".claude-plugin/plugin.json: missing version")

    plugins = marketplace.get("plugins")
    if not isinstance(plugins, list) or not plugins:
        error(".claude-plugin/marketplace.json: expected a non-empty plugins array")
        return

    entry = next((item for item in plugins if item.get("name") == plugin.get("name")), None)
    if entry is None:
        error(".claude-plugin/marketplace.json: missing dspy-skills plugin entry")
        return
    if entry.get("source") != "./":
        error(".claude-plugin/marketplace.json: dspy-skills source must be './'")
    if entry.get("version") != plugin.get("version"):
        error(".claude-plugin manifests: plugin versions do not match")


def validate_skills() -> None:
    skill_files = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    if not skill_files:
        error("skills/: no SKILL.md files found")
        return

    for skill_file in skill_files:
        relative = skill_file.relative_to(ROOT)
        metadata = parse_frontmatter(skill_file)
        skill_name = skill_file.parent.name

        if metadata.get("name") != skill_name:
            error(f"{relative}: name must match directory '{skill_name}'")
        if not metadata.get("description"):
            error(f"{relative}: missing description")
        if "allowed-tools" not in metadata:
            error(f"{relative}: missing allowed-tools")
        if skill_name not in EXTERNAL_SKILLS and metadata.get("dspy-compatibility") != STABLE_DSPY_VERSION:
            error(f"{relative}: expected dspy-compatibility '{STABLE_DSPY_VERSION}'")
        if skill_name in EXTERNAL_SKILLS and metadata.get("gepa-compatibility") != STABLE_GEPA_VERSION:
            error(f"{relative}: expected gepa-compatibility '{STABLE_GEPA_VERSION}'")


def validate_catalog() -> None:
    skill_names = sorted(path.parent.name for path in SKILLS_DIR.glob("*/SKILL.md"))
    skill_count = len(skill_names)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    plugin = json.loads(PLUGIN_PATH.read_text(encoding="utf-8"))
    marketplace = json.loads(MARKETPLACE_PATH.read_text(encoding="utf-8"))
    entry = next((item for item in marketplace.get("plugins", []) if item.get("name") == "dspy-skills"), {})
    expected_phrase = f"{skill_count} focused skills"

    for skill_name in skill_names:
        target = f"skills/{skill_name}/SKILL.md"
        if target not in readme:
            error(f"README.md: missing skill link '{target}'")

    if expected_phrase not in plugin.get("description", ""):
        error(f".claude-plugin/plugin.json: description must include '{expected_phrase}'")
    if expected_phrase not in entry.get("description", ""):
        error(f".claude-plugin/marketplace.json: description must include '{expected_phrase}'")
    if expected_phrase not in readme:
        error(f"README.md: introduction must include '{expected_phrase}'")


def validate_markdown_links() -> None:
    for markdown_file in sorted(ROOT.rglob("*.md")):
        if ".git" in markdown_file.parts or ".claude" in markdown_file.parts:
            continue
        text = markdown_file.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK_RE.findall(text):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = (markdown_file.parent / target).resolve()
            if not resolved.exists():
                error(f"{markdown_file.relative_to(ROOT)}: broken local link '{raw_target}'")


errors: list[str] = []
validate_manifests()
validate_skills()
validate_catalog()
validate_markdown_links()

if errors:
    print("Validation failed:")
    for item in errors:
        print(f"- {item}")
    sys.exit(1)

skill_count = len(list(SKILLS_DIR.glob("*/SKILL.md")))
print(f"Validation passed: {skill_count} skills, manifests, and local Markdown links")
