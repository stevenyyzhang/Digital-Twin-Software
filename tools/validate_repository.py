"""Validate the curated repository without running a GPU calculation."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
ignored = {".git", "__pycache__", ".venv", ".air-cache", ".coverage-cache",
           ".link-cache", ".display-cache", ".matplotlib-cache", ".pytest_cache"}
files = [p for p in root.rglob("*") if p.is_file() and not (set(p.relative_to(root).parts) & ignored)
         and p.name not in {".polyscope.ini", "imgui.ini"}]
errors = []
references = 0
for path in files:
    rel = path.relative_to(root).as_posix()
    if not rel.isascii():
        errors.append("Non-English filename: " + rel)
    if path.stat().st_size >= 100 * 1024 * 1024:
        errors.append("File exceeds 100 MiB: " + rel)
    if path.suffix == ".py":
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
        except SyntaxError as error:
            errors.append(str(error))
    if path.suffix in {".yaml", ".json", ".xml", ".ps1", ".bat"}:
        text = path.read_text(encoding="utf-8-sig")
        if re.search(r"[A-Za-z]:[\\/](?:Users|Radio-Map-UI)[\\/]", text):
            errors.append("Developer-specific absolute path: " + rel)
    # Do not print possible credentials; only identify the file if any are found.
    if path.suffix in {".py", ".json", ".yaml", ".md", ".ps1", ".bat"}:
        text = path.read_text(encoding="utf-8-sig")
        if re.search(r"(?:ghp_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9_-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----\r?\n)", text):
            errors.append("Possible private credential: " + rel)
for name in ["polyu-stylized", "mong-kok", "kam-tin"]:
    scene = root / name
    required = set()
    for node in ET.parse(scene / "scene.xml").iter("string"):
        if node.get("name") == "filename":
            required.add(scene / node.get("value"))
    for asset in json.loads((scene / "visuals.json").read_text()):
        for key in ("mesh", "texture"):
            if asset.get(key):
                required.add(scene / asset[key])
    for path in required:
        references += 1
        if not path.resolve().is_relative_to(root) or not path.is_file():
            errors.append("Missing or external asset: " + str(path))
for path in [root / "README.md", *list((root / "docs").glob("*.md"))]:
    if path.name == "UPSTREAM_README.md":
        continue
    for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" not in link and not link.startswith("#"):
            target = path.parent / link.split("#")[0]
            if not target.exists():
                errors.append(f"Broken documentation link: {path.name}: {link}")
result = dict(files=len(files), total_bytes=sum(p.stat().st_size for p in files),
              asset_references_checked=references, all_paths_ascii=not any("filename" in e for e in errors),
              errors=errors, largest_files=[dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size)
                  for p in sorted(files, key=lambda p:p.stat().st_size, reverse=True)[:5]])
print(json.dumps(result, indent=2))
sys.exit(bool(errors))
