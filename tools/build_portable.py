"""Bundle tracked source/assets with a supplied relocatable Windows runtime.

Build on Windows. The runtime must already contain the tested dependencies.
The output is separate from Git history; package licenses are retained.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(runtime, output, version):
    runtime = runtime.resolve()
    output = output.resolve()
    name = f"Digital-Twin-Studio-{version}-Windows-x64"
    folder = output / name
    archive = output / f"{name}.zip"
    if folder.exists() or archive.exists():
        raise FileExistsError("Choose a fresh output directory; existing builds are preserved.")
    if not (runtime / "python.exe").is_file() or not (runtime / "LICENSE.txt").is_file():
        raise ValueError("Expected a complete Windows Python runtime including its license.")
    folder.mkdir(parents=True)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for rel in filter(None, tracked):
        if rel in {".gitignore", ".gitattributes", "start.ps1"}:
            continue
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    shutil.copy2(ROOT / "tools/start-portable.ps1", folder / "start.ps1")
    shutil.copytree(runtime, folder / "runtime",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    # Relative to runtime/Lib/site-packages; also supports spawned RF workers.
    (folder / "runtime/Lib/site-packages/radio_map_source.pth").write_text(
        "../../../sionna-rt-gui/src\n", encoding="ascii")
    (folder / "START_HERE.txt").write_text(
        "Digital Twin Studio - Windows x64 portable edition\n\n"
        "1. Extract the entire ZIP to a writable folder.\n"
        "2. Double-click start.bat.\n\n"
        "Python and application dependencies are included. No Python installation,\n"
        "pip setup, administrator privileges or first-run dependency download is required.\n"
        "An NVIDIA CUDA-capable GPU and compatible NVIDIA driver are required.\n"
        "Do not run directly inside the ZIP. First launch generates local caches.\n"
        "If startup fails, the console stays open with the error details.\n\n"
        "The repository README also describes source installation; those Python/setup\n"
        "steps do not apply to this portable download. See THIRD_PARTY_NOTICES.md,\n"
        "the per-scene ASSET_LICENSE.md files and runtime package licenses.\n",
        encoding="utf-8")
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if any(not p.relative_to(folder).as_posix().isascii() for p in files):
        raise ValueError("All distribution filenames must be ASCII.")
    manifest = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(folder).as_posix()}\n"
        for p in files)
    (folder / "MANIFEST.sha256").write_text(manifest, encoding="ascii")
    print(f"Bundled {len(files)} files. Compressing...", flush=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError("ZIP CRC check failed: " + bad)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    checksum = archive.with_suffix(".zip.sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return dict(folder=str(folder), archive=str(archive), checksum=str(checksum),
                bytes=archive.stat().st_size, sha256=digest, files=len(files)+1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v0.1.0")
    args = parser.parse_args()
    if not args.version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in args.version):
        parser.error("Version must contain only ASCII letters, digits, dots and hyphens.")
    result = build(args.runtime, args.output, args.version)
    (args.output / "build-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
