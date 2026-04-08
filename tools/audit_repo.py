from __future__ import annotations

from pathlib import Path
from collections import defaultdict


IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "node_modules",
}

IGNORE_FILES = {
    ".DS_Store",
}

IMPORTANT_PATHS = {
    "README.md": "README",
    "Makefile": "MAKEFILE",
    "requirements.txt": "REQS",
    "pyproject.toml": "PYPROJECT",
    ".gitignore": "GITIGNORE",
    "pytest.ini": "PYTEST",
    "src": "SRC",
    "tests": "TESTS",
    "data": "DATA",
    "artifacts": "ARTIFACTS",
    "configs": "CONFIGS",
}


def should_ignore(path: Path) -> bool:
    if path.name in IGNORE_FILES:
        return True
    if path.is_dir() and path.name in IGNORE_DIRS:
        return True
    return False


def print_tree(root: Path, max_depth: int = 4) -> None:
    print(f"\n=== REPO TREE: {root} ===\n")

    def walk(current: Path, prefix: str = "", depth: int = 0) -> None:
        if depth > max_depth:
            return

        children = sorted(
            [p for p in current.iterdir() if not should_ignore(p)],
            key=lambda p: (p.is_file(), p.name.lower()),
        )

        for idx, child in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└── " if is_last else "├── "

            marker = ""
            rel_name = child.name
            if depth == 0 and rel_name in IMPORTANT_PATHS:
                marker = f" [{IMPORTANT_PATHS[rel_name]}]"

            print(prefix + connector + rel_name + marker)

            if child.is_dir():
                extension = "    " if is_last else "│   "
                walk(child, prefix + extension, depth + 1)

    walk(root)


def collect_summary(root: Path) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = defaultdict(list)

    for path in root.rglob("*"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.name in IGNORE_FILES:
            continue

        if path.is_file():
            suffix = path.suffix.lower() or "[no_ext]"
            rel = str(path.relative_to(root))
            summary[suffix].append(rel)

    return dict(summary)


def check_key_items(root: Path) -> None:
    print("\n=== KEY ITEMS CHECK ===\n")

    checks = [
        "README.md",
        "Makefile",
        "requirements.txt",
        "pyproject.toml",
        ".gitignore",
        "pytest.ini",
        "src",
        "tests",
        "data",
        "artifacts",
        "configs",
    ]

    for item in checks:
        exists = (root / item).exists()
        status = "OK" if exists else "MISSING"
        print(f"{status:8} {item}")


def suggest_structure(root: Path) -> None:
    print("\n=== QUICK STRUCTURE SUGGESTION ===\n")

    has_src = (root / "src").exists()
    has_tests = (root / "tests").exists()
    has_data = (root / "data").exists()
    has_artifacts = (root / "artifacts").exists()
    has_makefile = (root / "Makefile").exists()
    has_readme = (root / "README.md").exists()

    if has_src and has_tests and has_data and has_artifacts:
        print("Repo ma już sensowny szkielet. Raczej refactor niż przebudowa od zera.")
    else:
        print("Repo nie ma jeszcze pełnego szkieletu. Trzeba dołożyć minimum strukturalne.")

    if not has_makefile:
        print("- Dodać Makefile")
    if not has_readme:
        print("- Uzupełnić README")
    if not has_tests:
        print("- Dodać folder tests/")
    if not has_data:
        print("- Ujednolicić folder data/")
    if not has_artifacts:
        print("- Ujednolicić folder artifacts/")
    if not has_src:
        print("- Uporządkować kod pod src/")


def print_filetype_summary(root: Path, top_n: int = 8) -> None:
    summary = collect_summary(root)

    print("\n=== FILE TYPE SUMMARY ===\n")
    items = sorted(summary.items(), key=lambda kv: len(kv[1]), reverse=True)

    for suffix, files in items[:top_n]:
        print(f"{suffix:12} {len(files):4} files")

    print("\n=== SAMPLE FILES ===\n")
    for suffix, files in items[:top_n]:
        print(f"{suffix}:")
        for rel in files[:5]:
            print(f"  - {rel}")
        if len(files) > 5:
            print(f"  ... (+{len(files) - 5} more)")
        print()


def main() -> None:
    root = Path.cwd()

    print_tree(root, max_depth=4)
    check_key_items(root)
    print_filetype_summary(root)
    suggest_structure(root)


if __name__ == "__main__":
    main()