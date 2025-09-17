"""
Fix all import issues in the codebase
"""

import os
import re
from pathlib import Path


def fix_typing_imports(file_path):
    """Fix typing imports in a Python file"""

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if file uses typing annotations
    has_typing = bool(re.search(r"\b(List|Dict|Tuple|Optional|Union)\[", content))

    if not has_typing:
        return False  # No changes needed

    # Check if typing import exists
    has_import = "from typing import" in content or "import typing" in content

    if has_import:
        return False  # Already has import

    # Find import section
    lines = content.split("\n")
    import_end = 0

    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            import_end = i
        elif line.strip() == "" and import_end > 0:
            continue
        elif import_end > 0:
            break

    # Add typing import
    typing_imports = []
    if re.search(r"\bList\[", content):
        typing_imports.append("List")
    if re.search(r"\bDict\[", content):
        typing_imports.append("Dict")
    if re.search(r"\bTuple\[", content):
        typing_imports.append("Tuple")
    if re.search(r"\bOptional\[", content):
        typing_imports.append("Optional")
    if re.search(r"\bUnion\[", content):
        typing_imports.append("Union")

    if typing_imports:
        import_line = f"from typing import {', '.join(typing_imports)}"
        lines.insert(import_end + 1, import_line)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✓ Fixed typing imports in {file_path}")
        return True

    return False


def scan_and_fix_directory(directory):
    """Scan directory and fix all Python files"""

    fixed_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_typing_imports(file_path):
                    fixed_files.append(file_path)

    return fixed_files


def main():
    print("=== Fixing All Import Issues ===")

    # Fix scripts directory
    scripts_dir = Path(__file__).parent
    fixed_files = scan_and_fix_directory(scripts_dir)

    if fixed_files:
        print(f"Fixed {len(fixed_files)} files:")
        for file_path in fixed_files:
            print(f"  - {file_path}")
    else:
        print("✓ No typing import issues found in scripts")

    # Also check models directory
    models_dir = scripts_dir.parent / "models"
    fixed_files_models = scan_and_fix_directory(models_dir)

    if fixed_files_models:
        print(f"Fixed {len(fixed_files_models)} model files:")
        for file_path in fixed_files_models:
            print(f"  - {file_path}")
    else:
        print("✓ No typing import issues found in models")

    print("\n✓ All import issues have been checked and fixed!")


if __name__ == "__main__":
    main()
