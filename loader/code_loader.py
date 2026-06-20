from pathlib import Path
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".md",
    ".json",
    ".yaml",
    ".yml"
}

IGNORE_FOLDERS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build"
}


def load_repository(repo_path: str):

    documents = []

    repo_path = Path(repo_path)

    for file_path in repo_path.rglob("*"):

        if not file_path.is_file():
            continue

        if any(
            ignored in file_path.parts
            for ignored in IGNORE_FOLDERS
        ):
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:

            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            doc = Document(
                page_content=content,
                metadata={
                "file_name": file_path.name,
                "file_path": str(file_path.relative_to(repo_path)),
                "extension": file_path.suffix,
                "folder": file_path.parent.name
            }
            )

            documents.append(doc)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return documents