# Contributing

## Development Installation

**[uv](https://github.com/astral-sh/uv)** is the recommended tool for development.

Clone the repository and sync all packages:

```bash
git clone --recurse-submodules https://github.com/Jaded-Encoding-Thaumaturgy/vs-view.git
cd vs-view
uv sync --all-extras --all-groups
```

Run the development version:

```bash
uv run vsview
```

!!! note "Cython Extensions"

    If you are in an environment where you cannot compile Cython extensions, in `pyproject.toml`:

    - Remove `"src/vspackrgb"` from `tool.uv.workspace.members`
    - Comment out `vspackrgb = { workspace = true }` in `tool.uv.sources`

    You can now run `uv sync` to use the precompiled version from PyPi.

## Recommended Editor Settings

### VSCode / VSCodium

The settings below configure formatting, type-checking, and file associations consistently across the codebase.

Copy them into your `vsview/.vscode/settings.json`:

```json title="vsview/.vscode/settings.json"
{
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "[json]": {
        "editor.defaultFormatter": "vscode.json-language-features",
        "editor.tabSize": 2
    },
    "[toml]": {
        "editor.defaultFormatter": "tamasfe.even-better-toml"
    },
    "[github-actions-workflow]": {
        "editor.defaultFormatter": "esbenp.prettier-vscode"
    },
    "files.associations": {
        "*.vpy": "python"
    },
    "editor.codeActionsOnSave": {
        "source.organizeImports": "always"
    },
    "editor.formatOnSave": true,
    "mypy-type-checker.args": [
        "--fixed-format-cache"
    ],
    "mypy-type-checker.importStrategy": "fromEnvironment",
    "python.analysis.autoFormatStrings": true,
    "python.analysis.autoImportCompletions": true,
    "python.analysis.packageIndexDepths": [
        {
            "depth": 2,
            "name": "PySide6"
        }
    ],
    "python.analysis.stubPath": "stubs",
    "python.analysis.typeCheckingMode": "standard",
    "python.analysis.typeEvaluation.deprecateTypingAliases": true,
    "python.analysis.typeEvaluation.enableReachabilityAnalysis": true,
    "python.testing.pytestArgs": [
        "."
    ],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "search.exclude": {
        "**/submodules": true
    },
}
```
