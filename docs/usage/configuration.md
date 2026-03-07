---
icon: lucide/settings
title: Configuration & CLI
description: Command-line options and environment variables for VSView.
---

# Configuration & CLI

VSView can be configured through **command-line arguments** and **environment variables**.

!!! note

    Application settings (appearance, timeline, playback, etc.) are managed
    through the built-in **Settings Dialog** accessible from the menu bar.

---

## Arguments

```
vsview [OPTIONS] [INPUT ...]
```

`INPUT`
:   One or more file paths to open. VSView detects the type automatically:

    - `.py` / `.vpy` files open as VapourSynth scripts
    - Everything else opens as a video or image

    If omitted, VSView launches with default workspaces.

---

## Options

#### `--settings-path`
:   Print to stdout the resolved `global_settings.json` path and exit.

    The resolved path respects environment scoping if `--settings-env` is active.

    Default base directory:

    - `%LOCALAPPDATA%\vsview\` on Windows
    - `~/.config/vsview/` on Linux
    - `~/Library/Application Support/vsview/` on macOS

#### `--settings-wipe`
:   Delete the `global_settings.json` file (as shown by `--settings-path`) and exit.

#### `--settings-wipe-all`
:   Delete the entire settings directory (including all environment-scoped subdirectories) and exit.

#### `--no-settings`
:   Run without loading or saving any settings for this session.

#### `--settings-roaming`
:   **Windows only.** Store global settings in `%APPDATA%\vsview\` instead of `%LOCALAPPDATA%\vsview\`.

#### `--settings-env`
:   Scope settings to the active Python environment.

    Each environment gets its own subdirectory, preventing conflicts across virtual environments.

#### `--settings-env-copy`
:   If `--settings-env` is set and the scoped file doesn't exist yet, seed it from the base `global_settings.json`.

#### `--verbose` / `-v`
:   Enable verbose output. Repeat to increase verbosity (`-vv`, `-vvv`).

---

## Environment Variables

| Variable                                    | Equivalent flag        |
| :------------------------------------------ | :--------------------- |
| `VSVIEW_GLOBAL_SETTINGS_ROAMING`            | `--settings-roaming`   |
| `VSVIEW_GLOBAL_SETTINGS_ENVIRONMENT`        | `--settings-env`       |
| `VSVIEW_GLOBAL_SETTINGS_ENVIRONMENT_COPY`   | `--settings-env-copy`  |
