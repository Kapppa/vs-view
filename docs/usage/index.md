# Usage Overview

This page provides an overview of the **VSView** interface and its core components.

## Interface at a Glance

**VSView** uses a workspace-based layout to organize scripts and files and provides access to integrated tools.

<figure markdown="span">
    [![](../../assets/app_overview.png){ .lightboxOn }]
</figure>

---

## 1. The Sidebar

The sidebar on the left is your primary navigation hub. Each open workspace is represented by an icon.

### Navigation & Management
- **Switching Workspaces**: Click an icon to switch the main content area to that workspace.
- **Reordering**: You can click and drag icons to move them up or down.
- **Context Menu**: Right-click any sidebar icon to access workspace-specific actions:
    - **Reload**: Refresh the active workspace.
    - **Clear**: Reset the workspace to its default state.
    - **Delete**: Close the workspace and free up its resources.

---

## 2. The Menu Bar

The top menu bar provides global controls that affect the entire application.

### `New` Menu
Bring new content into **VSView**:

- **`Load Script...`**: Open a VapourSynth (`.vpy`) or Python (`.py`) script.
- **`Load File...`**: Open a video or image file.
- **Workspace Submenu**: Manually create an empty workspace of a specific type.

### `View` Menu
Manage your workspace environment:

- **`Tool Docks`**: Toggle the visibility of persistent dock tools.
- **`Tool Panels`**: Toggle the visibility of tabbed tools.

### `Settings`
Opens the **Settings Dialog**, where you can customize shortcuts, application and plugin settings.

---

## 3. The Main Content Area

The main area displays the active workspace's content.
See [Workspaces](workspaces/index.md) for more information.

---

## Quick Start: Launching VSView

### Command Line Interface

**VSView** detects input types automatically based on file extensions.

| Extension                            | Type               |
| ------------------------------------ | ------------------ |
| `.py`, `.vpy`                        | VapourSynth script |
| `.mkv`, `.mp4`, `.png`, `.jpg`, etc. | Video & Image file |

=== "uv"
    ```bash
    # Open VSView
    uv run vsview

    # Open specific files
    uv run vsview script.vpy video.mkv
    ```
=== "pip"
    ```bash
    # Open VSView
    vsview

    # Open specific files
    vsview script.vpy video.mkv
    ```

### Registering Outputs in Scripts

Register clips in VapourSynth scripts using `set_output()`:

```python title="example.vpy"
import vapoursynth as vs
from vsview import set_output

core = vs.core
clip = core.bs.VideoSource("video.mkv")

set_output(clip, "Source")
set_output(clip.std.Invert(), "Inverted")
```
