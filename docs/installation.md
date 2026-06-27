---
icon: lucide/download
---

# Installation

Choose your preferred package manager to install `vsview`.

We recommend the **[recommended](plugins/second-party.md#installation)** or **[full](plugins/second-party.md#installation)** bundle for most users
so that useful plugins are available out of the box.

=== "pip"
    ```bash title="Minimal installation"
    pip install vsview
    ```

    ```bash title="Install with recommended plugins"
    pip install vsview[recommended]
    ```

    ```bash title="Install with all plugins"
    pip install vsview[full]
    ```
=== "uv"
    ```bash title="Minimal installation"
    uv add vsview
    ```

    ```bash title="Install with recommended plugins"
    uv add vsview --extra recommended
    ```

    ```bash title="Install with all plugins"
    uv add vsview --extra full
    ```

## Development Installation

For contributing or local development, see the [Contributing](contributing.md) guide.
