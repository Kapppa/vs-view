# Installation

## Prerequisites

- **[Python](https://www.python.org/)** `>=3.12`
- **[VapourSynth](https://www.vapoursynth.com/)** `R69+`
    - **[BestSource](https://github.com/vapoursynth/bestsource)** (optional)

## Installation

Choose your preferred package manager to install `vsview`.
We recommend the **[essential](https://jaded-encoding-thaumaturgy.github.io/vs-view/plugins/second-party/#installation)** bundle for most users.

=== "pip"
    ```bash title="Standard installation"
    pip install vsview
    ```

    ```bash title="Install essential bundle"
    pip install vsview[essential]
    ```
=== "uv"
    ```bash title="Standard installation"
    uv add vsview
    ```

    ```bash title="Add essential bundle"
    uv add vsview --extra essential
    ```

## Development Installation

For contributing or local development, see the [Contributing](contributing.md) guide.
