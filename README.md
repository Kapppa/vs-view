# VSView

<div align="center">

<img src="https://github.com/Jaded-Encoding-Thaumaturgy/vs-view/raw/main/src/vsview/assets/loading.png" height="200"/>

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Lint](https://github.com/Jaded-Encoding-Thaumaturgy/vs-view/actions/workflows/lint.yml/badge.svg)](https://github.com/Jaded-Encoding-Thaumaturgy/vs-view/actions/workflows/lint.yml)
[![Discord](https://img.shields.io/discord/856381934052704266?label=Discord&logo=discord&logoColor=7F71FF)](https://discord.gg/XTpc6Fa9eB)

**The next-generation VapourSynth previewer**

</div>

Modern, extensible previewer for [VapourSynth](https://github.com/vapoursynth/vapoursynth), **VSView** lets you open scripts, videos or images in one interface, making it easier to preview, inspect and compare sources without switching tools.

<!-- prettier-ignore -->
> [!WARNING]
> **Beta Software**: This project is currently in early beta. Features are missing, bugs are expected, and the API is subject to breaking changes.

## Documentation

For comprehensive guides, feature overviews, and detailed usage instructions, visit the **[Official Documentation](https://jaded-encoding-thaumaturgy.github.io/vs-view/)**.

## Installation

### Prerequisites

- **[Python](https://www.python.org/)** `>=3.12`
- **[VapourSynth](https://www.vapoursynth.com/)** `R69+`
    - **[BestSource](https://github.com/vapoursynth/bestsource)** (optional)

### Install with pip

The quickest way to install `vsview`:

```bash
pip install vsview
```

For the recommended setup with commonly used plugins:

```bash
pip install vsview[essential]
```

### Install with uv

If you use [uv](https://docs.astral.sh/uv/) for package management:

```bash
uv add vsview
```

With essential plugins:

```bash
uv add vsview --extra essential
```

## Contributing

Contributions are welcome! Please check the [Discord server](https://discord.gg/XTpc6Fa9eB) or open an issue to discuss planned features.

For guidance on setting up a development environment, see [Contributing](https://jaded-encoding-thaumaturgy.github.io/vs-view/contributing/).
