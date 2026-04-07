---
icon: lucide/globe
title: Third-Party Plugins
---

# Third-Party Plugins

Third-party plugins are maintained by independent developers.

## Overview

<div class="grid cards" markdown>

- :lucide-box: **Native Resolution**

    ---

    Descale analysis suite for VSView, featuring `Get Native`, `Get Scaler`, and `Get Frequencies`.

    [:lucide-move-right: Details](#native-resolution) · [:fontawesome-brands-github: Source](https://github.com/Jaded-Encoding-Thaumaturgy/native-res)

</div>

---

## Native Resolution [ :fontawesome-brands-github: ](https://github.com/Jaded-Encoding-Thaumaturgy/native-res){ title="Source Code" }

=== "pip"
    ```bash
    pip install vsview-nativeres
    ```
=== "uv"
    ```bash
    uv add vsview-nativeres
    ```

`vsview-nativeres` provides a frontend for [nativeres](https://github.com/Jaded-Encoding-Thaumaturgy/nativeres), allowing resolution analysis within VSView.

### Features

- **Get Native**: Plots descale error across a specified dimension range.
- **Get Scaler**: Ranks kernels against target dimensions.
- **Get Frequencies**: Visualizes DCT-based frequency distributions.

More information can be found in the [nativeres documentation](https://github.com/Jaded-Encoding-Thaumaturgy/nativeres) and [vsview-nativeres documentation](https://github.com/Jaded-Encoding-Thaumaturgy/nativeres/blob/master/src/plugin/README.md).

### Requirements

- [**nativeres**](https://github.com/Jaded-Encoding-Thaumaturgy/nativeres)
