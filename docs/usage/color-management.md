---
icon: lucide/palette
title: Color Management
description: How VSView interprets clip color properties before converting frames for display.
---

# Color Management

When displaying VapourSynth clips, **VSView** must convert the output to packed RGB for rendering on the screen.

Currently, this conversion explicitly targets **BT.709** primaries and transfer characteristics.

## Display Conversion Limits

!!! warning "HDR and Out-of-Range Content"

    VSView does not perform tone mapping.
    
    Because display conversion is currently fixed to BT.709 targets,
    HDR content and out-of-range values will be clipped during preview.

This behavior affects preview only. It does not change the source clip in your script.

## Required Frame Properties

VSView relies on VapourSynth frame properties [`_Matrix`](https://www.vapoursynth.com/doc/apireference.html#reserved-frame-properties), [`_Primaries`](https://www.vapoursynth.com/doc/apireference.html#reserved-frame-properties) and [`_Transfer`](https://www.vapoursynth.com/doc/apireference.html#reserved-frame-properties) to properly interpret a clip's color space during conversion.

## Frame Property Policy

If `_Transfer` or `_Primaries` is missing or unspecified, VSView applies the selected **Frame property policy** from the **Settings Dialog**.

!!! note

    Before applying the policy, VSView normalizes an unspecified `_Matrix` value as follows:

    - `GRAY` input is treated as `MATRIX_BT709`.
    - `RGB` input is treated as `MATRIX_RGB`.
    - `YUV` input remains unspecified, which will likely result in the clip not being displayed at all.

### `Error` (default)

No property guessing is performed.

If `_Transfer` or `_Primaries` is missing, those properties remain unspecified.

During preview, the resize step will then usually fail because VapourSynth does not have enough information to convert between color spaces.

In that case, VSView reports the rendering error and displays the generated failure image for the frame instead of a normal preview.

### `Warn`

VSView attempts to infer missing `_Transfer` and `_Primaries` values from `_Matrix`.

A warning is emitted the first time a specific fallback value is assumed during preview.

### `Ignore`

VSView uses the same inference logic as `Warn`, but does not emit warnings.

## Property Guessing Rules

When the policy is `Warn` or `Ignore`, VSView infers missing [transfer characteristics](https://www.vapoursynth.com/doc/pythonreference.html#transfercharacteristics) and [color primaries](https://www.vapoursynth.com/doc/pythonreference.html#color-primaries) from the `_Matrix` property using the [mapping below](color-management.md#mapping-table).

If `_Matrix` is also missing or unspecified for a non-`GRAY` and non-`RGB` clip, VSView cannot infer the missing values.

!!! warning

    This inference is **heuristic**.
    
    `_Matrix` does not uniquely determine primaries or transfer characteristics.
    The rules reflect common modern encoding practices rather than strict historical standards.


### SDR Transfer Assumptions

For SDR content, the transfer code points:

* `TRANSFER_BT709` (1)
* `TRANSFER_BT601` (6)
* `TRANSFER_BT2020_10` (14)
* `TRANSFER_BT2020_12` (15)

are treated as **functionally equivalent** (per H.273 / [VapourSynth](https://www.vapoursynth.com/doc/functions/video/resize.html)).
VSView selects a conventional value based on the matrix family for consistency with [mpv](https://mpv.io/).

---

### Mapping Table

| Matrix (`_Matrix`)                               | Assumed Primaries (`_Primaries`) | Assumed Transfer (`_Transfer`) |
| :----------------------------------------------- | :------------------------------- | :----------------------------- |
| `MATRIX_RGB` (0)                                 | `PRIMARIES_BT709` (1)            | `TRANSFER_IEC_61966_2_1` (13)  |
| `MATRIX_BT709` (1)                               | `PRIMARIES_BT709` (1)            | `TRANSFER_BT709` (1)           |
| `MATRIX_BT470_BG` (4)                            | `PRIMARIES_BT709` (1)            | `TRANSFER_BT601` (6)           |
| `MATRIX_ST170_M` (5)                             | `PRIMARIES_BT709` (1)            | `TRANSFER_BT601` (6)           |
| `MATRIX_ST240_M` (6)                             | `PRIMARIES_BT709` (1)            | `TRANSFER_BT601` (6)           |
| `MATRIX_YCGCO` (8)                               | `PRIMARIES_BT709` (1)            | `TRANSFER_BT709` (1)           |
| `MATRIX_BT2020_NCL` / `MATRIX_BT2020_CL` (9, 10) | `PRIMARIES_BT2020` (9)           | `TRANSFER_BT2020_10` (14)      |
| `MATRIX_ICTCP` (14)                              | `PRIMARIES_BT709` (1)            | `TRANSFER_BT2020_10` (14)      |

!!! info
    These defaults are chosen to match **typical modern SDR signaling**, not to reconstruct original mastering intent.

    In particular, combinations such as `BT.470BG` matrix with `BT.709` primaries reflect a **pragmatic modern assumption** (as used by tools like [mpv](https://mpv.io/)), rather than a historically guaranteed pairing.

## Recommended Practice

!!! info

    Relying on inferred properties is convenient for previewing incomplete scripts, but it is not recommended if accurate colorimetry is required.

!!! tip

    Set `_Matrix`, `_Transfer`, and `_Primaries` explicitly in the VapourSynth script before outputting the clip, for example with `std.SetFrameProps`.
