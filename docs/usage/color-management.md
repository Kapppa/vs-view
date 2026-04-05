---
icon: lucide/palette
title: Color Management
description: How VSView handles color space conversion and frame properties.
---

# Color Management

When displaying VapourSynth clips, **VSView** must convert the output to packed RGB for rendering on the screen.

Currently, this conversion explicitly targets **BT.709** primaries and transfer characteristics.

!!! warning "HDR and Out-of-Range Content"

    VSView does not perform tone mapping.
    
    Because display conversion is currently fixed to BT.709 targets,
    HDR content and out-of-range values will be clipped during preview.

    This behavior affects preview only. It does not change the source clip in your script.

## Required Properties

VSView relies on VapourSynth frame properties [`_Matrix`](https://www.vapoursynth.com/doc/apireference.html#reserved-frame-properties),
[`_Primaries`](https://www.vapoursynth.com/doc/apireference.html#reserved-frame-properties)
and [`_Transfer`](https://www.vapoursynth.com/doc/apireference.html#reserved-frame-properties)
to properly interpret a clip's color space during conversion.

If these are missing, the conversion may fail or colors may appear incorrect.

!!! info "Explicit Handling"

    VSView does not infer or "guess" missing properties.

    You must ensure your clip has the correct properties set, or configure a fallback policy.

## Frame Property Policy

You can control how VSView reacts to missing properties via the **Settings Dialog**.

!!! danger "YUV Conversion"

    If the clip's color family is `YUV` and `_Matrix` is unknown or unspecified, VSView will always fail to convert the output to RGB.

- **Error (Default)**

    If `_Matrix`, `_Transfer`, or `_Primaries` are missing, VSView stops rendering and displays a failure image because VapourSynth does not have enough information to convert between color spaces.

-  **Warn**

    Preprocessing steps:

    - If the clip's color family is `RGB`, `_Matrix` is set to `MATRIX_RGB` (if unspecified).
    - If the clip's color family is `GRAY`, `_Matrix`, `_Transfer`, and `_Primaries` are forced to BT709 to prevent any conversion.
    - If the clip's color family is `YUV`, `_Matrix` is left as is.

    No conversion is performed if `_Transfer` or `_Primaries` is missing. A warning is logged instead.

-  **Ignore**

    Follows the same steps as **Warn**, but no warnings are emitted to the log.
