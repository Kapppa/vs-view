# VSPackRGB

RGB packing for VapourSynth frames.

Converts planar RGB VapourSynth clips into display-ready packed formats:

- **RGB24 → BGRA** (8-bit interleaved, stored in `GRAY32`)
- **RGB30 → A2R10G10B10** (10-bit packed, stored in `GRAY32`)
- **RGB48 → RGBA64** (16-bit interleaved, stored in `GRAY16`)
- **RGBH → RGBA16F** (16-bit float interleaved, stored in `GRAYH`)
- **RGBS → RGBA32F** (32-bit float interleaved, stored in `GRAYS`)

For higher-than-10-bit formats, the output clip is 4x wider than the input to accommodate the interleaved R, G, B, and A channels.

## Installation

Prebuilt wheels are provided for most platforms. If a compatible wheel is available, no compilation is required.

```bash
pip install vspackrgb
```

With `uv`:

```bash
uv add vspackrgb
```

## Benchmarks

### Blank clip with `keep=True`

```
            RGB24 Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃   Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ vszip.PackRGB      │  20000 │ 5.917s │ 3379.98 │
│ libp2p.Pack        │  20000 │ 5.975s │ 3347.32 │
│ akarin.Expr        │  20000 │ 5.962s │ 3354.39 │
│ vspackrgb (cython) │   6000 │ 5.556s │ 1079.89 │
│ vspackrgb (numpy)  │   3000 │ 8.605s │  348.63 │
│ vspackrgb (python) │     25 │ 9.761s │    2.56 │
└────────────────────┴────────┴────────┴─────────┘

             RGB30 Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ vszip.PackRGB      │  20000 │  5.911s │ 3383.25 │
│ libp2p.Pack        │  20000 │  5.915s │ 3381.34 │
│ akarin.Expr        │  20000 │  5.933s │ 3370.95 │
│ vspackrgb (cython) │   6000 │  6.437s │  932.18 │
│ vspackrgb (numpy)  │   3000 │ 13.371s │  224.37 │
│ vspackrgb (python) │     25 │  7.044s │    3.55 │
└────────────────────┴────────┴─────────┴─────────┘

             RGB48 Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ libp2p.Pack        │  20000 │ 15.819s │ 1264.30 │
│ vspackrgb (cython) │   6000 │  8.459s │  709.29 │
│ vspackrgb (numpy)  │   3000 │ 12.185s │  246.20 │
│ vspackrgb (python) │     25 │  9.870s │    2.53 │
└────────────────────┴────────┴─────────┴─────────┘

             RGBH Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   6000 │  8.883s │ 675.47 │
│ vspackrgb (numpy)  │   3000 │ 12.888s │ 232.78 │
│ vspackrgb (python) │     25 │  9.858s │   2.54 │
└────────────────────┴────────┴─────────┴────────┘

             RGBS Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   6000 │ 23.383s │ 256.60 │
│ vspackrgb (numpy)  │   3000 │ 24.899s │ 120.48 │
│ vspackrgb (python) │     25 │  9.750s │   2.56 │
└────────────────────┴────────┴─────────┴────────┘
```

### Real world scenario

```
            RGB24 Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vszip.PackRGB      │  20000 │ 32.561s │ 614.23 │
│ libp2p.Pack        │  20000 │ 32.851s │ 608.82 │
│ akarin.Expr        │  20000 │ 33.114s │ 603.97 │
│ vspackrgb (cython) │   6000 │  9.998s │ 600.09 │
│ vspackrgb (numpy)  │   3000 │ 10.169s │ 295.02 │
│ vspackrgb (python) │     25 │  9.957s │   2.51 │
└────────────────────┴────────┴─────────┴────────┘

            RGB30 Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vszip.PackRGB      │  20000 │ 35.110s │ 569.64 │
│ libp2p.Pack        │  20000 │ 35.950s │ 556.33 │
│ akarin.Expr        │  20000 │ 35.505s │ 563.31 │
│ vspackrgb (cython) │   6000 │ 10.987s │ 546.10 │
│ vspackrgb (numpy)  │   3000 │ 15.695s │ 191.14 │
│ vspackrgb (python) │     25 │  7.110s │   3.52 │
└────────────────────┴────────┴─────────┴────────┘

            RGB48 Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ libp2p.Pack        │  20000 │ 42.757s │ 467.76 │
│ vspackrgb (cython) │   6000 │ 13.680s │ 438.59 │
│ vspackrgb (numpy)  │   3000 │ 14.353s │ 209.02 │
│ vspackrgb (python) │     25 │  9.774s │   2.56 │
└────────────────────┴────────┴─────────┴────────┘

             RGBH Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   6000 │ 13.862s │ 432.83 │
│ vspackrgb (numpy)  │   3000 │ 14.358s │ 208.95 │
│ vspackrgb (python) │     25 │  9.785s │   2.55 │
└────────────────────┴────────┴─────────┴────────┘

             RGBS Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   6000 │ 32.984s │ 181.90 │
│ vspackrgb (numpy)  │   3000 │ 29.858s │ 100.48 │
│ vspackrgb (python) │     25 │  9.788s │   2.55 │
└────────────────────┴────────┴─────────┴────────┘
```

## Building

You only need a working C compiler/toolchain for your platform:

- Windows: Visual Studio Build Tools (Desktop development with C++)
- Linux: GCC/Clang and Python headers
- macOS: Xcode Command Line Tools

```bash
uv build --sdist --wheel --verbose
```
