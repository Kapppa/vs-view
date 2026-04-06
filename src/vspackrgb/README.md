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
│ vszip.PackRGB      │  20000 │ 5.936s │ 3369.12 │
│ libp2p.Pack        │  20000 │ 6.001s │ 3332.76 │
│ akarin.Expr        │  20000 │ 5.967s │ 3351.74 │
│ vspackrgb (cython) │   6000 │ 6.269s │  957.02 │
│ vspackrgb (numpy)  │   3000 │ 9.175s │  326.97 │
│ vspackrgb (python) │     25 │ 9.856s │    2.54 │
└────────────────────┴────────┴────────┴─────────┘

             RGB30 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ vszip.PackRGB      │  20000 │  6.161s │ 3245.97 │
│ libp2p.Pack        │  20000 │  5.919s │ 3378.69 │
│ akarin.Expr        │  20000 │  6.150s │ 3252.20 │
│ vspackrgb (cython) │   6000 │  7.797s │  769.55 │
│ vspackrgb (numpy)  │   3000 │ 14.281s │  210.06 │
│ vspackrgb (python) │     25 │  7.079s │    3.53 │
└────────────────────┴────────┴─────────┴─────────┘

             RGB48 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ libp2p.Pack        │  20000 │ 15.882s │ 1259.32 │
│ vspackrgb (cython) │   6000 │ 10.350s │  579.73 │
│ vspackrgb (numpy)  │   3000 │ 13.174s │  227.72 │
│ vspackrgb (python) │     25 │  9.813s │    2.55 │
└────────────────────┴────────┴─────────┴─────────┘

             RGBH Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   6000 │ 10.001s │ 599.94 │
│ vspackrgb (numpy)  │   3000 │ 13.225s │ 226.84 │
│ vspackrgb (python) │     25 │ 10.153s │   2.46 │
└────────────────────┴────────┴─────────┴────────┘

             RGBS Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   6000 │ 26.353s │ 227.68 │
│ vspackrgb (numpy)  │   3000 │ 27.185s │ 110.36 │
│ vspackrgb (python) │     25 │ 10.139s │   2.47 │
└────────────────────┴────────┴─────────┴────────┘
```

### Real world scenario

```
            RGB24 Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vszip.PackRGB      │  20000 │ 33.584s │ 595.52 │
│ libp2p.Pack        │  20000 │ 33.120s │ 603.87 │
│ akarin.Expr        │  20000 │ 33.273s │ 601.08 │
│ vspackrgb (cython) │   6000 │ 10.845s │ 553.26 │
│ vspackrgb (numpy)  │   3000 │ 12.321s │ 243.49 │
│ vspackrgb (python) │     25 │ 11.209s │   2.23 │
└────────────────────┴────────┴─────────┴────────┘

            RGB30 Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vszip.PackRGB      │  20000 │ 35.798s │ 558.69 │
│ libp2p.Pack        │  20000 │ 36.223s │ 552.13 │
│ akarin.Expr        │  20000 │ 35.643s │ 561.11 │
│ vspackrgb (cython) │   6000 │ 12.614s │ 475.64 │
│ vspackrgb (numpy)  │   3000 │ 17.685s │ 169.64 │
│ vspackrgb (python) │     25 │  7.733s │   3.23 │
└────────────────────┴────────┴─────────┴────────┘

            RGB48 Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ libp2p.Pack        │  20000 │ 42.315s │ 472.65 │
│ vspackrgb (cython) │   6000 │ 16.693s │ 359.43 │
│ vspackrgb (numpy)  │   3000 │ 17.214s │ 174.28 │
│ vspackrgb (python) │     25 │ 10.677s │   2.34 │
└────────────────────┴────────┴─────────┴────────┘

             RGBH Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   6000 │ 16.374s │ 366.43 │
│ vspackrgb (numpy)  │   3000 │ 16.887s │ 177.65 │
│ vspackrgb (python) │     25 │ 10.976s │   2.28 │
└────────────────────┴────────┴─────────┴────────┘

             RGBS Packing (1920x1080)             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   6000 │ 37.117s │ 161.65 │
│ vspackrgb (numpy)  │   3000 │ 34.515s │  86.92 │
│ vspackrgb (python) │     25 │ 11.092s │   2.25 │
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
