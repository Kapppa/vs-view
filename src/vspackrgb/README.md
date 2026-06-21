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

- CPU 9800X3D Windows 11 Pro 25H2 (26200.8655)
- Python 3.12.13
- VapourSynth R77 (With unlimited `max_cache_size`)
- vszip 15.0.0
- libp2p R2 (+ RGB48 packing fix)
- akarin 1.4.1
- cython 3.2.5
- numpy 2.4.6
- numba 0.65.1

### Blank clip with `keep=True`

```
             RGB24 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ vszip.PackRGB      │  20000 │  5.895s │ 3392.88 │
│ libp2p.Pack        │  20000 │  5.971s │ 3349.38 │
│ akarin.Expr        │  20000 │  5.951s │ 3360.97 │
│ vspackrgb (cython) │   7000 │  6.370s │ 1098.85 │
│ vspackrgb (numba)  │   7000 │  2.903s │ 2411.60 │
│ vspackrgb (numpy)  │   2000 │  5.754s │  347.57 │
│ vspackrgb (python) │     25 │ 10.029s │    2.49 │
└────────────────────┴────────┴─────────┴─────────┘

            RGB30 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃   Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ vszip.PackRGB      │  20000 │ 5.865s │ 3410.05 │
│ libp2p.Pack        │  20000 │ 5.903s │ 3388.30 │
│ akarin.Expr        │  20000 │ 5.911s │ 3383.56 │
│ vspackrgb (cython) │   7000 │ 7.801s │  897.35 │
│ vspackrgb (numba)  │   7000 │ 2.925s │ 2393.24 │
│ vspackrgb (numpy)  │   2000 │ 8.973s │  222.88 │
│ vspackrgb (python) │     25 │ 7.251s │    3.45 │
└────────────────────┴────────┴────────┴─────────┘

             RGB48 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ libp2p.Pack        │  20000 │ 12.004s │ 1666.09 │
│ vspackrgb (cython) │   7000 │  7.540s │  928.41 │
│ vspackrgb (numba)  │   7000 │  5.416s │ 1292.48 │
│ vspackrgb (numpy)  │   2000 │  8.061s │  248.12 │
│ vspackrgb (python) │     25 │  9.996s │    2.50 │
└────────────────────┴────────┴─────────┴─────────┘

             RGBH Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃   Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ vspackrgb (cython) │   7000 │ 7.513s │  931.70 │
│ vspackrgb (numba)  │   7000 │ 5.392s │ 1298.17 │
│ vspackrgb (numpy)  │   2000 │ 8.116s │  246.42 │
│ vspackrgb (python) │     25 │ 9.979s │    2.51 │
└────────────────────┴────────┴────────┴─────────┘

             RGBS Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   7000 │ 13.385s │ 522.98 │
│ vspackrgb (numba)  │   7000 │ 10.433s │ 670.92 │
│ vspackrgb (numpy)  │   2000 │ 13.386s │ 149.41 │
│ vspackrgb (python) │     25 │  9.890s │   2.53 │
└────────────────────┴────────┴─────────┴────────┘
```

### Real world scenario

Source clip is a 1080p `.m2ts` file muxed to `.mkv`,
indexed with BestSource R18 and resampled to the target format with `resize.Point`

```
            RGB24 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vszip.PackRGB      │   7000 │  9.451s │ 740.69 │
│ libp2p.Pack        │   7000 │  9.402s │ 744.52 │
│ akarin.Expr        │   7000 │  9.572s │ 731.30 │
│ vspackrgb (cython) │   7000 │ 10.504s │ 666.43 │
│ vspackrgb (numba)  │   7000 │  9.586s │ 730.22 │
│ vspackrgb (numpy)  │   2000 │  6.862s │ 291.46 │
│ vspackrgb (python) │     25 │ 10.065s │   2.48 │
└────────────────────┴────────┴─────────┴────────┘

            RGB30 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vszip.PackRGB      │   7000 │ 10.170s │ 688.27 │
│ libp2p.Pack        │   7000 │ 10.321s │ 678.25 │
│ akarin.Expr        │   7000 │ 10.471s │ 668.54 │
│ vspackrgb (cython) │   7000 │ 13.434s │ 521.06 │
│ vspackrgb (numba)  │   7000 │ 10.394s │ 673.43 │
│ vspackrgb (numpy)  │   2000 │ 10.672s │ 187.41 │
│ vspackrgb (python) │     25 │  7.350s │   3.40 │
└────────────────────┴────────┴─────────┴────────┘

            RGB48 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ libp2p.Pack        │   7000 │ 11.665s │ 600.06 │
│ vspackrgb (cython) │   7000 │ 13.496s │ 518.69 │
│ vspackrgb (numba)  │   7000 │ 12.391s │ 564.91 │
│ vspackrgb (numpy)  │   2000 │  9.071s │ 220.48 │
│ vspackrgb (python) │     25 │ 10.061s │   2.48 │
└────────────────────┴────────┴─────────┴────────┘

             RGBH Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   7000 │ 13.244s │ 528.53 │
│ vspackrgb (numba)  │   7000 │ 12.213s │ 573.18 │
│ vspackrgb (numpy)  │   2000 │  8.805s │ 227.14 │
│ vspackrgb (python) │     25 │ 10.002s │   2.50 │
└────────────────────┴────────┴─────────┴────────┘

             RGBS Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vspackrgb (cython) │   7000 │ 22.907s │ 305.58 │
│ vspackrgb (numba)  │   7000 │ 20.397s │ 343.19 │
│ vspackrgb (numpy)  │   2000 │ 17.765s │ 112.58 │
│ vspackrgb (python) │     25 │  9.974s │   2.51 │
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
