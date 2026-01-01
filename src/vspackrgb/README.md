# vspackrgb

RGB packing for VapourSynth frames.

Converts planar RGB VapourSynth clips into display-ready packed formats:

- **RGB24 → BGRA** (8-bit interleaved)
- **RGB30 → RGB30** (10-bit packed, 2-bit alpha)

Output is stored in a GRAY32 clip.

## Benchmarks

### Blank clip with `keep=True`

```
             RGB24 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ vszip.PackRGB      │   6000 │  2.186s │ 2745.14 │
│ libp2p.Pack        │   6000 │  2.867s │ 2092.89 │
│ akarin.Expr        │   6000 │  2.802s │ 2141.23 │
│ vspackrgb (cython) │   6000 │  8.998s │  666.84 │
│ vspackrgb (numpy)  │   6000 │ 20.182s │  297.30 │
│ vspackrgb (python) │     30 │ 13.706s │    2.19 │
└────────────────────┴────────┴─────────┴─────────┘

             RGB30 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃     FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ vszip.PackRGB      │   6000 │  2.842s │ 2111.21 │
│ libp2p.Pack        │   6000 │  2.817s │ 2129.78 │
│ akarin.Expr        │   6000 │  2.866s │ 2093.62 │
│ vspackrgb (cython) │   6000 │  8.750s │  685.72 │
│ vspackrgb (numpy)  │   6000 │ 30.640s │  195.83 │
│ vspackrgb (python) │     30 │ 11.968s │    2.51 │
└────────────────────┴────────┴─────────┴─────────┘
```

### Real world scenario

```
            RGB24 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vszip.PackRGB      │   6000 │  8.623s │ 695.79 │
│ libp2p.Pack        │   6000 │  9.450s │ 634.93 │
│ akarin.Expr        │   6000 │  9.395s │ 638.65 │
│ vspackrgb (cython) │   6000 │ 12.231s │ 490.57 │
│ vspackrgb (numpy)  │   6000 │ 23.421s │ 256.18 │
│ vspackrgb (python) │     30 │ 13.840s │   2.17 │
└────────────────────┴────────┴─────────┴────────┘

            RGB30 Packing (1920x1080)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Backend            ┃ Frames ┃    Time ┃    FPS ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ vszip.PackRGB      │   6000 │ 10.623s │ 564.79 │
│ libp2p.Pack        │   6000 │ 10.874s │ 551.77 │
│ akarin.Expr        │   6000 │ 10.655s │ 563.13 │
│ vspackrgb (cython) │   6000 │ 13.503s │ 444.35 │
│ vspackrgb (numpy)  │   6000 │ 36.062s │ 166.38 │
│ vspackrgb (python) │     30 │ 12.055s │   2.49 │
└────────────────────┴────────┴─────────┴────────┘

```
