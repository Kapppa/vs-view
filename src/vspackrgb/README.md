# vspackrgb

RGB packing for VapourSynth frames.

Converts planar RGB VapourSynth clips into display-ready packed formats:

- **RGB24 → BGRA** (8-bit interleaved)
- **RGB30 → RGB30** (10-bit packed, 2-bit alpha)

Output is stored in a GRAY32 clip.

## Backends

| Backend  | Speed     |
| -------- | --------- |
| `cython` | Fastest   |
| `numpy`  | Fast      |
| `python` | Very slow |
