"""NumPy-accelerated RGB packing functions."""

import ctypes


def pack_bgra_8bit(
    b_data: ctypes.Array[ctypes.c_uint8],
    g_data: ctypes.Array[ctypes.c_uint8],
    r_data: ctypes.Array[ctypes.c_uint8],
    width: int,
    height: int,
    src_stride: int,
    dest_ptr: int,
    dest_stride: int,
) -> None:
    """Pack planar 8-bit RGB to interleaved BGRA, writing to dest buffer."""
    import numpy as np

    b_arr = np.frombuffer(b_data, dtype=np.uint8).reshape((height, src_stride))[:, :width]
    g_arr = np.frombuffer(g_data, dtype=np.uint8).reshape((height, src_stride))[:, :width]
    r_arr = np.frombuffer(r_data, dtype=np.uint8).reshape((height, src_stride))[:, :width]

    bgra = np.empty((height, width, 4), dtype=np.uint8)
    bgra[:, :, 0] = b_arr
    bgra[:, :, 1] = g_arr
    bgra[:, :, 2] = r_arr
    bgra[:, :, 3] = 255  # Full alpha

    out = (ctypes.c_uint8 * (dest_stride * height)).from_address(dest_ptr)
    out_arr = np.frombuffer(out, dtype=np.uint8).reshape((height, dest_stride))

    row_bytes = width * 4
    out_arr[:, :row_bytes] = bgra.reshape((height, row_bytes))


def pack_rgb30_10bit(
    r_data: ctypes.Array[ctypes.c_uint16],
    g_data: ctypes.Array[ctypes.c_uint16],
    b_data: ctypes.Array[ctypes.c_uint16],
    width: int,
    height: int,
    samples_per_row: int,
    dest_ptr: int,
    dest_stride: int,
) -> None:
    """Pack planar 10-bit RGB to RGB30 (0xC0RRGGBB), writing to dest buffer."""
    import numpy as np

    r_arr = np.frombuffer(r_data, dtype=np.uint16).reshape((height, samples_per_row))[:, :width]
    g_arr = np.frombuffer(g_data, dtype=np.uint16).reshape((height, samples_per_row))[:, :width]
    b_arr = np.frombuffer(b_data, dtype=np.uint16).reshape((height, samples_per_row))[:, :width]

    dest_samples_per_row = dest_stride // 4
    out = (ctypes.c_uint32 * (dest_samples_per_row * height)).from_address(dest_ptr)
    out_arr = np.frombuffer(out, dtype=np.uint32).reshape((height, dest_samples_per_row))
    out_view = out_arr[:, :width]

    # Pack into RGB30: high 2 bits = 0b11 (alpha), R(10) | G(10) | B(10)
    temp = np.empty((height, width), dtype=np.uint32)

    # R << 20 into temp, then add alpha mask
    np.left_shift(r_arr, 20, out=temp, dtype=np.uint32)
    temp |= 0xC0000000

    # G << 10, add to temp
    np.left_shift(g_arr, 10, out=out_view, dtype=np.uint32)
    temp |= out_view

    # B (just cast) and final OR into output
    np.add(temp, b_arr, out=out_view, dtype=np.uint32)
