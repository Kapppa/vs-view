"""Pure Python RGB packing (reference only, too slow for real-time)."""

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
    out = (ctypes.c_uint8 * (dest_stride * height)).from_address(dest_ptr)

    for y in range(height):
        src_row = y * src_stride
        dst_row = y * dest_stride

        for x in range(width):
            dst_offset = dst_row + x * 4
            out[dst_offset + 0] = b_data[src_row + x]
            out[dst_offset + 1] = g_data[src_row + x]
            out[dst_offset + 2] = r_data[src_row + x]
            out[dst_offset + 3] = 255


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
    out = (ctypes.c_uint32 * ((dest_stride // 4) * height)).from_address(dest_ptr)
    dest_samples_per_row = dest_stride // 4

    for y in range(height):
        src_row = y * samples_per_row
        dst_row = y * dest_samples_per_row

        for x in range(width):
            r = r_data[src_row + x]
            g = g_data[src_row + x]
            b = b_data[src_row + x]
            out[dst_row + x] = 0xC0000000 | (r << 20) | (g << 10) | b
