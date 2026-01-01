# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython-accelerated RGB packing functions."""

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uintptr_t


cpdef void pack_bgra_8bit(
    const uint8_t[::1] b_data,
    const uint8_t[::1] g_data,
    const uint8_t[::1] r_data,
    int width,
    int height,
    int src_stride,
    uintptr_t dest_ptr,
    int dest_stride,
):
    """Pack planar 8-bit RGB to interleaved BGRA, writing to dest buffer."""
    cdef:
        uint8_t* out_base = <uint8_t*>dest_ptr
        uint8_t* out_row

        const uint8_t* b_ptr = &b_data[0]
        const uint8_t* g_ptr = &g_data[0]
        const uint8_t* r_ptr = &r_data[0]

        int x, y
        int src_row_offset

    with nogil:
        for y in range(height):
            src_row_offset = y * src_stride
            out_row = out_base + y * dest_stride

            for x in range(width):
                out_row[0] = b_ptr[src_row_offset + x]
                out_row[1] = g_ptr[src_row_offset + x]
                out_row[2] = r_ptr[src_row_offset + x]
                out_row[3] = 255
                out_row += 4


cpdef void pack_rgb30_10bit(
    const uint16_t[::1] r_data,
    const uint16_t[::1] g_data,
    const uint16_t[::1] b_data,
    int width,
    int height,
    int samples_per_row,
    uintptr_t dest_ptr,
    int dest_stride,
):
    """Pack planar 10-bit RGB to RGB30 (0xC0RRGGBB), writing to dest buffer."""
    cdef:
        uint8_t* out_base = <uint8_t*>dest_ptr
        uint32_t* out_row
        
        const uint16_t* r_ptr = &r_data[0]
        const uint16_t* g_ptr = &g_data[0]
        const uint16_t* b_ptr = &b_data[0]
        
        int x, y, src_row_offset
        uint32_t r, g, b
        
        uint32_t alpha_mask = 0xC0000000

    with nogil:
        for y in range(height):
            src_row_offset = y * samples_per_row
            out_row = <uint32_t*>(out_base + y * dest_stride)
            
            for x in range(width):
                r = r_ptr[src_row_offset + x]
                g = g_ptr[src_row_offset + x]
                b = b_ptr[src_row_offset + x]
                out_row[x] = alpha_mask | (r << 20) | (g << 10) | b
