"""Triton SHA-256 kernel for batched short-message hashing.

Scope: messages up to 55 bytes (single SHA-256 block after the 9-byte
pad). Covers brain-wallet passphrases and the intermediate SHA-256
of a compressed pubkey (33 bytes).

Layout: one Triton program per input. A single @triton.jit helper
(_rotr32) is used; everything else is inlined since Triton does NOT
allow nested function definitions inside @triton.jit.

Bit-exact correctness verified via `test_triton_correctness.py`
against `hashlib.sha256` on random inputs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


SHA256_K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)


@triton.jit
def _rotr32(x, n: tl.constexpr):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


@triton.jit
def _sha256_short_kernel(
    msg_ptr,                       # *u8  (N, 64)
    len_ptr,                       # *i32 (N,)
    out_ptr,                       # *u8  (N, 32)
    n_rows,
    K_ptr,                         # *u32 (64,)
):
    pid = tl.program_id(axis=0)
    if pid >= n_rows:
        return

    L_raw = tl.load(len_ptr + pid)
    L = tl.minimum(L_raw, 55).to(tl.uint32)

    # ---- Load the 64 input bytes for this row and apply SHA-256 padding ----
    byte_idx = tl.arange(0, 64)
    raw = tl.load(msg_ptr + pid * 64 + byte_idx).to(tl.uint32)

    # Zero out anything past the declared length L
    raw = tl.where(byte_idx < L.to(tl.int32), raw, 0)
    # Insert 0x80 at position L
    raw = tl.where(byte_idx == L.to(tl.int32), 0x80, raw)
    # Insert big-endian (L * 8) in the last four bytes (positions 60..63)
    bits = (L * 8)
    raw = tl.where(byte_idx == 60, (bits >> 24) & 0xFF, raw)
    raw = tl.where(byte_idx == 61, (bits >> 16) & 0xFF, raw)
    raw = tl.where(byte_idx == 62, (bits >>  8) & 0xFF, raw)
    raw = tl.where(byte_idx == 63, (bits      ) & 0xFF, raw)

    # ---- Pack 64 bytes into 16 big-endian u32 words ----
    # Standard Triton idiom for extracting one scalar from a vector:
    #   tl.sum(tl.where(byte_idx == k, raw, 0)).
    b00 = tl.sum(tl.where(byte_idx ==  0, raw, 0))
    b01 = tl.sum(tl.where(byte_idx ==  1, raw, 0))
    b02 = tl.sum(tl.where(byte_idx ==  2, raw, 0))
    b03 = tl.sum(tl.where(byte_idx ==  3, raw, 0))
    b04 = tl.sum(tl.where(byte_idx ==  4, raw, 0))
    b05 = tl.sum(tl.where(byte_idx ==  5, raw, 0))
    b06 = tl.sum(tl.where(byte_idx ==  6, raw, 0))
    b07 = tl.sum(tl.where(byte_idx ==  7, raw, 0))
    b08 = tl.sum(tl.where(byte_idx ==  8, raw, 0))
    b09 = tl.sum(tl.where(byte_idx ==  9, raw, 0))
    b10 = tl.sum(tl.where(byte_idx == 10, raw, 0))
    b11 = tl.sum(tl.where(byte_idx == 11, raw, 0))
    b12 = tl.sum(tl.where(byte_idx == 12, raw, 0))
    b13 = tl.sum(tl.where(byte_idx == 13, raw, 0))
    b14 = tl.sum(tl.where(byte_idx == 14, raw, 0))
    b15 = tl.sum(tl.where(byte_idx == 15, raw, 0))
    b16 = tl.sum(tl.where(byte_idx == 16, raw, 0))
    b17 = tl.sum(tl.where(byte_idx == 17, raw, 0))
    b18 = tl.sum(tl.where(byte_idx == 18, raw, 0))
    b19 = tl.sum(tl.where(byte_idx == 19, raw, 0))
    b20 = tl.sum(tl.where(byte_idx == 20, raw, 0))
    b21 = tl.sum(tl.where(byte_idx == 21, raw, 0))
    b22 = tl.sum(tl.where(byte_idx == 22, raw, 0))
    b23 = tl.sum(tl.where(byte_idx == 23, raw, 0))
    b24 = tl.sum(tl.where(byte_idx == 24, raw, 0))
    b25 = tl.sum(tl.where(byte_idx == 25, raw, 0))
    b26 = tl.sum(tl.where(byte_idx == 26, raw, 0))
    b27 = tl.sum(tl.where(byte_idx == 27, raw, 0))
    b28 = tl.sum(tl.where(byte_idx == 28, raw, 0))
    b29 = tl.sum(tl.where(byte_idx == 29, raw, 0))
    b30 = tl.sum(tl.where(byte_idx == 30, raw, 0))
    b31 = tl.sum(tl.where(byte_idx == 31, raw, 0))
    b32 = tl.sum(tl.where(byte_idx == 32, raw, 0))
    b33 = tl.sum(tl.where(byte_idx == 33, raw, 0))
    b34 = tl.sum(tl.where(byte_idx == 34, raw, 0))
    b35 = tl.sum(tl.where(byte_idx == 35, raw, 0))
    b36 = tl.sum(tl.where(byte_idx == 36, raw, 0))
    b37 = tl.sum(tl.where(byte_idx == 37, raw, 0))
    b38 = tl.sum(tl.where(byte_idx == 38, raw, 0))
    b39 = tl.sum(tl.where(byte_idx == 39, raw, 0))
    b40 = tl.sum(tl.where(byte_idx == 40, raw, 0))
    b41 = tl.sum(tl.where(byte_idx == 41, raw, 0))
    b42 = tl.sum(tl.where(byte_idx == 42, raw, 0))
    b43 = tl.sum(tl.where(byte_idx == 43, raw, 0))
    b44 = tl.sum(tl.where(byte_idx == 44, raw, 0))
    b45 = tl.sum(tl.where(byte_idx == 45, raw, 0))
    b46 = tl.sum(tl.where(byte_idx == 46, raw, 0))
    b47 = tl.sum(tl.where(byte_idx == 47, raw, 0))
    b48 = tl.sum(tl.where(byte_idx == 48, raw, 0))
    b49 = tl.sum(tl.where(byte_idx == 49, raw, 0))
    b50 = tl.sum(tl.where(byte_idx == 50, raw, 0))
    b51 = tl.sum(tl.where(byte_idx == 51, raw, 0))
    b52 = tl.sum(tl.where(byte_idx == 52, raw, 0))
    b53 = tl.sum(tl.where(byte_idx == 53, raw, 0))
    b54 = tl.sum(tl.where(byte_idx == 54, raw, 0))
    b55 = tl.sum(tl.where(byte_idx == 55, raw, 0))
    b56 = tl.sum(tl.where(byte_idx == 56, raw, 0))
    b57 = tl.sum(tl.where(byte_idx == 57, raw, 0))
    b58 = tl.sum(tl.where(byte_idx == 58, raw, 0))
    b59 = tl.sum(tl.where(byte_idx == 59, raw, 0))
    b60 = tl.sum(tl.where(byte_idx == 60, raw, 0))
    b61 = tl.sum(tl.where(byte_idx == 61, raw, 0))
    b62 = tl.sum(tl.where(byte_idx == 62, raw, 0))
    b63 = tl.sum(tl.where(byte_idx == 63, raw, 0))

    w0  = ((b00 << 24) | (b01 << 16) | (b02 << 8) | b03) & 0xFFFFFFFF
    w1  = ((b04 << 24) | (b05 << 16) | (b06 << 8) | b07) & 0xFFFFFFFF
    w2  = ((b08 << 24) | (b09 << 16) | (b10 << 8) | b11) & 0xFFFFFFFF
    w3  = ((b12 << 24) | (b13 << 16) | (b14 << 8) | b15) & 0xFFFFFFFF
    w4  = ((b16 << 24) | (b17 << 16) | (b18 << 8) | b19) & 0xFFFFFFFF
    w5  = ((b20 << 24) | (b21 << 16) | (b22 << 8) | b23) & 0xFFFFFFFF
    w6  = ((b24 << 24) | (b25 << 16) | (b26 << 8) | b27) & 0xFFFFFFFF
    w7  = ((b28 << 24) | (b29 << 16) | (b30 << 8) | b31) & 0xFFFFFFFF
    w8  = ((b32 << 24) | (b33 << 16) | (b34 << 8) | b35) & 0xFFFFFFFF
    w9  = ((b36 << 24) | (b37 << 16) | (b38 << 8) | b39) & 0xFFFFFFFF
    w10 = ((b40 << 24) | (b41 << 16) | (b42 << 8) | b43) & 0xFFFFFFFF
    w11 = ((b44 << 24) | (b45 << 16) | (b46 << 8) | b47) & 0xFFFFFFFF
    w12 = ((b48 << 24) | (b49 << 16) | (b50 << 8) | b51) & 0xFFFFFFFF
    w13 = ((b52 << 24) | (b53 << 16) | (b54 << 8) | b55) & 0xFFFFFFFF
    w14 = ((b56 << 24) | (b57 << 16) | (b58 << 8) | b59) & 0xFFFFFFFF
    w15 = ((b60 << 24) | (b61 << 16) | (b62 << 8) | b63) & 0xFFFFFFFF

    # Initial hash values H0..H7
    a = tl.cast(0x6A09E667, tl.uint32)
    b = tl.cast(0xBB67AE85, tl.uint32)
    c = tl.cast(0x3C6EF372, tl.uint32)
    d = tl.cast(0xA54FF53A, tl.uint32)
    e = tl.cast(0x510E527F, tl.uint32)
    f = tl.cast(0x9B05688C, tl.uint32)
    g = tl.cast(0x1F83D9AB, tl.uint32)
    h = tl.cast(0x5BE0CD19, tl.uint32)

    # 64 rounds. The message schedule is a 16-word rolling window
    # (w0..w15) updated each round once we pass the first 16.
    for t in tl.static_range(64):
        if t == 0:
            Wt = w0
        elif t == 1:
            Wt = w1
        elif t == 2:
            Wt = w2
        elif t == 3:
            Wt = w3
        elif t == 4:
            Wt = w4
        elif t == 5:
            Wt = w5
        elif t == 6:
            Wt = w6
        elif t == 7:
            Wt = w7
        elif t == 8:
            Wt = w8
        elif t == 9:
            Wt = w9
        elif t == 10:
            Wt = w10
        elif t == 11:
            Wt = w11
        elif t == 12:
            Wt = w12
        elif t == 13:
            Wt = w13
        elif t == 14:
            Wt = w14
        elif t == 15:
            Wt = w15
        else:
            # W_t = s1(w14) + w9 + s0(w1) + w0
            s0 = _rotr32(w1, 7) ^ _rotr32(w1, 18) ^ (w1 >> 3)
            s1 = _rotr32(w14, 17) ^ _rotr32(w14, 19) ^ (w14 >> 10)
            new = (s1 + w9 + s0 + w0) & 0xFFFFFFFF
            w0, w1, w2, w3, w4, w5, w6, w7 = w1, w2, w3, w4, w5, w6, w7, w8
            w8, w9, w10, w11, w12, w13, w14, w15 = w9, w10, w11, w12, w13, w14, w15, new
            Wt = new

        Kt = tl.load(K_ptr + t)
        S1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
        ch = (e & f) ^ ((~e) & g)
        T1 = (h + S1 + ch + Kt + Wt) & 0xFFFFFFFF
        S0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        T2 = (S0 + maj) & 0xFFFFFFFF
        h = g
        g = f
        f = e
        e = (d + T1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (T1 + T2) & 0xFFFFFFFF

    H0 = (0x6A09E667 + a) & 0xFFFFFFFF
    H1 = (0xBB67AE85 + b) & 0xFFFFFFFF
    H2 = (0x3C6EF372 + c) & 0xFFFFFFFF
    H3 = (0xA54FF53A + d) & 0xFFFFFFFF
    H4 = (0x510E527F + e) & 0xFFFFFFFF
    H5 = (0x9B05688C + f) & 0xFFFFFFFF
    H6 = (0x1F83D9AB + g) & 0xFFFFFFFF
    H7 = (0x5BE0CD19 + h) & 0xFFFFFFFF

    # Big-endian 32-byte output, byte-by-byte (no nested helper)
    tl.store(out_ptr + pid * 32 +  0, ((H0 >> 24) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 +  1, ((H0 >> 16) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 +  2, ((H0 >>  8) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 +  3, ((H0      ) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 +  4, ((H1 >> 24) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 +  5, ((H1 >> 16) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 +  6, ((H1 >>  8) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 +  7, ((H1      ) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 +  8, ((H2 >> 24) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 +  9, ((H2 >> 16) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 10, ((H2 >>  8) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 11, ((H2      ) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 12, ((H3 >> 24) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 13, ((H3 >> 16) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 14, ((H3 >>  8) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 15, ((H3      ) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 16, ((H4 >> 24) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 17, ((H4 >> 16) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 18, ((H4 >>  8) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 19, ((H4      ) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 20, ((H5 >> 24) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 21, ((H5 >> 16) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 22, ((H5 >>  8) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 23, ((H5      ) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 24, ((H6 >> 24) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 25, ((H6 >> 16) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 26, ((H6 >>  8) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 27, ((H6      ) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 28, ((H7 >> 24) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 29, ((H7 >> 16) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 30, ((H7 >>  8) & 0xFF).to(tl.uint8))
    tl.store(out_ptr + pid * 32 + 31, ((H7      ) & 0xFF).to(tl.uint8))


def sha256_batch(
    messages: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched SHA-256 for short messages (L <= 55 bytes).

    Parameters
    ----------
    messages
        `(N, 64)` uint8 tensor of raw bytes. Bytes past `lengths[i]`
        are ignored.
    lengths
        `(N,)` int32 tensor of byte counts (0 <= L <= 55).
    out
        Optional pre-allocated `(N, 32)` uint8 output.
    """
    assert messages.is_cuda, "sha256_batch needs a CUDA tensor"
    assert messages.dtype == torch.uint8
    assert messages.shape[1] == 64
    n = messages.shape[0]
    assert lengths.shape == (n,)
    if not lengths.is_cuda or lengths.dtype != torch.int32:
        lengths = lengths.to(device=messages.device, dtype=torch.int32)
    if out is None:
        out = torch.empty((n, 32), dtype=torch.uint8, device=messages.device)
    K_dev = torch.tensor(SHA256_K, dtype=torch.int64,
                          device=messages.device).to(torch.uint32)
    _sha256_short_kernel[(n,)](messages, lengths, out, n, K_dev)
    return out
