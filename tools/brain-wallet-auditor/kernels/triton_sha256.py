"""Triton SHA-256 kernel for batched short-message hashing.

Scope: messages up to 55 bytes (single SHA-256 block after the 9-byte
pad). This covers brain-wallet passphrases (typically 8-32 chars) and
the intermediate SHA-256 of a compressed pubkey (33 bytes) in the
hash160 step.

Layout: one Triton program per input. Each program loads up to 55
bytes (16 u32 words after padding), runs 64 rounds of the SHA-256
compression, writes 32 output bytes. Parallelism = batch size.

Bit-exact correctness is verified by the companion test against
`hashlib.sha256` on 10K random inputs of random lengths in [0, 55].
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
def _sha256_short_kernel(
    msg_ptr,                       # *u8  (N, MAX_BYTES)  raw input bytes (zero-padded)
    len_ptr,                       # *i32 (N,)             actual length per row
    out_ptr,                       # *u8  (N, 32)          output digest bytes
    n_rows,                        # scalar
    K_ptr,                         # *u32 (64,)           SHA-256 round constants
    BLOCK_BYTES: tl.constexpr,     # padded msg block size; must be 64
):
    """Single SHA-256 block on a short message.

    Each Triton program processes one message. The message must
    satisfy `length <= 55` (so the standard SHA-256 pad fits in one
    64-byte block).
    """
    pid = tl.program_id(axis=0)
    if pid >= n_rows:
        return

    # Load length and clamp to 55 (the kernel only supports single-block messages)
    L = tl.load(len_ptr + pid)
    L = tl.minimum(L, 55)

    # Load 64 bytes from this row and apply standard SHA-256 padding:
    #   byte[L] = 0x80
    #   bytes[L+1 .. 55] = 0x00
    #   bytes[56 .. 63] = big-endian (L * 8)
    byte_idx = tl.arange(0, BLOCK_BYTES)
    raw = tl.load(
        msg_ptr + pid * BLOCK_BYTES + byte_idx,
        mask=byte_idx < BLOCK_BYTES, other=0,
    )
    raw = raw.to(tl.uint32)
    # Zero out anything past L (so a row longer than its declared L is ignored)
    raw = tl.where(byte_idx < L, raw, 0)
    # Insert 0x80 at position L
    raw = tl.where(byte_idx == L, 0x80, raw)
    # Insert big-endian length-in-bits at bytes 56..63
    bits = (L * 8).to(tl.uint32)
    # Only bytes 60..63 carry non-zero length bits (since L <= 55, bits fits in u32)
    # but we follow the standard layout exactly.
    b_60 = (bits >> 24) & 0xFF
    b_61 = (bits >> 16) & 0xFF
    b_62 = (bits >>  8) & 0xFF
    b_63 = (bits      ) & 0xFF
    raw = tl.where(byte_idx == 60, b_60, raw)
    raw = tl.where(byte_idx == 61, b_61, raw)
    raw = tl.where(byte_idx == 62, b_62, raw)
    raw = tl.where(byte_idx == 63, b_63, raw)

    # Pack into 16 big-endian u32 words: W[0..15]
    # We compute each word from the byte vector. To keep things simple,
    # build w0..w15 individually.
    def word_at(offset):
        b0 = tl.sum(tl.where(byte_idx == offset + 0, raw, 0))
        b1 = tl.sum(tl.where(byte_idx == offset + 1, raw, 0))
        b2 = tl.sum(tl.where(byte_idx == offset + 2, raw, 0))
        b3 = tl.sum(tl.where(byte_idx == offset + 3, raw, 0))
        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3

    w0  = word_at(0)
    w1  = word_at(4)
    w2  = word_at(8)
    w3  = word_at(12)
    w4  = word_at(16)
    w5  = word_at(20)
    w6  = word_at(24)
    w7  = word_at(28)
    w8  = word_at(32)
    w9  = word_at(36)
    w10 = word_at(40)
    w11 = word_at(44)
    w12 = word_at(48)
    w13 = word_at(52)
    w14 = word_at(56)
    w15 = word_at(60)

    # Initial hash values H0..H7 (constants from SHA-256 spec)
    a = tl.cast(0x6A09E667, tl.uint32)
    b = tl.cast(0xBB67AE85, tl.uint32)
    c = tl.cast(0x3C6EF372, tl.uint32)
    d = tl.cast(0xA54FF53A, tl.uint32)
    e = tl.cast(0x510E527F, tl.uint32)
    f = tl.cast(0x9B05688C, tl.uint32)
    g = tl.cast(0x1F83D9AB, tl.uint32)
    h = tl.cast(0x5BE0CD19, tl.uint32)

    # We maintain w[0..15] as a "rotating window" of the last 16
    # message-schedule words via 16 local variables. After the first
    # 16 rounds, each new W_t = sigma1(W_{t-2}) + W_{t-7} +
    # sigma0(W_{t-15}) + W_{t-16}.
    def rotr32(x, n):
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def big_sigma0(x):
        return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)

    def big_sigma1(x):
        return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)

    def small_sigma0(x):
        return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3)

    def small_sigma1(x):
        return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10)

    # The 64 rounds, written out so Triton can fully unroll.
    # We track the rotating window via 16 named locals.
    for t in tl.static_range(64):
        # Compute W_t for t >= 16 and advance the window.
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
            # Extension step. New W = s1(W_{t-2}) + W_{t-7} + s0(W_{t-15}) + W_{t-16}.
            # Use the rolling window via 16 named locals.
            new = (small_sigma1(w14) + w9 + small_sigma0(w1) + w0) & 0xFFFFFFFF
            # Roll the window: drop w0, shift all left, store new in w15.
            w0  = w1
            w1  = w2
            w2  = w3
            w3  = w4
            w4  = w5
            w5  = w6
            w6  = w7
            w7  = w8
            w8  = w9
            w9  = w10
            w10 = w11
            w11 = w12
            w12 = w13
            w13 = w14
            w14 = w15
            w15 = new
            Wt = new

        Kt = tl.load(K_ptr + t)
        ch = (e & f) ^ ((~e) & g)
        T1 = (h + big_sigma1(e) + ch + Kt + Wt) & 0xFFFFFFFF
        maj = (a & b) ^ (a & c) ^ (b & c)
        T2 = (big_sigma0(a) + maj) & 0xFFFFFFFF
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

    # Write 32 output bytes big-endian
    def store_word(idx, w):
        tl.store(out_ptr + pid * 32 + idx * 4 + 0, ((w >> 24) & 0xFF).to(tl.uint8))
        tl.store(out_ptr + pid * 32 + idx * 4 + 1, ((w >> 16) & 0xFF).to(tl.uint8))
        tl.store(out_ptr + pid * 32 + idx * 4 + 2, ((w >>  8) & 0xFF).to(tl.uint8))
        tl.store(out_ptr + pid * 32 + idx * 4 + 3, ((w      ) & 0xFF).to(tl.uint8))

    store_word(0, H0)
    store_word(1, H1)
    store_word(2, H2)
    store_word(3, H3)
    store_word(4, H4)
    store_word(5, H5)
    store_word(6, H6)
    store_word(7, H7)


def sha256_batch(
    messages: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched SHA-256 for short messages.

    Parameters
    ----------
    messages
        `(N, 64)` uint8 tensor of zero-padded raw bytes. The padding
        bytes past `lengths[i]` are ignored (the kernel zero-fills
        internally), so callers may pass any value there.
    lengths
        `(N,)` int32 tensor of byte counts per message. Each value
        must satisfy `0 <= lengths[i] <= 55`.
    out
        Optional pre-allocated `(N, 32)` uint8 output. If None, one
        is allocated on the same device as `messages`.

    Returns
    -------
    `(N, 32)` uint8 tensor of digests.
    """
    assert messages.is_cuda, "sha256_batch needs a CUDA tensor"
    assert messages.dtype == torch.uint8, f"messages must be uint8, got {messages.dtype}"
    assert messages.shape[1] == 64, f"messages must be (N, 64); got {tuple(messages.shape)}"
    n = messages.shape[0]
    assert lengths.shape == (n,), f"lengths shape {lengths.shape} != ({n},)"
    if not lengths.is_cuda or lengths.dtype != torch.int32:
        lengths = lengths.to(device=messages.device, dtype=torch.int32)

    if out is None:
        out = torch.empty((n, 32), dtype=torch.uint8, device=messages.device)
    else:
        assert out.shape == (n, 32) and out.dtype == torch.uint8

    K_dev = torch.tensor(SHA256_K, dtype=torch.int64, device=messages.device).to(torch.uint32)

    grid = (n,)
    _sha256_short_kernel[grid](
        messages, lengths, out, n, K_dev,
        BLOCK_BYTES=64,
    )
    return out
