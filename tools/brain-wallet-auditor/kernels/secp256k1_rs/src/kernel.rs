//! CUDA kernel source for batched secp256k1 scalar multiplication.
//!
//! The kernel takes `n` 32-byte private keys and produces `n` 33-byte
//! compressed public keys (`pub = priv * G`).
//!
//! Implementation notes (Phase 3 baseline; will iterate for throughput):
//!
//! * 256-bit big-int as four u64 limbs in little-endian order.
//! * Modular reduction uses secp256k1's "pseudo-Mersenne" structure:
//!   p = 2^256 - 0x1000003D1.
//! * Modular inverse via Fermat little theorem: a^(p-2) mod p.
//!   Slower than extended-Euclidean but simpler and branch-free.
//! * Point ops in Jacobian coordinates (X, Y, Z); affine conversion
//!   happens once at the end.
//! * Scalar mul: double-and-add. The simplest possible algorithm;
//!   later iterations will switch to windowed-NAF for ~4x speedup.
//!
//! The kernel is correctness-first; the throughput floor will improve
//! substantially with later optimisations (windowed-NAF, precomputed
//! generator multiples, Montgomery form for field ops).

/// CUDA C source compiled at runtime via NVRTC.
pub const SECP256K1_CUDA_SRC: &str = r#"
// ============================================================================
// secp256k1 scalar multiplication kernel
// ============================================================================
//
// Each thread processes one private key. Inputs and outputs are big-endian
// byte arrays so they match the standard wire format used by libsecp256k1
// and by the Python pipeline.

extern "C" {

// ---------- 256-bit big-int as 4 u64 limbs, little-endian ----------

typedef struct { unsigned long long l[4]; } u256;

__device__ __forceinline__ void u256_zero(u256* x) {
    x->l[0] = 0; x->l[1] = 0; x->l[2] = 0; x->l[3] = 0;
}

__device__ __forceinline__ void u256_one(u256* x) {
    x->l[0] = 1; x->l[1] = 0; x->l[2] = 0; x->l[3] = 0;
}

__device__ __forceinline__ int u256_is_zero(const u256* x) {
    return (x->l[0] | x->l[1] | x->l[2] | x->l[3]) == 0;
}

__device__ __forceinline__ int u256_eq(const u256* a, const u256* b) {
    return a->l[0] == b->l[0] && a->l[1] == b->l[1]
        && a->l[2] == b->l[2] && a->l[3] == b->l[3];
}

__device__ __forceinline__ void u256_from_be32(u256* x, const unsigned char* be) {
    // be[0] is the most-significant byte. l[3] is the most-significant limb.
    for (int i = 0; i < 4; i++) {
        unsigned long long w = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            w = (w << 8) | (unsigned long long)be[i * 8 + j];
        }
        x->l[3 - i] = w;
    }
}

__device__ __forceinline__ void u256_to_be32(unsigned char* be, const u256* x) {
    for (int i = 0; i < 4; i++) {
        unsigned long long w = x->l[3 - i];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            be[i * 8 + j] = (unsigned char)(w >> (56 - 8 * j));
        }
    }
}

// ---------- secp256k1 field prime p = 2^256 - 2^32 - 977 ----------
// p = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
__device__ __constant__ unsigned long long P_LIMBS[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
};

// Generator G (compressed, x coord and y coord)
__device__ __constant__ unsigned long long GX_LIMBS[4] = {
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL,
};
__device__ __constant__ unsigned long long GY_LIMBS[4] = {
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL,
};

__device__ __forceinline__ void u256_p(u256* x) {
    x->l[0] = P_LIMBS[0]; x->l[1] = P_LIMBS[1];
    x->l[2] = P_LIMBS[2]; x->l[3] = P_LIMBS[3];
}

// Compare a >= b (unsigned), 256-bit.
__device__ __forceinline__ int u256_ge(const u256* a, const u256* b) {
    if (a->l[3] != b->l[3]) return a->l[3] > b->l[3];
    if (a->l[2] != b->l[2]) return a->l[2] > b->l[2];
    if (a->l[1] != b->l[1]) return a->l[1] > b->l[1];
    return a->l[0] >= b->l[0];
}

// r = a - b (unsigned, no underflow check). Returns the borrow.
__device__ __forceinline__ unsigned long long u256_sub(u256* r, const u256* a, const u256* b) {
    unsigned long long borrow = 0;
    unsigned long long t;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned long long ai = a->l[i], bi = b->l[i];
        t = ai - bi - borrow;
        // borrow out: t > ai when borrow happened
        borrow = (ai < bi + borrow) || (bi + borrow < bi) ? 1 : 0;
        r->l[i] = t;
    }
    return borrow;
}

// r = a + b (mod p). Adds, then conditionally subtracts p if r >= p or overflow.
__device__ __forceinline__ void fp_add(u256* r, const u256* a, const u256* b) {
    unsigned long long carry = 0;
    unsigned long long sum;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sum = a->l[i] + b->l[i];
        unsigned long long c1 = (sum < a->l[i]) ? 1 : 0;
        sum += carry;
        unsigned long long c2 = (sum < carry) ? 1 : 0;
        r->l[i] = sum;
        carry = c1 + c2;
    }
    // If overflow OR r >= p, subtract p.
    u256 p_val; u256_p(&p_val);
    if (carry || u256_ge(r, &p_val)) {
        u256_sub(r, r, &p_val);
    }
}

// r = a - b (mod p). If a < b, add p.
__device__ __forceinline__ void fp_sub(u256* r, const u256* a, const u256* b) {
    u256 p_val; u256_p(&p_val);
    if (u256_ge(a, b)) {
        u256_sub(r, a, b);
    } else {
        u256 tmp;
        u256_sub(&tmp, b, a);
        u256_sub(r, &p_val, &tmp);
    }
}

// 256-bit x 256-bit -> 512-bit multiply. r[0..3] low, r[4..7] high.
__device__ __forceinline__ void u256_mul_full(unsigned long long r[8],
                                                const u256* a, const u256* b) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r[i] = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned long long carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // r[i+j] += a[i] * b[j] + carry
            unsigned long long ah = a->l[i] >> 32;
            unsigned long long al = a->l[i] & 0xFFFFFFFFULL;
            unsigned long long bh = b->l[j] >> 32;
            unsigned long long bl = b->l[j] & 0xFFFFFFFFULL;
            unsigned long long ll = al * bl;
            unsigned long long lh = al * bh;
            unsigned long long hl = ah * bl;
            unsigned long long hh = ah * bh;
            // mid = lh + hl (with carry to hh)
            unsigned long long mid = lh + hl;
            unsigned long long mid_c = (mid < lh) ? (1ULL << 32) : 0;
            unsigned long long lo = ll + (mid << 32);
            unsigned long long lo_c = (lo < ll) ? 1 : 0;
            unsigned long long hi = hh + (mid >> 32) + mid_c + lo_c;
            // Add (hi:lo) + r[i+j] + carry
            unsigned long long old = r[i + j];
            unsigned long long new_lo = old + lo;
            unsigned long long c1 = (new_lo < old) ? 1 : 0;
            new_lo += carry;
            unsigned long long c2 = (new_lo < carry) ? 1 : 0;
            r[i + j] = new_lo;
            carry = hi + c1 + c2;
        }
        r[i + 4] = carry;
    }
}

// Modular reduction: r = x mod p where x is 512-bit.
// secp256k1's p = 2^256 - c where c = 0x1000003D1.
// So x = h * 2^256 + l, x mod p = l + h * c (mod p).
__device__ __forceinline__ void fp_reduce(u256* r, const unsigned long long x[8]) {
    // Constant c = 0x1000003D1 = (1 << 32) + 0x3D1
    // h * c: h is a 256-bit number (x[4..7]). We need h * c mod p.
    // c fits in 64 bits, so h * c is at most 320 bits (5 limbs of u64).

    u256 low;
    low.l[0] = x[0]; low.l[1] = x[1]; low.l[2] = x[2]; low.l[3] = x[3];

    u256 high;
    high.l[0] = x[4]; high.l[1] = x[5]; high.l[2] = x[6]; high.l[3] = x[7];

    // First-pass: compute low + high * c, where c = 0x1000003D1.
    // high * c overflows to a small high-part h2 (at most a few words).
    // We accumulate carefully.

    const unsigned long long C = 0x1000003D1ULL;

    // multiply high (4 limbs) by C (1 limb) -> 5-limb result
    unsigned long long prod[5];
    {
        unsigned long long carry = 0;
        for (int i = 0; i < 4; i++) {
            unsigned long long ah = high.l[i] >> 32;
            unsigned long long al = high.l[i] & 0xFFFFFFFFULL;
            unsigned long long ch = C >> 32;
            unsigned long long cl = C & 0xFFFFFFFFULL;
            unsigned long long ll = al * cl;
            unsigned long long lh = al * ch;
            unsigned long long hl = ah * cl;
            unsigned long long hh = ah * ch;
            unsigned long long mid = lh + hl;
            unsigned long long mid_c = (mid < lh) ? (1ULL << 32) : 0;
            unsigned long long lo = ll + (mid << 32);
            unsigned long long lo_c = (lo < ll) ? 1 : 0;
            unsigned long long hi = hh + (mid >> 32) + mid_c + lo_c;
            unsigned long long s = lo + carry;
            unsigned long long s_c = (s < carry) ? 1 : 0;
            prod[i] = s;
            carry = hi + s_c;
        }
        prod[4] = carry;
    }

    // Add prod[0..4] to low[0..3], producing a 5-limb intermediate
    unsigned long long acc[5];
    {
        unsigned long long carry = 0;
        for (int i = 0; i < 4; i++) {
            unsigned long long s = low.l[i] + prod[i];
            unsigned long long c1 = (s < low.l[i]) ? 1 : 0;
            s += carry;
            unsigned long long c2 = (s < carry) ? 1 : 0;
            acc[i] = s;
            carry = c1 + c2;
        }
        acc[4] = prod[4] + carry;
    }

    // Now acc is 5-limb, may need another reduction step (acc[4] * c).
    {
        unsigned long long h2 = acc[4];
        unsigned long long ch = C >> 32;
        unsigned long long cl = C & 0xFFFFFFFFULL;
        unsigned long long h2h = h2 >> 32;
        unsigned long long h2l = h2 & 0xFFFFFFFFULL;
        unsigned long long ll = h2l * cl;
        unsigned long long lh = h2l * ch;
        unsigned long long hl = h2h * cl;
        unsigned long long hh = h2h * ch;
        unsigned long long mid = lh + hl;
        unsigned long long mid_c = (mid < lh) ? (1ULL << 32) : 0;
        unsigned long long lo = ll + (mid << 32);
        unsigned long long lo_c = (lo < ll) ? 1 : 0;
        unsigned long long hi = hh + (mid >> 32) + mid_c + lo_c;

        // Add (hi:lo) to acc[0..1]
        unsigned long long s0 = acc[0] + lo;
        unsigned long long c1 = (s0 < acc[0]) ? 1 : 0;
        unsigned long long s1 = acc[1] + hi + c1;
        unsigned long long c2 = (s1 < acc[1]) ? 1 : 0;
        acc[0] = s0;
        acc[1] = s1;
        // Propagate carry through acc[2], acc[3]
        unsigned long long s2 = acc[2] + c2;
        unsigned long long c3 = (s2 < acc[2]) ? 1 : 0;
        acc[2] = s2;
        unsigned long long s3 = acc[3] + c3;
        unsigned long long c4 = (s3 < acc[3]) ? 1 : 0;
        acc[3] = s3;
        acc[4] = c4;
    }

    // One more carry pass: if acc[4] != 0, multiply by c (small now).
    if (acc[4] != 0) {
        unsigned long long h2 = acc[4];
        unsigned long long val = h2 * C;
        unsigned long long s0 = acc[0] + val;
        unsigned long long c1 = (s0 < acc[0]) ? 1 : 0;
        acc[0] = s0;
        for (int i = 1; i < 4 && c1; i++) {
            acc[i] += c1;
            c1 = (acc[i] < c1) ? 1 : 0;
        }
    }

    r->l[0] = acc[0]; r->l[1] = acc[1]; r->l[2] = acc[2]; r->l[3] = acc[3];

    // Final correction: if r >= p, subtract p (may need to do this twice)
    u256 p_val; u256_p(&p_val);
    if (u256_ge(r, &p_val)) u256_sub(r, r, &p_val);
    if (u256_ge(r, &p_val)) u256_sub(r, r, &p_val);
}

// r = a * b mod p
__device__ __forceinline__ void fp_mul(u256* r, const u256* a, const u256* b) {
    unsigned long long full[8];
    u256_mul_full(full, a, b);
    fp_reduce(r, full);
}

// r = a^2 mod p
__device__ __forceinline__ void fp_sqr(u256* r, const u256* a) {
    fp_mul(r, a, a);
}

// r = 1 / a mod p, via Fermat: a^(p-2) mod p.
// p - 2 = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D
// Square-and-multiply over the bits of (p - 2).
__device__ __forceinline__ void fp_inv(u256* r, const u256* a) {
    // p - 2 as 4 u64 limbs (little-endian).
    unsigned long long pm2[4] = {
        0xFFFFFFFEFFFFFC2DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
    };
    u256 base = *a;
    u256 result;
    u256_one(&result);
    // Iterate bits of pm2 from LSB to MSB. Branchless multiply via mask.
    for (int limb = 0; limb < 4; limb++) {
        unsigned long long w = pm2[limb];
        for (int b = 0; b < 64; b++) {
            if (w & 1ULL) {
                u256 tmp;
                fp_mul(&tmp, &result, &base);
                result = tmp;
            }
            u256 tmp2;
            fp_sqr(&tmp2, &base);
            base = tmp2;
            w >>= 1;
        }
    }
    *r = result;
}

// ---------- Point operations (Jacobian coordinates) ----------
//
// Jacobian: (X, Y, Z) represents the affine point (X/Z^2, Y/Z^3).
// Point at infinity: Z = 0.

typedef struct { u256 x, y, z; } jpoint;

__device__ __forceinline__ int jpoint_is_inf(const jpoint* p) {
    return u256_is_zero(&p->z);
}

__device__ __forceinline__ void jpoint_set_inf(jpoint* p) {
    u256_zero(&p->x);
    u256_one(&p->y);
    u256_zero(&p->z);
}

// Doubling: 2P. Cohen "Algorithm 3.21". (a = 0 for secp256k1 saves a step.)
// Reference: Hankerson "Guide to Elliptic Curve Cryptography", Algorithm 3.21.
__device__ __forceinline__ void jpoint_double(jpoint* r, const jpoint* p) {
    if (jpoint_is_inf(p) || u256_is_zero(&p->y)) {
        jpoint_set_inf(r);
        return;
    }
    u256 a, b, c, d, e, f, t1, t2;
    fp_sqr(&a, &p->y);                // A = Y^2
    fp_mul(&b, &p->x, &a);            // B = X*A
    fp_add(&b, &b, &b);
    fp_add(&b, &b, &b);               // B = 4*X*Y^2
    fp_sqr(&c, &a);
    fp_add(&c, &c, &c);
    fp_add(&c, &c, &c);
    fp_add(&c, &c, &c);               // C = 8*Y^4
    fp_sqr(&d, &p->x);                // D = X^2
    fp_add(&e, &d, &d);
    fp_add(&e, &e, &d);               // E = 3*X^2  (a = 0 for secp256k1)
    fp_sqr(&f, &e);                   // F = E^2

    // X' = F - 2*B
    fp_add(&t1, &b, &b);
    fp_sub(&r->x, &f, &t1);

    // Y' = E*(B - X') - C
    fp_sub(&t1, &b, &r->x);
    fp_mul(&t2, &e, &t1);
    fp_sub(&r->y, &t2, &c);

    // Z' = 2*Y*Z
    fp_mul(&t1, &p->y, &p->z);
    fp_add(&r->z, &t1, &t1);
}

// Addition: r = P + Q. Cohen "Algorithm 3.22" for mixed Jacobian + affine,
// but here we do general Jacobian + Jacobian.
__device__ __forceinline__ void jpoint_add(jpoint* r, const jpoint* p, const jpoint* q) {
    if (jpoint_is_inf(p)) { *r = *q; return; }
    if (jpoint_is_inf(q)) { *r = *p; return; }

    u256 u1, u2, s1, s2, h, rr, t, tmp;
    u256 z1z1, z2z2, z1z1z1, z2z2z2;

    fp_sqr(&z1z1, &p->z);             // Z1^2
    fp_sqr(&z2z2, &q->z);             // Z2^2
    fp_mul(&u1, &p->x, &z2z2);        // U1 = X1*Z2^2
    fp_mul(&u2, &q->x, &z1z1);        // U2 = X2*Z1^2
    fp_mul(&z1z1z1, &z1z1, &p->z);    // Z1^3
    fp_mul(&z2z2z2, &z2z2, &q->z);    // Z2^3
    fp_mul(&s1, &p->y, &z2z2z2);      // S1 = Y1*Z2^3
    fp_mul(&s2, &q->y, &z1z1z1);      // S2 = Y2*Z1^3

    if (u256_eq(&u1, &u2)) {
        if (!u256_eq(&s1, &s2)) {
            jpoint_set_inf(r);
            return;
        } else {
            jpoint_double(r, p);
            return;
        }
    }

    fp_sub(&h, &u2, &u1);             // H = U2 - U1
    fp_sub(&rr, &s2, &s1);            // R = S2 - S1

    u256 hsq, hcb;
    fp_sqr(&hsq, &h);                 // H^2
    fp_mul(&hcb, &hsq, &h);           // H^3
    fp_mul(&t, &u1, &hsq);            // U1*H^2

    // X3 = R^2 - H^3 - 2*U1*H^2
    fp_sqr(&tmp, &rr);
    fp_sub(&r->x, &tmp, &hcb);
    fp_add(&tmp, &t, &t);
    fp_sub(&r->x, &r->x, &tmp);

    // Y3 = R*(U1*H^2 - X3) - S1*H^3
    fp_sub(&tmp, &t, &r->x);
    fp_mul(&r->y, &rr, &tmp);
    fp_mul(&tmp, &s1, &hcb);
    fp_sub(&r->y, &r->y, &tmp);

    // Z3 = H*Z1*Z2
    fp_mul(&tmp, &p->z, &q->z);
    fp_mul(&r->z, &tmp, &h);
}

// Scalar multiplication: r = k * P, where k is a 256-bit scalar (BE bytes).
// Double-and-add from MSB to LSB.
__device__ __forceinline__ void jpoint_mul(jpoint* r, const u256* k, const jpoint* P) {
    jpoint Q;
    jpoint_set_inf(&Q);
    // Process bits MSB to LSB: scan limbs[3] down to limbs[0].
    for (int limb = 3; limb >= 0; limb--) {
        unsigned long long w = k->l[limb];
        for (int b = 63; b >= 0; b--) {
            jpoint tmp;
            jpoint_double(&tmp, &Q);
            Q = tmp;
            if ((w >> b) & 1ULL) {
                jpoint tmp2;
                jpoint_add(&tmp2, &Q, P);
                Q = tmp2;
            }
        }
    }
    *r = Q;
}

// Convert Jacobian point to affine (x, y).
__device__ __forceinline__ void jpoint_to_affine(u256* ax, u256* ay, const jpoint* p) {
    u256 zinv, zinv2, zinv3;
    fp_inv(&zinv, &p->z);
    fp_sqr(&zinv2, &zinv);
    fp_mul(&zinv3, &zinv2, &zinv);
    fp_mul(ax, &p->x, &zinv2);
    fp_mul(ay, &p->y, &zinv3);
}

// ---------- Public entry: derive compressed pubkey from 32-byte BE private key ----------

__global__ void secp256k1_priv_to_pub(
    const unsigned char* __restrict__ priv_be,  // (N, 32) BE bytes
    unsigned char* __restrict__ pub_compressed, // (N, 33) BE bytes
    int n_keys
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_keys) return;

    u256 k;
    u256_from_be32(&k, priv_be + tid * 32);

    jpoint G;
    G.x.l[0] = GX_LIMBS[0]; G.x.l[1] = GX_LIMBS[1];
    G.x.l[2] = GX_LIMBS[2]; G.x.l[3] = GX_LIMBS[3];
    G.y.l[0] = GY_LIMBS[0]; G.y.l[1] = GY_LIMBS[1];
    G.y.l[2] = GY_LIMBS[2]; G.y.l[3] = GY_LIMBS[3];
    u256_one(&G.z);  // Z = 1 -> point is in affine-as-Jacobian form

    jpoint P;
    jpoint_mul(&P, &k, &G);

    u256 ax, ay;
    jpoint_to_affine(&ax, &ay, &P);

    // Compressed pubkey: prefix byte = 0x02 (y even) or 0x03 (y odd),
    // then 32-byte big-endian x.
    pub_compressed[tid * 33 + 0] = (ay.l[0] & 1ULL) ? 0x03 : 0x02;
    u256_to_be32(pub_compressed + tid * 33 + 1, &ax);
}

}  // extern "C"
"#;
