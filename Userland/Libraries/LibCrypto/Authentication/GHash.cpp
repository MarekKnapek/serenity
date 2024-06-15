/*
 * Copyright (c) 2020, Ali Mohammad Pur <mpfard@serenityos.org>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/ByteReader.h>
#include <AK/Debug.h>
#include <AK/Platform.h>
#include <AK/Types.h>
#include <LibCrypto/Authentication/GHash.h>

namespace {

static u32 to_u32(u8 const* b)
{
    return AK::convert_between_host_and_big_endian(ByteReader::load32(b));
}

static void to_u8s(u8* b, u32 const* w)
{
    for (auto i = 0; i < 4; ++i) {
        ByteReader::store(b + i * 4, AK::convert_between_host_and_big_endian(w[i]));
    }
}

}

namespace Crypto::Authentication {

GHash::TagType GHash::process(ReadonlyBytes aad, ReadonlyBytes cipher)
{
    u32 tag[4] { 0, 0, 0, 0 };

    auto transform_one = [&](auto& buf) {
        size_t i = 0;
        for (; i < buf.size(); i += 16) {
            if (i + 16 <= buf.size()) {
                for (auto j = 0; j < 4; ++j) {
                    tag[j] ^= to_u32(buf.offset(i + j * 4));
                }
                galois_multiply(tag, m_key, tag);
            }
        }

        if (i > buf.size()) {
            u8 buffer[16] = {};
            Bytes buffer_bytes { buffer, 16 };
            buf.slice(i - 16).copy_to(buffer_bytes);

            for (auto j = 0; j < 4; ++j) {
                tag[j] ^= to_u32(buffer_bytes.offset(j * 4));
            }
            galois_multiply(tag, m_key, tag);
        }
    };

    transform_one(aad);
    transform_one(cipher);

    auto aad_bits = 8 * (u64)aad.size();
    auto cipher_bits = 8 * (u64)cipher.size();

    auto high = [](u64 value) -> u32 { return value >> 32; };
    auto low = [](u64 value) -> u32 { return value & 0xffffffff; };

    if constexpr (GHASH_PROCESS_DEBUG) {
        dbgln("AAD bits: {} : {}", high(aad_bits), low(aad_bits));
        dbgln("Cipher bits: {} : {}", high(cipher_bits), low(cipher_bits));
        dbgln("Tag bits: {} : {} : {} : {}", tag[0], tag[1], tag[2], tag[3]);
    }

    tag[0] ^= high(aad_bits);
    tag[1] ^= low(aad_bits);
    tag[2] ^= high(cipher_bits);
    tag[3] ^= low(cipher_bits);

    dbgln_if(GHASH_PROCESS_DEBUG, "Tag bits: {} : {} : {} : {}", tag[0], tag[1], tag[2], tag[3]);

    galois_multiply(tag, m_key, tag);

    TagType digest;
    to_u8s(digest.data, tag);

    return digest;
}

#if (ARCH(I386) || ARCH(X86_64))

#include <wmmintrin.h> /* _mm_clmulepi64_si128 */
#include <emmintrin.h> /* SSE2 _mm_load_si128 _mm_or_si128 _mm_set_epi8 _mm_slli_epi32 _mm_slli_si128 _mm_srli_epi32 _mm_srli_si128 _mm_store_si128 _mm_xor_si128 */
#include <tmmintrin.h> /* SSSE3 _mm_shuffle_epi8 */

static void __attribute__((target("pclmul,sse2,ssse3"))) galois_multiply_impl(u32 (&z)[4], u32 const (&x)[4], u32 const (&y)[4])
{
    __m128i bswap;
    __m128i bswapbe;
    __m128i a;
    __m128i b;
    __m128i ta;
    __m128i tb;
    __m128i tc;
    __m128i td;
    __m128i te;
    __m128i tf;
    __m128i tg;
    __m128i th;

    bswap = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    bswapbe = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    a = _mm_loadu_si128(((__m128i const*)(x)));
    b = _mm_loadu_si128(((__m128i const*)(y)));
    a = _mm_shuffle_epi8(_mm_shuffle_epi8(a, bswapbe), bswap);
    b = _mm_shuffle_epi8(_mm_shuffle_epi8(b, bswapbe), bswap);
    ta = _mm_clmulepi64_si128(a, b, 0x00);
    tb = _mm_clmulepi64_si128(a, b, 0x10);
    tc = _mm_clmulepi64_si128(a, b, 0x01);
    td = _mm_clmulepi64_si128(a, b, 0x11);
    tb = _mm_xor_si128(tb, tc);
    tc = _mm_slli_si128(tb, 8);
    tb = _mm_srli_si128(tb, 8);
    ta = _mm_xor_si128(ta, tc);
    td = _mm_xor_si128(td, tb);
    te = _mm_srli_epi32(ta, 31);
    tf = _mm_srli_epi32(td, 31);
    ta = _mm_slli_epi32(ta, 1);
    td = _mm_slli_epi32(td, 1);
    tg = _mm_srli_si128(te, 12);
    tf = _mm_slli_si128(tf, 4);
    te = _mm_slli_si128(te, 4);
    ta = _mm_or_si128(ta, te);
    td = _mm_or_si128(td, tf);
    td = _mm_or_si128(td, tg);
    te = _mm_slli_epi32(ta, 31);
    tf = _mm_slli_epi32(ta, 30);
    tg = _mm_slli_epi32(ta, 25);
    te = _mm_xor_si128(te, tf);
    te = _mm_xor_si128(te, tg);
    tf = _mm_srli_si128(te, 4);
    te = _mm_slli_si128(te, 12);
    ta = _mm_xor_si128(ta, te);
    th = _mm_srli_epi32(ta, 1);
    tb = _mm_srli_epi32(ta, 2);
    tc = _mm_srli_epi32(ta, 7);
    th = _mm_xor_si128(th, tb);
    th = _mm_xor_si128(th, tc);
    th = _mm_xor_si128(th, tf);
    ta = _mm_xor_si128(ta, th);
    td = _mm_xor_si128(td, ta);
    td = _mm_shuffle_epi8(_mm_shuffle_epi8(td, bswap), bswapbe);
    _mm_storeu_si128(((__m128i*)(z)), td);
}

#else

/// Galois Field multiplication using <x^127 + x^7 + x^2 + x + 1>.
/// Note that x, y, and z are strictly BE.
static void galois_multiply_impl(u32 (&_z)[4], u32 const (&_x)[4], u32 const (&_y)[4])
{

    // Note: Copied upfront to stack to avoid memory access in the loop.
    u32 x[4] { _x[0], _x[1], _x[2], _x[3] };
    u32 const y[4] { _y[0], _y[1], _y[2], _y[3] };
    u32 z[4] { 0, 0, 0, 0 };

    // Unrolled by 32, the access in y[3-(i/32)] can be cached throughout the loop.
#pragma GCC unroll 32
    for (ssize_t i = 127; i > -1; --i) {
        auto r = -((y[3 - (i / 32)] >> (i % 32)) & 1);
        z[0] ^= x[0] & r;
        z[1] ^= x[1] & r;
        z[2] ^= x[2] & r;
        z[3] ^= x[3] & r;
        auto a0 = x[0] & 1;
        x[0] >>= 1;
        auto a1 = x[1] & 1;
        x[1] >>= 1;
        x[1] |= a0 << 31;
        auto a2 = x[2] & 1;
        x[2] >>= 1;
        x[2] |= a1 << 31;
        auto a3 = x[3] & 1;
        x[3] >>= 1;
        x[3] |= a2 << 31;

        x[0] ^= 0xe1000000 & -a3;
    }

    memcpy(_z, z, sizeof(z));
}

#endif

void galois_multiply(u32 (&z)[4], u32 const (&x)[4], u32 const (&y)[4])
{
    galois_multiply_impl(z, x, y);
}

}
