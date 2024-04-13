/*
 * Copyright (c) 2020, Ali Mohammad Pur <mpfard@serenityos.org>
 * Copyright (c) 2023, Jelle Raaijmakers <jelle@gmta.nl>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/Endian.h>
#include <AK/Memory.h>
#include <AK/Platform.h>
#include <AK/Types.h>
#include <LibCrypto/Hash/SHA1.h>

namespace
{
    #if defined __GNUC__ && defined __GNUC_MINOR__
    #define gnuc_is_at_least(major, minor) (((__GNUC__) > (major)) || (((__GNUC__) == (major)) && ((__GNUC_MINOR__) >= (minor))))
    #define gnuc_attribute_target(x) __attribute__((__target__(x)))
    #else
    #define gnuc_is_at_least(major, minor) 0
    #define gnuc_attribute_target(x)
    #endif

    #define crypto_hash_sha1_x86_compiletime_test (ARCH(I386) || ARCH(X86_64)) && gnuc_is_at_least(11, 1)

    #if crypto_hash_sha1_x86_compiletime_test

    #include <emmintrin.h> /* SSE2 _mm_add_epi32 _mm_loadu_si128 _mm_set_epi32 _mm_set_epi64x _mm_setzero_si128 _mm_shuffle_epi32 _mm_storeu_si128 _mm_xor_si128 */
    #include <tmmintrin.h> /* SSSE3 _mm_shuffle_epi8 */
    #include <smmintrin.h> /* SSE4.1 _mm_extract_epi32 */
    #include <immintrin.h> /* SHA _mm_sha1msg1_epu32 _mm_sha1msg2_epu32 _mm_sha1nexte_epu32 _mm_sha1rnds4_epu32 */

    static bool crypto_hash_sha1_x86_inited = false;

    static inline bool crypto_hash_sha1_x86_runtime_test(void)
    {
        if(!crypto_hash_sha1_x86_inited)
        {
            __builtin_cpu_init();
            crypto_hash_sha1_x86_inited = true;
        }
        return
            __builtin_cpu_supports("sse2") &&
            __builtin_cpu_supports("sse3") &&
            __builtin_cpu_supports("sse4.1") &&
            __builtin_cpu_supports("sha");
    }

    static inline void gnuc_attribute_target("sse2,sse3,sse4.1,sha") crypto_hash_sha1_x86_transform(u32* const state, u8 const* const data)
    {
        #define sha1_x86_reverse_32_c ((0x0 << (3 * 2)) | (0x1 << (2 * 2)) | (0x2 << (1 * 2)) | (0x3 << (0 * 2)))

	    __m128i reverse_8;
	    __m128i abcdx;
	    __m128i e;
	    __m128i old_abcd;
	    __m128i old_e;
	    __m128i msg_0;
	    __m128i abcdy;
	    __m128i msg_1;
	    __m128i msg_2;
	    __m128i msg_3;

        VERIFY(state);
        VERIFY(data);
        VERIFY(crypto_hash_sha1_x86_runtime_test());

        reverse_8 = _mm_set_epi64x(0x0001020304050607ull, 0x08090a0b0c0d0e0full);
        abcdx = _mm_loadu_si128(((__m128i const*)(state)));
        abcdx = _mm_shuffle_epi32(abcdx, sha1_x86_reverse_32_c);
        e = _mm_set_epi32(*((int const*)(&state[4])), 0, 0, 0);

        old_abcd = abcdx;
        old_e = e;
        msg_0 = _mm_loadu_si128(((__m128i const*)(&data[0 * 16])));
        msg_0 = _mm_shuffle_epi8(msg_0, reverse_8);
        e = _mm_add_epi32(e, msg_0);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 0);
        msg_1 = _mm_loadu_si128(((__m128i const*)(&data[1 * 16])));
        msg_1 = _mm_shuffle_epi8(msg_1, reverse_8);
        e = _mm_sha1nexte_epu32(abcdx, msg_1);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 0);
        msg_2 = _mm_loadu_si128(((__m128i const*)(&data[2 * 16])));
        msg_2 = _mm_shuffle_epi8(msg_2, reverse_8);
        e = _mm_sha1nexte_epu32(abcdy, msg_2);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 0);
        msg_3 = _mm_loadu_si128(((__m128i const*)(&data[3 * 16])));
        msg_3 = _mm_shuffle_epi8(msg_3, reverse_8);
        e = _mm_sha1nexte_epu32(abcdx, msg_3);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 0);
        msg_0 = _mm_sha1msg1_epu32(msg_0, msg_1);
        msg_0 = _mm_xor_si128(msg_0, msg_2);
        msg_0 = _mm_sha1msg2_epu32(msg_0, msg_3);
        e = _mm_sha1nexte_epu32(abcdy, msg_0);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 0);
        msg_1 = _mm_sha1msg1_epu32(msg_1, msg_2);
        msg_1 = _mm_xor_si128(msg_1, msg_3);
        msg_1 = _mm_sha1msg2_epu32(msg_1, msg_0);
        e = _mm_sha1nexte_epu32(abcdx, msg_1);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 1);
        msg_2 = _mm_sha1msg1_epu32(msg_2, msg_3);
        msg_2 = _mm_xor_si128(msg_2, msg_0);
        msg_2 = _mm_sha1msg2_epu32(msg_2, msg_1);
        e = _mm_sha1nexte_epu32(abcdy, msg_2);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 1);
        msg_3 = _mm_sha1msg1_epu32(msg_3, msg_0);
        msg_3 = _mm_xor_si128(msg_3, msg_1);
        msg_3 = _mm_sha1msg2_epu32(msg_3, msg_2);
        e = _mm_sha1nexte_epu32(abcdx, msg_3);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 1);
        msg_0 = _mm_sha1msg1_epu32(msg_0, msg_1);
        msg_0 = _mm_xor_si128(msg_0, msg_2);
        msg_0 = _mm_sha1msg2_epu32(msg_0, msg_3);
        e = _mm_sha1nexte_epu32(abcdy, msg_0);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 1);
        msg_1 = _mm_sha1msg1_epu32(msg_1, msg_2);
        msg_1 = _mm_xor_si128(msg_1, msg_3);
        msg_1 = _mm_sha1msg2_epu32(msg_1, msg_0);
        e = _mm_sha1nexte_epu32(abcdx, msg_1);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 1);
        msg_2 = _mm_sha1msg1_epu32(msg_2, msg_3);
        msg_2 = _mm_xor_si128(msg_2, msg_0);
        msg_2 = _mm_sha1msg2_epu32(msg_2, msg_1);
        e = _mm_sha1nexte_epu32(abcdy, msg_2);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 2);
        msg_3 = _mm_sha1msg1_epu32(msg_3, msg_0);
        msg_3 = _mm_xor_si128(msg_3, msg_1);
        msg_3 = _mm_sha1msg2_epu32(msg_3, msg_2);
        e = _mm_sha1nexte_epu32(abcdx, msg_3);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 2);
        msg_0 = _mm_sha1msg1_epu32(msg_0, msg_1);
        msg_0 = _mm_xor_si128(msg_0, msg_2);
        msg_0 = _mm_sha1msg2_epu32(msg_0, msg_3);
        e = _mm_sha1nexte_epu32(abcdy, msg_0);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 2);
        msg_1 = _mm_sha1msg1_epu32(msg_1, msg_2);
        msg_1 = _mm_xor_si128(msg_1, msg_3);
        msg_1 = _mm_sha1msg2_epu32(msg_1, msg_0);
        e = _mm_sha1nexte_epu32(abcdx, msg_1);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 2);
        msg_2 = _mm_sha1msg1_epu32(msg_2, msg_3);
        msg_2 = _mm_xor_si128(msg_2, msg_0);
        msg_2 = _mm_sha1msg2_epu32(msg_2, msg_1);
        e = _mm_sha1nexte_epu32(abcdy, msg_2);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 2);
        msg_3 = _mm_sha1msg1_epu32(msg_3, msg_0);
        msg_3 = _mm_xor_si128(msg_3, msg_1);
        msg_3 = _mm_sha1msg2_epu32(msg_3, msg_2);
        e = _mm_sha1nexte_epu32(abcdx, msg_3);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 3);
        msg_0 = _mm_sha1msg1_epu32(msg_0, msg_1);
        msg_0 = _mm_xor_si128(msg_0, msg_2);
        msg_0 = _mm_sha1msg2_epu32(msg_0, msg_3);
        e = _mm_sha1nexte_epu32(abcdy, msg_0);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 3);
        msg_1 = _mm_sha1msg1_epu32(msg_1, msg_2);
        msg_1 = _mm_xor_si128(msg_1, msg_3);
        msg_1 = _mm_sha1msg2_epu32(msg_1, msg_0);
        e = _mm_sha1nexte_epu32(abcdx, msg_1);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 3);
        msg_2 = _mm_sha1msg1_epu32(msg_2, msg_3);
        msg_2 = _mm_xor_si128(msg_2, msg_0);
        msg_2 = _mm_sha1msg2_epu32(msg_2, msg_1);
        e = _mm_sha1nexte_epu32(abcdy, msg_2);
        abcdy = _mm_sha1rnds4_epu32(abcdx, e, 3);
        msg_3 = _mm_sha1msg1_epu32(msg_3, msg_0);
        msg_3 = _mm_xor_si128(msg_3, msg_1);
        msg_3 = _mm_sha1msg2_epu32(msg_3, msg_2);
        e = _mm_sha1nexte_epu32(abcdx, msg_3);
        abcdx = _mm_sha1rnds4_epu32(abcdy, e, 3);
        msg_0 = _mm_setzero_si128();
        e = _mm_sha1nexte_epu32(abcdy, msg_0);
        abcdx = _mm_add_epi32(abcdx, old_abcd);
        e = _mm_add_epi32(e, old_e);

        abcdx = _mm_shuffle_epi32(abcdx, sha1_x86_reverse_32_c);
        _mm_storeu_si128(((__m128i*)(&state[0])), abcdx);
        *((int*)(&state[4])) = _mm_extract_epi32(e, 3);

        #undef sha1_x86_reverse_32_c
    }

    #endif
}

namespace Crypto::Hash {

static constexpr auto ROTATE_LEFT(u32 value, size_t bits)
{
    return (value << bits) | (value >> (32 - bits));
}

inline void SHA1::transform(u8 const* data)
{
    u32 blocks[80];
    for (size_t i = 0; i < 16; ++i)
        blocks[i] = AK::convert_between_host_and_network_endian(((u32 const*)data)[i]);

    // w[i] = (w[i-3] xor w[i-8] xor w[i-14] xor w[i-16]) leftrotate 1
    for (size_t i = 16; i < Rounds; ++i)
        blocks[i] = ROTATE_LEFT(blocks[i - 3] ^ blocks[i - 8] ^ blocks[i - 14] ^ blocks[i - 16], 1);

    auto a = m_state[0], b = m_state[1], c = m_state[2], d = m_state[3], e = m_state[4];
    u32 f, k;

    for (size_t i = 0; i < Rounds; ++i) {
        if (i <= 19) {
            f = (b & c) | ((~b) & d);
            k = SHA1Constants::RoundConstants[0];
        } else if (i <= 39) {
            f = b ^ c ^ d;
            k = SHA1Constants::RoundConstants[1];
        } else if (i <= 59) {
            f = (b & c) | (b & d) | (c & d);
            k = SHA1Constants::RoundConstants[2];
        } else {
            f = b ^ c ^ d;
            k = SHA1Constants::RoundConstants[3];
        }
        auto temp = ROTATE_LEFT(a, 5) + f + e + k + blocks[i];
        e = d;
        d = c;
        c = ROTATE_LEFT(b, 30);
        b = a;
        a = temp;
    }

    m_state[0] += a;
    m_state[1] += b;
    m_state[2] += c;
    m_state[3] += d;
    m_state[4] += e;

    // "security" measures, as if SHA1 is secure
    a = 0;
    b = 0;
    c = 0;
    d = 0;
    e = 0;
    secure_zero(blocks, 16 * sizeof(u32));
}

void SHA1::update(u8 const* message, size_t length)
{
    while (length > 0) {
        size_t copy_bytes = AK::min(length, BlockSize - m_data_length);
        __builtin_memcpy(m_data_buffer + m_data_length, message, copy_bytes);
        message += copy_bytes;
        length -= copy_bytes;
        m_data_length += copy_bytes;
        if (m_data_length == BlockSize) {
            #if crypto_hash_sha1_x86_compiletime_test
            if(crypto_hash_sha1_x86_runtime_test())
            {
                crypto_hash_sha1_x86_transform(&m_state[0], m_data_buffer);
            }
            else
            #endif
            {
                transform(m_data_buffer);
            }
            m_bit_length += BlockSize * 8;
            m_data_length = 0;
        }
    }
}

SHA1::DigestType SHA1::digest()
{
    auto digest = peek();
    reset();
    return digest;
}

SHA1::DigestType SHA1::peek()
{
    DigestType digest;
    size_t i = m_data_length;

    // make a local copy of the data as we modify it
    u8 data[BlockSize];
    u32 state[5];
    __builtin_memcpy(data, m_data_buffer, m_data_length);
    __builtin_memcpy(state, m_state, 20);

    if (m_data_length < FinalBlockDataSize) {
        m_data_buffer[i++] = 0x80;
        while (i < FinalBlockDataSize)
            m_data_buffer[i++] = 0x00;

    } else {
        // First, complete a block with some padding.
        m_data_buffer[i++] = 0x80;
        while (i < BlockSize)
            m_data_buffer[i++] = 0x00;
        transform(m_data_buffer);

        // Then start another block with BlockSize - 8 bytes of zeros
        __builtin_memset(m_data_buffer, 0, FinalBlockDataSize);
    }

    // append total message length
    m_bit_length += m_data_length * 8;
    m_data_buffer[BlockSize - 1] = m_bit_length;
    m_data_buffer[BlockSize - 2] = m_bit_length >> 8;
    m_data_buffer[BlockSize - 3] = m_bit_length >> 16;
    m_data_buffer[BlockSize - 4] = m_bit_length >> 24;
    m_data_buffer[BlockSize - 5] = m_bit_length >> 32;
    m_data_buffer[BlockSize - 6] = m_bit_length >> 40;
    m_data_buffer[BlockSize - 7] = m_bit_length >> 48;
    m_data_buffer[BlockSize - 8] = m_bit_length >> 56;

    transform(m_data_buffer);

    for (i = 0; i < 4; ++i) {
        digest.data[i + 0] = (m_state[0] >> (24 - i * 8)) & 0x000000ff;
        digest.data[i + 4] = (m_state[1] >> (24 - i * 8)) & 0x000000ff;
        digest.data[i + 8] = (m_state[2] >> (24 - i * 8)) & 0x000000ff;
        digest.data[i + 12] = (m_state[3] >> (24 - i * 8)) & 0x000000ff;
        digest.data[i + 16] = (m_state[4] >> (24 - i * 8)) & 0x000000ff;
    }
    // restore the data
    __builtin_memcpy(m_data_buffer, data, m_data_length);
    __builtin_memcpy(m_state, state, 20);
    return digest;
}

}
