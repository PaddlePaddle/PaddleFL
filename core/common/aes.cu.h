// Modification 2021 PaddlePaddle Authors.

/**
 * AES.h
 *
 * The Advanced Encryption Standard (AES, aka AES) block cipher,
 * designed by J. Daemen and V. Rijmen.
 *
 * @author Paulo S. L. M. Barreto
 *
 * This software is hereby placed in the public domain.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

namespace common {

#ifndef USUAL_TYPES
#define USUAL_TYPES
typedef unsigned char   byte;
typedef uint32_t         u32;   /* assuming sizeof(u32) == 4 */
#endif /* USUAL_TYPES */

#ifndef AES_BLOCKBITS
#define AES_BLOCKBITS   128
#endif
#if AES_BLOCKBITS != 128
#error "AES_BLOCKBITS must be 128"
#endif

#ifndef AES_BLOCKSIZE
#define AES_BLOCKSIZE   16 /* bytes */
#endif
#if AES_BLOCKSIZE != 16
#error "AES_BLOCKSIZE must be 16"
#endif

#ifndef AES_MINKEYBITS
#define AES_MINKEYBITS  128
#endif
#if AES_MINKEYBITS != 128
#error "AES_MINKEYBITS must be 128"
#endif

#ifndef AES_MINKEYSIZE
#define AES_MINKEYSIZE  16 /* bytes */
#endif
#if AES_MINKEYSIZE != 16
#error "AES_MINKEYSIZE must be 16"
#endif

#ifndef AES_MAXKEYBITS
#define AES_MAXKEYBITS  256
#endif
#if AES_MAXKEYBITS != 256
#error "AES_MAXKEYBITS must be 256"
#endif

#ifndef AES_MAXKEYSIZE
#define AES_MAXKEYSIZE  32 /* bytes */
#endif
#if AES_MAXKEYSIZE != 32
#error "AES_MAXKEYSIZE must be 32"
#endif

#define MAXKC   (AES_MAXKEYBITS/32)
#define MAXKB   (AES_MAXKEYBITS/8)
#define MAXNR   14

#include "cuda_runtime.h"

class AES {

public:

    AES() = delete;

    AES(const void* cipherKey, u32 keyBits);

    ~AES();

    /**
     * Block size in bits.
     */
    static inline constexpr u32 blockBits() {
        return AES_BLOCKBITS;
    }

    /**
     * Block size in bytes.
     */
    static inline constexpr u32 blockSize() {
        return AES_BLOCKSIZE;
    }

    /**
     * Key size in bits.
     */
    inline u32 keyBits() const {
        // Nr for number of round
        return (Nr - 6) << 5;
    }

    /**
     * Key size in bytes.
     */
    inline u32 keySize() const {
        return (Nr - 6) << 2;
    }

    void makeKey(const void* cipherKey, u32 keyBits);

    // pt ct on device only
    void encrypt(const void *pt, void *ct, cudaStream_t stream = NULL) const;

    // decrypt not supported yet
    // void decrypt(const u32 *ct, u32 *pt);

    // pt ct on device only
    void encrypt_ecb(const void *pt, void *ct, size_t n = 1, cudaStream_t stream = NULL) const;

    // pt ct on device only
    void encrypt_ctr(const void *iv, void *ct, size_t n = 1, cudaStream_t stream = NULL) const;

    // pt ct on device only
    void encrypt_ctr_sub64(const void *iv, void *ct, size_t n = 1, cudaStream_t stream = NULL) const;

    // pt ct on device only
    void encrypt_ctr_xor(const void *iv, void *ct, size_t n = 1, cudaStream_t stream = NULL) const;

    // parallel stream not supported yet
    // void encrypt_ecb_async(const u32 *pt, u32 *ct, u32 n);

private:

    template<typename Func, typename Func2>
    void encrypt_impl(Func pt_functor, Func2 ct_functor, size_t n, cudaStream_t stream) const;

    void ExpandKey(const byte* cipherKey, u32 keyBits);

    void ExpandKeyBigEndian(const byte* cipherKey, u32 keyBits);

    // decrypt not supported yet
    // void InvertKey();

    // Nr for number of round
    // 128 bits: Nr = 10
    // 192 bits: Nr = 12
    // 256 bits: Nr = 14
    const u32 Nr;
    u32 e_sched[4 * (MAXNR + 1)];
    // u32 d_sched[4 * (MAXNR + 1)];

    // Pointers to GPU key schedules
    u32 *ce_sched;
    // u32 *cd_sched;
};
} // namespace common

