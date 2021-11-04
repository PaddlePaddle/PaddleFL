// Modification 2021 PaddlePaddle Authors.

/**
 * AES.cpp
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

#include "aes.cu.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace common {

#include "aes.tab"

#ifdef _MSC_VER
#define SWAP(x) (_lrotl(x, 8) & 0x00ff00ff | _lrotr(x, 8) & 0xff00ff00)
#define GETWORD(p) SWAP(*((u32 *)(p)))
#define PUTWORD(ct, st) (*((u32 *)(ct)) = SWAP((st)))
#else
#define GETWORD(pt) (((u32)(pt)[0] << 24) ^ ((u32)(pt)[1] << 16) ^ ((u32)(pt)[2] <<  8) ^ ((u32)(pt)[3]))
#define PUTWORD(ct, st) ((ct)[0] = (byte)((st) >> 24), (ct)[1] = (byte)((st) >> 16), (ct)[2] = (byte)((st) >>  8), (ct)[3] = (byte)(st), (st))
#endif

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define REVERSE_U32(n) ((unsigned long) (((n & 0x000000FF) << 24) | \
                                         ((n & 0x0000FF00) <<  8) | \
                                         ((n & 0x00FF0000) >>  8) | \
                                         ((n & 0xFF000000) >> 24)))
#define __AES_SWAP_ENDIAN 1

#else

#define __AES_SWAP_ENDIAN 0

#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

AES::AES(const void* cipherKey, u32 keyBits) : Nr(0), ce_sched(nullptr) {
    makeKey(cipherKey, keyBits);
    // cudaMalloc((void**)&cd_sched, sizeof(d_sched));
}

AES::~AES() {

    if (ce_sched) {
        cudaFree(ce_sched);
    }
    // cudaFree(cd_sched);
}

//////////////////////////////////////////////////////////////////////
// Support methods
//////////////////////////////////////////////////////////////////////

void AES::ExpandKey(const byte *cipherKey, u32 keyBits) {
    ExpandKeyBigEndian(cipherKey, keyBits);
#if __AES_SWAP_ENDIAN
    u32 *rek = e_sched;
    rek[0] = REVERSE_U32(rek[0]);
    rek[1] = REVERSE_U32(rek[1]);
    rek[2] = REVERSE_U32(rek[2]);
    rek[3] = REVERSE_U32(rek[3]);
    rek[4 * Nr + 0] = REVERSE_U32(rek[4 * Nr + 0]);
    rek[4 * Nr + 1] = REVERSE_U32(rek[4 * Nr + 1]);
    rek[4 * Nr + 2] = REVERSE_U32(rek[4 * Nr + 2]);
    rek[4 * Nr + 3] = REVERSE_U32(rek[4 * Nr + 3]);
#endif // __AES_SWAP_ENDIAN
}

void AES::ExpandKeyBigEndian(const byte *cipherKey, u32 keyBits) {
    u32 *rek = e_sched;
    u32 i = 0;
    u32 temp;
    rek[0] = GETWORD(cipherKey     );
    rek[1] = GETWORD(cipherKey +  4);
    rek[2] = GETWORD(cipherKey +  8);
    rek[3] = GETWORD(cipherKey + 12);
    if (keyBits == 128) {
        for (;;) {
            temp  = rek[3];
            rek[4] = rek[0] ^
                (Te4[(temp >> 16) & 0xff] & 0xff000000) ^
                (Te4[(temp >>  8) & 0xff] & 0x00ff0000) ^
                (Te4[(temp      ) & 0xff] & 0x0000ff00) ^
                (Te4[(temp >> 24)       ] & 0x000000ff) ^
                rcon[i];
            rek[5] = rek[1] ^ rek[4];
            rek[6] = rek[2] ^ rek[5];
            rek[7] = rek[3] ^ rek[6];
            if (++i == 10) {
                const_cast<u32&>(Nr) = 10;
                return;
            }
            rek += 4;
        }
    }
    rek[4] = GETWORD(cipherKey + 16);
    rek[5] = GETWORD(cipherKey + 20);
    if (keyBits == 192) {
        for (;;) {
            temp = rek[ 5];
            rek[ 6] = rek[ 0] ^
                (Te4[(temp >> 16) & 0xff] & 0xff000000) ^
                (Te4[(temp >>  8) & 0xff] & 0x00ff0000) ^
                (Te4[(temp      ) & 0xff] & 0x0000ff00) ^
                (Te4[(temp >> 24)       ] & 0x000000ff) ^
                rcon[i];
            rek[ 7] = rek[ 1] ^ rek[ 6];
            rek[ 8] = rek[ 2] ^ rek[ 7];
            rek[ 9] = rek[ 3] ^ rek[ 8];
            if (++i == 8) {
                const_cast<u32&>(Nr) = 12;
                return;
            }
            rek[10] = rek[ 4] ^ rek[ 9];
            rek[11] = rek[ 5] ^ rek[10];
            rek += 6;
        }
    }
    rek[6] = GETWORD(cipherKey + 24);
    rek[7] = GETWORD(cipherKey + 28);
    if (keyBits == 256) {
        for (;;) {
            temp = rek[ 7];
            rek[ 8] = rek[ 0] ^
                (Te4[(temp >> 16) & 0xff] & 0xff000000) ^
                (Te4[(temp >>  8) & 0xff] & 0x00ff0000) ^
                (Te4[(temp      ) & 0xff] & 0x0000ff00) ^
                (Te4[(temp >> 24)       ] & 0x000000ff) ^
                rcon[i];
            rek[ 9] = rek[ 1] ^ rek[ 8];
            rek[10] = rek[ 2] ^ rek[ 9];
            rek[11] = rek[ 3] ^ rek[10];
            if (++i == 7) {
                const_cast<u32&>(Nr) = 14;
                return;
            }
            temp = rek[11];
            rek[12] = rek[ 4] ^
                (Te4[(temp >> 24)       ] & 0xff000000) ^
                (Te4[(temp >> 16) & 0xff] & 0x00ff0000) ^
                (Te4[(temp >>  8) & 0xff] & 0x0000ff00) ^
                (Te4[(temp      ) & 0xff] & 0x000000ff);
            rek[13] = rek[ 5] ^ rek[12];
            rek[14] = rek[ 6] ^ rek[13];
            rek[15] = rek[ 7] ^ rek[14];
            rek += 8;
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Public Interface
//////////////////////////////////////////////////////////////////////

void AES::makeKey(const void* cipherKey, u32 keySize) {
    switch (keySize) {
    case 16:
    case 24:
    case 32:
        keySize <<= 3; // key size is now in bits
        break;
    case 128:
    case 192:
    case 256:
        break;
    default:
        throw std::invalid_argument("Invalid AES key size: " + std::to_string(keySize));
    }

    // set enc key
    ExpandKey((const byte*)cipherKey, keySize);
    if (!ce_sched) {
        cudaMalloc((void**)&ce_sched, sizeof(e_sched));
    }
    cudaMemcpy(ce_sched, e_sched, sizeof(e_sched), cudaMemcpyHostToDevice);

    // // set dec key
    // InvertKey();
    // cudaMemcpy(cd_sched, d_sched, sizeof(e_sched), cudaMemcpyHostToDevice);
}

void AES::encrypt(const void *pt, void* ct, cudaStream_t stream ) const {
    encrypt_ecb(pt, ct, 1, stream);
}

// template implementation of loop
// usage:
//
// auto loop = Loop<begin, end>();
// loop(func);
//
// which is equivalent to:
//
// for (int i = begin; i < end; ++i) {
//     func(i);
// }
template<int begin, int end>
struct Loop{
    template<typename Func>
    __device__ void operator()(Func func) {
        func(begin);
        auto next = Loop<begin + 1, end>();
        next(func);
    }
};

// recursion end point
template<int end>
struct Loop<end, end>{
    template<typename Func>
    __device__ void operator()(Func func) {
    }
};

template<int Nr, typename Func, typename Func2>
__global__ void AES_encrypt(Func pt_accessor, Func2 ct_accessor, u32 *rek, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {

        size_t offset = i << 2;

        u32 s[4];
        u32 t[4];
        /*
         * map byte array block to cipher state
         * and add initial round key:
         */
        s[0] = pt_accessor(offset, 0) ^ rek[0];
        s[1] = pt_accessor(offset, 1) ^ rek[1];
        s[2] = pt_accessor(offset, 2) ^ rek[2];
        s[3] = pt_accessor(offset, 3) ^ rek[3];

#define __AES_ROUND_FUNC(out, in, idx) \
    do { \
        (out)[0] = cTe0[(in)[0] >> 24] ^ cTe1[((in)[1] >> 16) & 0xff] ^ cTe2[((in)[2] >>  8) & 0xff] ^ cTe3[(in)[3] & 0xff] ^ rek[4 * (idx)];     \
        (out)[1] = cTe0[(in)[1] >> 24] ^ cTe1[((in)[2] >> 16) & 0xff] ^ cTe2[((in)[3] >>  8) & 0xff] ^ cTe3[(in)[0] & 0xff] ^ rek[4 * (idx) + 1]; \
        (out)[2] = cTe0[(in)[2] >> 24] ^ cTe1[((in)[3] >> 16) & 0xff] ^ cTe2[((in)[0] >>  8) & 0xff] ^ cTe3[(in)[1] & 0xff] ^ rek[4 * (idx) + 2]; \
        (out)[3] = cTe0[(in)[3] >> 24] ^ cTe1[((in)[0] >> 16) & 0xff] ^ cTe2[((in)[1] >>  8) & 0xff] ^ cTe3[(in)[2] & 0xff] ^ rek[4 * (idx) + 3]; \
    } while (0)

        /* round 1: */
#if __AES_SWAP_ENDIAN
        t[0] = cTe0[s[0] & 0xff] ^ cTe1[(s[1] >> 8) & 0xff] ^ cTe2[(s[2] >> 16) & 0xff] ^ cTe3[s[3] >> 24] ^ rek[4];
        t[1] = cTe0[s[1] & 0xff] ^ cTe1[(s[2] >> 8) & 0xff] ^ cTe2[(s[3] >> 16) & 0xff] ^ cTe3[s[0] >> 24] ^ rek[5];
        t[2] = cTe0[s[2] & 0xff] ^ cTe1[(s[3] >> 8) & 0xff] ^ cTe2[(s[0] >> 16) & 0xff] ^ cTe3[s[1] >> 24] ^ rek[6];
        t[3] = cTe0[s[3] & 0xff] ^ cTe1[(s[0] >> 8) & 0xff] ^ cTe2[(s[1] >> 16) & 0xff] ^ cTe3[s[2] >> 24] ^ rek[7];
#else  // __AES_SWAP_ENDIAN
        __AES_ROUND_FUNC(t, s, 1);
#endif // __AES_SWAP_ENDIAN

        auto lambda = [&s, &t, &rek] (int i) {
            __AES_ROUND_FUNC(s, t, 2 * i);
            __AES_ROUND_FUNC(t, s, 2 * i + 1);
        };

        // compiling phase loop unroll
        auto loop = Loop<1, Nr / 2>();
        loop(lambda);

        rek += Nr << 2;

        /*
         * apply last round and
         * map cipher state to byte array block:
         */

        u32 ct[4];

#if __AES_SWAP_ENDIAN
#define __AES_FINAL_ROUND_FUNC(i) \
    do {                                                     \
    ct[i] =                                         \
        (cTe4[(t[(i + 0) % 4] >> 24)       ] & 0x000000ff) ^ \
        (cTe4[(t[(i + 1) % 4] >> 16) & 0xff] & 0x0000ff00) ^ \
        (cTe4[(t[(i + 2) % 4] >>  8) & 0xff] & 0x00ff0000) ^ \
        (cTe4[(t[(i + 3) % 4]      ) & 0xff] & 0xff000000) ^ \
        rek[i];                                              \
    } while (0)

#else  // __AES_SWAP_ENDIAN
#define __AES_FINAL_ROUND_FUNC(i) \
    do {                                                     \
    ct[i] =                                         \
        (cTe4[(t[(i + 0) % 4] >> 24)       ] & 0xff000000) ^ \
        (cTe4[(t[(i + 1) % 4] >> 16) & 0xff] & 0x00ff0000) ^ \
        (cTe4[(t[(i + 2) % 4] >>  8) & 0xff] & 0x0000ff00) ^ \
        (cTe4[(t[(i + 3) % 4]      ) & 0xff] & 0x000000ff) ^ \
        rek[i];                                              \
    } while (0)
#endif // __AES_SWAP_ENDIAN

        __AES_FINAL_ROUND_FUNC(0);
        __AES_FINAL_ROUND_FUNC(1);
        __AES_FINAL_ROUND_FUNC(2);
        __AES_FINAL_ROUND_FUNC(3);

        ct_accessor(ct, offset);
    }
}

#define AES_CUDA_THREAD_SIZE 512

template<typename Func, typename Func2>
void AES::encrypt_impl(Func pt_functor, Func2 ct_functor, size_t n, cudaStream_t stream) const {

    dim3 block_size = dim3(AES_CUDA_THREAD_SIZE, 1);
    dim3 grid_size = dim3((n + AES_CUDA_THREAD_SIZE - 1) / AES_CUDA_THREAD_SIZE, 1);

    switch (Nr) {
    case 10: AES_encrypt<10><<<grid_size, block_size, 0, stream>>>(
                 pt_functor, ct_functor, ce_sched, n);
             break;
    case 12: AES_encrypt<12><<<grid_size, block_size, 0, stream>>>(
                 pt_functor, ct_functor, ce_sched, n);
             break;
    case 14: AES_encrypt<14><<<grid_size, block_size, 0, stream>>>(
                 pt_functor, ct_functor, ce_sched, n);
             break;
    }
    // cudaStreamSynchronize(stream);
}

void AES::encrypt_ecb(const void* pt, void* ct, size_t n, cudaStream_t stream) const {
    auto pt_functor = [pt] __device__ (size_t offset, int idx) -> u32 {
        return reinterpret_cast<const u32*>(pt)[offset + idx]; };

    auto ct_functor = [ct] __device__ (u32* ct_ ,size_t offset) {
        reinterpret_cast<u32*>(ct)[offset]     = ct_[0];
        reinterpret_cast<u32*>(ct)[offset + 1] = ct_[1];
        reinterpret_cast<u32*>(ct)[offset + 2] = ct_[2];
        reinterpret_cast<u32*>(ct)[offset + 3] = ct_[3];
    };

    encrypt_impl(pt_functor, ct_functor, n , stream);
}

void AES::encrypt_ctr(const void* iv, void* ct, size_t n, cudaStream_t stream) const {

    auto iv_functor = [iv] __device__ (size_t offset, int idx) -> u32 {
        return reinterpret_cast<const u32*>(iv)[idx] + (offset >> 2) * (idx == 3);
    };

    auto ct_functor = [ct] __device__ (u32* ct_ ,size_t offset) {
        reinterpret_cast<u32*>(ct)[offset]     = ct_[0];
        reinterpret_cast<u32*>(ct)[offset + 1] = ct_[1];
        reinterpret_cast<u32*>(ct)[offset + 2] = ct_[2];
        reinterpret_cast<u32*>(ct)[offset + 3] = ct_[3];
    };

    encrypt_impl(iv_functor, ct_functor, n , stream);
}

void AES::encrypt_ctr_sub64(const void* iv, void* ct, size_t n, cudaStream_t stream) const {

    auto iv_functor = [iv] __device__ (size_t offset, int idx) -> u32 {
        return reinterpret_cast<const u32*>(iv)[idx] + (offset >> 2) * (idx == 3);
    };

    auto ct_functor = [ct] __device__ (u32* ct_ ,size_t offset) {
        uint64_t* dest_ = reinterpret_cast<uint64_t*>(ct) + (offset >> 1);
        dest_[0] -= reinterpret_cast<uint64_t*>(ct_)[0];
        dest_[1] -= reinterpret_cast<uint64_t*>(ct_)[1];
    };

    encrypt_impl(iv_functor, ct_functor, n , stream);
}

void AES::encrypt_ctr_xor(const void* iv, void* ct, size_t n, cudaStream_t stream) const {

    auto iv_functor = [iv] __device__ (size_t offset, int idx) -> u32 {
        return reinterpret_cast<const u32*>(iv)[idx] + (offset >> 2) * (idx == 3);
    };

    auto ct_functor = [ct] __device__ (u32* ct_ ,size_t offset) {
        reinterpret_cast<u32*>(ct)[offset]     ^= ct_[0];
        reinterpret_cast<u32*>(ct)[offset + 1] ^= ct_[1];
        reinterpret_cast<u32*>(ct)[offset + 2] ^= ct_[2];
        reinterpret_cast<u32*>(ct)[offset + 3] ^= ct_[3];
    };

    encrypt_impl(iv_functor, ct_functor, n , stream);
}
} // namespace common
