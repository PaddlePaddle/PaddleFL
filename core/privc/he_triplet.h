/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifndef SMC_MAIN_COMMON_HE_TRIPLET_H
#define SMC_MAIN_COMMON_HE_TRIPLET_H

#include <gmp.h>
#include <gmpxx.h>
#include <omp.h>
#include <queue>
#include <chrono>

#include "seal/seal.h"
#include "glog/logging.h"

#include "core/common/prng.h"
#include "core/common/tensor_adapter.h"
#include "core/paddlefl_mpc/mpc_protocol/abstract_network.h"

namespace privc {

using seal::Encryptor;
using seal::Evaluator;
using seal::Decryptor;
using seal::EncryptionParameters;
using seal::RelinKeys;
using seal::scheme_type;
using seal::SEALContext;
using seal::PublicKey;
using seal::SecretKey;
using seal::Plaintext;
using seal::Ciphertext;
using seal::BatchEncoder;
using seal::KeyGenerator;
using seal::CoeffModulus;
using seal::PlainModulus;

using seal::Serializable;

using paddle::mpc::AbstractNetwork;
using common::PseudorandomNumberGenerator;
using common::TensorAdapter;

template<typename T, size_t N>
T fixed_mult(T a, T b) {
    __int128_t ret = (__int128_t) a * b;
    return (T) (ret >> N);
}

// type T canbe uint64_t, uint32_t that decides
//        generating triplets bits length (64-bit or 32-bit)
// size N is decimal bits
template<typename T, size_t N>
class HETriplet {
public:
    HETriplet() = delete;

    // @param poly_modulus_degree: seal's poly_modulus_degree
    //                              can be 4096, 8192, 16384
    // @param max_seal_plain_bit: max plain_modulus bit len
    //                            must less than 61
    // @ param num_thread: number of threadss
    HETriplet(size_t party,
              AbstractNetwork* io,
              PseudorandomNumberGenerator& prng,
              size_t poly_modulus_degree = 8192,
              size_t max_seal_plain_bit = 60,
              size_t num_thread = 0);

    ~HETriplet() {}

    // init seal context
    void init();

    // get triplet
    std::array<T, 3> get_triplet();

    template <typename U>
    void get_triplet(TensorAdapter<U>* ret);

    // get penta triplet
    std::array<T, 5> get_penta_triplet();

    template <typename U>
    void get_penta_triplet(TensorAdapter<U>* ret);

    // recover result from Chinese Remainder Theorem (CRT)
    static void recover_crt(const std::vector<std::vector<uint64_t>>& in,
                            const std::vector<uint64_t>& plain_modulus,
                            std::vector<mpz_class>& out,
                            const mpz_class& triplet_modulus);

    // fill triplet into queue
    void fill_triplet_buffer(std::queue<std::array<T, 3>> &queue);

    // fill penta triplet into queue
    void fill_penta_triplet_buffer(std::queue<std::array<T, 5>> &queue);

private:

    void send(const void* data, size_t size) {
        _io->send(1 - _party, data, size);
    }

    void recv(void* data, size_t size) {
        _io->recv(1 - _party, data, size);
    }

    template <class U>
    void send(const U& val) {
        send(&val, sizeof(U));
    }

    template <class U>
    U recv() {
        U val;
        recv(&val, sizeof(U));
        return val;
    }

    void send_str(const std::string& str, size_t size) {
        send<size_t>(size);
        if (size != 0) {
            send(str.data(), size);
        }
    }

    void recv_str(std::string& str) {
        size_t size = recv<size_t>();
        str.resize(size);
        if (size != 0) {
            recv(&str.at(0), size);
        }
    }

    // calc modulus op for vector
    template<typename U>
    void vec_mod(const std::vector<U>& in,
                 std::vector<uint64_t>& out,
                 uint64_t modulus) {
        std::transform(in.begin(), in.end(), out.begin(),
                              [&modulus](U a) {
                                   return a % modulus;
                               });
    }

    // calc modulus op for vector
    void vec_mod(const std::vector<mpz_class>& in,
                 std::vector<uint64_t>& out,
                 uint64_t modulus) {
        std::transform(in.begin(), in.end(), out.begin(),
                        [&modulus](const mpz_class& a) {
                            mpz_class ret;
                            mpz_mod(ret.get_mpz_t(), a.get_mpz_t(),
                                    mpz_class(modulus).get_mpz_t());
                            return ret.get_ui();
                        });
    }

    template<typename U>
    U rand_val() {
        U val;
        _prng.get_array(&val, sizeof(val));
        return val;
    }

    // calc triplet element 'c' based CRT
    void calc_triplet_c(const std::vector<uint64_t>& r_vec,
                        const Plaintext& a_plain,
                        const Plaintext& b_plain,
                        Ciphertext& recv_a_cipher,
                        Ciphertext& recv_b_cipher,
                        Evaluator& evaluator,
                        BatchEncoder& batch_encoder,
                        Encryptor& encryptor,
                        RelinKeys& relin_keys,
                        Ciphertext& c_cipher);

    // calc triplet element 'c' based CRT
    void calc_penta_triplet_c(
                        const std::vector<uint64_t>& r_vec,
                        const std::vector<uint64_t>& r_alpha_vec,
                        const Plaintext& a_plain,
                        const Plaintext& a_alpha_plain,
                        const Plaintext& b_plain,
                        Ciphertext& recv_a_cipher,
                        Ciphertext& recv_alpha_cipher,
                        Ciphertext& recv_b_cipher,
                        Evaluator& evaluator,
                        BatchEncoder& batch_encoder,
                        Encryptor& encryptor,
                        RelinKeys& relin_keys,
                        Ciphertext& c_cipher,
                        Ciphertext& c_alpha_cipher);

    static const int _s_statistcal_security_bit = 40;

    AbstractNetwork* _io;

    PseudorandomNumberGenerator& _prng;

    const size_t _party;

    int _total_plain_bit;

    gmp_randclass _prng_gmp;

    size_t _triplet_step;

    size_t _num_thread;

    bool _quiet;

    // seal ctx
    std::vector<std::shared_ptr<SEALContext>> _contexts;
    std::vector<PublicKey> _public_keys;
    std::vector<SecretKey> _secret_keys;
    std::vector<RelinKeys> _relin_keys;
    std::vector<uint64_t> _plain_modulus;
    mpz_class _triplet_modulus;
    size_t _max_seal_plain_bit;

    std::queue<std::array<T, 3>> _triplet_buffer;
    std::queue<std::array<T, 5>> _penta_triplet_buffer;
};

} // namespace privc

#include "./he_triplet_impl.h"

#endif
