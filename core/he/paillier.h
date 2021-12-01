// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <emmintrin.h>
#include <gmpxx.h>
#include <wmmintrin.h>

namespace feature {

using block = __m128i;

class Paillier {
public:

    Paillier();

    explicit Paillier(const Paillier& other);

    Paillier& operator=(const Paillier& rhs);

    // seems no need of move semantics

    ~Paillier() {}

    void keygen(size_t n_len = 3072);

    // return pubkey byte length give bit key size
    static size_t pubkey_byte_len(size_t keysize_bit_len);

    // return privkey byte length give bit key size
    static size_t privkey_byte_len(size_t keysize_bit_len);

    // returns paillier ciphertext
    // plain required to be integer
    // plain in long, unsigned long, double type also can be accepted
    mpz_class encrypt(const mpz_class& plain);

    // returns paillier plaintext
    mpz_class decrypt(const mpz_class& cipher) const;

    // Homomorphic addition, inputs are encrypted by same pk
    mpz_class homm_add(const mpz_class& op0, const mpz_class& op1) const;

    // Homomorphic subtraction
    mpz_class homm_minus(const mpz_class& op0, const mpz_class& op1) const;

    // Homomorphic multiplication
    mpz_class homm_mult(const mpz_class& cipher, const mpz_class& plain) const;

    // seed internal prng
    void prng_seed(const block& in);

    // returns serialized pk
    std::string export_pk() const;

    // deserialize pk
    void import_pk(const std::string& in);

    // returns serialized sk
    std::string export_sk() const;

    // deserialize sk
    void import_sk(const std::string& in);

    // returns paillier ciphertext
    // plain required to be int64_t type
    mpz_class encrypt_int64_t(int64_t plain);

    // returns signed paillier plaintext mod 2^64
    int64_t decrypt_int64_t(const mpz_class& cipher) const {
        return to_int64_t(decrypt(cipher));
    }

    // returns serialized cipher
    std::string encode_cipher(const mpz_class& in) const;

    // returns deserialized mpz_class, can be cipher
    static mpz_class decode(const std::string& in);

    // returns byte_len of n^(1+n_square)
    size_t byte_len(bool n_square) const;

    mpz_class get_random_bits(size_t bits) {
        return _prng.get_z_bits(bits);
    }

    mpz_class n() {
        return _n;
    }
private:
    std::string encode(const mpz_class& in) const;

    std::string padding_leading_zero(const std::string& in,
                                     bool n_square = false) const;

    int64_t to_int64_t(const mpz_class& in) const;

    // set mismatched n will disable sk
    void set_pk(const mpz_class& n, const mpz_class& g);

    // set sk will enable pk implicitly
    void set_sk(const mpz_class& n,
                const mpz_class& lambda,
                const mpz_class& alpha);

    void prng_seed();

private:
    size_t _n_len;

    gmp_randclass _prng;

    bool _pk_set;
    bool _sk_set;

    mpz_class _n;
    mpz_class _n_square;
    mpz_class _g;
    mpz_class _lambda;

    mpz_class _alpha; // order of a subgroup of nth-residue group

    mpz_class _p;
    mpz_class _q;
    mpz_class _p_inv;
    mpz_class _q_inv;

    mpz_class _mu_p;
    mpz_class _mu_q;

    mpz_class _p_sqr;
    mpz_class _q_sqr;
    mpz_class _p_sqr_inv;
    mpz_class _q_sqr_inv;

    mpz_class _nth_residue;

    inline mpz_class powm(const mpz_class& base,
                          const mpz_class& exp) const;

};

}

