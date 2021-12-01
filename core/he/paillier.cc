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

#include "paillier.h"

#include <gmp.h>
#include <gmpxx.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>

namespace feature {

block read_block_from_dev_urandom() {
    block ret;
    std::ifstream in("/dev/urandom");
    in.read(reinterpret_cast<char *>(&ret), sizeof(ret));
    return ret;
}

// T stores an integer, native endianness
template<typename T>
mpz_class import_from_value(const T& in) {
    mpz_class out;
    mpz_import(out.get_mpz_t(), 1, 1, sizeof(in), 0, 0, &in);
    return out;
}

// T required to be sequence container
// most significant element first, and host byte order within each value
template<typename T>
mpz_class import_from_container(const T& in) {
    mpz_class out;
    mpz_import(out.get_mpz_t(), in.size(), 1, sizeof(typename T::value_type), 0, 0, in.data());
    return out;
}

mpz_class import_from_string(const std::string& in) {
    return import_from_container(in);
}

// mod in to fit T
template<typename T>
T export_to_value(const mpz_class& in) {
    T out(0);
    mpz_class mod(1);
    size_t numb = 8 * sizeof(T);
    mod <<= numb;
    mod = in % mod;
    mpz_export(&out, nullptr, 1, sizeof(T), 0, 0, mod.get_mpz_t());
    return out;
}

template<typename T>
T export_to_container(const mpz_class& in) {
    size_t type_size = sizeof(typename T::value_type);
    size_t numb = 8 * type_size;
    size_t cnt = (mpz_sizeinbase(in.get_mpz_t(), 2) + numb - 1) / numb;
    T out;
    out.resize(cnt);
    mpz_export(out.data(), nullptr, 1, type_size, 0, 0, in.get_mpz_t());
    return out;
}

std::string export_to_string(const mpz_class& in) {
    auto r = export_to_container<std::vector<char>>(in);
    return std::string(r.begin(), r.end());
}

Paillier::Paillier() :
        _n_len(0),
        _prng(gmp_randinit_default),
        _pk_set(false),
        _sk_set(false) {
    prng_seed();
}

Paillier::Paillier(const Paillier& other) :
        _prng(gmp_randinit_default), _pk_set(false), _sk_set(false) {
    prng_seed();
    if (other._sk_set) {
        set_sk(other._n, other._lambda, other._alpha);
    } else if (other._pk_set) {
        set_pk(other._n, other._g);
    }
}

Paillier& Paillier::operator=(const Paillier& rhs) {
    _sk_set = false;
    _pk_set = false;

    if (rhs._sk_set) {
        set_sk(rhs._n, rhs._lambda, rhs._alpha);
    } else if (rhs._pk_set) {
        set_pk(rhs._n, rhs._g);
    }
    return *this;
}

void Paillier::prng_seed(const block& in) {
    _prng.seed(import_from_value(in));
}

void Paillier::prng_seed() {
    prng_seed(read_block_from_dev_urandom());
}

// M-R test iteration number from openssl/bn.h
inline size_t miller_rabin_iteration_num(size_t len) {
    return len >= 3747 ? 3 :
        len >= 1345 ? 4 :
        len >= 476 ? 5 :
        len >= 400 ? 6 :
        len >= 347 ? 7 :
        len >= 308 ? 4 :
        len >= 55 ? 27 : 34;
}

int prime_test(const mpz_class& op) {
    size_t len = mpz_sizeinbase(op.get_mpz_t(), 2);
    return mpz_probab_prime_p(op.get_mpz_t(),
                              miller_rabin_iteration_num(len));
}

mpz_class gen_prime(size_t len, gmp_randclass& prng) {
    mpz_class p;
    for (;;) {
        p = prng.get_z_bits(len);
        p |= mpz_class(1) << len - 1;
        if (prime_test(p)) {
            break;
        }
    }
    return p;
}

mpz_class gen_dsa_prime(size_t len, const mpz_class& q, gmp_randclass& prng) {
    mpz_class p;
    size_t q_len = mpz_sizeinbase(q.get_mpz_t(), 2);
    for (;;) {
        size_t p_len = len - q_len;
        p = prng.get_z_bits(p_len);
        p |= mpz_class(1) << p_len - 1;
        p *= q;
        p += 1;
        p_len = mpz_sizeinbase(p.get_mpz_t(), 2);
        if (prime_test(p) && p_len == len) {
            break;
        }
    }
    return p;
}

inline size_t get_symmetric_key_size(size_t rsa_n_len) {
    if (rsa_n_len >= 3072) {
        return 128;
    } else if (rsa_n_len >= 2048) {
        return 112;
    } else if (rsa_n_len >= 1024) {
        return 80;
    }
    return 0;
}

void Paillier::keygen(size_t n_len) {
    if (n_len < 1024) {
        throw std::logic_error("key too short, not safe");
    }

    size_t p_len = (n_len + 1) / 2;

    // double length required for asymmetric key
    size_t alpha_len = 2 * get_symmetric_key_size(n_len);

    for (_n_len = 0; _n_len != n_len;) {
        _alpha = gen_prime(alpha_len, _prng);
        _p = gen_dsa_prime(p_len, _alpha, _prng);
        _q = gen_dsa_prime(p_len, _alpha, _prng);

        _n = _p * _q;
        _n_len = mpz_sizeinbase(_n.get_mpz_t(), 2);
    }

    _lambda = (_p - 1) * (_q - 1);

    set_sk(_n, _lambda, _alpha);
}

void Paillier::set_pk(const mpz_class& n, const mpz_class& g) {
    if (n <= 1 || g <= 1) {
        throw std::logic_error("invalid pk");
    }
    // get bit len
    size_t len = mpz_sizeinbase(n.get_mpz_t(), 2);
    if (len < 1024) {
        throw std::logic_error("key len too short, not safe");
    }
    _n_len = len;
    _sk_set &= n == _n;
    _n = n;
    _g = g;
    _n_square = n * n;

    _nth_residue = powm(_g, _n);
    _pk_set = true;
}

std::pair<mpz_class, mpz_class> p_q_from_phi_n(const mpz_class& phi,
                                              const mpz_class& n) {
    mpz_class b = n + 1 -phi;
    mpz_class root = b * b - 4 * n;
    mpz_sqrt(root.get_mpz_t(), root.get_mpz_t());
    return {(b + root) / 2, (b - root) / 2};
}

void Paillier::set_sk(const mpz_class& n,
                      const mpz_class& lambda,
                      const mpz_class& alpha){
    if (n <= 1) {
        throw std::logic_error("invalid pk");
    }
    mpz_class verify = 2;
    mpz_powm(verify.get_mpz_t(), verify.get_mpz_t(),
             lambda.get_mpz_t(), n.get_mpz_t());
    if ((verify != 1)) {
        throw std::logic_error("invalid sk");
    }

    auto pq = p_q_from_phi_n(lambda, n); // lambda == phi

    _p = pq.first;
    _q = pq.second;

    if (_p * _q != n) {
        throw std::logic_error("invalid sk");
    }

    _n = n;
    _n_len = mpz_sizeinbase(_n.get_mpz_t(), 2);
    _n_square = n * n;
    _lambda = lambda;
    _alpha = alpha;

    _p_sqr = _p * _p;
    _q_sqr = _q * _q;

    mpz_invert(_p_inv.get_mpz_t(),
               _p.get_mpz_t(), _q.get_mpz_t());

    mpz_invert(_q_inv.get_mpz_t(),
               _q.get_mpz_t(), _p.get_mpz_t());

    _mu_p = _alpha * _q;
    _mu_q = _alpha * _p;

    mpz_invert(_mu_p.get_mpz_t(),
               _mu_p.get_mpz_t(), _p.get_mpz_t());

    mpz_invert(_mu_q.get_mpz_t(),
               _mu_q.get_mpz_t(), _q.get_mpz_t());

    mpz_invert(_p_sqr_inv.get_mpz_t(),
               _p_sqr.get_mpz_t(), _q_sqr.get_mpz_t());

    mpz_invert(_q_sqr_inv.get_mpz_t(),
               _q_sqr.get_mpz_t(), _p_sqr.get_mpz_t());

    _sk_set = true; // to enable crt powm

    // generate a element of order n * alpha
    auto gen_g = [this]() {
        mpz_class lcm;
        mpz_class p_ = _p - 1;
        mpz_class q_ = _q - 1;

        mpz_lcm(lcm.get_mpz_t(), p_.get_mpz_t(), q_.get_mpz_t());

        mpz_class order = lcm / _alpha;

        mpz_class ret;
        for (;;) {
            // order won't be to large
            // and 1 / order elements satisfies
            ret = _prng.get_z_bits(_n_len);
            ret = powm(ret, order);
            if (powm(ret, _n) != 1 &&
                powm(ret, _alpha) != 1) {
                return ret;
            }
        }
    };

    _g = gen_g();
    set_pk(n, _g);
}

inline mpz_class powm_crt(const mpz_class& base,
                          const mpz_class& exp,
                          const mpz_class& m0,
                          const mpz_class& m0_inv,
                          const mpz_class& m1,
                          const mpz_class& m1_inv,
                          const mpz_class& m) {
    mpz_class y0;
    mpz_class y1;
    mpz_powm(y0.get_mpz_t(), base.get_mpz_t(),
             exp.get_mpz_t(), m0.get_mpz_t());

    mpz_powm(y1.get_mpz_t(), base.get_mpz_t(),
             exp.get_mpz_t(), m1.get_mpz_t());

    return (y0 * m1_inv * m1 + y1 * m0_inv * m0) % m;
}

inline mpz_class Paillier::powm(const mpz_class& base,
                                const mpz_class& exp) const {
    mpz_class ret;
    if (_sk_set) {
        ret = powm_crt(base, exp,
                       _p_sqr, _p_sqr_inv,
                       _q_sqr, _q_sqr_inv,
                       _n_square);
    } else {
        mpz_powm(ret.get_mpz_t(), base.get_mpz_t(),
                 exp.get_mpz_t(), _n_square.get_mpz_t());
    }
    return ret;
}

mpz_class Paillier::encrypt(const mpz_class& plain) {
    if (!_pk_set) {
        throw std::logic_error("pk not set");
    }

    // entropy as 2 * symmetric key size
    mpz_class r = _prng.get_z_bits(2 * get_symmetric_key_size(_n_len));

    r = powm(_nth_residue, r);

    mpz_class cipher = 1 + plain * _n;

    return r * cipher % _n_square;
}

mpz_class Paillier::decrypt(const mpz_class& cipher) const {
    if (!_sk_set) {
        throw std::logic_error("sk not set");
    }
    auto dec_p = [this, &cipher](const mpz_class& p,
                                 const mpz_class& p_sqr,
                                 const mpz_class& mu) -> mpz_class {
        mpz_class exp = _alpha;
        mpz_powm(exp.get_mpz_t(), cipher.get_mpz_t(),
                 exp.get_mpz_t(), p_sqr.get_mpz_t());
        exp = (exp - 1) / p; // now exp = L(x) = (x - 1) / p
        return exp * mu % p;
    };
    mpz_class c_p = dec_p(_p, _p_sqr, _mu_p);
    mpz_class c_q = dec_p(_q, _q_sqr, _mu_q);
    return (c_p * _q_inv * _q + c_q * _p_inv * _p) % _n;
}

mpz_class Paillier::homm_add(const mpz_class& op0, const mpz_class& op1) const {
    if (!_pk_set) {
        throw std::logic_error("pk not set");
    }
    return op0 * op1 % _n_square;
}

mpz_class Paillier::homm_mult(const mpz_class& cipher, const mpz_class& plain) const {
    if (!_pk_set) {
        throw std::logic_error("pk not set");
    }
    mpz_class ret = powm(cipher, plain);
    return ret;
}

mpz_class Paillier::homm_minus(const mpz_class& op0, const mpz_class& op1) const {
    auto neg = homm_mult(op1, -1);
    return homm_add(op0, neg);
}

size_t Paillier::byte_len(bool n_square) const {
    size_t byte_len = (_n_len + 7) / 8;
    return byte_len * (1 + n_square);
}

std::string Paillier::padding_leading_zero(const std::string& in, bool n_square) const {
    if (!_pk_set) {
        throw std::logic_error("pk not set");
    }
    size_t zero_num = byte_len(n_square) - in.size();
    return std::string(zero_num, '\x00') + in;
}

std::string Paillier::encode(const mpz_class& in) const {
    return padding_leading_zero(export_to_string(in));
}

std::string Paillier::encode_cipher(const mpz_class& in) const {
    return padding_leading_zero(export_to_string(in), true);
}

int64_t Paillier::to_int64_t(const mpz_class& in) const {
    int sign = in > _n / 2 ? -1 : 1;
    mpz_class neg = _n - in;
    int64_t val = export_to_value<int64_t>(sign > 0 ? in : neg);
    return sign * val;
}

mpz_class Paillier::decode(const std::string& in) {
    return import_from_string(in);
}

std::string Paillier::export_pk() const {
    // check if pk set internally
    // bitlen(_g) = 2 * n_len
    return encode(_n) + encode_cipher(_g);
}

size_t Paillier::pubkey_byte_len(size_t keysize_bit_len) {
    size_t n_byte_len = (keysize_bit_len + 7) / 8;
    // 1 for _n, 2 for _g, see above
    return 3 * n_byte_len;
}

std::string Paillier::export_sk() const {
    if (!_sk_set) {
        throw std::logic_error("sk not set");
    }
    return encode(_n) + encode(_lambda) + encode(_alpha);
}

size_t Paillier::privkey_byte_len(size_t keysize_bit_len) {
    size_t n_byte_len = (keysize_bit_len + 7) / 8;
    // 1 for _n, 1 for _lambda, 1 for _alpha
    return 3 * n_byte_len;
}

void Paillier::import_pk(const std::string& in) {
    if (in.size() % 3) {
        throw std::logic_error("invalid pk");
    }
    size_t n_byte_len = in.size() / 3;
    set_pk(decode(std::string(in, 0, n_byte_len)),
           decode(std::string(in, n_byte_len)));
}

void Paillier::import_sk(const std::string& in) {
    if (in.size() % 3) {
        throw std::logic_error("invalid sk");
    }
    size_t n_byte_len = in.size() / 3;
    set_sk(decode(std::string(in, 0, n_byte_len)),
           decode(std::string(in, n_byte_len, n_byte_len)),
           decode(std::string(in, 2 * n_byte_len)));
}

mpz_class Paillier::encrypt_int64_t(int64_t plain) {
    auto p_ = import_from_value(plain);
    return encrypt(p_);
}

}
