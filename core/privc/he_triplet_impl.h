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

#include <gmp.h>
#include <gmpxx.h>

#include "seal/seal.h"
#include "glog/logging.h"

namespace privc {

#ifdef VERBOSE_MODE
using std::chrono::system_clock;

inline double duration(system_clock::time_point& end,
                       system_clock::time_point& begin) {
    return double(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                    end - begin).count());
}
#endif

template<typename T, size_t N>
HETriplet<T, N>::HETriplet(size_t party,
                        AbstractNetwork* io,
                        PseudorandomNumberGenerator& prng,
                        size_t poly_modulus_degree,
                        size_t max_seal_plain_bit,
                        size_t num_thread):
    _prng(prng), _party(party), _io(io),
    _prng_gmp(gmp_randinit_default),

    // slot count (batch size) for seal is equal to poly_modulus_degree
    // so triplet step is set to poly_modulus_degree
    _triplet_step(poly_modulus_degree),
    _max_seal_plain_bit(max_seal_plain_bit),
    _num_thread(num_thread) {

    size_t bit_len = sizeof(T) * 8;
    _triplet_modulus = mpz_class(1) << bit_len;

    // total plaintext modulus length
    _total_plain_bit = 2 * bit_len + _s_statistcal_security_bit + 2;

    _prng_gmp.seed(this->rand_val<uint64_t>());

    if (_num_thread != 0) {
        omp_set_num_threads(_num_thread);
    }
}

// calc triplet element 'c' based CRT
// that is, c0 = a0b1 + b0a1 + r
// all elements (a0, b0, a1, b1, r) have been mod CRT modulus
template<typename T, size_t N>
void HETriplet<T, N>::calc_triplet_c(const std::vector<uint64_t>& r_vec,
                    const Plaintext& local_a_plain,
                    const Plaintext& local_b_plain,
                    Ciphertext& remote_a_cipher,
                    Ciphertext& remote_b_cipher,
                    Evaluator& evaluator,
                    BatchEncoder& batch_encoder,
                    Encryptor& encryptor,
                    RelinKeys& relin_keys,
                    Ciphertext& c_cipher) {
    Plaintext r_plain;

    batch_encoder.encode(r_vec, r_plain);

    evaluator.multiply_plain_inplace(remote_a_cipher, local_b_plain);

    evaluator.multiply_plain_inplace(remote_b_cipher, local_a_plain);

    evaluator.add(remote_a_cipher, remote_b_cipher, c_cipher);

    evaluator.relinearize_inplace(c_cipher, relin_keys);

    evaluator.add_plain_inplace(c_cipher, r_plain);

}

// calc triplet element 'c' 'c_alpha' based CRT
// that is, c0 = a0 * b1 + b0 * a1 + r
//          c0_alpha = alpha0 * b1 + b0 * alpha1 + r'
// all elements (a0, alpha0, b0, a1, alpha1, b1, r, r') have been mod CRT modulus
template<typename T, size_t N>
void HETriplet<T, N>::calc_penta_triplet_c(
                    const std::vector<uint64_t>& r_vec,
                    const std::vector<uint64_t>& r_alpha_vec,
                    const Plaintext& local_a_plain,
                    const Plaintext& local_alpha_plain,
                    const Plaintext& local_b_plain,
                    Ciphertext& remote_a_cipher,
                    Ciphertext& remote_alpha_cipher,
                    Ciphertext& remote_b_cipher,
                    Evaluator& evaluator,
                    BatchEncoder& batch_encoder,
                    Encryptor& encryptor,
                    RelinKeys& relin_keys,
                    Ciphertext& c_cipher,
                    Ciphertext& c_alpha_cipher) {
    Plaintext r_plain;
    Plaintext r_alpha_plain;

    batch_encoder.encode(r_vec, r_plain);
    batch_encoder.encode(r_alpha_vec, r_alpha_plain);

    evaluator.multiply_plain_inplace(remote_a_cipher, local_b_plain);
    evaluator.multiply_plain_inplace(remote_alpha_cipher, local_b_plain);

    Ciphertext tmp;
    Ciphertext tmp_alpha;

    evaluator.multiply_plain(remote_b_cipher, local_a_plain, tmp);
    evaluator.multiply_plain(remote_b_cipher, local_alpha_plain, tmp_alpha);

    evaluator.add(tmp, remote_a_cipher, c_cipher);
    evaluator.add(tmp_alpha, remote_alpha_cipher, c_alpha_cipher);

    evaluator.relinearize_inplace(c_cipher, relin_keys);
    evaluator.relinearize_inplace(c_alpha_cipher, relin_keys);

    evaluator.add_plain_inplace(c_cipher, r_plain);
    evaluator.add_plain_inplace(c_alpha_cipher, r_alpha_plain);

}

template<typename T, size_t N>
void HETriplet<T, N>::init() {

    _num_thread = omp_get_max_threads();

    // alice setup seal ctx
    if (_party == 0) {
        // sync _num_thread
        size_t remote_num_thread = recv<size_t>();
        send<size_t>(_num_thread);
        _num_thread = std::max(remote_num_thread, _num_thread);

        // using BFV scheme
        EncryptionParameters parms(seal::scheme_type::bfv);

        // set poly_modulus_degree
        size_t poly_modulus_degree = _triplet_step;

        parms.set_poly_modulus_degree(poly_modulus_degree);

        std::vector<uint64_t> plain_modulus;

        int seal_plain_bit = _max_seal_plain_bit;
        int plain_bit_count = 0;

        // find enough seal instances for CRT
        // (length CRT_M(accumulate plain_modulus) > total_plain_bit)
        while (_total_plain_bit - plain_bit_count > 0) {

            // set coeff_modulus
            auto coeff_modulus = CoeffModulus::BFVDefault(poly_modulus_degree);
            parms.set_coeff_modulus(coeff_modulus);

            // generate different plain modulus
            auto plain_modulus = PlainModulus::Batching(poly_modulus_degree,
                                                    seal_plain_bit--);
            plain_bit_count += floor(std::log2(plain_modulus.value()));
            parms.set_plain_modulus(plain_modulus);

            std::shared_ptr<SEALContext> context = std::make_shared<SEALContext>(parms);

            KeyGenerator keygen(*context);

            SecretKey sk = keygen.secret_key();

            PublicKey pk;
            keygen.create_public_key(pk);

            Serializable<RelinKeys> relin_keys = keygen.create_relin_keys();

            // tell bob to recv ctx stream
            send<bool>(true);

            // send parms
            std::stringstream parms_str;
            auto size = parms.save(parms_str);
            send_str(parms_str.str(), size);

            // send public key
            std::stringstream pk_str;
            auto size_pk = pk.save(pk_str);
            send_str(pk_str.str(), size_pk);

            // send relin key
            std::stringstream relin_keys_ss;
            size_t size_relin_key = relin_keys.save(relin_keys_ss);
            send_str(relin_keys_ss.str(), size_relin_key);

            _contexts.emplace_back(context);
            _public_keys.emplace_back(pk);
            _secret_keys.emplace_back(sk);

            _plain_modulus.emplace_back(parms.plain_modulus().value());

        }
        // tell bob end to recv ctx stream
        send<bool>(false);

    } else {
        // sync num_thread
        send<size_t>(_num_thread);
        size_t remote_num_thread = recv<size_t>();
        _num_thread = std::max(remote_num_thread, _num_thread);

        // get enough seal instances for CRT from alice
        bool recv_ctx_stream = recv<bool>();
        while (recv_ctx_stream) {
            // recv parms and public key from receiver
            std::string parms_str;
            recv_str(parms_str);

            std::string pk_str;
            recv_str(pk_str);

            std::string relin_keys_str;
            recv_str(relin_keys_str);

            EncryptionParameters parms(seal::scheme_type::bfv);
            auto parms_ss = std::stringstream(parms_str);
            parms.load(parms_ss);

            std::shared_ptr<SEALContext> context = std::make_shared<SEALContext>(parms);

            _contexts.emplace_back(context);

            auto pk_ss = std::stringstream(pk_str);

            PublicKey pk;
            pk.load(*context, pk_ss);
            _public_keys.emplace_back(pk);

            std::stringstream relin_keys_ss(relin_keys_str);
            RelinKeys relin_key;
            relin_key.load(*context, relin_keys_ss);
            _relin_keys.emplace_back(relin_key);

            // whether has more seal instance
            recv_ctx_stream = recv<bool>();

            _plain_modulus.emplace_back(parms.plain_modulus().value());
        }
    }

    omp_set_num_threads(_num_thread);

    #ifdef VERBOSE_MODE
    LOG(INFO) << "total plain bits : " << _total_plain_bit;
    LOG(INFO) << "The number of threads for openMP is set to " << _num_thread;
    #endif
}

template<typename T, size_t N>
std::array<T, 3> HETriplet<T, N>::get_triplet() {
    auto &queue = _triplet_buffer;

    if (queue.empty()) {
        fill_triplet_buffer(queue);
    }

    auto ret = queue.front();
    queue.pop();

    return ret;
}

template<typename T, size_t N>
template<typename U>
void HETriplet<T, N>::get_triplet(TensorAdapter<U>* ret) {
  size_t num_trip = ret->numel() / 3;

  for (int i = 0; i < num_trip; ++i) {
    auto triplet = get_triplet();
    auto ret_ptr = ret->data() + i;
    *ret_ptr = triplet[0];
    *(ret_ptr + num_trip) = triplet[1];
    *(ret_ptr + 2 * num_trip) = triplet[2];
  }
}

template<typename T, size_t N>
std::array<T, 5> HETriplet<T, N>::get_penta_triplet() {
    auto &queue = _penta_triplet_buffer;

    if (queue.empty()) {
        fill_penta_triplet_buffer(queue);
    }

    auto ret = queue.front();
    queue.pop();

    return ret;
}

template<typename T, size_t N>
template<typename U>
void HETriplet<T, N>::get_penta_triplet(TensorAdapter<U>* ret) {
  size_t num_trip = ret->numel() / 5;

  for (int i = 0; i < num_trip; ++i) {
    auto triplet = get_penta_triplet();
    auto ret_ptr = ret->data() + i;
    *ret_ptr = triplet[0];
    *(ret_ptr + num_trip) = triplet[1];
    *(ret_ptr + 2 * num_trip) = triplet[2];
    *(ret_ptr + 3 * num_trip) = triplet[3];
    *(ret_ptr + 4 * num_trip) = triplet[4];
  }
}

// CRT algorithm
// for detail, see https://oi-wiki.org/math/crt/
template<typename T, size_t N>
void HETriplet<T, N>::recover_crt(
                    const std::vector<std::vector<uint64_t>>& in,
                    const std::vector<uint64_t>& plain_modulus,
                    std::vector<mpz_class>& out,
                    const mpz_class& triplet_modulus) {

    size_t crt_size = plain_modulus.size();

    mpz_class crt_M(1);

    for (int i = 0; i < crt_size; ++i) {
        crt_M *= plain_modulus[i];
    }

    auto invert = [](const mpz_class& m,
                        const uint64_t mod_m) -> mpz_class {
        mpz_class res;
        mpz_class mod_m_mpz(mod_m);
        mpz_invert(res.get_mpz_t(), m.get_mpz_t(), mod_m_mpz.get_mpz_t());
        return res;
    };

    out.resize(in[0].size(), mpz_class(0));

    for (int i = 0; i < crt_size; ++i) {
        mpz_class crt_m = crt_M / plain_modulus[i];

        mpz_class crt_m_invert = invert(crt_m, plain_modulus[i]);

        for (int j = 0; j < out.size(); ++j) {
            out[j] += in[i][j] * crt_m * crt_m_invert;
        }
    }

    for (int j = 0; j < out.size(); ++j) {
        out[j] = out[j] % crt_M;
    }

}

// abtain triplet for Alice: a0, b0, c0; Bob: a1, b1, c1
// that (a0 + a1) * (b0 + b1) = c0 + c1
// in following algorithm c0 = a0 * b0 + a0 * b1 + b0 * a1 + r
//                        c1 = a1 * b1 - r
template<typename T, size_t N>
void HETriplet<T, N>::fill_triplet_buffer(std::queue<std::array<T, 3>> &queue) {
    size_t crt_size = _contexts.size();

    // thread local variables
    std::vector<std::vector<T>> a_vec_tloc(_num_thread);
    std::vector<std::vector<T>> b_vec_tloc(_num_thread);
    std::vector<std::vector<T>> c_vec_tloc(_num_thread);

    std::vector<std::vector<uint64_t>> c_crt;

    for (int t = 0; t < _num_thread; ++t) {

        for (int j = 0; j < _triplet_step; ++j) {
            a_vec_tloc[t].emplace_back(this->rand_val<T>());
            b_vec_tloc[t].emplace_back(this->rand_val<T>());
        }
    }

    if (_party == 0) {

        // alice encrypt a0, b0 and decrypt c0 based CRT
        for (int i = 0; i < crt_size; ++i) {
            auto& context = *(_contexts[i]);
            Encryptor encryptor(*(_contexts[i]), _public_keys[i]);

            BatchEncoder batch_encoder(context);

            Decryptor decryptor(context, _secret_keys[i]);

            std::vector<Ciphertext> a_cipher_tloc(_num_thread);
            std::vector<Ciphertext> b_cipher_tloc(_num_thread);

            #ifdef VERBOSE_MODE
            auto start = system_clock::now();
            #endif

            // encrypt a0, b0
            #pragma omp parallel for schedule(static) num_threads(_num_thread)
            for (int t = 0; t < _num_thread; ++t) {
                std::vector<uint64_t> a_vec_(_triplet_step);
                std::vector<uint64_t> b_vec_(_triplet_step);

                vec_mod<T>(a_vec_tloc[t], a_vec_, _plain_modulus[i]);
                vec_mod<T>(b_vec_tloc[t], b_vec_, _plain_modulus[i]);

                Plaintext a_plain;
                Plaintext b_plain;
                batch_encoder.encode(a_vec_, a_plain);
                batch_encoder.encode(b_vec_, b_plain);

                encryptor.encrypt(a_plain, a_cipher_tloc[t]);
                encryptor.encrypt(b_plain, b_cipher_tloc[t]);
            }

            #ifdef VERBOSE_MODE
            auto end_enc = system_clock::now();
            #endif

            std::stringstream cipher_ss;

            size_t size = 0;

            // reduce thread variables
            for (int t = 0; t < _num_thread; ++t) {
                size += a_cipher_tloc[t].save(cipher_ss);
                size += b_cipher_tloc[t].save(cipher_ss);
            }

            // send encrypted a0, b0
            send_str(cipher_ss.str(), size);

            #ifdef VERBOSE_MODE
            auto end_send = system_clock::now();
            #endif

            // recv encrypted c0
            std::string c_cipher_str;
            recv_str(c_cipher_str);

            #ifdef VERBOSE_MODE
            auto end_recv = system_clock::now();
            #endif

            std::stringstream c_cipher_ss(c_cipher_str);

            std::vector<Ciphertext> c_cipher_tloc(_num_thread);


            for (int t = 0; t < _num_thread; ++t) {
                c_cipher_tloc[t].load(context, c_cipher_ss);

            }

            std::vector<std::vector<uint64_t>> c_vec_tloc_(_num_thread);

            // decrypt c0
            #pragma omp parallel for schedule(static) num_threads(_num_thread)
            for (int t = 0; t < _num_thread; ++t) {

                Plaintext c_plain;

                if ( decryptor.invariant_noise_budget(c_cipher_tloc[t]) == 0) {
                    throw std::runtime_error("noise budget is not enough for decrypt");
                }
                decryptor.decrypt(c_cipher_tloc[t], c_plain);

                batch_encoder.decode(c_plain, c_vec_tloc_[t]);
            }

            #ifdef VERBOSE_MODE
            auto end_dec = system_clock::now();

            LOG_FIRST_N(INFO, 1) << "alice enc time cost (ms): " << duration(end_enc, start) <<
                ", send time: " << duration(end_send, end_enc) <<
                ", recv time: " << duration(end_recv, end_send) <<
                ", dec time: " << duration(end_dec, end_recv) << "\n";
            #endif

            // reduce c_vec_tloc
            std::vector<uint64_t> c_vec_(_num_thread * _triplet_step);
            for (int t = 0; t < _num_thread; ++t) {
                std::copy(c_vec_tloc_[t].begin(), c_vec_tloc_[t].end(), c_vec_.begin() + t * _triplet_step);
            }

            c_crt.emplace_back(c_vec_);
        }

        std::vector<mpz_class> c_crt_out;

        // recover c0 from CRT
        recover_crt(c_crt, _plain_modulus, c_crt_out, _triplet_modulus);

        // calc final c0
        for (int t = 0; t < _num_thread; ++t) {
            for (int i = 0; i < _triplet_step; ++i) {
                // fixedpoint triplet with N decimal bit
                mpz_class c_rshift = c_crt_out[t * _triplet_step + i] >> N;

                c_vec_tloc[t].emplace_back(fixed_mult<T, N>(a_vec_tloc[t][i], b_vec_tloc[t][i])
                                           + (T) c_rshift.get_ui());
            }

        }

    } else {

        // thread local variable
        std::vector<std::vector<mpz_class>> r_vec_tloc(_num_thread);

        // calc final c1
        for(int t = 0; t < _num_thread; ++t) {
            for (int i = 0; i < _triplet_step; ++i) {
                r_vec_tloc[t].emplace_back(_prng_gmp.get_z_bits(_total_plain_bit));

                // fixedpoint triplet with N decimal bit
                mpz_class r_rshift = r_vec_tloc[t][i] >> N;

                c_vec_tloc[t].emplace_back(fixed_mult<T, N>(a_vec_tloc[t][i], b_vec_tloc[t][i])
                                                            - (T) r_rshift.get_ui());
            }

        }

        // bob evaluate cipher alice's triplet c0 based CRT
        for (int i = 0; i < crt_size; ++i) {
            auto& context = *(_contexts[i]);

            Encryptor encryptor(*(_contexts[i]), _public_keys[i]);

            BatchEncoder batch_encoder(context);

            Evaluator evaluator(context);

            std::vector<Ciphertext> remote_a_cipher_tloc(_num_thread);
            std::vector<Ciphertext> remote_b_cipher_tloc(_num_thread);

            std::vector<Ciphertext> c_cipher_tloc(_num_thread);

            #ifdef VERBOSE_MODE
            auto start = system_clock::now();
            #endif

            // recv a0, b0 from alice
            std::string cipher_str;
            recv_str(cipher_str);
            std::stringstream cipher_ss(cipher_str);

            #ifdef VERBOSE_MODE
            auto end_recv = system_clock::now();
            #endif

            for (int t = 0; t < _num_thread; ++t) {
                remote_a_cipher_tloc[t].load(context, cipher_ss);
                remote_b_cipher_tloc[t].load(context, cipher_ss);
            }

            // calc encrypted c0 based CRT
            #pragma omp parallel for schedule(static) num_threads(_num_thread)
            for (int t = 0; t < _num_thread; ++t) {

                std::vector<uint64_t> a_vec_(_triplet_step);
                std::vector<uint64_t> b_vec_(_triplet_step);
                std::vector<uint64_t> r_vec_(_triplet_step);

                std::vector<uint64_t> c_vec_(_triplet_step);

                vec_mod<T>(a_vec_tloc[t], a_vec_, _plain_modulus[i]);
                vec_mod<T>(b_vec_tloc[t], b_vec_, _plain_modulus[i]);
                vec_mod(r_vec_tloc[t], r_vec_, _plain_modulus[i]);

                Plaintext a_plain_local;
                Plaintext b_plain_local;

                batch_encoder.encode(a_vec_, a_plain_local);
                batch_encoder.encode(b_vec_, b_plain_local);

                calc_triplet_c(r_vec_, a_plain_local, b_plain_local,
                            remote_a_cipher_tloc[t], remote_b_cipher_tloc[t],
                            evaluator, batch_encoder, encryptor,
                            _relin_keys[i], c_cipher_tloc[t]);
            }

            #ifdef VERBOSE_MODE
            auto end_calc = system_clock::now();
            #endif

            std::stringstream c_cipher_ss;
            size_t size = 0;
            for (int t = 0; t < _num_thread; ++t) {
                auto& c_ = c_cipher_tloc[t];
                size += c_.save(c_cipher_ss);
            }

            // send encrypted c0 to alice
            send_str(c_cipher_ss.str(), size);

            #ifdef VERBOSE_MODE
            auto end_send = system_clock::now();

            LOG_FIRST_N(INFO, 1) << " bob recv time (ms): " << duration(end_recv, start) <<
                ", calc time: " << duration(end_calc, end_recv) <<
                ", send time: " << duration(end_send, end_calc) << "\n";
            #endif
        }
    }

    for (int t = 0; t < _num_thread; ++t) {
        for (int i = 0; i < _triplet_step; ++i) {
            queue.emplace(std::array<T, 3>{
                    a_vec_tloc[t][i], b_vec_tloc[t][i], c_vec_tloc[t][i]});
        }
    }
}

// abtain penta triplet for Alice: a0, alpha0, b0, c0, c1_alpha;
// Bob: a1, alpha1, b1, c1, c1_alpha
// that (a0 + a1) * (b0 + b1) = c0 + c1
//       (alpha0 + alpha1) * (b0 + b1) = (c0_alpha + c1_alpha)
// in following algorithm c0 = a0 * b0 + a0 * b1 + b0 * a1 + r
//                        c1 = a1 * b1 - r
//                        c0_alpha = alpha0 * b0 + alpha0 * b1 + b0 * alpha1 + r'
//                        c1_alpha = alpha1 * b1 - r'
template<typename T, size_t N>
void HETriplet<T, N>::fill_penta_triplet_buffer(std::queue<std::array<T, 5>> &queue) {
    size_t crt_size = _contexts.size();

    // thread local variables
    std::vector<std::vector<T>> a_vec_tloc(_num_thread);
    std::vector<std::vector<T>> alpha_vec_tloc(_num_thread);
    std::vector<std::vector<T>> b_vec_tloc(_num_thread);
    std::vector<std::vector<T>> c_vec_tloc(_num_thread);
    std::vector<std::vector<T>> c_alpha_vec_tloc(_num_thread);

    std::vector<std::vector<uint64_t>> c_crt;
    std::vector<std::vector<uint64_t>> c_alpha_crt;

    for (int t = 0; t < _num_thread; ++t) {

        for (int j = 0; j < _triplet_step; ++j) {
            a_vec_tloc[t].emplace_back(this->rand_val<T>());
            alpha_vec_tloc[t].emplace_back(this->rand_val<T>());
            b_vec_tloc[t].emplace_back(this->rand_val<T>());
        }
    }

    if (_party == 0) {

        // alice encrypt a0, alpha0, b0 and decrypt c0, c_alpha0 based CRT
        for (int i = 0; i < crt_size; ++i) {
            auto& context = *(_contexts[i]);
            Encryptor encryptor(*(_contexts[i]), _public_keys[i]);

            BatchEncoder batch_encoder(context);

            Decryptor decryptor(context, _secret_keys[i]);

            std::vector<Ciphertext> a_cipher_tloc(_num_thread);
            std::vector<Ciphertext> alpha_cipher_tloc(_num_thread);
            std::vector<Ciphertext> b_cipher_tloc(_num_thread);

            #ifdef VERBOSE_MODE
            auto start = system_clock::now();
            #endif

            // encrypt a0, alpha0, b0
            #pragma omp parallel for schedule(static) num_threads(_num_thread)
            for (int t = 0; t < _num_thread; ++t) {
                std::vector<uint64_t> a_vec_(_triplet_step);
                std::vector<uint64_t> alpha_vec_(_triplet_step);
                std::vector<uint64_t> b_vec_(_triplet_step);

                vec_mod<T>(a_vec_tloc[t], a_vec_, _plain_modulus[i]);
                vec_mod<T>(alpha_vec_tloc[t], alpha_vec_, _plain_modulus[i]);
                vec_mod<T>(b_vec_tloc[t], b_vec_, _plain_modulus[i]);

                Plaintext a_plain;
                Plaintext alpha_plain;
                Plaintext b_plain;
                batch_encoder.encode(a_vec_, a_plain);
                batch_encoder.encode(alpha_vec_, alpha_plain);
                batch_encoder.encode(b_vec_, b_plain);

                encryptor.encrypt(a_plain, a_cipher_tloc[t]);
                encryptor.encrypt(alpha_plain, alpha_cipher_tloc[t]);
                encryptor.encrypt(b_plain, b_cipher_tloc[t]);
            }

            #ifdef VERBOSE_MODE
            auto end_enc = system_clock::now();
            #endif

            std::stringstream cipher_ss;

            size_t size = 0;

            // reduce thread variables
            for (int t = 0; t < _num_thread; ++t) {
                size += a_cipher_tloc[t].save(cipher_ss);
                size += alpha_cipher_tloc[t].save(cipher_ss);
                size += b_cipher_tloc[t].save(cipher_ss);
            }

            // send encrypted a0, alpha0, b0
            send_str(cipher_ss.str(), size);

            #ifdef VERBOSE_MODE
            auto end_send = system_clock::now();
            #endif

            // recv encrypted c0, c_alpha0
            std::string c_cipher_str;
            recv_str(c_cipher_str);

            #ifdef VERBOSE_MODE
            auto end_recv = system_clock::now();
            #endif

            std::stringstream c_cipher_ss(c_cipher_str);

            std::vector<Ciphertext> c_cipher_tloc(_num_thread);
            std::vector<Ciphertext> c_alpha_cipher_tloc(_num_thread);


            for (int t = 0; t < _num_thread; ++t) {
                c_cipher_tloc[t].load(context, c_cipher_ss);
                c_alpha_cipher_tloc[t].load(context, c_cipher_ss);
            }

            std::vector<std::vector<uint64_t>> c_vec_tloc_(_num_thread);
            std::vector<std::vector<uint64_t>> c_alpha_vec_tloc_(_num_thread);

            // decrypt c0, c_alpha0
            #pragma omp parallel for schedule(static) num_threads(_num_thread)
            for (int t = 0; t < _num_thread; ++t) {

                Plaintext c_plain;
                Plaintext c_alpha_plain;

                if ((decryptor.invariant_noise_budget(c_cipher_tloc[t]) == 0) ||
                    (decryptor.invariant_noise_budget(c_alpha_cipher_tloc[t]) == 0)) {
                    throw std::runtime_error("noise budget is not enough for decrypt");
                }
                decryptor.decrypt(c_cipher_tloc[t], c_plain);
                decryptor.decrypt(c_alpha_cipher_tloc[t], c_alpha_plain);

                batch_encoder.decode(c_plain, c_vec_tloc_[t]);
                batch_encoder.decode(c_alpha_plain, c_alpha_vec_tloc_[t]);
            }

            #ifdef VERBOSE_MODE
            auto end_dec = system_clock::now();

            LOG_FIRST_N(INFO, 1) << "alice enc time cost (ms): " << duration(end_enc, start) <<
                ", send time: " << duration(end_send, end_enc) <<
                ", recv time: " << duration(end_recv, end_send) <<
                ", dec time: " << duration(end_dec, end_recv) << "\n";
            #endif

            // reduce c_vec_tloc, c_alpha_vec_tloc
            std::vector<uint64_t> c_vec_(_num_thread * _triplet_step);
            std::vector<uint64_t> c_alpha_vec_(_num_thread * _triplet_step);
            for (int t = 0; t < _num_thread; ++t) {
                std::copy(c_vec_tloc_[t].begin(), c_vec_tloc_[t].end(), c_vec_.begin() + t * _triplet_step);
                std::copy(c_alpha_vec_tloc_[t].begin(), c_alpha_vec_tloc_[t].end(), c_alpha_vec_.begin() + t * _triplet_step);
            }

            c_crt.emplace_back(c_vec_);
            c_alpha_crt.emplace_back(c_alpha_vec_);
        }

        std::vector<mpz_class> c_crt_out;
        std::vector<mpz_class> c_alpha_crt_out;

        // recover c0 c_alpha0 from CRT
        recover_crt(c_crt, _plain_modulus, c_crt_out, _triplet_modulus);
        recover_crt(c_alpha_crt, _plain_modulus, c_alpha_crt_out, _triplet_modulus);

        // calc final c0, c_alpha0
        for (int t = 0; t < _num_thread; ++t) {
            for (int i = 0; i < _triplet_step; ++i) {
                // fixedpoint triplet with N decimal bit
                mpz_class c_rshift = c_crt_out[t * _triplet_step + i] >> N;
                mpz_class c_alpha_rshift = c_alpha_crt_out[t * _triplet_step + i] >> N;

                c_vec_tloc[t].emplace_back(fixed_mult<T, N>(a_vec_tloc[t][i], b_vec_tloc[t][i])
                                           + (T) c_rshift.get_ui());
                c_alpha_vec_tloc[t].emplace_back(fixed_mult<T, N>(alpha_vec_tloc[t][i], b_vec_tloc[t][i])
                                           + (T) c_alpha_rshift.get_ui());
            }

        }

    } else {

        // thread local variable
        std::vector<std::vector<mpz_class>> r_vec_tloc(_num_thread);
        std::vector<std::vector<mpz_class>> r_alpha_vec_tloc(_num_thread);

        // calc final c1 c_alpha1
        for(int t = 0; t < _num_thread; ++t) {
            for (int i = 0; i < _triplet_step; ++i) {
                r_vec_tloc[t].emplace_back(_prng_gmp.get_z_bits(_total_plain_bit));
                r_alpha_vec_tloc[t].emplace_back(_prng_gmp.get_z_bits(_total_plain_bit));

                // fixedpoint triplet with N decimal bit
                mpz_class r_rshift = r_vec_tloc[t][i] >> N;
                mpz_class r_alpha_rshift = r_alpha_vec_tloc[t][i] >> N;

                c_vec_tloc[t].emplace_back(fixed_mult<T, N>(a_vec_tloc[t][i], b_vec_tloc[t][i])
                                                            - (T) r_rshift.get_ui());
                c_alpha_vec_tloc[t].emplace_back(fixed_mult<T, N>(alpha_vec_tloc[t][i], b_vec_tloc[t][i])
                                                            - (T) r_alpha_rshift.get_ui());
            }

        }

        // bob evaluate cipher alice's triplet c0 c_alpha0 based CRT
        for (int i = 0; i < crt_size; ++i) {
            auto& context = *(_contexts[i]);

            Encryptor encryptor(*(_contexts[i]), _public_keys[i]);

            BatchEncoder batch_encoder(context);

            Evaluator evaluator(context);

            std::vector<Ciphertext> remote_a_cipher_tloc(_num_thread);
            std::vector<Ciphertext> remote_alpha_cipher_tloc(_num_thread);
            std::vector<Ciphertext> remote_b_cipher_tloc(_num_thread);

            std::vector<Ciphertext> c_cipher_tloc(_num_thread);
            std::vector<Ciphertext> c_alpha_cipher_tloc(_num_thread);

            #ifdef VERBOSE_MODE
            auto start = system_clock::now();
            #endif

            // recv a0, a_alpha0, b0 from alice
            std::string cipher_str;
            recv_str(cipher_str);
            std::stringstream cipher_ss(cipher_str);

            #ifdef VERBOSE_MODE
            auto end_recv = system_clock::now();
            #endif

            for (int t = 0; t < _num_thread; ++t) {
                remote_a_cipher_tloc[t].load(context, cipher_ss);
                remote_alpha_cipher_tloc[t].load(context, cipher_ss);
                remote_b_cipher_tloc[t].load(context, cipher_ss);
            }

            // calc encrypted c0 c_alpha0 based CRT
            #pragma omp parallel for schedule(static) num_threads(_num_thread)
            for (int t = 0; t < _num_thread; ++t) {

                std::vector<uint64_t> a_vec_(_triplet_step);
                std::vector<uint64_t> alpha_vec_(_triplet_step);
                std::vector<uint64_t> b_vec_(_triplet_step);
                std::vector<uint64_t> r_vec_(_triplet_step);
                std::vector<uint64_t> r_alpha_vec_(_triplet_step);

                std::vector<uint64_t> c_vec_(_triplet_step);
                std::vector<uint64_t> c_alpha_vec_(_triplet_step);

                vec_mod<T>(a_vec_tloc[t], a_vec_, _plain_modulus[i]);
                vec_mod<T>(alpha_vec_tloc[t], alpha_vec_, _plain_modulus[i]);
                vec_mod<T>(b_vec_tloc[t], b_vec_, _plain_modulus[i]);
                vec_mod(r_vec_tloc[t], r_vec_, _plain_modulus[i]);
                vec_mod(r_alpha_vec_tloc[t], r_alpha_vec_, _plain_modulus[i]);

                Plaintext a_plain_local;
                Plaintext alpha_plain_local;
                Plaintext b_plain_local;

                batch_encoder.encode(a_vec_, a_plain_local);
                batch_encoder.encode(alpha_vec_, alpha_plain_local);
                batch_encoder.encode(b_vec_, b_plain_local);

                calc_penta_triplet_c(r_vec_, r_alpha_vec_, a_plain_local,
                                     alpha_plain_local, b_plain_local,
                                    remote_a_cipher_tloc[t], remote_alpha_cipher_tloc[t],
                                    remote_b_cipher_tloc[t],
                                    evaluator, batch_encoder, encryptor,
                                    _relin_keys[i], c_cipher_tloc[t], c_alpha_cipher_tloc[t]);
            }

            #ifdef VERBOSE_MODE
            auto end_calc = system_clock::now();
            #endif

            std::stringstream c_cipher_ss;
            size_t size = 0;
            for (int t = 0; t < _num_thread; ++t) {
                auto& c_ = c_cipher_tloc[t];
                auto& c_alpha_ = c_alpha_cipher_tloc[t];

                size += c_.save(c_cipher_ss);
                size += c_alpha_.save(c_cipher_ss);
            }

            // send encrypted c0 c_alpha0 to alice
            send_str(c_cipher_ss.str(), size);

            #ifdef VERBOSE_MODE
            auto end_send = system_clock::now();

            LOG_FIRST_N(INFO, 1) << " bob recv time (ms): " << duration(end_recv, start) <<
                ", calc time: " << duration(end_calc, end_recv) <<
                ", send time: " << duration(end_send, end_calc) << "\n";
            #endif
        }
    }

    for (int t = 0; t < _num_thread; ++t) {
        for (int i = 0; i < _triplet_step; ++i) {
            queue.emplace(std::array<T, 5>{
                    a_vec_tloc[t][i], alpha_vec_tloc[t][i],
                    b_vec_tloc[t][i],
                    c_vec_tloc[t][i], c_alpha_vec_tloc[t][i]});
        }
    }
}

} // namespace privc
