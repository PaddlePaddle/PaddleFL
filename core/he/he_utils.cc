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

#include <atomic>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "paillier.h"

namespace py = pybind11;

namespace feature {

PYBIND11_MODULE(he_utils, m) {

    m.doc() = "mpc feature engineering he utils ";

    py::class_<mpz_class>(m, "mpz_class")
        .def(py::init<>())
        .def(py::init<const mpz_class &>())
        .def(py::init<const char *>())
        .def(py::init<const std::string &>())
        .def(py::init<unsigned long int>())
        .def(py::init<signed long int>())
        .def(py::init<float>())
        .def(py::init<double>());
    
    py::class_<Paillier>(m, "Paillier")
        .def(py::init<>())
        .def(py::init<const Paillier &>())
        .def("keygen", &Paillier::keygen)
        .def("encrypt", &Paillier::encrypt)
        .def("decrypt", &Paillier::decrypt)
        .def("homm_add", &Paillier::homm_add)
        .def("homm_minus", &Paillier::homm_minus)
        .def("homm_mult", &Paillier::homm_mult)
        .def("export_pk", &Paillier::export_pk)
        .def("export_pk_bytes", [](Paillier &paillier){
            std::string pk = paillier.export_pk();
            return py::bytes(pk);
        })
        .def("import_pk", &Paillier::import_pk)
        .def("export_sk", &Paillier::export_sk)
        .def("export_sk_bytes", [](Paillier &paillier){
            std::string sk = paillier.export_sk();
            return py::bytes(sk);
        })
        .def("import_sk", &Paillier::import_sk)
        .def("encrypt_int64_t", &Paillier::encrypt_int64_t)
        .def("batch_encrypt_int64_t", [](Paillier &paillier, const std::vector<int64_t> & plains){
            std::vector<mpz_class> ciphers;
            ciphers.reserve(plains.size());
            for (auto plain : plains) {
                ciphers.emplace_back(paillier.encrypt_int64_t(plain));
            }
            return ciphers;
        })
        .def("decrypt_int64_t", &Paillier::decrypt_int64_t)
        .def("batch_decrypt_int64_t", [](Paillier &paillier, const std::vector<mpz_class> & ciphers){
            std::vector<int64_t> plains;
            plains.reserve(ciphers.size());
            for (auto cipher : ciphers) {
                plains.emplace_back(paillier.decrypt_int64_t(cipher));
            }
            return plains;
        })
        .def("encode_cipher", &Paillier::encode_cipher)
        .def("encode_cipher_bytes", [](Paillier &paillier, mpz_class &cipher){
            std::string cipher_ = paillier.encode_cipher(cipher);
            return py::bytes(cipher_);
        })
        .def("batch_encode_cipher_bytes", [](Paillier &paillier, std::vector<mpz_class> &ciphers){
            std::vector<py::bytes> encode_ciphers;
            encode_ciphers.reserve(ciphers.size());
            for (auto cipher : ciphers) {
                encode_ciphers.emplace_back(py::bytes(paillier.encode_cipher(cipher)));
            }
            return encode_ciphers;
        })
        .def_static("decode", &Paillier::decode)
        .def("batch_decode", [](Paillier &paillier, std::vector<std::string> &encode_ciphers){
            std::vector<mpz_class> decode_ciphers;
            decode_ciphers.reserve(encode_ciphers.size());
            for (auto cipher : encode_ciphers) {
                decode_ciphers.emplace_back(paillier.decode(cipher));
            }
            return decode_ciphers;
        })
        .def("byte_len", &Paillier::byte_len)
        .def("get_random_bits", &Paillier::get_random_bits)
        .def("n", &Paillier::n);

    m.def("cal_pos_ratio", [](const mpz_class & a, const mpz_class &b) {
        mpf_class result = mpf_class(a) / (mpf_class(a) + mpf_class(b));
        return result.get_d();
    });

    m.def("cal_woe", [](const mpz_class & a, const mpz_class &b, 
                        const int64_t &total_pos, const int64_t &total_neg){
        if (a == 0) {
            return -20.0;
        } else if (b == 0) {
            return 20.0;
        } else {
            mpf_class result = mpf_class(a) * mpf_class(total_neg) 
                                / (mpf_class(b) * mpf_class(total_pos));
            double woe = log(result.get_d());
            if (fabs(woe) < 1.0e-8) {
                woe = 0.0;
            }
            return woe;
        }
    });

    // choose scaling_factor 64-bit to meet the float precision requirement
    // suppose millions data  20(data size) + 23(float fraction) < 64
    m.def("cal_blind_iv", [](const mpz_class & a, const mpz_class &b, 
                       const int64_t &total_pos, const int64_t &total_neg){
        double woe = 0.0;
        if (a == 0) {
            woe = -20.0;
        } else if (b == 0) {
            woe = 20.0;
        } else {
            mpf_class temp = mpf_class(a) * mpf_class(total_neg) 
                            / (mpf_class(b) * mpf_class(total_pos));
            woe = log(temp.get_d());
            if (fabs(woe) < 1.0e-8) {
                woe = 0.0;
            }
        } 
        mpz_class scaling_factor = mpz_class(1) << 64;
        mpz_class woe_f = mpz_class(mpf_class(woe) * scaling_factor);
        mpz_class weight = a * mpz_class(scaling_factor / total_pos)
                           - b * mpz_class(scaling_factor / total_neg);
        mpz_class blind_iv = mpz_class(woe_f * weight);
        return blind_iv;
    });

    m.def("mod_inv", [](const mpz_class & a, const mpz_class &n) {
        mpz_class inv;
        mpz_invert(inv.get_mpz_t(), a.get_mpz_t(), n.get_mpz_t());
        return inv;
    });

    m.def("cal_unblind_iv",[](const mpz_class &a){
        mpf_class result = mpf_class(a) / (mpf_class(1) << 128);
        return result.get_d();
    });

    m.def("cal_blind_ks", [](const mpz_class & a, const mpz_class &b, 
                       const int64_t &total_pos, const int64_t &total_neg){
 
        mpz_class scaling_factor = mpz_class(1) << 64;
        mpz_class blind_ks = abs(a * mpz_class(scaling_factor / total_pos)
                             - b * mpz_class(scaling_factor / total_neg));
        return blind_ks;
    });

    m.def("cal_max_ks", [](const std::vector<mpz_class> & ks) {
        mpz_class max_ks(-1);
        for (auto _ks : ks) {
            max_ks = max_ks > _ks ? max_ks : _ks;
        }
        mpf_class result = mpf_class(max_ks) / (mpf_class(1) << 64);
        return result.get_d();
    });

    m.def("cal_blind_auc", [](const std::vector<mpz_class> &pos, 
                              const std::vector<mpz_class> &neg){
        mpz_class auc(0);
        for(uint32_t i = 0; i < pos.size(); ++i) {
            auc += pos[i] * neg[i];
        }
        return auc;
    });

    m.def("cal_unblind_auc", [](const std::vector<mpz_class> &blind_auc,
                                const int64_t &total_pos,
                                const int64_t &total_neg){
        std::vector<double> auc;
        auc.reserve(blind_auc.size());
        for(uint32_t i = 0; i < blind_auc.size(); ++i) {
            mpf_class temp = mpf_class(blind_auc[i]) / mpf_class(2 * total_pos * total_neg);
            auc.emplace_back(temp.get_d());
        }
        return auc;
    });
}
}


