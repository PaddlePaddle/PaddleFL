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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "he_triplet.h"
#include "core/paddlefl_mpc/mpc_protocol/network/mesh_network.h"
#include "core/common/crypto.h"

namespace privc {

class HETripletTest : public ::testing::Test {
public:
    int _port = 33455;
    const char * _addr = "127.0.0.1";

    std::thread _t[2];

    static std::shared_ptr<gloo::rendezvous::HashStore> _store;
    PseudorandomNumberGenerator _prng;

    static std::shared_ptr<paddle::mpc::MeshNetwork> _io[2];

    size_t _party[2]{ 0, 1 };

    HETripletTest() : _prng(common::ZeroBlock) {}

    ~HETripletTest() {}

    static inline std::shared_ptr<paddle::mpc::MeshNetwork> gen_network(size_t idx) {
        return std::make_shared<paddle::mpc::MeshNetwork>(idx,
                                                          "127.0.0.1",
                                                          2,
                                                          "test_prefix_privc",
                                                          _store);
    }

    static void gen_network() {

        std::thread t[2];

        for (size_t i = 0; i < 2; ++i) {
            t[i] = std::thread([i]() {
                                _io[i] = std::make_shared<paddle::mpc::MeshNetwork>(
                                    i, "127.0.0.1", 2, "test_prefix_privc", _store);
                                _io[i]->init();
                                });
        }
        for (auto& ti : t) {
            ti.join();
        }
    }

    static void SetUpTestCase() {
        _store = std::make_shared<gloo::rendezvous::HashStore>();
        gen_network();
    }

};

std::shared_ptr<gloo::rendezvous::HashStore> HETripletTest::_store;
std::shared_ptr<paddle::mpc::MeshNetwork> HETripletTest::_io[2];

template<typename T, size_t N>
inline void verify_triplet(std::array<T, 3> t0,
                    std::array<T, 3> t1,
                    double abs_error = 5000) {
    uint64_t c_expect = fixed_mult<T, N>(t0[0], t0[1])
        + fixed_mult<T, N>(t0[0], t1[1])
        + fixed_mult<T, N>(t1[0], t0[1])
        + fixed_mult<T, N>(t1[0], t1[1]);

    uint64_t c_actual = t0[2] + t1[2];

    EXPECT_NEAR(c_expect, c_actual, abs_error);
}

template<typename T, size_t N>
inline void verify_penta_triplet(std::array<T, 5> t0,
                    std::array<T, 5> t1,
                    double abs_error = 5000) {
    uint64_t c_expect = fixed_mult<T, N>(t0[0], t0[2])
        + fixed_mult<T, N>(t0[0], t1[2])
        + fixed_mult<T, N>(t1[0], t0[2])
        + fixed_mult<T, N>(t1[0], t1[2]);

    uint64_t c_alpha_expect = fixed_mult<T, N>(t0[1], t0[2])
        + fixed_mult<T, N>(t0[1], t1[2])
        + fixed_mult<T, N>(t1[1], t0[2])
        + fixed_mult<T, N>(t1[1], t1[2]);

    uint64_t c_actual = t0[3] + t1[3];
    uint64_t c_alpha_actual = t0[4] + t1[4];

    EXPECT_NEAR(c_expect, c_actual, abs_error);
    EXPECT_NEAR(c_alpha_expect, c_alpha_actual, abs_error);
}


TEST_F(HETripletTest, recover_crt_test) {

    std::vector<std::vector<uint64_t>> in{{2}, {10}};

    std::vector<uint64_t> modulus{7,11};
    std::vector<mpz_class> out(1);

    size_t bit_len = sizeof(uint64_t) * 8;
    mpz_class triplet_modulus(1);
    triplet_modulus = triplet_modulus << bit_len;

    mpz_class crt_M = mpz_class(7) * 11;

    out[0] = out[0] % crt_M;

    HETriplet<uint64_t, 32>::recover_crt(in, modulus, out, triplet_modulus);

    EXPECT_EQ(out[0].get_ui(), 65);

}

TEST_F(HETripletTest, integer64_test) {

    std::array<uint64_t, 3> triplet[2];

    _t[0] = std::thread([&triplet, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 0> tripletor(_party[0],
                                         io.get(),
                                         _prng);
        tripletor.init();
        triplet[0] = tripletor.get_triplet();
    });

    _t[1] = std::thread([&triplet, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 0> tripletor(_party[1],
                                         io.get(),
                                         _prng);
        tripletor.init();
        triplet[1] = tripletor.get_triplet();
    });

    for (auto& i : _t) {
        i.join();
    }

    verify_triplet<uint64_t, 0>(triplet[0], triplet[1]);
}

TEST_F(HETripletTest, fixed64_test) {

    std::array<uint64_t, 3> triplet[2];


    size_t poly_modulus_degree = 8192;
    size_t max_seal_plain_bit = 50;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 32> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[0] = tripletor.get_triplet();
    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 32> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[1] = tripletor.get_triplet();
    });
    for (auto& i : _t) {
        i.join();
    }

    verify_triplet<uint64_t, 32>(triplet[0], triplet[1]);
}

TEST_F(HETripletTest, fixed64_test_small_degree) {

    std::array<uint64_t, 3> triplet[2];


    size_t poly_modulus_degree = 4096;
    size_t max_seal_plain_bit = 25;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 32> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[0] = tripletor.get_triplet();
    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 32> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[1] = tripletor.get_triplet();
    });
    for (auto& i : _t) {
        i.join();
    }

    verify_triplet<uint64_t, 32>(triplet[0], triplet[1]);
}

TEST_F(HETripletTest, fixed64_test_batch) {

    std::vector<std::array<uint64_t, 3>> triplet[2];


    size_t poly_modulus_degree = 8192;
    size_t max_seal_plain_bit = 50;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 32> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < poly_modulus_degree; ++i) {
            triplet[0].emplace_back(tripletor.get_triplet());
        }

    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 32> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < poly_modulus_degree; ++i) {
            triplet[1].emplace_back(tripletor.get_triplet());
        }
    });
    for (auto& i : _t) {
        i.join();
    }

    for (int i = 0; i < poly_modulus_degree; ++i) {
        verify_triplet<uint64_t, 32>(triplet[0][i], triplet[1][i]);
    }

}

TEST_F(HETripletTest, fixed64_test_multi_batch) {

    std::vector<std::array<uint64_t, 3>> triplet[2];


    size_t poly_modulus_degree = 8192;
    size_t max_seal_plain_bit = 50;
    size_t batch_size = poly_modulus_degree * 2 + 1;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 32> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[0].emplace_back(tripletor.get_triplet());
        }

    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 32> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[1].emplace_back(tripletor.get_triplet());
        }
    });
    for (auto& i : _t) {
        i.join();
    }

    for (int i = 0; i < batch_size; ++i) {
        verify_triplet<uint64_t, 32>(triplet[0][i], triplet[1][i]);
    }

}

TEST_F(HETripletTest, fixed32_test) {

    std::array<uint32_t, 3> triplet[2];

    _t[0] = std::thread([&triplet, this](){
        auto io = _io[0];
        HETriplet<uint32_t, 16> tripletor(_party[0],
                                         io.get(),
                                         _prng);
        tripletor.init();
        triplet[0] = tripletor.get_triplet();
    });

    _t[1] = std::thread([&triplet, this](){
        auto io = _io[1];
        HETriplet<uint32_t, 16> tripletor(_party[1],
                                         io.get(),
                                         _prng);
        tripletor.init();
        triplet[1] = tripletor.get_triplet();
    });

    for (auto& i : _t) {
        i.join();
    }

    verify_triplet<uint32_t, 16>(triplet[0], triplet[1]);
}

TEST_F(HETripletTest, fixed32_test_large_deg) {

    std::array<uint32_t, 3> triplet[2];

    size_t poly_modulus_degree = 16384;
    size_t max_seal_plain_bit = 60;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint32_t, 16> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[0] = tripletor.get_triplet();
    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint32_t, 16> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[1] = tripletor.get_triplet();
    });

    for (auto& i : _t) {
        i.join();
    }

    verify_triplet<uint32_t, 16>(triplet[0], triplet[1]);
}


TEST_F(HETripletTest, fixed32_test_small_deg) {

    std::array<uint32_t, 3> triplet[2];

    size_t poly_modulus_degree = 4096;
    size_t max_seal_plain_bit = 25;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint32_t, 16> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[0] = tripletor.get_triplet();
    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint32_t, 16> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[1] = tripletor.get_triplet();
    });

    for (auto& i : _t) {
        i.join();
    }

    verify_triplet<uint32_t, 16>(triplet[0], triplet[1]);
}

TEST_F(HETripletTest, fixed_u64_test) {

    std::array<uint64_t, 3> triplet[2];

    size_t poly_modulus_degree = 8192;
    size_t max_seal_plain_bit = 60;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 32> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[0] = tripletor.get_triplet();
    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 32> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[1] = tripletor.get_triplet();
    });

    for (auto& i : _t) {
        i.join();
    }

    verify_triplet<uint64_t, 32>(triplet[0], triplet[1]);
}

TEST_F(HETripletTest, penta_fixed64_test_multi_batch) {

    std::vector<std::array<uint64_t, 5>> triplet[2];


    size_t poly_modulus_degree = 8192;
    size_t max_seal_plain_bit = 60;
    size_t batch_size = 524288;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 32> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[0].emplace_back(tripletor.get_penta_triplet());
        }

    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 32> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[1].emplace_back(tripletor.get_penta_triplet());
        }
    });
    for (auto& i : _t) {
        i.join();
    }

    for (int i = 0; i < batch_size; ++i) {
        verify_penta_triplet<uint64_t, 32>(triplet[0][i], triplet[1][i]);
    }

}

TEST_F(HETripletTest, penta_fixed64_test_small_deg) {

    std::vector<std::array<uint64_t, 5>> triplet[2];


    size_t poly_modulus_degree = 4096;
    size_t max_seal_plain_bit = 25;
    size_t batch_size = 524288;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 32> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[0].emplace_back(tripletor.get_penta_triplet());
        }

    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 32> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[1].emplace_back(tripletor.get_penta_triplet());
        }
    });
    for (auto& i : _t) {
        i.join();
    }

    for (int i = 0; i < batch_size; ++i) {
        verify_penta_triplet<uint64_t, 32>(triplet[0][i], triplet[1][i]);
    }

}

TEST_F(HETripletTest, penta_fixed64_test_high_deg) {

    std::vector<std::array<uint64_t, 5>> triplet[2];


    size_t poly_modulus_degree = 8192 * 2;
    size_t max_seal_plain_bit = 60;
    size_t batch_size = 524288;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 32> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[0].emplace_back(tripletor.get_penta_triplet());
        }

    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 32> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[1].emplace_back(tripletor.get_penta_triplet());
        }
    });
    for (auto& i : _t) {
        i.join();
    }

    for (int i = 0; i < batch_size; ++i) {
        verify_penta_triplet<uint64_t, 32>(triplet[0][i], triplet[1][i]);
    }

}


TEST_F(HETripletTest, penta_fixed64_test) {

    std::array<uint64_t, 5> triplet[2];


    size_t poly_modulus_degree = 8192;
    size_t max_seal_plain_bit = 40;
    size_t batch_size = 524288;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint64_t, 32> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[0] = tripletor.get_penta_triplet();


    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint64_t, 32> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        triplet[1] = tripletor.get_penta_triplet();
    });
    for (auto& i : _t) {
        i.join();
    }

    verify_penta_triplet<uint64_t, 32>(triplet[0], triplet[1]);

}

TEST_F(HETripletTest, penta_fixed32_test) {

    std::vector<std::array<uint32_t, 5>> triplet[2];


    size_t poly_modulus_degree = 8192;
    size_t max_seal_plain_bit = 60;
    size_t batch_size = 524288;

    _t[0] = std::thread([&, this](){
        auto io = _io[0];
        HETriplet<uint32_t, 16> tripletor(_party[0],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[0].emplace_back(tripletor.get_penta_triplet());
        }

    });

    _t[1] = std::thread([&, this](){
        auto io = _io[1];
        HETriplet<uint32_t, 16> tripletor(_party[1],
                                         io.get(),
                                         _prng,
                                          poly_modulus_degree,
                                          max_seal_plain_bit);
        tripletor.init();
        for (int i = 0; i < batch_size; ++i) {
            triplet[1].emplace_back(tripletor.get_penta_triplet());
        }
    });
    for (auto& i : _t) {
        i.join();
    }

    for (int i = 0; i < batch_size; ++i) {
        verify_penta_triplet<uint32_t, 16>(triplet[0][i], triplet[1][i]);
    }

}

} // namespace privc
