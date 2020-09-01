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

#include "crypto.h"

#include <cstring>
#include <string>

#include "gtest/gtest.h"

namespace psi {

TEST(crypto, hash) {
    std::string input = "abc";
    char output[HASH_DIGEST_LEN + 1];

    output[HASH_DIGEST_LEN] = '\0';

    const char *standard_vec =
        "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e"
        "\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";

    hash(input.data(), input.size(), output);

    EXPECT_STREQ(standard_vec, output);

}

TEST(crypto, enc) {
    std::string input = "abc";
    std::string iv = "0123456789ab";
    std::string key = "0123456789abcdef"; // aes_128_gcm, key_len = 128bit

    unsigned int cipher_len = GCM_IV_LEN + GCM_TAG_LEN + input.size();
    auto *output = new unsigned char [cipher_len];

    int enc_ret = encrypt((unsigned char *)input.data(), input.size(),
                          (unsigned char *)key.data(),
                          (unsigned char *)iv.data(), output);

    ASSERT_EQ(cipher_len, (size_t)enc_ret);

    char *plaintext = new char [input.size() + 1];
    plaintext[input.size()] = '\0';
    int dec_ret = decrypt(output, enc_ret, (unsigned char *)key.data(),
                          (unsigned char *)plaintext);

    ASSERT_EQ(input.size(), (size_t)dec_ret);

    EXPECT_STREQ(input.c_str(), plaintext);

    delete output;
    delete plaintext;
}


TEST(crypto, ecdh) {
    ECDH alice;
    ECDH bob;

    auto ga = alice.generate_key();
    auto gb = bob.generate_key();

    auto ka = alice.get_shared_secret(gb);
    auto kb = bob.get_shared_secret(ga);

    ASSERT_EQ(ka.size(), kb.size());
    EXPECT_TRUE(0 == std::memcmp(ka.data(), kb.data(), ka.size()));
}

TEST(crypto, hash_block) {

    block in = ZeroBlock;

    for (size_t i = 0; i < 1e6; ++i) {
        hash_block(in);
    }
}

TEST(crypto, hash_blocks) {

    block in = ZeroBlock;

    for (size_t i = 0; i < 1e6; ++i) {
        hash_blocks({in, in});
    }
}

};
