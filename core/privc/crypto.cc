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

#include <openssl/ecdh.h>
#include <string.h>

#include "glog/logging.h"

namespace psi {

u8 *hash(const void *d, u64 n, void *md) {
    return SHA1(reinterpret_cast<const u8 *>(d), n, reinterpret_cast<u8 *>(md));
}

int encrypt(const unsigned char *plaintext, int plaintext_len,
            const unsigned char *key, const unsigned char *iv,
            unsigned char *ciphertext) {
    EVP_CIPHER_CTX *ctx = NULL;
    int len = 0;
    int aes_ciphertext_len = 0;
    int ret = 0;

    memcpy(ciphertext, iv, GCM_IV_LEN);

    unsigned char *aes_ciphertext = ciphertext + GCM_IV_LEN;
    unsigned char *tag = ciphertext + GCM_IV_LEN + plaintext_len;

    ctx = EVP_CIPHER_CTX_new();
    if (ctx == NULL) {
        LOG(ERROR) << "openssl error";
        return 0;
    }

    ret = EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, key, iv);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        return 0;
    }

    ret = EVP_EncryptUpdate(ctx, NULL, &len, iv, GCM_IV_LEN);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        return 0;
    }

    ret =
        EVP_EncryptUpdate(ctx, aes_ciphertext, &len, plaintext, plaintext_len);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        return 0;
    }
    aes_ciphertext_len = len;

    ret = EVP_EncryptFinal_ex(ctx, aes_ciphertext + len, &len);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        return 0;
    }
    aes_ciphertext_len += len;

    if (aes_ciphertext_len != plaintext_len) {
        LOG(ERROR) << "encrypt error: ciphertext len mismatched";
        return 0;
    }

    ret = EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, GCM_TAG_LEN, tag);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        return 0;
    }

    EVP_CIPHER_CTX_free(ctx);

    return aes_ciphertext_len + GCM_IV_LEN + GCM_TAG_LEN;
}

int decrypt(const unsigned char *ciphertext, int ciphertext_len,
            const unsigned char *key, unsigned char *plaintext) {
    EVP_CIPHER_CTX *ctx = NULL;

    int len = 0;
    int plaintext_len = 0;
    int ret = 0;

    const unsigned char *iv = ciphertext;
    const unsigned char *aes_ciphertext = ciphertext + GCM_IV_LEN;

    int aes_ciphertext_len = ciphertext_len - GCM_IV_LEN - GCM_TAG_LEN;

    unsigned char tag[GCM_TAG_LEN];

    memcpy(tag, ciphertext + ciphertext_len - GCM_TAG_LEN, GCM_TAG_LEN);

    ctx = EVP_CIPHER_CTX_new();
    if (ctx == NULL) {
        LOG(ERROR) << "openssl error";
        return -1;
    }

    ret = EVP_DecryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, key, iv);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        return -1;
    }

    ret = EVP_DecryptUpdate(ctx, NULL, &len, iv, GCM_IV_LEN);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        return -1;
    }

    ret = EVP_DecryptUpdate(ctx, plaintext, &len, aes_ciphertext,
                            aes_ciphertext_len);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        return -1;
    }
    plaintext_len = len;

    if (plaintext_len != ciphertext_len - GCM_IV_LEN - GCM_TAG_LEN) {
        LOG(ERROR) << "openssl error";
        return -1;
    }

    ret = EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, GCM_TAG_LEN, tag);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        return -1;
    }

    ret = EVP_DecryptFinal_ex(ctx, plaintext + len, &len);

    EVP_CIPHER_CTX_free(ctx);

    if (ret > 0) {
        plaintext_len += len;
        return plaintext_len;
    } else {
        return -1;
    }
}

ECDH::ECDH() {
    _error = false;
    int ret = 0;

    _group = EC_GROUP_new_by_curve_name(CURVE_ID);
    if (_group == NULL) {
        LOG(ERROR) << "openssl error";
        _error = true;
        return;
    }

    _key = EC_KEY_new();
    if (_key == NULL) {
        LOG(ERROR) << "openssl error";
        _error = true;
        return;
    }

    ret = EC_KEY_set_group(_key, _group);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        _error = true;
        return;
    }

    _remote_key = EC_POINT_new(_group);
    if (_remote_key == NULL) {
        LOG(ERROR) << "openssl error";
        _error = true;
        return;
    }
}

ECDH::~ECDH() {
    EC_POINT_free(_remote_key);
    EC_KEY_free(_key);
    EC_GROUP_free(_group);
}

std::array<u8, POINT_BUFFER_LEN> ECDH::generate_key() {
    int ret = 0;
    std::array<u8, POINT_BUFFER_LEN> output;

    if (_error) {
        LOG(ERROR) << "internal error";
        return output;
    }

    ret = EC_KEY_generate_key(_key);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        _error = true;
        return output;
    }

    const EC_POINT *key_point = EC_KEY_get0_public_key(_key);
    if (key_point == NULL) {
        LOG(ERROR) << "openssl error";
        _error = true;
        return output;
    }


    ret = EC_POINT_point2oct(_group, key_point, POINT_CONVERSION_COMPRESSED,
                             output.data(), POINT_BUFFER_LEN, NULL);
    if (ret == 0) {
        LOG(ERROR) << "openssl error";
        _error = true;
        return output;
    }

    return output;
}

std::array<u8, POINT_BUFFER_LEN - 1>
ECDH::get_shared_secret(const std::array<u8, POINT_BUFFER_LEN> &remote_key) {
    int ret = 0;
    std::array<u8, POINT_BUFFER_LEN - 1> secret;

    ret = EC_POINT_oct2point(_group, _remote_key, remote_key.data(),
                             remote_key.size(), NULL);
    if (ret != 1) {
        LOG(ERROR) << "openssl error";
        _error = true;
        return secret;
    }


    int secret_len = POINT_BUFFER_LEN - 1;
    // compressed flag not included in secret, see
    // http://www.secg.org/sec1-v2.pdf chapter 2.2.3

    ret = ECDH_compute_key(secret.data(), secret_len, _remote_key, _key, NULL);

    if (ret <= 0) {
        LOG(ERROR) << "openssl error";
        _error = true;
        return secret;
    }

    return secret;
}
} // namespace psi
