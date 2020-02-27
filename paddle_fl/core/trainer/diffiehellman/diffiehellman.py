# coding=utf-8

#
# The MIT License (MIT)
#
# Copyright (c) 2016 Chris von Csefalvay
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
"""
diffiehellmann declares the main key exchange class.
"""

__version__ = '0.13.3'

from hashlib import sha256

from .decorators import requires_private_key
from .exceptions import MalformedPublicKey, RNGError
from .primes import PRIMES
import os

#python3
try:
    from ssl import RAND_bytes
    rng = RAND_bytes
except (AttributeError, ImportError):
    rng = os.urandom


class DiffieHellman:
    """
    Implements the Diffie-Hellman key exchange protocol.

    """

    def __init__(self, group=18, key_length=640):

        self.key_length = max(200, key_length)
        self.generator = PRIMES[group]["generator"]
        self.prime = PRIMES[group]["prime"]

    def load_private_key(self, priv_key_filepath="priv_key.txt"):
        f = open(priv_key_filepath, "r")
        self.private_key = int(f.read())
        # self.private_key = 1236621350910932696206938487330072474688096146032487063733488274339542368951034578546908207981931

    def generate_private_key(self):
        """
        Generates a private key of key_length bits and attaches it to the object as the __private_key variable.

        :return: void
        :rtype: void
        """
        key_length = self.key_length // 8 + 8
        key = 0

        try:
            key = int.from_bytes(rng(key_length), byteorder='big')
        except:
            key = int(hex(rng(key_length)), base=16)

        self.private_key = key

    def verify_public_key(self, other_public_key):
        return self.prime - 1 > other_public_key > 2 and pow(
            other_public_key, (self.prime - 1) // 2, self.prime) == 1

    @requires_private_key
    def generate_public_key(self):
        """
        Generates public key.

        :return: void
        :rtype: void
        """
        self.public_key = pow(self.generator, self.private_key, self.prime)

    @requires_private_key
    def generate_shared_secret(self, other_public_key, echo_return_key=False):
        """
        Generates shared secret from the other party's public key.

        :param other_public_key: Other party's public key
        :type other_public_key: int
        :param echo_return_key: Echo return shared key
        :type bool
        :return: void
        :rtype: void
        """
        if self.verify_public_key(other_public_key) is False:
            raise MalformedPublicKey

        self.shared_secret = pow(other_public_key, self.private_key,
                                 self.prime)
        try:
            #python3
            shared_secret_as_bytes = self.shared_secret.to_bytes(
                self.shared_secret.bit_length() // 8 + 1, byteorder='big')
        except:
            #python2
            length = self.shared_secret.bit_length() // 8 + 1
            shared_secret_as_bytes = ('%%0%dx' % (
                length << 1) % self.shared_secret).decode('hex')[-length:]
        _h = sha256()
        _h.update(bytes(shared_secret_as_bytes))

        self.shared_key = _h.hexdigest()

        if echo_return_key is True:
            return self.shared_key
