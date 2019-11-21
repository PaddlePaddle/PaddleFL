# coding=utf-8

# 
# (c) Chris von Csefalvay, 2015.

"""
exceptions is responsible for exception handling etc.
"""


class MalformedPublicKey(BaseException):
    """
    The public key is malformed as it does not meet the Legendre symbol criterion. The key might have been tampered with or might have been damaged in transit.
    """

    def __str__(self):
        return "Public key malformed: fails Legendre symbol verification."


class RNGError(BaseException):
    """
    Thrown when RNG could not be obtained.
    """

    def __str__(self):
        return "RNG could not be obtained. This module currently only works with Python 3."