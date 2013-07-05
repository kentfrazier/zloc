import math
from itertools import product
import sys

_INT_BITS = int(math.log(sys.maxint, 2) + 1)
_HALF_BITS = _INT_BITS // 2

_LOW_ORDER_MASK = (1 << _HALF_BITS) - 1

def explicit_interleave(x, y):
    for var in (x, y):
        assert isinstance(var, int)
        assert var >> _HALF_BITS == 0

    z = 0
    for offset in xrange(_HALF_BITS):
        bit_pair = (
            (((x >> offset) & 1) << 1)
            |
            ((y >> offset) & 1)
        )
        z |= bit_pair << (offset * 2)

    return z


def _build_twiddling_steps():
    """
    Calculate the bitwise "magic numbers" needed to efficiently interleave
    numbers.

    Each step here essentially splits the bits in half.

    With the following 16-bit string, where each letter is a discrete bit,
    and a dash indicates a bit whose value we can disregard, we would do the
    following:

    x = 00000000abcdefgh
    x = x | (x << 4)      # 0000abcd----efgh
    x = x & 0x0F0F        # 0000abcd0000efgh
    x = x | (x << 2)      # 00ab--cd00ef--gh
    x = x & 0x3333        # 00ab00cd00ef00gh
    x = x | (x << 1)      # 0a-b0c-d0e-f0g-h
    x = x & 0x5555        # 0a0b0c0d0e0f0g0h

    We can then interleave two numbers so spread out easily

    x = x << 1            # a0b0c0d0e0f0g0h0
    y = 0i0j0k0l0m0n0o0p
    interleaved = x | y   # aibjckdlemfngohp

    This can easily be extended to more bits by adding prior steps with
    larger widths and masks.

    And in reverse:

    z = aibjckdlemfngohp
    x = z >> 1            # 0aibjckdlemfngoh

    x = x & 0x5555        # 0a0b0c0d0e0f0g0h
    x = x | (x >> 1)      # 0aabbccddeeffggh
    x = x & 0x3333        # 00ab00cd00ef00gh
    x = x | (x >> 2)      # 00ababcdcdefefgh
    x = x & 0x0F0F        # 0000abcd0000efgh
    x = x | (x >> 4)      # 0000abcdabcdefgh

    Then we just have to mask out the high-order bits, and we are back to the
    original value for x.

    x = x & 0x00FF        # 00000000abcdefgh
    """

    steps = []
    chunk_size = _HALF_BITS
    while chunk_size > 1:
        mask = 0
        width = chunk_size // 2
        submask = (1 << width) - 1
        for offset in xrange(0, _INT_BITS, chunk_size):
            mask |= submask << offset
        steps.append((width, mask))
        chunk_size >>= 1
    return tuple(steps)



_BIT_TWIDDLING_STEPS = _build_twiddling_steps()


def interleave(x, y):
    for stride, mask in _BIT_TWIDDLING_STEPS:
        x = (x | (x << stride)) & mask
        y = (y | (y << stride)) & mask
    z = (x << 1) | y
    return z


def explicit_deinterleave(z):
    x = 0
    y = 0
    for offset in xrange(_HALF_BITS):
        bit_pair = (z >> (offset * 2)) & 3
        x |= (bit_pair >> 1) << offset
        y |= (bit_pair & 1) << offset
    return (x, y)


def deinterleave(z):
    x = z >> 1
    y = z

    for stride, mask in reversed(_BIT_TWIDDLING_STEPS):
        x = x & mask
        x = x | (x >> stride)
        y = y & mask
        y = y | (y >> stride)

    x &= _LOW_ORDER_MASK
    y &= _LOW_ORDER_MASK

    return x, y

def test_symmetric_interleave(num_tests=10000):
    import random
    max_half_int = (1 << _HALF_BITS) - 1

    for _ in xrange(num_tests):
        x = random.randint(0, max_half_int)
        y = random.randint(0, max_half_int)
        assert (
            deinterleave(interleave(x, y)) == (x, y)
        ), 'Fail for x: {0}, y: {1}'.format(x, y)


def neighbors(z, length):

    mask = (1 << length) - 1

    def adjacent(n):
        return tuple((n + mod) & mask for mod in (-1, 0, 1))

    xs, ys = map(adjacent, deinterleave(z))
    return tuple(interleave(x, y) for x, y in product(xs, ys))


_ZLOC_MULTIPLIER = 1. / (360. / (1 << _HALF_BITS))
_ZLOC_MULTIPLIER = 1. / (360. / (1 << (_HALF_BITS - 6)))
_LATITUDE_OFFSET = 90  # -90 to 90 degrees -> 0 to 180 degrees
_LONGITUDE_OFFSET = 180  # -180 to 180 degrees -> 0 to 360 degrees

def lat_lng_to_zloc(lat, lng):
    lat = int((_LATITUDE_OFFSET + lat) * _ZLOC_MULTIPLIER)
    lng = int((_LONGITUDE_OFFSET + lng) * _ZLOC_MULTIPLIER)
    zloc = interleave(lat, lng)
    return zloc

def zloc_to_lat_lng(zloc):
    lat, lng = deinterleave(zloc)
    lat = float(lat) / _ZLOC_MULTIPLIER - _LATITUDE_OFFSET
    lng = float(lng) / _ZLOC_MULTIPLIER - _LONGITUDE_OFFSET
    return lat, lng


# TODO: encode bit length in the first 6 bits of the int, leaving 58 for pairs
#       this allows the bit string to also encode its scope and doesn't result
#       in an important loss of precision at the scale we are talking
# TODO: create functions for geographic distance from lat/lng
# TODO: return a range from zloc_to_lat_lng? Due to the varying distance of
#       lng, maybe this would be inaccurate. Probably not super necessary.
