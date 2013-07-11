from itertools import product
from math import (
    atan2,
    ceil,
    cos,
    log,
    pi,
    radians,
    sin,
    sqrt,
)


_INT_BITS = 64
_PAIR_LENGTH_BITS = 6
_PAIR_BITS = _INT_BITS - _PAIR_LENGTH_BITS
MAX_PRECISION = _PAIR_BITS // 2
_NUM_ZLOC_VALUES = 1 << MAX_PRECISION

_PAIR_MASK = (1 << _PAIR_BITS) - 1
_LOW_ORDER_MASK = (1 << MAX_PRECISION) - 1

_RADIANS_PER_ZLOC_VALUE = 2 * pi / _NUM_ZLOC_VALUES
_DEGREES_PER_ZLOC_VALUE = 360. / _NUM_ZLOC_VALUES
_LATITUDE_DEGREE_OFFSET = 90.     # -90 to 90 degrees     -> 0 to 180 degrees
_LONGITUDE_DEGREE_OFFSET = 180.   # -180 to 180 degrees   -> 0 to 360 degrees
_LATITUDE_RADIAN_OFFSET = pi / 2  # -pi/2 to pi/2 radians -> 0 to pi radians
_LONGITUDE_RADIAN_OFFSET = pi     # -pi to pi radians     -> 0 to 2*pi radians

# According to WGS-84 reference ellipsoid
EARTH_RADIUS_EQUATORIAL = 6378137.0
EARTH_RADIUS_POLAR = 6356752.314245

# Per the formula from the International Union of Geodesy and Geophysics
EARTH_RADIUS_MEAN = (2 * EARTH_RADIUS_EQUATORIAL + EARTH_RADIUS_POLAR) / 3


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
    stride = _INT_BITS // 4
    while stride > 0:
        mask = (1 << stride) - 1
        span = stride * 2
        while span < _INT_BITS:
            mask |= mask << span
            span *= 2
        steps.append((stride, mask))
        stride //= 2
    return tuple(steps)


_BIT_TWIDDLING_STEPS = _build_twiddling_steps()


def interleave(x, y):
    for stride, mask in _BIT_TWIDDLING_STEPS:
        x = (x | (x << stride)) & mask
        y = (y | (y << stride)) & mask
    z = (x << 1) | y
    return z


def deinterleave(z):
    x = z >> 1
    y = z

    for stride, mask in reversed(_BIT_TWIDDLING_STEPS):
        x &= mask
        x |= x >> stride
        y &= mask
        y |= y >> stride

    x &= _LOW_ORDER_MASK
    y &= _LOW_ORDER_MASK

    return x, y


def _show_bits(n):
    print bin((1 << 64) | n)[3:]


_NEIGHBOR_OFFSETS = tuple(product((-1, 0, 1), (-1, 0, 1)))

def neighbors(zloc):
    return relative_zlocs(zloc, _NEIGHBOR_OFFSETS)


def relative_zlocs(zloc, offset_pairs):
    """
    Given a zloc and a sequence of 2-tuples of offset pairs (which should
    contain only integer values), return zlocs with those offsets at the
    precision level of `zloc`.
    """
    precision = zloc >> _PAIR_BITS
    pair_bits = precision * 2
    pair_bit_offset = _PAIR_BITS - pair_bits

    z = (zloc & _PAIR_MASK) >> pair_bit_offset
    header = precision << _PAIR_BITS
    bit_mask = (1 << pair_bits) - 1

    x, y = deinterleave(z)
    neighbor_zlocs = []

    for x_offset, y_offset in offset_pairs:
        neighbor_z = interleave(
            (x + x_offset) & bit_mask,
            (y + y_offset) & bit_mask,
        )
        neighbor_zlocs.append((neighbor_z << pair_bit_offset) | header)

    return tuple(neighbor_zlocs)


def zloc_precision(zloc):
    return zloc >> _PAIR_BITS


def zloc_midpoint(zloc):
    precision = zloc >> _PAIR_BITS
    if precision >= MAX_PRECISION:
        return zloc

    new_header = MAX_PRECISION << _PAIR_BITS

    sub_pair_offset = (precision - 1) * 2
    sub_mask = 0x03 << sub_pair_offset

    return (zloc & _PAIR_MASK) | new_header | sub_mask


def zloc_at_precision(zloc, precision):
    new_pair_bits = 2 * precision
    mask = ((1 << new_pair_bits) - 1) << (_PAIR_BITS - new_pair_bits)
    zloc &= mask
    zloc |= precision << _PAIR_BITS
    return zloc


def ancestor(zloc, level=1):
    new_precision = (zloc >> _PAIR_BITS) - level
    return zloc_at_precision(zloc, new_precision)


def lat_lng_to_zloc(lat, lng):
    zlat = int((_LATITUDE_DEGREE_OFFSET + lat) / _DEGREES_PER_ZLOC_VALUE)
    zlng = int((_LONGITUDE_DEGREE_OFFSET + lng) / _DEGREES_PER_ZLOC_VALUE)
    zloc = interleave(zlat, zlng)
    zloc |= MAX_PRECISION << _PAIR_BITS
    return zloc


def lat_lng_radians_to_zloc(lat, lng):
    zlat = int((_LATITUDE_RADIAN_OFFSET + lat) / _RADIANS_PER_ZLOC_VALUE)
    zlng = int((_LONGITUDE_RADIAN_OFFSET + lng) / _RADIANS_PER_ZLOC_VALUE)
    zloc = interleave(zlat, zlng)
    zloc |= MAX_PRECISION << _PAIR_BITS
    return zloc


def zloc_to_lat_lng(zloc):
    zlat, zlng = deinterleave(zloc)
    lat = zlat * _DEGREES_PER_ZLOC_VALUE - _LATITUDE_DEGREE_OFFSET
    lng = zlng * _DEGREES_PER_ZLOC_VALUE - _LONGITUDE_DEGREE_OFFSET
    return lat, lng


def zloc_to_lat_lng_radians(zloc):
    lat, lng = deinterleave(zloc)
    lat = lat * _RADIANS_PER_ZLOC_VALUE - _LATITUDE_RADIAN_OFFSET
    lng = lng * _RADIANS_PER_ZLOC_VALUE - _LONGITUDE_RADIAN_OFFSET
    return lat, lng


def zloc_to_lat_lng_with_range(zloc):
    pass


def zloc_to_lat_lng_radians_with_range(zloc):
    pass


def lat_lng_distance(lat1, lng1, lat2, lng2):

    return lat_lng_radian_distance(
        lat1=radians(lat1),
        lng1=radians(lng1),
        lat2=radians(lat2),
        lng2=radians(lng2),
    )


def _haversine(theta):
    return sin(theta / 2.)**2


def lat_lng_radian_distance(lat1, lng1, lat2, lng2):
    """
    Calculate great circle distance using the haversine formula

    See:
        http://www.movable-type.co.uk/scripts/latlong.html
        and
        http://en.wikipedia.org/wiki/Haversine_formula
    """

    lat_diff = lat2 - lat1
    lng_diff = lng2 - lng1

    a = _haversine(lat_diff) + _haversine(lng_diff) * cos(lat1) * cos(lat2)
    angular_distance = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = EARTH_RADIUS_MEAN * angular_distance

    return distance


def zloc_distance(z1, z2):
    lat1, lng1 = zloc_to_lat_lng_radians(z1)
    lat2, lng2 = zloc_to_lat_lng_radians(z2)
    return lat_lng_radian_distance(lat1, lng1, lat2, lng2)


def zloc_range_meters(zloc):
    """
    Return a pair of (lat_range, top_lng_range, bottom_lng_range) in meters
    """
    topleft, topright, bottomleft, bottomright = zloc_corners(zloc)
    return (
        zloc_distance(topleft, bottomleft),
        zloc_distance(topleft, topright),
        zloc_distance(bottomleft, bottomright),
    )


def zloc_corners(zloc):
    """
    Return max-precision zlocs for the four corner points of a zloc.

    Order is: top-left, top-right, bottom-right, bottom-left
    """
    precision = zloc >> _PAIR_BITS
    range_ = 1 << (MAX_PRECISION - precision)
    zloc_max_precision = zloc_at_precision(zloc, MAX_PRECISION)
    return (zloc_max_precision,) + relative_zlocs(
        zloc_max_precision,
        (
            (0, range_),
            (range_, 0),
            (range_, range_),
        ),
    )


def zloc_blocks_for_radius(center_zloc, radius):
    current_precision = zloc_precision(center_zloc)
    lat_range, _, _ = zloc_range_meters(center_zloc)
    num_lat_blocks = radius / lat_range
    lat_precision = int(ceil(log(num_lat_blocks, 2)))

    expanded_zloc = zloc_at_precision(center_zloc,
                                      current_precision - lat_precision)

    _, top_lng_range, bottom_lng_range = zloc_range_meters(expanded_zloc)
    lng_range = (top_lng_range + bottom_lng_range) / 2
    lng_blocks_needed = int(ceil((radius * 2 / lng_range) / 2)) * 2

    lng_blocks_per_side = lng_blocks_needed / 2

    offsets = product(
        (1, 0),
        xrange(-lng_blocks_per_side, lng_blocks_per_side),
    )

    return relative_zlocs(expanded_zloc, offsets)


def test_symmetric_interleave(num_tests=10000):
    import random
    MAX_HALF_INT = (1 << MAX_PRECISION) - 1

    for _ in xrange(num_tests):
        x = random.randint(0, MAX_HALF_INT)
        y = random.randint(0, MAX_HALF_INT)
        assert (
            deinterleave(interleave(x, y)) == (x, y)
        ), 'Fail for x: {0}, y: {1}'.format(x, y)


def test_symmetric_lat_lng_conversion(num_tests=10000):
    import random

    max_error = 360. * _DEGREES_PER_ZLOC_VALUE

    for _ in xrange(num_tests):
        lat = -90. + random.random() * 180.
        lng = -180. + random.random() * 360.
        zloc = lat_lng_to_zloc(lat, lng)
        lat2, lng2 = zloc_to_lat_lng(zloc)
        lat_diff = abs(lat2 - lat)
        lng_diff = abs(lng2 - lng)
        assert (
            lat_diff <= max_error and lng_diff <= max_error
        ), 'Fail for lat: {0}, lng: {1}'.format(lat, lng)


def test_symmetric_lat_lng_radians_conversion(num_tests=10000):
    import random

    max_error = 2 * pi * _RADIANS_PER_ZLOC_VALUE

    for _ in xrange(num_tests):
        lat = -pi / 2 + random.random() * pi
        lng = -pi + random.random() * 2 * pi
        zloc = lat_lng_radians_to_zloc(lat, lng)
        lat2, lng2 = zloc_to_lat_lng_radians(zloc)
        lat_diff = abs(lat2 - lat)
        lng_diff = abs(lng2 - lng)
        assert (
            lat_diff <= max_error and lng_diff <= max_error
        ), 'Fail for lat: {0}, lng: {1}'.format(lat, lng)


def _zloc_to_d3_point(zloc):
    lat, lng = zloc_to_lat_lng(zloc)
    return lng, lat


def _color_mod(shape, fill=None, stroke=None):
    if fill is not None:
        shape['fill'] = fill
    if stroke is not None:
        shape['stroke'] = stroke
    return shape
