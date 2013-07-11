from math import (
    asin,
    atan2,
    cos,
    degrees,
    pi,
    sin,
)
from numpy import linspace
from zloc import (
    MAX_PRECISION,
    EARTH_RADIUS_MEAN,
    zloc_blocks_for_radius,
    zloc_corners,
    zloc_precision,
    zloc_to_lat_lng,
    zloc_to_lat_lng_radians,
)
import geojson


def zloc_to_geojson(zloc):
    if zloc_precision(zloc) == MAX_PRECISION:
        lat, lng = zloc_to_lat_lng(zloc)
        return geojson.Point([lng, lat])
    tl, tr, bl, br = map(
        tuple,
        map(reversed, map(zloc_to_lat_lng, zloc_corners(zloc))))
    return geojson.Polygon([tl, tr, br, bl, tl])


def zloc_circle(zloc, radius_m, num_points=100):
    angles = linspace(0.0, 2 * pi, num_points, endpoint=False)

    radius_rad = radius_m / EARTH_RADIUS_MEAN
    origin_lat, origin_lng = zloc_to_lat_lng_radians(zloc)

    cos_r = cos(radius_rad)
    sin_r = sin(radius_rad)
    cos_olat = cos(origin_lat)
    sin_olat = sin(origin_lat)

    points = []
    for theta in angles:
        lat = asin(sin_olat * cos_r + cos_olat * sin_r * cos(theta))
        dlng = atan2(sin(theta) * sin_r * cos_olat, cos_r - sin_olat * sin(lat))
        lng = ((origin_lng - dlng + pi) % (2 * pi)) - pi

        points.append(map(degrees, [lng, lat]))

    points.append(points[0])

    return geojson.Polygon(points)


def zloc_circle_coverage(zloc, radius):
    circle = zloc_circle(zloc, radius)
    blocks = map(zloc_to_geojson, zloc_blocks_for_radius(zloc, radius))
    origin = zloc_to_geojson(zloc)

    collection = geojson.FeatureCollection(
        features=[
            geojson.Feature(
                id='search area',
                geometry=circle,
                properties={
                    'origin': zloc,
                    'radius': radius,
                },
            ),
            geojson.Feature(
                id='zloc blocks',
                geometry=geojson.GeometryCollection(
                    geometries=blocks,
                ),
                properties={},
            ),
            geojson.Feature(
                id='origin',
                geometry=origin,
                properties={
                    'zloc': zloc,
                },
            ),
        ],
        properties={},
    )

    return collection
