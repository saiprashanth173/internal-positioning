import math
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.2f')


def calculate_dist_from_rssi(rssi, txPower=-12, n=4.5):
    d = 10 ** ((txPower - rssi) / (10 * n))
    return d * 0.115


class base_station(object):
    def __init__(self, lat, lon, dist):
        self.lat = lat
        self.lon = lon
        self.dist = dist


class point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class circle(object):
    def __init__(self, point, radius):
        self.center = point
        self.radius = radius


class json_data(object):
    def __init__(self, circles, inner_points, center):
        self.circles = circles
        self.inner_points = inner_points
        self.center = center


def serialize_instance(obj):
    d = {}
    d.update(vars(obj))
    return d


def get_two_points_distance(p1, p2):
    return math.sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2))


def get_two_circles_intersecting_points(c1, c2):
    p1 = c1.center
    p2 = c2.center
    r1 = c1.radius
    r2 = c2.radius

    d = get_two_points_distance(p1, p2)
    # if to far away, or self contained - can't be done
    if d >= (r1 + r2) or d <= math.fabs(r1 - r2):
        return None

    a = (pow(r1, 2) - pow(r2, 2) + pow(d, 2)) / (2 * d)
    h = math.sqrt(pow(r1, 2) - pow(a, 2))
    x0 = p1.x + a * (p2.x - p1.x) / d
    y0 = p1.y + a * (p2.y - p1.y) / d
    rx = -(p2.y - p1.y) * (h / d)
    ry = -(p2.x - p1.x) * (h / d)
    return [point(x0 + rx, y0 - ry), point(x0 - rx, y0 + ry)]


def get_all_intersecting_points(circles):
    points = []
    num = len(circles)
    for i in range(num):
        j = i + 1
        for k in range(j, num):
            res = get_two_circles_intersecting_points(circles[i], circles[k])
            if res:
                points.extend(res)
    return points


def is_contained_in_circles(point, circles):
    for i in range(len(circles)):
        if (get_two_points_distance(point, circles[i].center) > (circles[i].radius)):
            return False
    return True


def get_polygon_center(points):
    center = point(0, 0)
    num = len(points)
    for i in range(num):
        center.x += points[i].x
        center.y += points[i].y
    center.x /= num
    center.y /= num
    return center


def get_points():
    p1 = point(4.5, 8.5)
    p2 = point(10, 4)
    p3 = point(14, 4)
    p4 = point(18.5, 4)
    p5 = point(10, 7)
    p6 = point(14, 7)
    p7 = point(18.5, 7)
    p8 = point(10, 10)
    p9 = point(4, 15)
    p10 = point(10, 15)
    p11 = point(14, 15)
    p12 = point(18, 15)
    p13 = point(23, 15)

    return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]


def get_center(circle_list):
    inner_points = []
    for p in get_all_intersecting_points(circle_list):
        inner_points.append(p)
    center = get_polygon_center(inner_points)
    return center


if __name__ == '__main__':
    r2 = calculate_dist_from_rssi(-70)
