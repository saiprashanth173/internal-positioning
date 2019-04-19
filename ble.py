import json
import math
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

def calculate_dist_from_rssi(rssi, txPower=-12, n=4.5):  
#   if rssi == -200:
#     return -1.0

#   ratio = rssi*1.0/txPower
#   if ratio < 1.0:
#     return ratio**10
#   else:
#     accuracy =  (0.89976)*(ratio**7.7095) + 0.111
#     return accuracy
    d = 10 ** ((txPower - rssi) / (10 * n))
    return d*0.115

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
    if d >= (r1 + r2) or d <= math.fabs(r1 -r2):
        return None

    a = (pow(r1, 2) - pow(r2, 2) + pow(d, 2)) / (2*d)
    h  = math.sqrt(pow(r1, 2) - pow(a, 2))
    x0 = p1.x + a*(p2.x - p1.x)/d 
    y0 = p1.y + a*(p2.y - p1.y)/d
    rx = -(p2.y - p1.y) * (h/d)
    ry = -(p2.x - p1.x) * (h / d)
    return [point(x0+rx, y0-ry), point(x0-rx, y0+ry)]

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

if __name__ == '__main__' :
    p2 = point(10, 4)
    p3 = point(14, 2)
    p4 = point(18.5, 3.5)
    p5 = point(10, 7)
    p7 = point(18.5, 7)
    p8 = point(10, 10)
    #354 I06
    # r2 = calculate_dist_from_rssi(-70)
    # r5 = calculate_dist_from_rssi(-73)
    # r8 = calculate_dist_from_rssi(-82)   
    # c1 = circle(p2, r2)
    # c2 = circle(p5, r5)
    # c3 = circle(p8, r8)



    #1113 K04
    # r2 = calculate_dist_from_rssi(-67)
    # r3 = calculate_dist_from_rssi(-81)
    # r5 = calculate_dist_from_rssi(-81)
    # c1 = circle(p2, r2)
    # c2 = circle(p3, r3)
    # c3 = circle(p5, r5)

    # 1252 N03
    # r3 = calculate_dist_from_rssi(-71)
    # r5 = calculate_dist_from_rssi(-82)
    # r7 = calculate_dist_from_rssi(-76)
    # c1 = circle(p3, r3)
    # c2 = circle(p5, r5)
    # c3 = circle(p7, r7)

    # 1068 K05
    # r2 = calculate_dist_from_rssi(-78)
    # r4 = calculate_dist_from_rssi(-78)
    # r5 = calculate_dist_from_rssi(-77)
    # c1 = circle(p2, r2)
    # c2 = circle(p4, r4)
    # c3 = circle(p5, r5)


    # circle_list = [c1, c2, c3]

    inner_points = []
    for p in get_all_intersecting_points(circle_list):
        # if is_contained_in_circles(p, circle_list):
        inner_points.append(p) 
    
    center = get_polygon_center(inner_points)

    in_json = json_data([c1, c2, c3], [p2, p5, p8], center)

    out_json = json.dumps(in_json, sort_keys=True,
                     indent=4, default=serialize_instance)

    print(out_json)

# 10^((x+70)/(10*n)) = 1.73205, 10^((x+73)/(10*n)) = 1.41421
