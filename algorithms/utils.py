import math

def euclidean_dist(p, q):
    coord_pairs = zip(p, q)
    dist = math.sqrt(sum([(x - y) ** 2 for x, y in coord_pairs]))
    return dist

def manhattan_dist(p, q):
    coord_pairs = zip(p, q)
    dist = sum([abs(x - y) for x, y in coord_pairs])
    return dist

def haversine_dist(p, q):
    earth_radius = 6373.0
    p_lat = math.radians(p[0])
    p_lon = math.radians(p[1])
    q_lat = math.radians(q[0])
    q_lon = math.radians(q[1])
    lon_diff = q_lon - p_lon
    lat_diff = q_lat - p_lat
    a = math.sin(lat_diff / 2) ** 2 + math.cos(p_lat) * math.cos(q_lat) * math.sin(lon_diff / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c * 1000
    return distance
