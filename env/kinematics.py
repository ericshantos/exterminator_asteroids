from dataclasses import dataclass
import math

NO_COLLISION: float = math.inf


@dataclass(slots=True)
class RelativeState:
    forward: float
    right: float
    vel_forward: float
    vel_right: float

    @property
    def distance(self) -> float:
        return math.hypot(self.forward, self.right)

    @property
    def bearing(self) -> float:
        return math.atan2(self.right, self.forward)


def ship_axes(angle_degrees: float) -> tuple[tuple[float, float], tuple[float, float]]:
    radians = math.radians(angle_degrees)

    heading = (math.sin(radians), -math.cos(radians))
    right = (math.cos(radians), math.sin(radians))

    return heading, right


def to_ship_frame(
    dx: float,
    dy: float,
    dvx: float,
    dvy: float,
    heading: tuple[float, float],
    right: tuple[float, float],
) -> RelativeState:
    return RelativeState(
        forward=dx * heading[0] + dy * heading[1],
        right=dx * right[0] + dy * right[1],
        vel_forward=dvx * heading[0] + dvy * heading[1],
        vel_right=dvx * right[0] + dvy * right[1],
    )


def _smallest_positive_root(a: float, b: float, c: float) -> float:
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return NO_COLLISION

        t = -c / b

        return t if t >= 0 else NO_COLLISION

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return NO_COLLISION

    sqrt_d = math.sqrt(discriminant)

    roots = sorted(((-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)))

    for t in roots:
        if t >= 0:
            return t

    return NO_COLLISION


def time_to_collision(state: RelativeState, contact_distance: float) -> float:
    px, py = state.forward, state.right
    vx, vy = state.vel_forward, state.vel_right

    c = px * px + py * py - contact_distance * contact_distance

    if c <= 0:
        return 0.0

    b = 2 * (px * vx + py * vy)

    # Afastando-se: a distância só cresce.
    if b >= 0:
        return NO_COLLISION

    a = vx * vx + vy * vy

    return _smallest_positive_root(a, b, c)


def closest_approach(state: RelativeState) -> float:
    px, py = state.forward, state.right
    vx, vy = state.vel_forward, state.vel_right

    speed_sq = vx * vx + vy * vy

    if speed_sq < 1e-9:
        return state.distance

    t = max(0.0, -(px * vx + py * vy) / speed_sq)

    return math.hypot(px + vx * t, py + vy * t)


@dataclass(slots=True)
class Intercept:
    reachable: bool
    lead_angle: float
    flight_time: float


def intercept(state: RelativeState, bullet_speed: float, max_flight: float) -> Intercept:
    px, py = state.forward, state.right
    vx, vy = state.vel_forward, state.vel_right

    a = vx * vx + vy * vy - bullet_speed * bullet_speed
    b = 2 * (px * vx + py * vy)
    c = px * px + py * py

    t = _smallest_positive_root(a, b, c)

    if t == NO_COLLISION or t > max_flight:
        return Intercept(False, state.bearing, NO_COLLISION)

    aim_forward = px + vx * t
    aim_right = py + vy * t

    return Intercept(True, math.atan2(aim_right, aim_forward), t)


def angle_difference(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi
