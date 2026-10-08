import math

from env.kinematics import (
    NO_COLLISION,
    RelativeState,
    angle_difference,
    closest_approach,
    escape_time_to_collision,
    intercept,
    ship_axes,
    time_to_collision,
)


def test_ship_axes_follow_screen_convention() -> None:
    heading, right = ship_axes(0.0)
    assert heading == (0.0, -1.0)
    assert right == (1.0, 0.0)

    heading, right = ship_axes(90.0)
    assert math.isclose(heading[0], 1.0) and abs(heading[1]) < 1e-9
    assert abs(right[0]) < 1e-9 and math.isclose(right[1], 1.0)


def test_time_to_collision_head_on() -> None:
    state = RelativeState(forward=100.0, right=0.0, vel_forward=-2.0, vel_right=0.0)
    assert math.isclose(time_to_collision(state, 10.0), 45.0)


def test_time_to_collision_receding_and_touching() -> None:
    receding = RelativeState(100.0, 0.0, 2.0, 0.0)
    assert time_to_collision(receding, 10.0) == NO_COLLISION

    touching = RelativeState(5.0, 0.0, 2.0, 0.0)
    assert time_to_collision(touching, 10.0) == 0.0


def test_time_to_collision_near_miss() -> None:
    passing = RelativeState(100.0, 30.0, -2.0, 0.0)
    assert time_to_collision(passing, 10.0) == NO_COLLISION
    assert math.isclose(closest_approach(passing), 30.0)


def test_escape_without_turn_or_gain_is_time_to_collision() -> None:
    head_on = RelativeState(100.0, 0.0, -2.0, 0.0)
    assert math.isclose(escape_time_to_collision(head_on, 10.0, 0.0, 0.0, 0.0), 45.0)


def test_escape_collides_during_turn() -> None:
    head_on = RelativeState(100.0, 0.0, -2.0, 0.0)
    assert math.isclose(escape_time_to_collision(head_on, 10.0, 60.0, 0.0, 3.0), 45.0)


def test_escape_sideways_avoids_collision() -> None:
    head_on = RelativeState(100.0, 0.0, -2.0, 0.0)
    assert escape_time_to_collision(head_on, 10.0, 0.0, 0.0, 3.0) == NO_COLLISION
    assert escape_time_to_collision(head_on, 10.0, 20.0, 0.0, 3.0) == NO_COLLISION


def test_escape_too_late_collides_after_turn() -> None:
    head_on = RelativeState(100.0, 0.0, -2.0, 0.0)
    ttc = escape_time_to_collision(head_on, 10.0, 44.0, 0.0, 3.0)
    assert math.isclose(ttc, 44.0 + 22.0 / 13.0)


def test_intercept_stationary_target_ahead() -> None:
    shot = intercept(RelativeState(100.0, 0.0, 0.0, 0.0), 10.0, 60.0)
    assert shot.reachable
    assert math.isclose(shot.lead_angle, 0.0, abs_tol=1e-9)
    assert math.isclose(shot.flight_time, 10.0)


def test_intercept_leads_a_moving_target() -> None:
    shot = intercept(RelativeState(100.0, 0.0, 0.0, 3.0), 10.0, 60.0)
    assert shot.reachable
    assert shot.lead_angle > 0


def test_intercept_out_of_range() -> None:
    shot = intercept(RelativeState(1000.0, 0.0, 0.0, 0.0), 10.0, 60.0)
    assert not shot.reachable


def test_angle_difference_wraps() -> None:
    assert math.isclose(angle_difference(math.radians(170), math.radians(-170)),
                        math.radians(-20))
