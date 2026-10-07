from enum import Enum


class DeathCause(str, Enum):
    ASTEROID = "asteroid"
    HYPERSPACE_LANDING = "hyperspace_landing"
    HYPERSPACE_FAILURE = "hyperspace_failure"
    SAUCER_BULLET = "saucer_bullet"
    SAUCER_COLLISION = "saucer_collision"
