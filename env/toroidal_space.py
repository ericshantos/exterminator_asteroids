import numpy as np


class ToroidalSpace:
    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height

    def wrap_x(
        self,
        x: float,
    ) -> float:
        return x % self.width

    def wrap_y(
        self,
        y: float,
    ) -> float:
        return y % self.height

    def ghost_offsets(
        self,
        x: float,
        y: float,
        radius: float,
    ) -> list[tuple[float, float]]:
        xs = [0.0]
        ys = [0.0]

        if x - radius < 0:
            xs.append(float(self.width))
        elif x + radius > self.width:
            xs.append(-float(self.width))

        if y - radius < 0:
            ys.append(float(self.height))
        elif y + radius > self.height:
            ys.append(-float(self.height))

        return [(ox, oy) for ox in xs for oy in ys]

    def delta_x(
        self,
        dx: float,
    ) -> float:
        if dx > self.width / 2:
            return dx - self.width

        if dx < -self.width / 2:
            return dx + self.width

        return dx

    def delta_y(
        self,
        dy: float,
    ) -> float:
        if dy > self.height / 2:
            return dy - self.height

        if dy < -self.height / 2:
            return dy + self.height

        return dy

    def relative_vector(
        self,
        source_x: float,
        source_y: float,
        target_x: float,
        target_y: float,
    ) -> tuple[float, float]:

        dx = self.delta_x(
            target_x - source_x
        )

        dy = self.delta_y(
            target_y - source_y
        )

        return dx, dy

    def distance(
        self,
        source_x: float,
        source_y: float,
        target_x: float,
        target_y: float,
    ) -> float:

        dx, dy = self.relative_vector(
            source_x,
            source_y,
            target_x,
            target_y
        )

        return float(np.hypot(dx, dy))

    def direction(
        self,
        source_x: float,
        source_y: float,
        target_x: float,
        target_y: float,
    ) -> np.ndarray:

        dx, dy = self.relative_vector(
            source_x,
            source_y,
            target_x,
            target_y
        )

        norm = np.hypot(dx, dy)

        if norm <= 1e-8:
            return np.zeros(2, dtype=np.float32)

        return np.asarray(
            [dx / norm, dy / norm],
            dtype=np.float32
        )
