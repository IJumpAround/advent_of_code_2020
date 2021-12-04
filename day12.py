import math
from enum import Enum
from typing import NamedTuple
import numpy as np
from utils.input_loader import load_file_as_list


class Command(NamedTuple):
    action: 'Action'
    amount: int


class Coordinate:

    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def scale(self, x: int = 1, y: int = 1, scale=0, inplace=True):
        if scale:
            x = self.x * scale
            y = self.y * scale
        else:
            x = self.x * x
            y = self.y * y

        if inplace:
            self.x = x
            self.y = y
            return self

        return Coordinate(x, y)

    def manhattan(self, other: 'Coordinate') -> int:
        x_abs = abs(other.x + self.x)
        y_abs = abs(other.y + self.y)

        return x_abs + y_abs

    @property
    def value(self):
        return self.x, self.y

    def __add__(self, other: 'Coordinate'):
        return Coordinate(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Coordinate'):
        return Coordinate(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: int):
        return Coordinate(self.x * scalar, self.y * scalar)

    def __floordiv__(self, scalar):
        return Coordinate(self.x // scalar, self.y // scalar)

    def __truediv__(self, scalar):
        return Coordinate(self.x / scalar, self.y / scalar)

    def __eq__(self, other: 'Coordinate'):
        if isinstance(other, tuple):
            other = Coordinate(*other)
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f'({self.x:.1f},{self.y:.1f})'

    def __repr__(self):
        return f"<Coordinate(x={self.x}, y={self.y})>"

    def __hash__(self):
        return hash((self.x, self.y))


origin = Coordinate()


class Direction(Enum):
    N = (0, 1)
    S = (0, -1)
    W = (-1, 0)
    E = (1, 0)
    NE = (1, 1)
    SE = (1, -1)
    SW = (-1, -1)
    NW = (-1, 1)


def prettify_unit_vector(vector: Coordinate) -> str:
    vector_char_map = {
        Direction.N.value: '↑',
        Direction.NE.value: '↗',
        Direction.E.value: '→',
        Direction.SE.value: '↘',
        Direction.S.value: '↓',
        Direction.SW.value: '↙',
        Direction.W.value: '←',
        Direction.NW.value: '↖',
    }

    return vector_char_map.get(vector, np.angle(vector.value, deg=True)[0])


class Rotate(Enum):
    L = 'L'
    R = 'R'


class Move(Enum):
    F = 1


class Action(Enum):
    N = Direction.N
    S = Direction.S
    W = Direction.W
    E = Direction.E
    L = Rotate.L
    R = Rotate.R
    F = Move.F


ccw_90_matrix = np.array([[0, -1],
                          [1, 0]])

ccw_270_matrx = np.array([[0, 1],
                          [-1, 0]])


def unit_vector_from_facing(facing: Rotate):
    return Direction.W.value if facing.L else Direction.E.value


def _counter_clockwise_rotate(unit_vector: Coordinate, degrees: int):
    theta = math.radians(degrees)
    theta += np.angle(theta)
    rot_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta), math.cos(theta)]])

    out = rot_matrix.dot(unit_vector.value)
    return Coordinate(*out)


def rotate_right(unit_vector: Coordinate, degrees: int):
    degrees = 360 - degrees
    return _counter_clockwise_rotate(unit_vector, degrees)


def rotate_left(unit_vector: Coordinate, degrees: int):
    return _counter_clockwise_rotate(unit_vector, degrees)


def rotate_vector(vector: Coordinate, direction: Rotate, degrees) -> Coordinate:
    if direction == Rotate.R:
        vector = rotate_right(vector, degrees)
    else:
        vector = rotate_left(vector, degrees)

    return vector


def calc_unit_vector(position: Coordinate):
    norm = np.linalg.norm(position.value)

    new = position / norm
    return new


class Ship:

    def __init__(self, use_waypoint=False):
        self.use_waypoint = use_waypoint
        self.facing_unit_vector = Coordinate(1, 0)
        self._ship_position = Coordinate()
        self._waypoint = self._ship_position + Coordinate(10, 1)
        self.history = []

    def execute_cmd(self, cmd: Command):
        action = cmd.action.value
        if isinstance(action, Rotate):
            self.adjust_facing(action, cmd.amount)
        elif action == Move.F:
            self.move_forward(cmd.amount)
        elif isinstance(action, Direction):
            self.move(action, cmd.amount)
        else:
            raise NotImplementedError(action)

        self.history.append((str(self), cmd))

    def _rotate_waypoint(self, direction: Rotate, degrees: int):
        ship_relative_waypoint = self._waypoint - self._ship_position
        rotated = rotate_vector(ship_relative_waypoint, direction, degrees)

        self._waypoint = rotated + self._ship_position

    def _rotate_ship(self, direction: Rotate, degrees: int):
        ship_unit_vector = self.facing_unit_vector

        self.facing_unit_vector = rotate_vector(ship_unit_vector, direction, degrees)

    def adjust_facing(self, direction: Rotate, degrees: int):
        if self.use_waypoint:
            self._rotate_waypoint(direction, degrees)
        else:
            self._rotate_ship(direction, degrees)

    def move_forward(self, amount):

        if self.use_waypoint:
            move_direction = self.waypoint_position - self._ship_position
        else:
            move_direction = self.facing_unit_vector

        move_vector = move_direction.scale(scale=amount, inplace=False)
        self._ship_position += move_vector

        if self.use_waypoint:
            self._waypoint += move_vector

    def move(self, move: Direction, amount: int):
        move = Coordinate(*move.value)
        move.scale(scale=amount, inplace=True)

        if self.use_waypoint:
            self._waypoint += move
        else:
            self._ship_position += move

    def show_history(self):
        hist_string = '\n'.join([f"{turn + 1}: {str(h[0])} {str(h[1])}" for turn, h in enumerate(self.history)])
        manhattan = origin.manhattan(self._ship_position)
        hist_string = f"Ship's history: (state after action, action)\n{hist_string}"

        return hist_string + '\nManhattan Distance: ' + str(manhattan)

    @property
    def manhattan(self):
        return origin.manhattan(self._ship_position)

    @property
    def waypoint_position(self):
        return self._waypoint

    @property
    def ship_unit_vector(self):
        return calc_unit_vector(self._ship_position)

    def __str__(self):
        waypoint_state = f"⛿: {self.waypoint_position}" if self.use_waypoint else ''
        ship_state = f"🛥 {prettify_unit_vector(self.ship_unit_vector)} {self._ship_position}"
        return f"{ship_state:12}  {waypoint_state if waypoint_state else ''}"


def action_from_text(text):
    actions = {
        'N': Action.N,
        'S': Action.S,
        'W': Action.W,
        'E': Action.E,
        'L': Action.L,
        'R': Action.R,
        'F': Action.F

    }

    return actions[text]


def solve(sample, part=1):
    input_lines = map(lambda x: (Command(action_from_text(x[0]), int(x[1:]))), load_file_as_list(12, sample))
    input_lines = list(input_lines)

    waypoint = (part == 2)
    ship = Ship(use_waypoint=waypoint)

    for command in input_lines:
        ship.execute_cmd(command)

    print(ship.show_history())

    return ship.manhattan


if __name__ == '__main__':
    assert round(solve(False)) == 1319
    assert round(solve(False, 2)) == 62434
