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
        else:
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

    def __eq__(self, other: 'Coordinate'):
        if isinstance(other, tuple):
            other = Coordinate(*other)
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f'({self.x},{self.y})'

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

    return vector_char_map[vector]

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
    out = out.round()
    return Coordinate(*out)


def rotate_right(unit_vector: Coordinate, degrees: int):
    degrees = 360 - degrees
    return _counter_clockwise_rotate(unit_vector, degrees)


def rotate_left(unit_vector: Coordinate, degrees: int):

    return _counter_clockwise_rotate(unit_vector, degrees)


class Ship:

    def __init__(self):
        self.facing_unit_vector = Coordinate(1, 0)
        self.position = Coordinate()
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

    def adjust_facing(self, direction: Rotate, degrees: int):
        unit_vector = self.facing_unit_vector

        if direction == Rotate.R:
            unit_vector = rotate_right(unit_vector, degrees)
        else:
            unit_vector = rotate_left(unit_vector, degrees)

        self.facing_unit_vector = unit_vector

    def move_forward(self, amount):
        move_vector = self.facing_unit_vector.scale(scale=amount, inplace=False)
        self.position += move_vector

    def move(self, move: Direction, amount: int):
        move = Coordinate(*move.value)
        self.position += move.scale(scale=amount)

    def show_history(self):
        hist_string = '\n'.join([f"{turn+1}: {str(h[0])} {str(h[1])}" for turn, h in enumerate(self.history)])
        manhattan = origin.manhattan(self.position)
        hist_string = f"Ship's history: (state after action, action)\n{hist_string}"

        return hist_string + '\nManhattan Distance: ' + str(manhattan)

    def __str__(self):
        return f"{prettify_unit_vector(self.facing_unit_vector)} {self.position})"


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


def solve(sample):
    input_lines = map(lambda x: (Command(action_from_text(x[0]), int(x[1:]))), load_file_as_list(12, sample))
    input_lines = list(input_lines)
    print(input_lines)

    ship = Ship()

    # ship.execute_cmd(input_lines[0])

    for command in input_lines:
        ship.execute_cmd(command)

    print(ship)
    print(ship.show_history())


if __name__ == '__main__':
    solve(False)
