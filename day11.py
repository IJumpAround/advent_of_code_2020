import copy
from enum import Enum
from typing import Tuple

from utils.input_loader import load_file_as_list

X = 0
Y = 1

Coords = Tuple[int, int]


class Tile(Enum):
    FLOOR = '.'
    EMPTY = 'L'
    OCCUPIED = '#'
    OOB = None


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    RUP = (-1, 1)
    LUP = (-1, -1)
    LDOWN = (1, -1)
    RDOWN = (1, 1)


def print_board(board):
    for row in board:
        print(row)


def offset_coordinates(coordinates: Coords, direction: Direction, amount=1) -> Coords:
    direction = direction.value
    new_coords = (coordinates[0] + direction[0] * amount, coordinates[1] + direction[1] * amount)

    return new_coords


# num occupied next to adjacent decides action (8 positions cardinal plus diagonal)
# rules
# empty seat with no adjacent occupied becomes occupied
# occupied with 4 adjacent occupied -> empty

def coords_out_of_bounds(board, coords: Coords):
    return coords[0] < 0 or coords[1] < 0 or coords[0] > len(board[0]) or coords[1] > len(board)


def visible_count(board, tile: Coords, looking_for: Tile, max_dist=None):
    count = 0
    for direction in Direction:
        distance = 1
        other_tile = None
        while other_tile != Tile.OOB and distance != max_dist:
            other_tile = get_tile(board, tile, direction, distance)
            if other_tile == looking_for:
                count += 1
                break
            elif other_tile != Tile.FLOOR:
                break
            distance += 1

    return count


def get_tile(board, start_tile_coords: Coords, direction: Direction, distance) -> Tile:
    new_coords = offset_coordinates(start_tile_coords, direction, distance)

    if coords_out_of_bounds(board, new_coords):
        return Tile.OOB
    try:
        tile = board[new_coords[X]][new_coords[Y]]
    except:
        return Tile.OOB

    return Tile(tile)


def count_all_seats_type(board, looking_for: Tile):
    total = 0
    for row in board:
        total += sum([1 if looking_for.value == val else 0 for val in row])

    return total


def solve(sample):
    file_content = load_file_as_list(11, sample, line_as_list=True)

    current_board = file_content

    prev_board = None
    pass_num = 1
    while prev_board != current_board:
        pending_board = copy.deepcopy(current_board)
        prev_board = current_board
        print(f"Pass: {pass_num}")
        print_board(current_board)
        for row in range(len(current_board)):
            for col in range(len(current_board[row])):
                tile = Tile(current_board[row][col])

                if tile.value == Tile.FLOOR:
                    continue

                if tile == Tile.EMPTY and visible_count(current_board, (row, col), Tile.OCCUPIED) == 0:
                    pending_board[row][col] = Tile.OCCUPIED.value
                elif tile == Tile.OCCUPIED and visible_count(current_board, (row, col), Tile.OCCUPIED) >= 5:
                    pending_board[row][col] = Tile.EMPTY.value
        pass_num += 1
        current_board = pending_board

    answer = count_all_seats_type(current_board, Tile.OCCUPIED)

    print(answer)
    return answer


if __name__ == '__main__':
    solve(False)
