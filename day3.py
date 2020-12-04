from functools import reduce

import numpy as np

def read_input():
    lines = []
    with open('puzzle_input/day3/input.txt', 'r') as f:
        for line in f:
            squares = [square for square in line.strip()]
            lines.append(squares)

    return np.array(lines)

TREE = '#'

terrain = read_input()
print(terrain)
PAGE_WIDTH = len(terrain[0])


moves = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
height = len(terrain)

def move(x,y):
    x = (x + RIGHT) % PAGE_WIDTH
    y+= DOWN

    return x,y


tree_counts = []
for RIGHT, DOWN in moves:
    tree_count = 0
    y_pos = 0
    x_pos = 0
    while y_pos < height - DOWN:
        x_pos, y_pos = move(x_pos, y_pos)

        if terrain[y_pos,x_pos] == TREE:
            tree_count += 1
            # terrain[y_pos, x_pos] = 'X'
    tree_counts.append(tree_count)

res = reduce(lambda x,y: x * y, tree_counts)

print (tree_counts)
print(res)