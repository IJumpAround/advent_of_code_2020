import math
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import random
from typing import List

import numpy as np

from tabulate import tabulate

def get_monster_bit_mask():
    sea_monster_literal = """                  # 
#    ##    ##    ###
 #  #  #  #  #  #   """
    print(sea_monster_literal)
    l = [[1 if char == '#' else 0 for char in row] for row in  sea_monster_literal.splitlines()]
    monster = np.array([np.array(r) for r in l], dtype='bool')

    print(monster.shape)
    print(tabulate(monster))
    return monster

# with open('puzzle_input/day20/test_input.txt') as f:
#     contents = f.read()#

with open('puzzle_input/day20/input.txt') as f:
    contents = f.read()


raw_tiles = contents.split('\n\n')

tiles = [raw.split('\n') for raw in raw_tiles]


# print(tiles)
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


class Graph:
    class Node:

        def __init__(self, tile: 'Tile'):
            self.tile = tile
            self.dist = math.inf

        @property
        def id(self):
            if isinstance(self.tile, Tile):
                return self.tile.id
            else:
                return self.tile

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other: 'Graph.Node'):
            return self.id == other.id

        def __str__(self):
            return f'{self.id}'

        def __repr__(self):
            return self.__str__()

    class Edge:
        def __init__(self, left: 'Graph.Node', right: 'Graph.Node', edge_connections: tuple):
            self.left = left
            self.right = right
            self.left_edge = edge_connections[0]
            self.right_edge = edge_connections[0]

        def __eq__(self, other: 'Graph.Edge'):
            return self.left.id == other.left.id and self.right.id == other.right.id

        def __hash__(self):
            return hash((self.left.id, self.right.id))

    def __init__(self, size):
        self.edges = []
        self.mtx = [[] for i in range(size)]
        self.seen = []
        self.graph = defaultdict(set)
        self.potential_paths = []
        self.shortest = []
        self.dist = {}
        self.v = 0

    def add_edge(self, left: 'Tile', right: 'Tile'):
        # right_node = self.Node(right)
        # left_node = self.Node(left)
        # edge = self.Edge(left_node, right_node, (left_side, right_side))
        self.graph[left.id].add(right.id)
        self.graph[right.id].add(left.id)

    def find_pdath(self, start: 'Tile', end: 'Tile'):
        seen = set()
        self.potential_paths = []
        self.shortest = []
        # self.v =

        current_path = []

        start = self.Node(start)

        self._rfind(start, end.id, seen, current_path)
        print(f'Returning: {self.shortest}')
        return self.shortest

    def _add_path(self, path, t='dfs'):
        temp = deepcopy(path)
        # path = reduce(lambda x, y: x if len(x) < len(y) else y, self.potential_paths)
        if len(temp) < len(self.shortest) or not self.shortest:
            self.shortest = temp

    def minDistance(self, dist, queue):
        # Initialize min value and min_index as -1
        minimum = float("Inf")
        min_index = -1

        # from the dist array,pick one which
        # has min value and is till in queue
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index


    # def bfs(self, start, target):
    #     s = self.Node(start)
    #     return self._bfs(s, target)

    def _bfs(self, src, dest, pred, dist):
        # a queue to maintain queue of vertices whose
        # adjacency list is to be scanned as per normal
        # DFS algorithm
        queue = []

        # boolean array visited[] which stores the
        # information whether ith vertex is reached
        # at least once in the Breadth first search
        visited = defaultdict(None)

        # initially all vertices are unvisited
        # so v[i] for all i is false
        # and as no path is yet constructed
        # dist[i] for all i set to infinity
        for i in self.graph:
            dist[i] = 1000000
            pred[i] = -1

        # now source is first to be visited and
        # distance from source to itself should be 0
        visited[src] = True
        dist[src] = 0
        queue.append(src)

        # standard BFS algorithm
        while queue:
            u = queue.pop(0)
            # for i in range(len(self.graph[u])):
            for vertex in self.graph[u]:

                if vertex not in visited:
                    visited[vertex] = True
                    dist[vertex] = dist[u] + 1
                    pred[vertex] = u
                    queue.append(vertex)

                    # We stop BFS when we find
                    # destination.
                    if (vertex == dest):
                        return True

        return False


    # utility function to print the shortest distance
    # between source vertex and destination vertex
    def bfs(self, s, dest):
        s = s.id
        dest = dest.id
        # predecessor[i] array stores predecessor of
        # i and distance array stores distance of i
        # from s
        pred = {i: 0 for i in self.graph}
        dist = {v: 0 for v in self.graph}

        if not self._bfs(s, dest, pred, dist):
            print("Given source and destination are not connected")

        # vector path stores the shortest path
        path = []
        crawl = dest
        crawl = dest
        path.append(crawl)

        while pred[crawl] != -1:
            path.append(pred[crawl])
            crawl = pred[crawl]

        # distance from source is in distance array
        print("Shortest path length is : " + str(len(path)), end='')

        # printing path from source to destination
        print("\nPath is : : ")

        for i in range(len(path) - 1, -1, -1):
            print(path[i], end=' ')

        return path


    def _rfind(self, node: 'Node', target: int, seen: set, current_path: list):
        found = False
        if node.id in seen or node in current_path:
            return False

        seen.add(node.id)
        current_path.append(node)

        if node.id == target:
            print(current_path)
            self._add_path(current_path)

            current_path.pop()
            if node.id in seen:
                seen.remove(node.id)
            return True

        for vertex in self.graph[node.id]:
            # print(f'{vertex.id} ', end='')
            # if self.shortest and len(current_path) > len(self.shortest):
            #     break
            if vertex != current_path[-1]:
                found = self._rfind(vertex, target, seen, current_path)
            # if found:
            #     current_path.pop()
            #     return
            if found or self.shortest and len(current_path) > len(self.shortest):
                break


        if node.id in seen:
            seen.remove(node.id)
        if current_path:
            current_path.pop()


class Board:

    border_size = 1
    _board: np.array
    def side(self, side):

        lookup = {
            NORTH: self.north,
            WEST: self.west,
            SOUTH: self.south,
            EAST: self.east
        }
        return lookup[side]

    def __init__(self, arr):
        self._board = []

        if isinstance(arr, np.ndarray):
            self._board = arr
        else:
            self._board = np.array(arr)

        self.height = self.board.shape[0] - self.border_size * 2
        self.width = self.board.shape[1] - self.border_size * 2

    @property
    def board(self) -> np.array:
        return self._board

    @board.setter
    def board(self, value):
        self._board = value

    def _align_edge_tile(self, tile, null_sides):
        target_sides = [tile.nulled_sides[side] for side in null_sides]

        while not all(target_sides):
            tile.clockwise()
            target_sides = [tile.nulled_sides[side] for side in null_sides]

    def align_edge_null_side(self):
        corner1 = self.constrained_board[0,0]
        corner2 = self.constrained_board[0, -1]
        corner3 = self.constrained_board[-1, -1]
        corner4 = self.constrained_board[-1, 0]

        self._align_edge_tile(corner1, (NORTH, WEST))
        self._align_edge_tile(corner2, (NORTH, EAST))
        self._align_edge_tile(corner3, (EAST, SOUTH))
        self._align_edge_tile(corner4, (SOUTH, WEST))

        for i in range(1, len(self.constrained_board[0]) - 1):
            self._align_edge_tile(self.constrained_board[0,i], (NORTH,))
            self._align_edge_tile(self.constrained_board[-1,i], (SOUTH,))

        for i in range(1, len(self.constrained_board[0]) -1):
            self._align_edge_tile(self.constrained_board[i,0], (WEST,))
            self._align_edge_tile(self.constrained_board[i,-1], (EAST,))

    def orient_tiles(self):
        def get_sides():
            target_edges = []
            for i, d in enumerate(directions):
                truth = []
                d_tile = mtx_filter[d]

                if d_tile is None:
                    target_edges.append(None)
                else:
                    found = curr_cell.cmpr(d_tile, i, flipped=True)
                    if found and ((i + 2) % 4 == found[0] or ((i - 2) % 4 == found[0])):
                        target_edges.append(True)
                    elif not found:
                        target_edges.append(False)
                    else:
                        target_edges += (curr_cell.cmpr(d_tile, i, flipped=True))
            return target_edges
        aligned = [False] * self.constrained_board.size

        while not all(aligned) and not self.check_alignments():
            idx = -1
            for row in range(1, self.board.shape[0] - 1):
                for col in range(1, self.board.shape[0] - 1):
                    idx += 1
                    directions = ((0,1), (1,-1), (-1,1), (1,0))

                    curr_cell: 'Tile' = self.board[row, col]
                    mtx_filter = self.board[row-1:row+2,col-1:col+2]
                    print(mtx_filter.shape)
                    target_edges = []
                    if curr_cell.locked:
                        continue

                    for i, d in enumerate(directions):
                        truth = []
                        d_tile = mtx_filter[d]


                        if d_tile is None:
                            target_edges.append(None)
                        else:
                            found = curr_cell.cmpr(d_tile, i, flipped=True)
                            if found and ((i+2)%4 == found[0] or ((i-2)%4 == found[0])):
                                target_edges.append(True)
                            elif not found:
                                target_edges.append(False)
                            else:
                                target_edges += (curr_cell.cmpr(d_tile, i, flipped=True))


                    print(curr_cell.id)
                    # print(target_edges)

                    truth = map(lambda x: False if x == False or isinstance(x, int) else True, target_edges)
                    if all(truth):
                        print(f'{curr_cell.id} is aligned')
                        curr_cell.locked = True
                        aligned[idx] = True
                    else:
                        aligned[idx] = False

                        if None not in target_edges:
                            if not any(target_edges):
                                curr_cell.clockwise()
                            else:
                                print('test')


                        else:
                            null_indices = [i for i,v in enumerate(target_edges) if v is None]

                            if len(null_indices) == 1:
                                print('flipping an edge piece')
                                if set(null_indices) & {NORTH, SOUTH}:
                                    curr_cell.flip(1)
                                else:
                                    curr_cell.flip(0)
                            elif len(null_indices) > 1:
                                print('flipping a corner piece')
                                if set(null_indices).issuperset({NORTH, WEST} or set(null_indices).issuperset({SOUTH, EAST})):
                                    curr_cell.clockwise()
                                    curr_cell.flip(1)
                                elif set(null_indices).issuperset({NORTH,EAST}) or set(null_indices).issuperset({SOUTH, WEST}):
                                    curr_cell.clockwise()
                                    curr_cell.flip(0)
                                test123 = get_sides()
                                print('test')



    def check_alignments(self):
        for row in range(1, self.board.shape[0]-1):

            for col in range(1 ,self.board.shape[1] - 1):
                curr_cell: 'Tile' = self.board[row, col]
                cell_to_right = self.board[row, col + 1]
                cell_below = self.board[row + 1, col]

                if 0 < col < self.board.shape[1] - 2 and not curr_cell.cmpr(cell_to_right, EAST, WEST):
                    matches = curr_cell.cmpr(cell_to_right, EAST)
                    return False
                if 0 < row < self.board.shape[0] - 2 and not curr_cell.cmpr(cell_below, SOUTH, NORTH):
                    return False
        return True

    def get_tile(self, tile_id):
        for row in self.constrained_board:
            for tile in row:
                if tile.id == tile_id:
                    return tile

    @property
    def sides(self):
        sides = [self.north, self.east, self.south, self.west]
        return sides

    @property
    def north(self):
        return self.board[np.newaxis, 0, :]

    @property
    def east(self):
        return self.board[np.newaxis, :, -1].T

    @property
    def south(self):
        return self.board[np.newaxis, -1,:]

    @property
    def west(self):
        return self.board[np.newaxis, :, 0].T

    @property
    def constrained_board(self):
        return self.board[self.border_size:-self.border_size, self.border_size: -self.border_size]

    def __str__(self):
        return str(self.board)

class Tile(Board):

    def __init__(self, tile):
        raw_tile = tile[1:]
        self.id = tile[0].split(' ')[-1].split(':')[0]
        two_d_tile = [list(row) for row in raw_tile]
        self._locked = False
        self._nulled_sides = defaultdict(int)
        self.corner = False
        self.edge_piece = False
        super(Tile, self).__init__(two_d_tile)


    def __str__(self):
        return str(f'{self.id}\n {self.board}')

    def __repr__(self):
        return self.__str__()


    def cmpr(self, other: 'Tile', side_enum, target=None, flipped=False):

        if not other:
            return []

        this_side = self.side(side_enum)
        if this_side.shape[0] != 1:
            this_side = this_side.T

        if target:
            other_side = other.side(target)
            return this_side.shape == other_side.shape and np.array_equal(this_side, other_side)

        other_sides = [side if side.shape[0] == 1 else side.T for side  in other.sides ]
        matches = []



        for i, other_side in enumerate(other_sides):
            if this_side.shape == other_side.shape and  (np.array_equal(this_side, other_side) or flipped and np.array_equal(this_side, np.fliplr(other_side))):
                matches.append(i)

        return matches

    def null_side(self, side):
        sides = [np.reshape(side, side.size).tolist() for side in self.sides]
        idx = sides.index(side)

        self._nulled_sides[idx] = True

    @property
    def nulled_sides(self):
        return self._nulled_sides

    @property
    def locked(self):
        return self._locked

    @locked.setter
    def locked(self, value):
        self._locked = value

    def clockwise(self):
        self.board = np.rot90(self.board, axes=(1,0))
        temp = deepcopy(self._nulled_sides)
        for k in range(4):
            self._nulled_sides[k] = temp[(k - 1) % 4]


    def counter_clockwise(self, d=None):
        self.board = np.rot90(self.board)
        temp = deepcopy(self._nulled_sides)
        for k in range(4):
            self._nulled_sides[k] = temp[(k + 1) % 4]


    def flip(self, axis=0):
        if axis == 0:
            self.board = np.flipud(self.board)
            t = self._nulled_sides[NORTH]
            self._nulled_sides[NORTH] = self._nulled_sides[SOUTH]
            self._nulled_sides[SOUTH] = t
        else:
            self.board = np.fliplr(self.board)
            t = self._nulled_sides[WEST]
            self._nulled_sides[WEST] = self._nulled_sides[EAST]
            self._nulled_sides[EAST] = t

    def randomize(self):
        chs = random.randint(1,3)
        if chs < 3:
            self.flip(chs)

        if chs < 3:
            if chs == 1:
                self.clockwise()
            else:
                self.counter_clockwise()



def arrange(board: np.array, tiles: tuple, randomize=False):
    tile_idx = 0



    for i in range(1, len(board) - 1):
        for j in range(1, len(board)-1):
            board[i,j] = tiles[tile_idx]
            tile_idx += 1

            if randomize:
                board[i,j].randomize()

def arrange_test_answer(board, tiles):
    copy_tiles = [tile for tile in tiles]

    while copy_tiles:
        tile = copy_tiles[0]
        tile_num = int(tile.id)

        if tile_num == 1951:
            board[1,1] = tile
            board[1,1].flip()
        elif tile_num == 2311:
            tile.flip()
            board[1,2] = tile

        elif tile_num == 3079:
            board[1,3] = tile
        elif tile_num  == 2729:
            tile.flip()
            board[2,1] = tile
        elif tile_num == 1427:
            tile.flip()
            board[2,2] = tile
        elif tile_num == 2473:
            tile.clockwise()
            tile.flip(0)
            board[2,3] = tile
        elif tile_num == 2971:
            tile.flip()
            board[3,1] = tile
        elif tile_num == 1489:
            tile.flip()
            board[3,2] = tile
        else:
            tile.flip(1)
            board[3,3] = tile

        copy_tiles.pop(0)


def find_edges_without_matches(all_edges):
    unmatched = []
    for i, edge in enumerate(all_edges):
        edge_view = all_edges[:i] + all_edges[i +1:]
        if edge not in edge_view and list(reversed(edge)) not in edge_view:
            unmatched.append(edge)
    return unmatched

def find_corner_tiles(unmatched, board: Board):
    corners = []
    side_piece = []
    for row in board.constrained_board:
        for tile in row:
            edge_count = []
            for side in tile.sides:
                s = np.reshape(side, side.size).tolist()
                if s in unmatched or list(reversed(s)) in unmatched:
                    edge_count.append(s)
            if len(edge_count) >= 2:
                corners.append(tile)
                [tile.null_side(a)  for a in edge_count]
            elif len(edge_count) == 1:
                side_piece.append(tile)
                [tile.null_side(a)  for a in edge_count]
    return corners, side_piece


def find_common_edges(board: Board, unique=None):
    tile_lookup = {}
    for row in board.constrained_board:
        for tile in row:
            tile_lookup[tile.id] = tile

    unique_edges = unique or []
    tile_edge_table = defaultdict(list)
    all_edges = []
    if not unique_edges:
        for row in board.constrained_board:
            for tile in row:
                sides = tile.sides
                for side in sides:
                    t = side.tolist()[0]
                    if side.shape[0] > 1:
                        t = side.T.tolist()[0]

                    all_edges.append(t)
                    if t not in unique_edges and list(reversed(t)) not in unique_edges:
                        unique_edges.append(t)
        print('unique')
        # pprint(unique_edges)
        print(f'total edges: {board.width * board.height * 4} unique edges: {len(unique_edges)}')

    unmatched = find_edges_without_matches(all_edges)
    print(unmatched)

    corners, side_pieces = find_corner_tiles(unmatched, board)

    # print(reduce(lambda x,y: x * y, [int(tile.id) for tile in corners]))

    for row in board.constrained_board:
        for tile in row:
            for s in tile.sides:
                if s.shape[0] > 1:
                    s = s.T

                tile_edge_table[tile.id].append(get_edge_id(unique_edges, s.tolist()[0]))

    # pprint(tile_edge_table)
    # print(tabulate(board.board))
    graph = Graph(len(tile_lookup.keys()))
    overlaps = defaultdict(list)
    for i in range(len(tile_edge_table)):
        tile_id = list(tile_edge_table.keys())[i]
        edge_ids = tile_edge_table[tile_id]

        for j in range(i + 1, len(tile_edge_table)):
            other_tile_id = list(tile_edge_table.keys())[j]
            other_edge_ids = tile_edge_table[other_tile_id]
            if tile_id != other_tile_id:
                overlap = list(filter(lambda x: x in other_edge_ids, edge_ids))
                overlap1 = [(tile_edge_table[tile_id].index(curr_edge), tile_edge_table[other_tile_id].index(curr_edge), curr_edge) for curr_edge in overlap]
                for curr_edge in overlap:
                    touching_edges = (tile_edge_table[tile_id].index(curr_edge), tile_edge_table[other_tile_id].index(curr_edge))
                    graph.add_edge(tile_lookup[tile_id], tile_lookup[other_tile_id])
                    # graph.add_edge(tile_lookup[other_tile_id], tile_lookup[tile_id])
                matched_side = overlap1[1:]
                overlap2 = [(tile_edge_table[other_tile_id].index(curr_edge), tile_edge_table[tile_id].index(curr_edge), curr_edge) for curr_edge in overlap]
                if overlap:
                    overlaps[tile_id].append((other_tile_id, overlap1))
                    overlaps[other_tile_id].append((tile_id, overlap2))
                    # print(f'{tile_id} {other_tile_id} overlap: {overlap}')

    pprint(overlaps)
    return unique_edges, graph, corners, side_pieces, tile_lookup, overlaps

def sides_aligned(l_enum, r_enum):
    sides = [l_enum, r_enum]

    return NORTH in sides and SOUTH in sides or WEST in sides and EAST in sides



def align(tile1: Tile, tile2: Tile, side1, side2):

    which = random.randint(0,1)
    while not sides_aligned(side1, side2):

        if which:
            tile1.clockwise()
            side1 = (side1 + 1) % 4
        elif not tile2.locked:
            tile2.clockwise()
            side2 = (side2 +1) % 4


        if sides_aligned(side1, side2) and side2 not in tile1.cmpr(tile2, side1):
            if side1 in [NORTH, SOUTH]:
                if which:
                    tile1.flip(1)
                else:
                    tile1.flip(1)
            else:
                if which:
                    tile1.flip(0)
                else:
                    tile2.flip(0)

def align_tile_rotation(board, unique_edges, overlaps):
    rotations = 0
    aligned = False
    pprint(overlaps)
    while not aligned:
        aligned = True
        for tile_id in overlaps:
            connections = overlaps[tile_id]

            for connection in connections:
                other_tile_id = connection[0]
                faces = connection[1]
                if not sides_aligned(*faces[0][:-1]):
                    align(board.get_tile(tile_id), board.get_tile(other_tile_id), *faces[0][:-1])
                    rotations += 1
                    aligned = False
        unique_edges, x, corners, side_pieces, tile_lookup, overlaps = find_common_edges(board, unique_edges)

    print('Tile rotations complete')
    print(f'{rotations} tiles rotated')

def get_edge_id(unique_edges: list, edge):
    try:
        idx = unique_edges.index(edge)
    except ValueError:
        idx = unique_edges.index(list(reversed(edge)))
    return idx

def apply_corners(board, corners: List[Tile]):
    arr = board.constrained_board

    first = corners.pop()
    while not first.nulled_sides.get(NORTH) or not first.nulled_sides.get(WEST):
        first.clockwise()
    arr[0,0] = first

    second = corners.pop()
    while not second.nulled_sides.get(NORTH) or not second.nulled_sides.get(EAST):
        second.clockwise()
    arr[0,-1] = second


    third = corners.pop()
    while not third.nulled_sides.get(SOUTH) or not third.nulled_sides.get(EAST):
        third.clockwise()
    arr[-1,-1] = third

    fourth = corners.pop()
    while not fourth.nulled_sides.get(SOUTH) or not fourth.nulled_sides.get(WEST):
        fourth.clockwise()
    arr[-1, 0] = fourth




    # def _rfind(self, curr, target):





tiles = [Tile(tile) for tile in tiles]
count = len(tiles)

edge = int(math.sqrt(count))

board = np.ndarray(shape=(edge+2,edge+2), dtype='object')

board = Board(board)
arrange(board.board, tiles)
# arrange_test_answer(board.board, tiles)
# print(board.check_alignments())
# exit()

# arrange(board.board, tiles)

print("Brute forcing tile placement")
iteration = 1
perms = []

unique, graph, corners, side_pieces, tile_lookup, overlaps = find_common_edges(board)

for corner in corners:
    corner.corner = True
board_copy = deepcopy(board)
for side in side_pieces:
    side.side_piece = True
apply_corners(board, corners)

# print(tabulate(board.board))


corners = ((0,0), (0,-1), (-1,-1), (-1,0))
paths = []

# for i in range(len(corners)):


corner_swaps = []
for i in range(len(corners)):
    idx1 = corners[i]
    idx2 = corners[(i+1)%4]
    corner1 = board.constrained_board[idx1]
    corner2 = board.constrained_board[idx2]
    print(f'{corners[i]} {corners[(i+1)%4]}')
    print(f'Finding path from {corner1.id} to {corner2.id}')


    path = graph.bfs(corner1, corner2)
    # path = graph.bfs(corner1, corner2)
    # print(f'visited: {visited}')
    # path = visited[:visited.index(corner2.id)]
    # exit()
    # path = graph.find_path(corner1, corner2)

    paths.append([tile_lookup[t_id] for t_id in path])
    pprint(path)

    if len(path) > board.width:
        corner_swaps.append((idx1,idx2))
# exit(())

print(tabulate(board.board))
if corner_swaps:
    first = corner_swaps[0][0]
    second = corner_swaps[1][1]
    temp = board.constrained_board[first]
    board.constrained_board[first] = board.constrained_board[second]
    board.constrained_board[second] = temp
pprint(paths)


left_side = graph.bfs(board.constrained_board[0,0], board.constrained_board[-1, 0])
right_side = graph.bfs(board.constrained_board[0,-1], board.constrained_board[-1,-1])

for i in range(1, len(left_side) - 1):
    board.constrained_board[i,0] = tile_lookup[left_side[i]]
    board.constrained_board[i,-1] = tile_lookup[right_side[i]]

for i, row in enumerate(board.constrained_board):
    path = graph.bfs(row[0], row[-1])
    for j in range(1, len(row) - 1):
        print(f'Path: {path}')
        board.constrained_board[i,j] = tile_lookup[path[j]]

# align_tile_rotation(board, unique, overlaps)
# exit()


board.align_edge_null_side()
print(tabulate(board.board))
board.orient_tiles()

print(tabulate(board.constrained_board))

bit_mask = get_monster_bit_mask()




# undivided_board = np.array([[col for col in row] for row in board.constrained_board ])
dims = (board.board[1,1].width * board.constrained_board.shape[0], board.board[1,1].width * board.constrained_board.shape[0])
undivided_board = np.ndarray(dims)


# Remove edges
side_len = board.constrained_board[0,0].board.shape[0] - 2
new_side_len = side_len * board.constrained_board.shape[0]
new_matrix = np.ndarray((new_side_len, new_side_len), dtype='object')
resized_matrix = np.ndarray((new_side_len, new_side_len), dtype='object')

# new_row = []

def convert_to_bits(x):
    return x == '#'
vfunc = np.vectorize(convert_to_bits)

for r, row in enumerate(board.constrained_board):
    new_row = np.ndarray((side_len,side_len * len(row)), dtype='bool')
    resized_row = np.ndarray((side_len,side_len * len(row)), dtype='object')
    offset = side_len
    for i, tile in enumerate(row):
        tile.board = tile.board[1:-1,1:-1]
        resized_row[:,i * offset: i * offset + offset] = tile.board
        new_row[:,i * offset: i * offset + offset] = vfunc(tile.board)



    new_matrix[r * offset: r*offset + offset,:] = new_row
    resized_matrix[r * offset: r*offset + offset,:] = resized_row




print(tabulate(new_matrix))
print(tabulate(resized_matrix))
# f    # np.vstack(undivided_board, new_row)
print(undivided_board.shape)


mask_width = bit_mask.shape[1]
mask_height = bit_mask.shape[0]

total_roughness = 0
monster_count = 0

def nop(*args):
    pass

print(f'Checking board alignements: {board.check_alignments()}')
# exit()
# for fn in (nop,) + (np.rot90,) * 3 +  (np.fliplr,)  + (np.rot90,) * 3 :
# fn(new_matrix)
for i in range(new_matrix.shape[0] - bit_mask.shape[0]+ 1):
    for j in range(new_matrix.shape[1] - bit_mask.shape[1] + 1):
        mask_area = new_matrix[i: mask_height + i, j: mask_width + j]

        # print(tabulate((bit_mask)))
        # print(tabulate(mask_area))
        masked = np.bitwise_and(bit_mask, mask_area)
        if np.equal(masked, bit_mask).all():
            roughness = np.bitwise_and(mask_area, np.bitwise_not(bit_mask))
            print(tabulate(resized_matrix[i: mask_height + i, j: mask_width + j]))
            print(f"Found a monster! Top left corner at {i, j}")
            print(tabulate(masked))
            print('masked')
            total_roughness += np.sum(roughness)
            monster_count += 1
            print(tabulate(roughness))
            print('roughness')

            # mask_area = roughness
            new_matrix[i: mask_height + i, j: mask_width + j] = roughness
            print(tabulate(mask_area))

final_count = 0
for r in new_matrix:
    for c in r:
        if c:
            final_count += 1


print(f'Found {monster_count} monsters ')
print(f'Total roughness: {final_count}')
exit()
while not board.check_alignments():
    randomize = False


    align_tile_rotation(board,unique,overlaps)
    iteration += 1

print(tabulate(board.board))
print(f'Found working configuration after flailing around for {iteration} passes')
