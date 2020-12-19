from collections import defaultdict
from pprint import pprint

rules = """light red bags contain 1 bright white bag, 2 muted yellow bags.
dark orange bags contain 3 bright white bags, 4 muted yellow bags.
bright white bags contain 1 shiny gold bag.
muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.
shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.
dark olive bags contain 3 faded blue bags, 4 dotted black bags.
vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
faded blue bags contain no other bags.
dotted black bags contain no other bags."""
import re

f = open('puzzle_input/day7/input.txt')
rules = [line.strip() for line in f]
f.close()


bag_regex = re.compile('(\w+ \w+) bag contain')
contained_bag_regex = re.compile('bag contain (\d \w+ \w+ bag )+')



adjacency = defaultdict(set)
edges = []
def parse_rule(rule):
    rule = rule.replace(',', '.')
    rule = rule.replace('bags','bag')
    container_match = bag_regex.search(rule)
    container = container_match.groups()[0]

    contained = rule[container_match.end():-1].split('.')

    for c in contained:
        if "no other bag" not in c:
            edges.append((container, c.strip()))

def create_adjacency_list(edges):

    for edge in edges:
        outer = edge[0]
        inner = edge[1]

        s = adjacency.setdefault(outer, set())

        for i in inner.split('.'):
            s.add(i)

def create_reverse_adjacency_list(edges):
    for edge in edges:
        outer = edge[0]
        inner = edge[1]


        for i in inner.split('.'):
            temp = " ".join(i.split(' ')[1:-1])
            s = adjacency.setdefault(temp.strip(), set())
            s.add(outer)

target = "shiny gold"

class Graph:

    def __init__(self, graph):
        self.graph = graph
        self._seen = set()

    def search(self, vertex=target):
        seen = set()

        self._dfs(vertex)


    def _dfs(self, vertex):

        self._seen.add(vertex)

        for neighbor in self.graph[vertex]:
            if neighbor not in self._seen:
                self._seen.add(neighbor)
                self._dfs(neighbor)

    @property
    def visited(self):
        return self._seen




for rule in rules:
    parse_rule(rule)
create_adjacency_list(edges)
create_reverse_adjacency_list(edges)
# for rule in rules:
graph = Graph(adjacency)
print(adjacency)
print(f'{edges=}')

pprint(adjacency)

graph.search(target)
parents = graph.visited

pprint(f'{parents=}')
pprint(parents)
print(len(parents) -1)
