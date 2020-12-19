import re
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
dotted black bags contain no other bags.""".split('\n')


rules = """shiny gold bags contain 2 dark red bags.
dark red bags contain 2 dark orange bags.
dark orange bags contain 2 dark yellow bags.
dark yellow bags contain 2 dark green bags.
dark green bags contain 2 dark blue bags.
dark blue bags contain 2 dark violet bags.
dark violet bags contain no other bags.""".split('\n')

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

    def search(self, vertex=target):
        seen = set()

        bag_count = self._dfs(vertex)
        print('bag count: ', bag_count)


    def _dfs(self, vertex):

        # self._seen.add(vertex)
        bag_count = 0
        for neighbor in self.graph[vertex]:
            n = neighbor.split(' ')
            num_neighbor_bags = int(n[0])
            neighbor_bag = " ".join(n[1:-1])

            # if not neighbor_bag in self._seen:
            # bag_count += count
            # self._seen.add(neighbor_bag)
            neighbor_bag_contains = self._dfs(neighbor_bag)

            if neighbor_bag_contains == 1:
                bag_count += neighbor_bag_contains * num_neighbor_bags
            else:
                bag_count += (neighbor_bag_contains * num_neighbor_bags) + num_neighbor_bags

        return bag_count or 1


for rule in rules:
    parse_rule(rule)
create_adjacency_list(edges)

graph = Graph(adjacency)
print(adjacency)
print(f'{edges=}')

pprint(adjacency)

graph.search(target)


