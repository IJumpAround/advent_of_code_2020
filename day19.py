import datetime
import re
from collections import defaultdict

start = datetime.datetime.now()
def get_rules():
    global test
    if test:
        return \
"""0: 4 1 5
1: 2 3 | 3 2
2: 4 4 | 5 5
3: 4 5 | 5 4
4: "a"
5: "b" """.split('\n')

    with open('puzzle_input/day19/rules.txt') as f:
        return [line.strip() for line in f]

def get_messages():
    global test
    if test:
        return """ababbb
bababa
abbbab
aaabbb
aaaabbb""".split('\n')
    with  open('puzzle_input/day19/messages.txt') as f:
        return [line.strip() for line in f]



class DiGraph:

    class _Vertex:
        def __init__(self, number, dependencies: list, value=None):
            self._rule_number = number
            self.dependencies = dependencies # list, each item is an or rule
                                            # 0: 4 1 5 -> ['4 1 5']
                                            # 1: 2 3 | 3 2 -> ['2 3', '3 2']
            self.value = value

        def _inject_dependencies(self, resolved_dependencies):

            injected_dependencies = []
            for dep in self.dependencies:

                temp =''
                for req in dep.split(' '):
                    for rule_number in resolved_dependencies:
                        if str(rule_number) == req:
                            temp += resolved_dependencies[rule_number]

                injected_dependencies.append(temp)
            return injected_dependencies

        def resolve(self, resolved_dependencies: dict):
            injected = self._inject_dependencies(resolved_dependencies)

            resolution = "|".join(injected)
            resolution = f'({resolution})'

            return resolution


        @property
        def rule_number(self):
            return self._rule_number

        @rule_number.setter
        def rule_number(self, value):
            self._rule_number = value

        @property
        def terminal(self):
            return self.value

        def __str__(self):
            return f'Rule: {self.rule_number} = {(self.dependencies or self.terminal)}'


    class _Edge:
        def __init__(self, tail: int, head: int):
            self.tail = tail
            self.head = head


    def __init__(self):
        self.graph = defaultdict(set)
        self.vertices = {}




    def _parse_rule(self, rule: str):
        rule_number = rule[:rule.index(':')]
        terminal = None
        dependencies = []
        if '"' in rule:
            terminal = rule[rule.find('"')+1: rule.rfind('"')]
        else:
            dependencies = rule[rule.index(':')+1:].strip().split('|')

        return self._Vertex(rule_number, dependencies, terminal)

    def _parse_vertex_dependencies(self, vertex: 'DiGraph._Vertex'):
        dependencies = vertex.dependencies

        rule_dependencies = []
        for d in dependencies:
            rule_dependencies += [int(num) for num in d.strip().split(' ')]

        return rule_dependencies

    def add_edge(self, rule: str):
        vertex = self._parse_rule(rule)
        self.vertices[int(vertex.rule_number)] = vertex
        heads = self._parse_vertex_dependencies(vertex)

        for head in heads:
            self.graph[vertex.rule_number].add(head)


    def resolve(self, vertex: int):
        vertex = self.vertices[vertex]
        return self._resolve(vertex)


    def _resolve(self, vertex: 'DiGraph._Vertex'):
        # self.seen.add(vertex.rule_number)
        if vertex.terminal:
            return vertex.terminal

        resolutions = {}
        vertex_neighbors = self.graph[vertex.rule_number]
        for neighbor_num in vertex_neighbors:
            neighbor = self.vertices[neighbor_num]
            resolutions[neighbor_num] = self._resolve(neighbor)


        resolved_string = vertex.resolve(resolutions)
        print(f'{vertex} resolves to : {resolved_string}')
        return resolved_string

test = False

messages = get_messages()
rules = get_rules()


graph = DiGraph()

for rule in rules:
    graph.add_edge(rule)

print(graph.graph)

regex = graph.resolve(0)
print(f"Resolved regex: {regex}")
count = 0

regex = re.compile(regex)

print(messages)
for message in messages:

    match = regex.fullmatch(message)

    if match and match.string == message:
        count += 1

print(f"total matches: {count}")
print(f'Done in {datetime.datetime.now() - start}')
