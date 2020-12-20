import re
from collections import defaultdict


def get_rules():
    global test
    if test:
        return \
"""42: 9 14 | 10 1
9: 14 27 | 1 26
10: 23 14 | 28 1
1: "a"
11: 42 31
5: 1 14 | 15 1
19: 14 1 | 14 14
12: 24 14 | 19 1
16: 15 1 | 14 14
31: 14 17 | 1 13
6: 14 14 | 1 14
2: 1 24 | 14 4
0: 8 11
13: 14 3 | 1 12
15: 1 | 14
17: 14 2 | 1 7
23: 25 1 | 22 14
28: 16 1
4: 1 1
20: 14 14 | 1 15
3: 5 14 | 16 1
27: 1 6 | 14 18
14: "b"
21: 14 1 | 1 14
25: 1 1 | 1 14
22: 14 14
8: 42
26: 14 22 | 1 20
18: 15 15
7: 14 5 | 1 21
24: 14 1""".split('\n')

    with open('puzzle_input/day19/rules.txt') as f:
        return [line.strip() for line in f]

def get_messages():
    global test
    if test:
        return """abbbbbabbbaaaababbaabbbbabababbbabbbbbbabaaaa
bbabbbbaabaabba
babbbbaabbbbbabbbbbbaabaaabaaa
aaabbbbbbaaaabaababaabababbabaaabbababababaaa
bbbbbbbaaaabbbbaaabbabaaa
bbbababbbbaaaaaaaabbababaaababaabab
ababaaaaaabaaab
ababaaaaabbbaba
baabbaaaabbaaaababbaababb
abbbbabbbbaaaababbbbbbaaaababb
aaaaabbaabaaaaababaa
aaaabbaaaabbaaa
aaaabbaabbaaaaaaabbbabbbaaabbaabaaa
babaaabbbaaabaababbaabababaaab
aabbbbbaabbbaaaaaabbbbbababaaaaabbaaabba""".split('\n')
    with  open('puzzle_input/day19/messages.txt') as f:
        return [line.strip() for line in f]

def update_rules(rules):
    for i in range(len(rules)):
        rule = rules[i]
        if rule[:2] == '8:':
            rules[i] = '8: 42 +'
        elif rule[:3] == '11:':
            # rules[i] = '11: 42 31 | 42 11 31'
            rules[i] = '11: 42 31 +'
    return rules

class DiGraph:

    class _Vertex:
        def __init__(self, number, dependencies: list, value=None):
            self._rule_number = number
            self.dependencies = dependencies # list, each item is an or rule
                                            # 0: 4 1 5 -> ['4 1 5']
                                            # 1: 2 3 | 3 2 -> ['2 3', '3 2']
            self.value = value
            self._repeating = False

        def _inject_dependencies(self, resolved_dependencies):


            injected_dependencies = []
            for dep in self.dependencies:

                temp =''
                for req in dep.split(' '):
                    for rule_number in resolved_dependencies:
                        if str(rule_number) == req:
                            temp += resolved_dependencies[rule_number]
                            if self.rule_number == '11':
                                temp += '{5}'
        # Incrementing the repetition value from 1-5 allowed me to sum the actual matches.
        # nearly fucked myself writing a Regex generator for a language that can be described by CFGs but not regular language.
        # 331 + 34 + 6 + 3
                injected_dependencies.append(temp)
            return injected_dependencies

        def resolve(self, resolved_dependencies: dict):

            injected = self._inject_dependencies(resolved_dependencies)


            resolution = "|".join(injected)

            if self.repeating and self.rule_number != '11':
                resolution = f'({resolution}+)'
            else:
                resolution = f'({resolution})'

            return resolution


        @property
        def repeating(self):
            return self._repeating

        @repeating.setter
        def repeating(self, value):
            self._repeating = value

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
        repeating = False
        rule_dependencies = []
        for d in dependencies:
            temp = []
            if '+' in d:
                d= d.replace(' +', '')
                # temp = ['+']
                vertex.repeating = True


            rule_dependencies += [int(num) for num in d.strip().split(' ')]
            rule_dependencies += temp


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
            if neighbor_num != '+':
                neighbor = self.vertices[neighbor_num]
                resolutions[neighbor_num] = self._resolve(neighbor)
            else:
                resolutions[neighbor_num] = '+'


        resolved_string = vertex.resolve(resolutions)
        print(f'{vertex} resolves to : {resolved_string}')
        return resolved_string

test = False

messages = get_messages()
rules = get_rules()

rules = update_rules(rules)

print(f'{len(messages)=}')
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

    if match:
        count += 1

print(f"total matches: {count}")

