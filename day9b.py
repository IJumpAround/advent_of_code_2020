import sys

target = 23278925
test_input = """35
20
15
25
47
40
62
55
65
95
102
117
150
182
127
219
299
277
309
576"""

preamble_length = 25


def load(test=False):
    if test:
        global preamble_length
        global target
        target = 127
        preamble_length = 5
        return [int(i) for i in test_input.split('\n')]
    else:
        with open('puzzle_input/day9/input.txt') as f:
            return [int(line.strip()) for line in f]


def split_values(values):
    return values[:preamble_length], values[preamble_length:]


def sum_preamble(preamble):
    sums = []
    for i in range(len(preamble)):
        for j in range(len(preamble)):
            sums.append(preamble[i] + preamble[j])
    return sums


values = load(False)

for i, value in enumerate(values[preamble_length:]):
    preamble, nums = split_values(values[i:])
    combos = sum_preamble(preamble)

    if value not in combos:
        print(value)
        break



contiguous = []


def contig(numbers, target):
    trying = list()
    return _contig(numbers, target, trying)


def _contig(numbers, target, trying):
    for i, num in enumerate(numbers):
        trying.append(num)
        s = sum(trying)
        if s == target:
            return min(trying) + max(trying)
        elif s > target:
            trying= []
            return _contig(numbers[1:], target, trying)


sys.setrecursionlimit(2000)

answer = contig(values, target)
print(answer)