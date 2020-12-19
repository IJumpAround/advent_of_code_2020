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

values = load()


for i, value in enumerate(values[preamble_length:]):
    preamble, nums = split_values(values[i:])
    combos = sum_preamble(preamble)

    if value not in combos:
        print(value)
        break






