import time
from collections import deque
from functools import reduce

from utils import input_loader


def exist(a0, a1, b0, b1, n0, n1):
    print(f'{n0=} {n1=}')
    x = a0 * b1 * n1 + a1 * b0 * n0
    print('running existence algorithm')
    print(f'X= a0 * b1 * n1 + a1 * b0 * n0 = '
          f' {a0} * {b1} * {n1} + {a1} * {b0} * {n0} = {x}')

    orig = x
    mult = n0 * n1
    m_abs = abs(mult)
    while True:
        x_abs = abs(x)
        if m_abs - x_abs >= x_abs:
            break
        if x < 1:
            sign = m_abs > 1
        else:
            sign = m_abs < 1
        sign = int(sign) or -1
        x += (mult * sign)

    print(f'Existence reduced {orig} -> {x}')
    return x


def euc(a, b):
    """Return bezouts coefficients
    take n0,n1
    """
    print(f'bez input: {a, b}')
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = (r, old_r - quotient * r)
        old_s, s = (s, old_s - quotient * s)
        old_t, t = (t, old_t - quotient * t)

    print('bez output ', old_s, old_t)
    return old_s, old_t


def pair_algo(n0, n1, a0, a1):
    # n1 is either last bus or combined value
    # n0 is always a bus

    b0, b1 = euc(n0, n1)
    print(f'{b0=},{b1=}')
    print(f'Sanity check b0*n0 + b1*n1 = {b0} * {n0} + {b1} * {n1} = {b0 * n0 + b1 * n1}')

    a0 = exist(a0, a1, b0, b1, n0, n1)

    print('pair algo returning: ', a0)
    return a0


def get_bus_a_value(bus, original_buses):
    a = bus - original_buses.index(bus)  # puzzle algo
    # a = original_buses.index(bus)  # wikipedia example algo
    return a


def doit(last, cur):
    n0, a0 = last
    n1, a1 = cur
    s = pair_algo(n0, n1, a0, a1)
    n_prod = n0 * n1

    return n_prod, s


def calculate_unique_x(buses):
    orig_buses = buses
    active_buses = list(filter(lambda bus: bus != -1, buses))

    print(f'{orig_buses=}')
    N = reduce(lambda x, y: x * y, [bus for bus in orig_buses if bus != -1])

    print('\n\n')
    print('Pairing')
    comparable = deque(sorted(active_buses))
    pairs = deque()

    # can't divide all values into pairs, so take the largest value and have it be merged last for efficiency
    if len(comparable) % 2 == 1:
        p = comparable.pop()
        pairs.append(((p, get_bus_a_value(p, orig_buses)), None))

    # appendleft all other pairs
    while len(comparable) > 1:
        l, r = comparable.popleft(), comparable.pop()
        pairs.appendleft(((l, get_bus_a_value(l, orig_buses)),
                          (r, get_bus_a_value(r, orig_buses))))

        print(pairs)
    remaining = None

    print(f'{pairs=}')

    s = 0
    print()
    print("Beginning apply algo to pair loop")

    l_last, r_last = pairs.popleft()

    for pair in pairs:
        l, r = pair
        print('\n***Loop pair', (l, r))

        if r:
            r_last = doit(r_last, r)
        else:
            remaining = r_last

        l_last = doit(l_last, l)

    if remaining:
        print(f'\n\nRunning final pair: {remaining}')
        n, s = doit(l_last, remaining)
        r_last = (n, s)
        print()
    elif l_last and r_last:
        n, s = doit(l_last, r_last)
        l_last = (n, s)

    print(f'{l_last=}')
    print(f'{r_last=}')

    print('final s', s)
    print(f'{N=}')
    print(f'n_l * n_r {l_last[0] * r_last[0]}')

    answer = s if s > 0 else N + s
    print(f'N+s{N + s}')
    print(f'answer= {s if s > 0 else N + s}')

    return answer


def solve(sample):
    start = time.time()
    file_content = input_loader.load_file_as_list(13, sample)

    earliest_departure, bus_numbers = file_content
    # bus_numbers = '17,x,13,19'
    # 3417   =  0 (mod 17)
    # 3417   = -2 (mod 13)
    # 3417   = -4 (mod 19)
    # bus_numbers = '11,x,x,x,x,x,11,x,x,21,x,x,x,16,x,x,x,x,x,25'
    # bus_numbers = '3,x,x,4,5'
    # bus_numbers = '3,x,x,4'
    # bus_numbers = '67,7,59,61'
    # bus_numbers = '67,x,7,59,61'
    # bus_numbers = '67,7,x,59,61'
    # bus_numbers = '1789,37,47,1889'

    earliest_departure = int(earliest_departure)
    buses = [int(bus) if bus != 'x' else -1 for bus in bus_numbers.split(',')]
    active_buses = list(filter(lambda bus: bus != -1, buses))

    at_time = calculate_unique_x(buses)

    soonest_bus = None
    candidate_offset = -999
    for bus in active_buses:
        offset = earliest_departure % -bus
        if offset > candidate_offset:
            soonest_bus = bus
            candidate_offset = offset

    p1_answer = soonest_bus * -candidate_offset
    print(p1_answer)
    if sample:
        assert at_time == 1068781
        assert p1_answer == 295
    else:
        assert p1_answer == 4782
        assert at_time == 1118684865113056

    print(f"Finished in {time.time() - start}seconds")

if __name__ == '__main__':
    solve(True)
    solve(False)
