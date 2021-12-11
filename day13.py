import math
import multiprocessing
import os
import sys
import threading
import time
from collections import deque
from functools import reduce
from pathlib import Path

import numpy
import numpy as np

from utils import input_loader


# bus id indicates departure frequency
# schedules timestamp = number of minutes since ref point
# at 0 all busses leave, eventually return
# time loop = bus id number

# line1 earliest possible departure
# line2 bus ids in service
# x = out of service

def answer(bus_id: int, minutes_waiting: int):
    return bus_id * minutes_waiting

def fastest_bus(buses) -> int:
    return reduce(lambda x, y: x if x < y else y, buses)


def scheduled(current_time, bus):
    return 1 if bus == -1 else current_time % bus == 0




def exist(a0, a1, b0, b1, n0, n1):
    # print(a0,a1,b0,b1,n0,n1)
    # x =a1*m2*n2 + a2*m1*n1
    print(f'{n0=} {n1=}')
    X = a0 * b1 * n1 + a1 * b0 * n0
    print('running existence algorithm')
    print(f'X= a0 * b1 * n1 + a1 * b0 * n0 = '
          f' {a0} * {b1} * {n1} + {a1} * {b0} * {n0} = {X}')

    # N = -1
    # print(f'{X=}')
    # while N < 0:
    #    N = X + n0 * n1
    # s = X + (n0 * n1)
    # s = X
    orig = X
    mult = n0 * n1
    m_abs = abs(mult)
    while True:
        x_abs = abs(X)
        if m_abs - x_abs >= x_abs:
            break
        if X < 1:
            sign = m_abs > 1
        else:
            sign = m_abs < 1
        sign = int(sign) or -1
        X += (mult * sign)

    print(f'Existince reduced {orig} -> {X}')
    return X

def pair_algo(n0, n1, a0, a1):
    # n1 is either last bus or combined value
    # n0 is always a bus

    # print('orig: n0,n1 ', n0, n1)
    # bus = n0
    # n0 = n1 * n0
    # n1 = bus
    # print('next: n0,n1', n0, n1)
    # a0 = a1
    # a1 = bus - orig_buses.index(bus)
    # print(f'a0,a1 {a0,a1}')
    b0, b1 = euc(n0, n1)
    print(f'{b0=},{b1=}')
    print(f'Sanity check b0*n0 + b1*n1 = {b0} * {n0} + {b1} * {n1} = {b0*n0 + b1*n1}')
    # print(f'n0*b0+n1*b1={}\n')

    # assert (n0 * b0) + (n1 * b1) == 1
    a0 = exist(a0, a1, b0, b1, n0, n1)

    print('pair algo returning: ', a0)
    return a0

def get_bus_a_value(bus, original_buses):
    a = bus - original_buses.index(bus)  # puzzle algo
    # a = original_buses.index(bus)  # wikipedia example algo
    return a

def get_schedule(buses, earliest_departure):
    start = time.time()
    fastest = fastest_bus(buses)
    orig_buses = buses
    num_orig_buses = len(orig_buses)
    timestamp = earliest_departure = 100000000000000
    earliest_occurance = -1
    active_buses = list(filter(lambda bus: bus != -1, buses))
    new_buses = []
    consecutive_xs = 0
    alignment_offset = buses[0]

    final_bus = buses[-1]
    for bus in buses:
        if bus >= 0:
            if consecutive_xs != 0:
                new_buses.append(-consecutive_xs)
                consecutive_xs = 0
            new_buses.append(bus)
        else:
            consecutive_xs += 1

    buses = new_buses
    num_active = len(active_buses)
    streak = list()
    bus = buses[0]
    last_departed = buses[-1]

    large_number_for_increment = bus if bus > last_departed else last_departed
    # timestamp += bus - (timestamp % bus)
    # new attempt:



    # bus0_factors = prime_factor(bus)
    # last_bus_factors = prime_factor(last_departed)

    # bus_factors = []
    # print("Finding prime factors")
    # for bus in buses:
    #     if bus > 0:
    #         print(f"Finding prime factors of {bus}")
    #         bus_factors.extend(prime_factor(bus))
    # bus_factors = [prime_factor(bus) for bus in buses]

    # lcm = reduce(lambda x,y: x * y, bus_factors)
    # print(f"Reducing to get LCM: {lcm}")

    num_threads = 12


    #
    #
    # while timestamp % lcm != 0 and timestamp % last_departed != 0:
    #     timestamp += lcm - (timestamp % lcm)
    # timestamp += alignment_offset
    loops = 0
    # lcm = 1

    largest_bus = reduce(lambda x,y: y if x < y else x, buses)
    largest_bus_time_offset = orig_buses.index(largest_bus)
    # print(f'Bus time offset: {largest_bus_time_offset}')
    # threads = []
    # cpu_count = multiprocessing.cpu_count() - 2
    # # cpu_count = 1
    # a = distribute(lcm, cpu_count)




    # print(f'LCM: {lcm}')

    # print('test')
    # out = Path(r'C:\Users\Jump_Around\PycharmProjects\advent_of_code_2021\log_test')
    # f = open(out / (str(os.getpid()) + ".out"), 'w', buffering=0)
    # f_err = open(out / (str(os.getpid()) + "_err.out"), 'w', buffering=0)
    # stdout.write('test')
    # sys.stdout = f
    # sys.stderr = f_err
    timestamp = start
    if timestamp == 0:
        timestamp = 1
    # print(f"Largest bus: {largest_bus}")
    loops = 0
    start = time.time()
    # timestamp += offset - (timestamp % offset)
    print(timestamp)
    while timestamp % largest_bus != 0:
        timestamp += largest_bus - (timestamp % largest_bus)
    print(timestamp)

    # bez = euc(3,4)

    AN = []


    for i, bus in enumerate(orig_buses):
        if bus != -1:
            AN.append((i, bus))

    # print(f'{AN=}')

    print(f'{orig_buses=}')
    N = reduce(lambda x,y: x* y, [bus for bus in orig_buses if bus != -1])
    # print(f'{N=}')
    # bez = euc(3,4)
    # print(f'{bez=}')?


    X = 0
    # first_




    print('\n\n')

    print('Pairing')
    # print(f'{s=}')
    computed = deque()
    comparable = deque(sorted(active_buses))
    pairs = deque()
    while comparable:
        if len(comparable) % 2 == 0:
            l,r = comparable.popleft(), comparable.pop()
            li = get_bus_a_value(l, orig_buses)
            ri = get_bus_a_value(r, orig_buses)
            pairs.appendleft(((l, li), (r, ri)))
        else:
            p = comparable.pop()
            pairs.appendleft(((p, get_bus_a_value(p, orig_buses)), None))
        print(pairs)
    remaining = None
    # if isinstance(pairs[-1], int):
    #     remaining = pairs.pop()
    #     remaining = (remaining, get_bus_a_value(remaining, orig_buses))

    print(f'{pairs=}')
    # print(remaining)
    # c3 = [x * y for x,y in pairs]
    # print(c3)
    s = 0
    print()
    # if remaining:
    #     print('remaining',remaining)
    #     ne = pairs.pop()
    #     l = ne[:2]
    #     n1,a1 = ne[2:]
    #     n0 = remaining[0]
    #     a0 = remaining[1]
    #     print(f'n0,n1', n0,n1)
    #     print(f'a0,a1 {a0,a1}')
    #     b0,b1 = euc(n0, n1)
    #     print(f'n0*b0+n1*b1={(n0 * b0) + (n1 * b1)}')
    #     s = exist(s, a1, b0, b1, n0, n1)
    #
    #     pairs.append((*l, a0, a1 ))


    print("Beginning apply algo to pair loop")
    done = False
    # s = pairs[0][2]
    # while not done and pairs:
    # last_s = None
    # last_a1 = None
    #
    # l_last, r_last = pairs.pop()
    # # l_last_n0, l_last_a0
    #
    # l_last_s = None
    last_s = None
    def doit(last, cur):
        n0, a0 = last
        n1, a1 = cur
        # r_n1, r_a1 = r
        # if last_s is not None:
        #     s = last_s

            # else:
            #     n1, n0 = l,r
            #     a1, a0 = last

        s = pair_algo(n0, n1, a0, a1)
        # solution is x === a0,1 (mod n0n1)
        # so x becomes the previous a value when we iterate
        n_prod = n0 * n1

        return n_prod, s

    r_res = None
    l_res = None
    l_last, r_last = pairs.popleft()

    # if pairs[0][1] is None:
    #     # exit()
    #     p = pairs.pop()
    #     remaining = p
    #     # pairs.append((p,None))
    #     pass


    for pair in pairs:
        l, r = pair
        print('\n***Loop pair', (l,r))
        # last = (l, r)
        # if not isinstance(l, int):
            # n0,n1,a0,a1 =
        # l_n1, l_a1 = l
        # r_n1, r_a1 = r


        # if l_last_s is not None:
        #     l_a0 = l_last_s
        if r:
            r_res = doit(r_last, r)
        else:
            # if len(pairs) == 1:
            remaining = r_last
            # else:
            #     remaining = None

        l_res = doit(l_last, l)

        # else:
        #     n1, n0 = l,r
        #     a1, a0 = last
        r_last = r_res
        l_last = l_res
        #
        # l_s = pair_algo(l_n0, l_n1, l_a0, l_a1)
        # r_s = pair_algo(r_n0, r_n1, r_a0, r_a1)
        # # solution is x === a0,1 (mod n0n1)
        # # so x becomes the previous a value when we iterate
        # l_n_prod = l_n0 * l_n1
        # r_n_prod = r_n0 * r_n1
        # l_last_s, r_last_s = l_s, r_s

    if remaining:
        print(f'\n\nRunning final pair: {remaining}')
        n = remaining[0]
        a = remaining[1]
        print('n,a',n,a)
        print('n2, a2',l_res)
        n, s = doit(l_res, remaining)
        r_res = (n, s)
        print()
    elif l_res and r_res:
        n, s = doit(l_last, r_last)
        l_res = (n, s)
        # if r_res:
            # n, s = doit((n, s), r_res)
        # s = pair_algo(l[0], remaining[0], last_s, remaining[1])
    elif l_res and r_res:

        n, s = doit(l_res, r_res)


    print(f'{l_res=}')
    print(f'{r_res=}')
    # for i in range(active_buses.index(n1)+1, len(active_buses)):
    #     bus = active_buses[i]
    #     print()
    #     print('n0,n1', n0, n1)
    #     n0 = n1 * n0
    #     n1 = bus
    #     print('n0,n1', n0, n1)
    #     a0 = a1
    #     a1 = bus - orig_buses.index(active_buses[i])
    #     print(f'a0,a1 {a0,a1}')
    #     b0, b1 = euc(n0, n1)
    #     print(f'{b0=},{b1=}')
    #     print(f'n0*b0+n1*b1={(n0 * b0) + (n1 * b1)}')
    #
    #     s = exist(s, a1, b0, b1, n0, n1)
    #
    #     print(s)

        # computed.append((n0,n1,a0,a1, b0,b1 ))
        # Ni = N / bus
        # print(f'X += {ai} * {b0} * {Ni}' )
        # X += ai * b0 * Ni

    print('final s', s)
    print('final small n ', n)
    print(f'{N=}')
    print(f'n_l * n_r {l_res[0]*r_res[0]}')
    # return
    # print(f'{As=}')
    # print(f'{N=}')
    print(f'N+s{N+s}')
    print(f'answer= {s if s>0 else N+s}')
    # print()
    # print(f'{X=}')
    # while X < 0:
    #     X += N
    # L = active_buses[0] * bez[0]
    # R = active_buses[1] * bez[1]
    # X = orig_buses.index(active_buses[0]) * active_buses[1] * L + orig_buses.index(active_buses[1]) * active_buses[0] * R
    # X = X + (active_buses[0] * active_buses[1])
    # X = exist(0,3, *bez, 3, 4)
    # print(f'{X=}')
    # # X =
    return
    while True:

        # if not streak:
        #     bus = buses[0]
        #     timestamp += bus - (timestamp % bus)
        #
        #     if not streak and timestamp % bus == 0:
        #         earliest_occurance = timestamp
        #         streak.append(bus)
        #         last_departed = bus
        #         # timestamp += 1
        #
        #         if last_departed > 0:
        #             start_idx = buses.index(last_departed) + 1
        #         else:
        #             start_idx = 0
        # prime_factors

        # if not streak and timestamp % bus == 0:

        # vertical_window_height = len(active_buses) + sum(filter(lambda bus: bus > 0, buses))
        # vertical_window_height = num_orig_buses
        # bitmask = np.zeros((vertical_window_height , len(orig_buses)))
        # # bitmask = np.vstack((np.array(orig_buses), bitmask))
        #
        # # y_axis = np.array(range(timestamp, timestamp + vertical_window_height))
        #
        # # np.index
        #
        # for i, row in enumerate(bitmask):
        #     row[i] = scheduled(timestamp + i, orig_buses[i])

        done = True
        for i in range(num_orig_buses):
            if not scheduled((timestamp - offset) + i, orig_buses[i]):
                done = False
                break

        if done:
            print(f"Found match at timestamp {timestamp}, offset {offset}")
            return timestamp - offset
            # for j in range(num_orig_buses):
            #     if scheduled(timestamp + j, orig_buses[i])
            # if all(scheduled(timestamp + j, orig_buses[j]) for j in range(num_orig_buses)):
            #     return timestamp

        # if all(diagonal):
        #     return timestamp
        # bitmask = []
        # if bitmask.all():
        #     return timestamp

        # np.apply_along_axis(scheduled, 0, bitmask)

        # timestamp += alignment_offset
        # for bus in buses[start_idx:]:
        timestamp += largest_bus
        # timestamp += 1
        loops += 1

        if loops % 10000000 == 0:
            print(
                f'pid={os.getpid()} Loops: {loops}, timestamp: {timestamp} {end - timestamp} left to check {time.time() - start:.2f}s elapsed')
        #
        #     if bus < 0:  # collapse x's into one negative value representing the number of columns
        #         timestamp -= bus
        #         continue
        #
        #     if not streak or bus == -1:
        #         break
        #
        #     is_scheduled = timestamp % bus == 0
        #
        #     if not is_scheduled and streak:
        #         streak = []
        #         last_departed = -2
        #         break
        #     elif is_scheduled and streak and last_departed == streak[-1]:
        #         last_departed = bus
        #         streak.append(bus)
        #         timestamp += 1
        #         continue
        #
        # if streak and len(streak) == num_active:
        #     print(f'Done after {time.time() - start}s')
        #     return bus, earliest_occurance
        #
        # # if round(timestamp, -5) % 10000000 == 0:
        # #     print(f'Timestamp: {timestamp}, elapsed: {time.time() - start}')
        # timestamp += 1
    return 'nope'


# def do_search(num_orig_buses, start, end, offset, orig_buses, largest_bus):



def euc(a, b):
    """Return bezouts coefficients
    take n0,n1
    """
    print(f'bez input: {a,b}')
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
    buses = [int(bus) if bus != 'x' else -1 for bus in bus_numbers.split(',') ]

    # print(earliest_departure)
    # print(buses)

    at_time = get_schedule(buses, earliest_departure)

    # mtx = []
    # row_range = list(range(at_time-10, at_time +  len(buses) + 5))
    # for i in row_range:
    #     row = [scheduled(i, bus) for bus in buses]
    #     mtx.append(row)
    #
    # mtx = np.array(mtx)
    # mtx = numpy.vstack((buses, mtx))
    # idx = [[0] + row_range]
    # mtx = np.append(np.array(idx).T, mtx, 1)
    # # numpy.concatenate((mtx, idx), axis=0)
    # print(mtx)
    #
    #
    # # print(f'{the_bus=}')
    # print(f'{at_time=} after {time.time() - start}')
    # # pprint(schedule)
    # time_waiting = at_time - earliest_departure
    # # print(answer(the_bus, time_waiting))

if __name__ == '__main__':
    # brute_is_prime(1889)
    solve(False)