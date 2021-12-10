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

def brute_is_prime(number):
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True

def prime_factor(number):
    orig = number
    primes = []

    while number % 2 == 0:
        primes.append(2)
        number /= 2

    candidate = 3
    while number != 1:
        while not brute_is_prime(candidate):
            candidate += 2

        q, rem = divmod(number, candidate)
        if rem == 0:
            primes.append(candidate)
            number /= candidate
            candidate = 3
        candidate += 2

    assert reduce(lambda x,y: x * y, primes) == orig
    return primes

def distribute(number, num_chunks=1):
    l = number

    n,r  = divmod(number, num_chunks)

    range_chunks = []
    for ndx in range(100000000000000, l, n):
        range_chunks.append((ndx,min(ndx + n, l)))

    return range_chunks


def exist(a0, a1, b0, b1, n0, n1):
    # print(a0,a1,b0,b1,n0,n1)
    # x =a1*m2*n2 + a2*m1*n1

    X = a0 * b1 * n1 + a1 * b0 * n0

    print(f'X= a0 * b1 * n1 + a1 * b0 * n0')
    print(f'{X} = {a0} * {b1} * {n1} + {a1} * {b0} * {n0}')

    # N = -1
    # print(f'{X=}')
    # while N < 0:
    #    N = X + n0 * n1
    # s = X + (n0 * n1)
    # s = X
    last = X
    c = last
    while True:
        if c > 0:
            c = c - (n0 * n1)
        else:
            c = c + (n0 * n1)
        if abs(c) > abs(last):
            s = last
            break
        last = c

    print(f'reduced {X}->{s}')
    return s

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
    print(f'Bus time offset: {largest_bus_time_offset}')
    # threads = []
    # cpu_count = multiprocessing.cpu_count() - 2
    # # cpu_count = 1
    # a = distribute(lcm, cpu_count)




    # print(f'LCM: {lcm}')

    print('test')
    # out = Path(r'C:\Users\Jump_Around\PycharmProjects\advent_of_code_2021\log_test')
    # f = open(out / (str(os.getpid()) + ".out"), 'w', buffering=0)
    # f_err = open(out / (str(os.getpid()) + "_err.out"), 'w', buffering=0)
    # stdout.write('test')
    # sys.stdout = f
    # sys.stderr = f_err
    timestamp = start
    if timestamp == 0:
        timestamp = 1
    print(f"Largest bus: {largest_bus}")
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
    print(f'{N=}')
    # bez = euc(3,4)
    # print(f'{bez=}')?


    X = 0
    # first_

    print('\nrunnning algo')
    n0 = active_buses[0]
    n1 = active_buses[1]
    a0 = 0
    a1 = (n1 - orig_buses.index(n1))
    print(f'n0,n1', n0,n1)
    print(f'a0,a1 {a0,a1}')
    b0,b1 = euc(n0, n1)
    print(f'n0*b0+n1*b1={(n0 * b0) + (n1 * b1)}')
    s = exist(a0, a1, b0, b1, n0, n1)



    As = [a0, a1]
    print(f'{s=}')
    computed = deque()
    for i in range(active_buses.index(n1)+1, len(active_buses)):
        bus = active_buses[i]
        print()
        print('n0,n1', n0, n1)
        n0 = n1 * n0
        n1 = bus
        print('n0,n1', n0, n1)
        a0 = a1
        a1 = bus - orig_buses.index(active_buses[i])
        As.append(a1)
        print(f'a0,a1 {a0,a1}')
        b0, b1 = euc(n0, n1)
        print(f'{b0=},{b1=}')
        print(f'n0*b0+n1*b1={(n0 * b0) + (n1 * b1)}')

        s = exist(s, a1, b0, b1, n0, n1)

        print(s)

        computed.append((n0,n1,a0,a1, b0,b1 ))
        # Ni = N / bus
        # print(f'X += {ai} * {b0} * {Ni}' )
        # X += ai * b0 * Ni

    print('final s', s)
    return
    # print(f'{As=}')
    # print(f'{N=}')
    # print(f'N+s{N+s}')
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
    # bus_numbers = '67,7,59,61'
    # bus_numbers = '67,x,7,59,61'
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