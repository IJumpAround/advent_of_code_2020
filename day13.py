from functools import reduce
from pprint import pprint

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


def scheduled(current_time, bus, time_zero=0):
    return current_time % bus == 0


def get_schedule(buses, earliest_departure):
    fastest = fastest_bus(buses)
    headers = ['time'] + [f'bus {bus_number}' for bus_number in buses]

    timestamp = earliest_departure

    schedule = [headers]
    for i in range(fastest * 2):
        row = []
        for bus in buses:
            if scheduled(timestamp, bus):
                row.append(1)
                return bus, timestamp
            else:
                row.append(0)
        row = [timestamp] + [1 if scheduled(timestamp, bus) else 0 for bus in buses]
        schedule.append(row)
        timestamp+=1

    return schedule


def solve(sample):

    file_content = input_loader.load_file_as_list(13, sample)

    earliest_departure, bus_numbers = file_content

    earliest_departure = int(earliest_departure)
    buses = [int(bus) for bus in bus_numbers.split(',') if bus != 'x']

    print(earliest_departure)
    print(buses)

    the_bus, at_time = get_schedule(buses, earliest_departure)

    # pprint(schedule)
    time_waiting = at_time - earliest_departure
    print(answer(the_bus, time_waiting))

if __name__ == '__main__':
    solve(False)