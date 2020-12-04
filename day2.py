from pathlib import Path
import re


def read_input(filename: Path):
    lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    parts = [line.strip().split(' ') for line in lines]

    return parts


parts_list = read_input('puzzle_input/day2/input.txt')
print(f'total lines {len(parts_list)}')
valid = []
for part in parts_list:

    min_count = int(part[0].split('-')[0])
    max_count = int(part[0].split('-')[-1])

    letter = part[1][0]
    password = part[-1]

    index1 = min_count - 1
    index2 = max_count - 1

    first = False
    second = False
    try:
        first = (password[index1] == letter)
    except Exception:
        continue

    try:
        second = (password[index2] == letter)
    except Exception:
        continue

    if first ^ second:
        valid.append(password)
    #
    # print(letter)
    # print(found)
    # print(min_count)
    # print(max_count)
    #
    # if min_count <= len(found) <= max_count:
    #     valid.append(password)


print(f'{len(valid)} passwords were valid')