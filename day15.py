from collections import defaultdict, deque

from utils import input_loader





def memorize(number, turn, number_watch):
    if len(number_watch[number]) == 0:
        speak = 0
    else:
        speak = turn - number_watch[number][0]
        number_watch[number].popleft()
    number_watch[number].append(turn)

    return speak

def solve(sample, stop_at):
    number_watch = defaultdict(deque)

    content = input_loader.load_file_as_string(15, sample)

    numbers = [int(n) for n in content.split(',')]
    print(numbers)

    turn_number = len(numbers) + 1
    for i, number in enumerate(numbers):
        number_watch[number].append(i+1)

    last_spoken = 0
    while turn_number < stop_at:

        number = last_spoken


        speak = memorize(number, turn_number, number_watch)
        last_spoken = speak

        turn_number += 1

        # print(number_watch)

    print('turn ',turn_number, ' last spoken', last_spoken)


    return last_spoken

if __name__ == '__main__':
    assert solve(False, 2020) == 1194
    assert solve(False, 30000000) == 48710
