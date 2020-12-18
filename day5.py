import math


def run(override=None):

    if not override:
        with open('puzzle_input/day5/input.txt', 'r') as f:
            coded_seats = [line.strip() for line in f]
    else:
        coded_seats = [override]

    LOWER = 'lower'
    UPPER = 'upper'

    symbols = {
        'F': LOWER,
        'B': UPPER,
        'L': LOWER,
        'R': UPPER
    }

    def recursion(nums, seat):

        if len(nums) == 1:
            return nums[0]

        char = seat[0]

        direction = symbols[char]


        if direction == UPPER:
            nums = nums[len(nums) // 2:]
        elif direction == LOWER:

            nums = nums[:len(nums) // 2]

        return recursion(nums, seat[1:])


    highest = -1
    taken_seats = []
    for seat in coded_seats:

        coded_row = seat[0:8]
        coded_col = seat[7:]

        row = recursion(list(range(0,128)), coded_row)
        col = recursion(list(range(0,8)), coded_col)

        seat_id = row * 8 + col
        taken_seats.append(seat_id)
        highest = seat_id if highest < seat_id else highest
        print(f"{seat=} {row=} {col=} {seat_id=}")
    taken_seats = sorted(taken_seats)

    print(f'{highest=}')
    print(f'{taken_seats=}')

    for i in range(len(taken_seats)):
        if taken_seats[i] + 1 != taken_seats[i + 1]:
            print(f'{i=}')
            print(f'{taken_seats[i]=}')

    for i in range(len(taken_seats)):
        seat = taken_seats[i]
        if taken_seats[i + 2] == seat + 2 and taken_seats[i + 1] != seat + 1:
            print(f"Here's your seat: {seat + 1}")
            break



if __name__ == '__main__':
    run()