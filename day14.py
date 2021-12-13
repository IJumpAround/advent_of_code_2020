import re
from collections import defaultdict

from utils import input_loader

mem2 = defaultdict(int)


def apply(bit_index, x_mask, current_address, value):
    # bit index starts at 36 so we start with smallest bit
    global mem2
    while x_mask[bit_index-1] != 'X' and bit_index > 0:
        bit_index -= 1

    if bit_index == 0:
        return

    x_bit = 2 ** (36- bit_index)
    x_int_mask = 1 * x_bit

    addr1 = current_address
    addr2 = current_address ^ x_int_mask

    mem2[addr1] = value
    mem2[addr2] = value

    apply(bit_index-1, x_mask, addr1, value)
    apply(bit_index-1, x_mask, addr2, value)



def solve(sample):
    mem = defaultdict(int)
    file_content = input_loader.load_file_as_list(14, sample)

    and_mask = None
    or_mask = None
    mask = None
    for line in file_content:
        if line.startswith('mask'):
            mask = line.split(' = ')[1]

            and_mask = int("".join(['1' if bit != '0' else '0' for bit in mask]), 2)
            or_mask = int("".join(['0' if bit != '1' else '1' for bit in mask]), 2)

            memory_mask = int("".join(['1' if bit == '1' else '0' for bit in mask]), 2) # apply with or

            continue

        dest, value = line.split(' = ')
        value = int(value)
        address = int(re.search(r'\[(\d+)\]', dest).groups()[0])

        mem[address] = (value & and_mask) | or_mask
        x_idx = 36
        print(mask)

        base_address = address | memory_mask
        apply(x_idx, mask, base_address, value)


    print(file_content)
    print(mem)
    print(mem2)
    answer = sum(mem.values())
    p2_answer = sum(mem2.values())

    print('p1 answer: ', answer)
    print('p2 answer: ', p2_answer)
    assert p2_answer == 5724245857696
    assert answer == 15018100062885


if __name__ == '__main__':
    solve(sample=False)
