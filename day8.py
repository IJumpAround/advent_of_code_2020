from typing import Tuple

accumulator = 0
execution_index = 0

def acc(arg):
    global accumulator
    accumulator += int(arg)

def jmp(offset):
    global execution_index
    execution_index += int(offset)

def nop(arg):
    pass

ops = {
    'acc': acc,
    'jmp': jmp,
    'nop': nop,
}


def load_program(test=False):

    if test:
        return """nop +0
acc +1
jmp +4
acc +3
jmp -3
acc -99
acc +1
jmp -4
acc +6""".split('\n')
    f = open('puzzle_input/day8/input.txt')
    instructions = [line.strip() for line in f]
    f.close()
    return instructions

def parse_line(line: str) -> Tuple[callable, str]:
    cmd, arg = line.split(' ')

    cmd = ops[cmd]

    return cmd, arg

def done(idx, prg):
    return idx >= len(prg)


def restart():
    global accumulator
    global execution_index
    global prev_ex
    global jump_stack

    jump_stack = []
    accumulator = 0
    execution_index = 0
    prev_ex = set()

def restore_patched_line(idx):
    global nop_stack
    global program
    idx, inst =  nop_stack.pop()

    program[idx] = inst

prev_ex = set()

program = load_program(test=False)
jump_stack = []
nop_stack = []
excluded = set()
while not done(execution_index, program):

    instruction = program[execution_index]

    # loop detected
    if execution_index in prev_ex:
        while True:
            last_jump_addr = jump_stack.pop()
            idx = last_jump_addr + execution_index

            if last_jump_addr not in excluded:
                break
            else:
                restore_patched_line(last_jump_addr)

        nop_stack.append((last_jump_addr, program[last_jump_addr]))
        program[last_jump_addr] = 'nop +0'
        excluded.add(last_jump_addr)
        restart()
        continue

    prev_ex.add(execution_index)
    cmd, arg = parse_line(instruction)


    if cmd != jmp:
        execution_index += 1
    else:
        jump_stack.append(execution_index)

    cmd(arg)

print(f'{accumulator=}')
print(f'{program=}')