test_cases = ["1 + (2 * 3) + (4 * (5 + 6))", "5 + (8 * 3 + 9 + 3 * 4 * 3)",
"2 * 3 + (4 * 5)", "5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))",  "((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2"]

def add(l, r):
    return l + r

def mult(l, r):
    return l * r

stack = []

operations = {
    '+': add,
    '*': mult
}

value = {
    '+': 1,
    '*': 0
}

parens = ['(',')']

f = open('puzzle_input/day18/input.txt', 'r')
actual_input = [line.strip() for line in f]
f.close()


def to_postfix(line):
    output = []
    stack = []
    line = line.replace(' ', '')

    for char in line:
        if char  not in operations and char not in parens:
            output.append(char)
        elif char in operations and stack and (stack[-1] == '(' or value[stack[-1]] < value[char]):
            stack.append(char)
        elif char in operations and stack and stack[-1] in operations:
            while stack and stack[-1] in operations and value[stack[-1]] >= value[char]:
                curr = stack.pop()
                if curr in operations:
                    output.append(curr)
                elif curr in parens:
                    break
            stack.append(char)
        elif char in operations:
            stack.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack:
                top = stack.pop()
                if top != '(':
                    output.append(top)
                else:
                    break
    while stack:
        output.append(stack.pop())
    return "".join(output)


def execute(postfix):
    stack = []
    for curr in postfix:
        # curr = postfix.pop()

        if curr not in operations:
            stack.append(curr)
        else:
            r = stack.pop()
            l = stack.pop()
            op_res = operations[curr](int(l),int(r))
            stack.append(op_res)
    return stack[0]

total = 0
for test_input in actual_input:
    res = to_postfix(test_input)
    print(res)
    res = execute(res)
    print(res)
    total += res

print(total)