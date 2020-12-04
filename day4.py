lines = []
batches = []
batch = []
with open('puzzle_input/day4/input.txt') as f:
    batch = []
    for line in f:
        if line.strip() == "":
            batches.append(batch)
            batch = []
        else:
            batch += line.strip().split(' ')
batches.append(batch)
req = ['hgt','hcl','ecl','pid','byr','iyr','eyr']
opt = ['cid']

valid = 0
for batch in batches:
    d = {}
    nope=False
    for cmd in batch:
        l = cmd.split(':')
        cmd = l[0]
        val = l[1]
        d[cmd]=val

    for r in req:
        if r not in d:
            nope=True
            break
        nope=False
    if not nope:
        valid +=1

print(valid)

