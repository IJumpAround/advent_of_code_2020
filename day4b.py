import marshmallow
from marshmallow import ValidationError
from marshmallow.fields import Field, String, Number, Validator, Integer
from marshmallow.validate import Range, ContainsOnly, Length, Regexp, OneOf


class MySchema(marshmallow.Schema):
    #
    byr = Number(required=True,validate=Range(min=1920,max=2002))
    iyr = Number(required=True,validate=Range(min=2010,max=2020))
    eyr = Number(required=True,validate=Range(min=2020,max=2030))
    hgt = String(required=True)
    hcl = String(required=True,validate=Regexp('#[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'))
    ecl  = String(required=True,validate=OneOf(['amb','blu','brn','gry','grn','hzl','oth']))
    pid = Integer(required=True)
    cid = String(required=False)

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
# req = {'hgt':,'hcl','ecl','pid','byr':[1920,2002],'iyr','eyr'}
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

    # for r in req:
    #     if r not in d:
    #         nope=True
    #         break
    #     nope=False

    schema = MySchema()

    try:

        assert len(d['pid']) == 9
        schema.load(d)

        hgt = d['hgt']
        print(f'{hgt=}')
        if 'cm' in hgt:
            num = hgt[:-2]
            print(f'{num=}')
            if 150 <= int(num) <= 193:
                valid += 1
        elif 'in' in hgt:
            num = hgt[:-2]
            if 59 <= int(num) <= 76:
                valid +=1
    except ValidationError as e:
        print(e.messages)
        print(d)
        print()
    except Exception as e:
        print(e)
        print(0)



print(valid)

