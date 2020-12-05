import marshmallow
from marshmallow import ValidationError
from marshmallow.fields import String, Number
from marshmallow.validate import Range, Regexp, OneOf


def validate_hgt(value):
    if 'cm' in value and (150 <= int(value[:-2]) <= 193):
        return True
    elif 'in' in value and (59 <= int(value[:-2]) <= 76):
        return True
    else:
        raise ValidationError("Not correct")

def number_len_validator(value):
    if len(str(value)) != 9:
        raise ValidationError("bad")

class MySchema(marshmallow.Schema):
    byr = Number(required=True,validate=Range(min=1920,max=2002))
    iyr = Number(required=True,validate=Range(min=2010,max=2020))
    eyr = Number(required=True,validate=Range(min=2020,max=2030))
    hgt = String(required=True, validate=validate_hgt)
    hcl = String(required=True,validate=Regexp('#[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'))
    ecl  = String(required=True,validate=OneOf(['amb','blu','brn','gry','grn','hzl','oth']))
    pid = String(required=True, validate=number_len_validator)
    cid = String(required=False)

schema = MySchema()

batches = []
with open('../puzzle_input/day4/input.txt') as f:
    batch = []
    for line in f:
        if line.strip() == "":
            batches.append(batch)
            batch = []
        else:
            batch += line.strip().split(' ')
batches.append(batch)

valid = 0
for batch in batches:
    d = {}
    for cmd in batch:
        l = cmd.split(':')
        cmd = l[0]
        val = l[1]
        d[cmd]=val

    try:
        schema.load(d)
        valid += 1
    except ValidationError as e:
        pass
    except Exception as e:
        print(e)

print(valid)
assert valid == 167
print('Success!')


