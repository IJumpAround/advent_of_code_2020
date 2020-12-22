import argparse
from pathlib import Path

import requests

base = "https://adventofcode.com/2020/day"

in_endpoint = '/input'
def download(args):
    day = args.day

    url = f'{base}/{day}/input'

    headers = {
        'cookie': 'session=53616c7465645f5f5f8b95e4178a34de1ef26a92de0f79cf9239c8a3ff0d48da1aad8b797708d79007a90ca051bd6af6'
    }
    s = requests.Session()

    response = requests.get(url, headers)
    login_response = requests.

    puzzle_input = response.content

    a = 'https://github.com/session'

    username_id = 'login_field'
    pw_id = 'password'
    print(puzzle_input)
    return puzzle_input


def save(text, day):
    here = Path(__file__).parent.parent
    input_folder = here / 'puzzle_input'

    day_folder = input_folder / f'day{day}'

    if not day_folder.is_dir():
        day_folder.mkdir()

    target = day_folder/'input.txt'


    with open(target, 'wb') as f:
        f.write(text)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('day')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    inp = download(args)
    save(inp, args.day)