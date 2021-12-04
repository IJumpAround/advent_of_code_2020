import logging
from pathlib import Path

logger = logging.getLogger(__name__)

INPUT_FILE = 'input.txt'
SAMPLE_FILE = 'sample.txt'


def day_int_to_str(day):
    return f'day{day}'


def get_filename(sample=False):
    return INPUT_FILE if not sample else SAMPLE_FILE


def input_root() -> Path:
    return Path(__file__).parent.parent / 'puzzle_input'


def get_day_input_path(day: str, raising: bool = True):
    """Return path to input file for day

    If raising and path doesn't exist, raise exception
    """
    day_folder = input_root() / day
    try:
        assert day_folder.is_dir()
    except AssertionError as e:
        logger.error(day_folder)
        if raising:
            raise e

    return day_folder


def _load_file_as_string(day: str, filename=INPUT_FILE) -> str:

    file = get_day_input_path(day) / filename

    logger.debug(f'Loading file: {file}')
    assert file.is_file()

    file_text = file.read_text()

    return file_text


def load_file_as_string(day: int, sample: bool = False):
    day = day_int_to_str(day)
    filename = get_filename(sample)

    text = _load_file_as_string(day, filename)
    logger.debug(f'File string content\n{text}\n')

    return text


def load_file_as_list(day, sample=False, line_as_list=False):
    day = day_int_to_str(day)
    filename = get_filename(sample)

    file = _load_file_as_string(day, filename)

    lines = file.splitlines()

    if line_as_list:
        new_lines = []
        for line in lines:
            line = [ch for ch in line]
            new_lines.append(line)
        lines = new_lines

    logger.debug(f"File list content: \n{lines}\n")
    return lines
