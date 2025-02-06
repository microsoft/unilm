import re


def standard_cleaner(completion: str):
    # Regular expression to match code blocks with or without a language indicator
    pattern = r'```(?:\w+)?\n(.*?)```'
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def standard_cleaner_default(completion: str):
    # Regular expression to match code blocks with or without a language indicator
    pattern = r'```(?:\w+)?\n(.*?)```'
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return completion.strip()


def tag_cleaner(completion: str):
    # Regular expression to match content between [BEGIN] and [END]
    pattern = r'<BEGIN>(.*?)<END>'

    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


_code_cleaners = {
    "standard": standard_cleaner,
    "standard_default": standard_cleaner_default,
    "tag": tag_cleaner,
}


def get(name: str):
    return _code_cleaners[name]
