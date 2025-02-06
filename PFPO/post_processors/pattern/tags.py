import re


def think_tag_cleaner(completion: str):
    # Regular expression to match content between [BEGIN] and [END]
    pattern = r'<thinking>(.*?)</thinking>'

    # Find all matches
    matches = re.findall(pattern, completion, re.DOTALL)

    # Check if there's exactly one pair of tags
    if len(matches) == 1:
        return matches[0].strip()  # Return the content of the single match
    else:
        return False  # Return False for 0 or more than 1 pair of tags


def solution_tag_cleaner(completion: str):
    # Regular expression to match content between [BEGIN] and [END]
    pattern = r'<solution>(.*?)</solution>'

    matches = re.findall(pattern, completion, re.DOTALL)

    # Check if there's exactly one pair of tags
    if len(matches) == 1:
        return matches[0].strip()  # Return the content of the single match
    else:
        return False  # Return False for 0 or more than 1 pair of tags


def output_tag_cleaner(completion: str):
    # Regular expression to match content between [BEGIN] and [END]
    pattern = r'<Output>(.*?)</Output>'

    matches = re.findall(pattern, completion, re.DOTALL)

    # Check if there's exactly one pair of tags
    if len(matches) == 1:
        return matches[0].strip()  # Return the content of the single match
    else:
        return False  # Return False for 0 or more than 1 pair of tags


_code_cleaners = {
    "think": think_tag_cleaner,
    "solution": solution_tag_cleaner,
    "output": output_tag_cleaner
}


def get(name: str):
    return _code_cleaners[name]
