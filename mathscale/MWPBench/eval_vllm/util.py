import re

def last_boxed_only(sample):
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception as e:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def unbox_and_extract(text):
    start_indices = [m.start() for m in re.finditer(r'\\boxed{', text)]
    extracted_contents = []
    for start in start_indices:
        brace_count = 0
        for i, char in enumerate(text[start:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    extracted_contents.append(text[start+7:end-1])  # +7 to skip '\\boxed{'
                    break
    # Replace '\\boxed{...}' with the content inside it
    unboxed_text = re.sub(r'\\boxed{(.*?)}', r'\1', text)
    return unboxed_text, extracted_contents

def convert_to_latex_fraction(text: str) -> str:
    # Use regex to find all occurrences of ((num)/(denom))
    pattern = re.compile(r"\(\(([\d]+)\)/\(([\d]+)\)\)")
    
    matches = pattern.findall(text)
    
    for match in matches:
        num, denom = match
        latex_frac = f"\\\\frac{{{num}}}{{{denom}}}"
        
        # Replace the old expression with the LaTeX fraction
        text = text.replace(f"(({num})/({denom}))", latex_frac)
    
    return text

def strip_string(string):
    # convert ((3)/(4)) -> \\frac{3}{4}
    string = convert_to_latex_fraction(string)

    # remove ,
    string = string.replace(",", "")

    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # My own
    string = string.replace("\\quad", " ")
    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def is_number(s):
    s = s.strip("$")

    try:
        # Try to convert the string to an integer
        int(s)
        return True
    except ValueError:
        try:
            # Try to convert the string to a float
            float(s)
            return True
        except ValueError:
            return False

def is_single_inline_math(expression: str) -> bool:
    # Use regex to check for a pattern that starts and ends with dollar signs,
    # and contains no other dollar signs in between.
    pattern = re.compile(r"^\$[^$]+\$$")
    
    match = pattern.match(expression)
    
    return bool(match)

def is_equiv(prediction_ans, reference_ans, verbose=False):
    if prediction_ans is None and reference_ans is None:
        print("WARNING: Both None")
        return True, prediction_ans, reference_ans
    if prediction_ans is None or reference_ans is None:
        return False, prediction_ans, reference_ans

    try:
        clean_prediction_ans = strip_string(prediction_ans)
        clean_reference_ans = strip_string(reference_ans)

        if is_number(clean_prediction_ans) and is_number(clean_reference_ans):
            judge = float(clean_prediction_ans.strip("$")) == float(clean_reference_ans.strip("$"))
            # print(f"1 judge: {judge}")
        elif is_single_inline_math(clean_reference_ans):
            judge = (clean_reference_ans.strip("$") in clean_prediction_ans.strip("$"))
            # print(f"2 judge: {judge}")
        elif (len(clean_prediction_ans) >= 3) and (not is_number(clean_prediction_ans)) and (not clean_prediction_ans.startswith("-")) and (not clean_reference_ans.startswith("-")) and (clean_prediction_ans in clean_reference_ans):
            judge = True
            # print(f"3 judge: {judge}")
        elif (len(clean_reference_ans) >= 3) and (not is_number(clean_reference_ans)) and (not clean_prediction_ans.startswith("-")) and (not clean_reference_ans.startswith("-")) and (clean_reference_ans in clean_prediction_ans):
            judge = True
            # print(f"4 judge: {judge}")
        else:
            judge = clean_prediction_ans == clean_reference_ans
            # print(f"5 judge: {judge}")
        if verbose:
            print(f"clean_prediction_ans: {clean_prediction_ans} | clean_reference_ans: {clean_reference_ans} | judge: {judge}")
        return judge, clean_prediction_ans, clean_reference_ans
    except Exception as e:
        print(e)
        return prediction_ans == reference_ans, prediction_ans, reference_ans

def is_correct(completion, answer, verbose=False):
    completion = completion.lower()
    answer = answer.lower()

    # Extract short answer from completion
    extract_ans = None

    clean_reference_ans = strip_string(answer)
    is_reference_ans_number = is_number(clean_reference_ans)

    # First extract boxed answer
    unbox_long_answer, box_short_answers = unbox_and_extract(completion)
    if box_short_answers != []:
        extract_ans = box_short_answers[-1].strip()
        # print(f"1 extract_ans: {extract_ans}")
    # extract the last number answer
    elif is_reference_ans_number:
        numbers = re.findall(r"[\-+]?\d*[\.,/]?\d+", completion)
        if numbers: 
            extract_ans = numbers[-1]
        # print(f"2 extract_ans: {extract_ans}")
    # extract "the answer is ..." answer
    elif ("answer is" in completion) or ("solution is" in completion):
        if "answer is" in completion:
            split_ans = completion.split('answer is')
        else:
            split_ans = completion.split('solution is')
        ans = split_ans[-1].strip().lstrip(":").strip()
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        extract_ans_temp = extract_ans_temp.strip('.')
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        # print(f"3 extract_ans: {extract_ans}")
    # extract "therefore xx is xxx" answer
    elif "is" in completion:
        pos = completion.rfind("is")
        ans = completion[pos+2:].strip().lstrip(":").strip()
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        extract_ans_temp = extract_ans_temp.strip('.')
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        # print(f"4 extract_ans: {extract_ans}")
    else:
        return False, f"failed extracting answer from completion", clean_reference_ans

    judge, clean_prediction_ans, clean_reference_ans = is_equiv(extract_ans, answer, verbose=verbose)
    return judge, clean_prediction_ans, clean_reference_ans

if __name__ == "__main__":
    reference_ans = "$2$"
    prediction_ans = "To find the value of $a$, we need to evaluate the function $f$ at $f(\\sqrt{6})$ and set it equal to 3.\n\nFirst, let's find the value of $f(\\sqrt{6})$. Since $\\sqrt{6}$ is not in the domain of the first piece of the function, we move on to the second piece. \n\nFor $x \\leq 2$, the function becomes $f(x) = |x-3| + a$. Plugging in $\\sqrt{6}$, we have $f(\\sqrt{6}) = |(\\sqrt{6})-3| + a$.\n\nNext, we set $f(\\sqrt{6})$ equal to 3 and solve for $a$. \n\n$|(\\sqrt{6})-3| + a = 3$\n\nSince we are looking for a real number value of $a$, we can ignore the absolute value and solve for $a$.\n\n$(\\sqrt{6})-3 + a = 3$\n\n$\\sqrt{6} - 3 + a = 3$\n\n$\\sqrt{6} + a = 3 + 3$\n\n$\\sqrt{6} + a = 6$\n\nSubtracting 6 from both sides, we get:\n\n$a = 6 - \\sqrt{6}$\n\nTherefore, the value of $a$ is $6 - \\sqrt{6}$.\n\nThe answer is $a = 6 - \\sqrt{6}$."
    print(is_correct(prediction_ans, reference_ans, verbose=True))