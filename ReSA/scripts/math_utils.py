import re
import os
import io
import json
import copy
import regex
import pickle
import datetime
import traceback
import numpy as np
from tqdm import tqdm
from math import isclose
from pathlib import Path
from contextlib import redirect_stdout
from concurrent.futures import TimeoutError
from functools import partial
import multiprocessing
import multiprocess
from multiprocess import Pool
from typing import List, Tuple, Optional, Type, TypeVar, Any, Iterable, Union, Dict

import dateutil.relativedelta
from pebble import ProcessPool
from timeout_decorator import timeout
from word2number import w2n

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

# from utils.math_examples import get_examples


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


# EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
    "direct": ("Problem: {input}\nAnswer: ", "{output}", "\n\n"),
    "o1": (
        "### Instruction:\n Return your final response within \\boxed{{}}. {input}\n\n### Think:\n ",
        "{output}",
        "\n\n",
    ),
}


def construct_prompt(example, data_name, args):
    if args.adapt_few_shot and data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        demos = load_prompt(data_name, args.prompt_type, 5)
    else:
        demos = load_prompt(data_name, args.prompt_type, args.num_shots)
    prompt_type = args.prompt_type
    
    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    
    demo_prompt = splitter.join(
        [
            input_template.format(input=q) + output_template.format(output=a)
            for q, a in demos
        ]
    )
    
    context = input_template.format(input=example["question"])
    if len(demo_prompt) == 0 or (
        args.adapt_few_shot and example["gt_ans"] not in ["A", "B", "C", "D", "E"]
    ):
        full_prompt = context
    else:
        full_prompt = demo_prompt + splitter + context

    return full_prompt.strip(" ")  # important!










# parser.py

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
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


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_multi_choice_answer(pred_str):
    # TODO: SFT models
    if "Problem:" in pred_str:
        pred_str = pred_str.split("Problem:", 1)[0]
    pred_str = pred_str.replace("choice is", "answer is")
    patt = regex.search(r"answer is \(?(?P<ans>[abcde])\)?", pred_str.lower())
    if patt is not None:
        return patt.group("ans").upper()
    return "placeholder"


direct_answer_trigger_for_fewshot = ("choice is", "answer is")


def choice_answer_clean(pred: str):
    pred = pred.strip("\n")

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    # Split the trigger to find the answer.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def clean_units(pred_str: str):
    """Clean the units in the number."""

    def convert_pi_to_number(code_string):
        code_string = code_string.replace("\\pi", "π")
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
        code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("¥", "")
    pred_str = pred_str.replace("°C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("°", "")
    return pred_str


def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    if any([option in pred.lower() for option in ["yes", "true"]]):
        pred = "True"
    elif any([option in pred.lower() for option in ["no", "false"]]):
        pred = "False"
    elif any(
        [
            option in pred.lower()
            for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        ]
    ):
        pass
    else:
        # Some of the models somehow get used to boxed output from pre-training
        if "boxed" in pred:
            pred = find_box(pred)

        if answer_flag:
            # Extract the numbers out of the string
            pred = pred.split("=")[-1].strip()
            pred = clean_units(pred)
            try:
                tmp = str(latex2sympy(pred))
                pred = str(eval(tmp))
            except Exception:
                if re.match(r"-?[\d\.]+\s\D+$", pred):
                    pred = pred.split(" ")[0]
                elif re.match(r"-?[\d\.]+\s[^\s]+$", pred):
                    pred = pred.split(" ")[0]
        else:
            # desparate search over the last number
            preds = re.findall(r"-?\d*\.?\d+", pred)
            if len(preds) >= 1:
                pred = preds[-1]
            else:
                pred = ""

    return pred


def extract_answer(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if data_name in ["mmlu_stem", "sat_math", "aqua", "gaokao2023"]:
        # TODO check multiple choice
        return choice_answer_clean(pred_str)

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # choice answer
    if (
        data_name in ["sat_math", "aqua"]
        or "mmlu" in data_name
    ):
        tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
        if tmp:
            pred = tmp[-1]
        else:
            pred = pred.strip().strip(".")

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]


def parse_ground_truth(example: Dict[str, Any], data_name):
    if "gt_cot" in example and "gt" in example:
        if data_name in ["math"]:
            gt_ans = extract_answer(example["gt_cot"], data_name)
        elif data_name in STRIP_EXCEPTIONS:
            gt_ans = example["gt"]
        else:
            gt_ans = strip_string(example["gt"])
        return example["gt_cot"], gt_ans

    # parse ground truth
    if data_name in ["math", "minerva_math"]:
        gt_cot = example["solution"]
        gt_ans = extract_answer(gt_cot, data_name)
    elif data_name == "gsm8k":
        gt_cot, gt_ans = example["answer"].split("####")
    elif data_name == "svamp":
        gt_cot, gt_ans = example["Equation"], example["Answer"]
    elif data_name == "asdiv":
        gt_cot = example["formula"]
        gt_ans = re.sub(r"\(.*?\)", "", example["answer"])
    elif data_name == "mawps":
        gt_cot, gt_ans = None, example["target"]
    elif data_name == "tabmwp":
        gt_cot = example["solution"]
        gt_ans = example["answer"]
        if example["ans_type"] in ["integer_number", "decimal_number"]:
            if "/" in gt_ans:
                gt_ans = int(gt_ans.split("/")[0]) / int(gt_ans.split("/")[1])
            elif "," in gt_ans:
                gt_ans = float(gt_ans.replace(",", ""))
            elif "%" in gt_ans:
                gt_ans = float(gt_ans.split("%")[0]) / 100
            else:
                gt_ans = float(gt_ans)
    elif data_name == "carp_en":
        gt_cot, gt_ans = example["steps"], example["answer"]
    elif data_name == "mmlu_stem":
        abcd = "ABCD"
        gt_cot, gt_ans = None, abcd[example["answer"]]
    elif data_name == "sat_math":
        gt_cot, gt_ans = None, example["Answer"]
    elif data_name == "aqua":
        gt_cot, gt_ans = None, example["correct"]
    elif data_name in ["gaokao2023en", "college_math", "gaokao_math_cloze"]:
        gt_cot, gt_ans = None, example["answer"].replace("$", "").strip()
    elif data_name == "gaokao_math_qa":
        gt_cot, gt_ans = None, example["label"]
    elif data_name in ["gaokao2024_mix", "cn_middle_school"]:
        if len(example["choice_answer"]) > 0:
            gt_cot, gt_ans = None, example["choice_answer"]
        else:
            gt_cot, gt_ans = None, example["answer"]
    elif data_name == "olympiadbench":
        gt_cot, gt_ans = None, example["final_answer"][0].strip("$")
    elif data_name in [
        "aime24",
        "amc23",
        "cmath",
        "gaokao2024_I",
        "gaokao2024_II",
        "imo2024",
    ]:
        gt_cot, gt_ans = None, example["answer"]
    else:
        raise NotImplementedError(f"`{data_name}`")
    # post process
    gt_cot = str(gt_cot).strip()
    if data_name not in STRIP_EXCEPTIONS:
        gt_ans = strip_string(gt_ans, skip_unit=data_name == "carp_en")
    else:
        gt_ans = (
            gt_ans.replace("\\neq", "\\ne")
            .replace("\\leq", "\\le")
            .replace("\\geq", "\\ge")
        )
    return gt_cot, gt_ans


def parse_question(example, data_name):
    question = ""
    if data_name == "asdiv":
        question = f"{example['body'].strip()} {example['question'].strip()}"
    elif data_name == "svamp":
        body = example["Body"].strip()
        if not body.endswith("."):
            body = body + "."
        question = f'{body} {example["Question"].strip()}'
    elif data_name == "tabmwp":
        title_str = (
            f'regarding "{example["table_title"]}" ' if example["table_title"] else ""
        )
        question = f"Read the following table {title_str}and answer a question:\n"
        question += f'{example["table"]}\n{example["question"]}'
        if example["choices"]:
            question += (
                f' Please select from the following options: {example["choices"]}'
            )
    elif data_name == "carp_en":
        question = example["content"]
    elif data_name == "mmlu_stem":
        options = example["choices"]
        assert len(options) == 4
        for i, (label, option) in enumerate(zip("ABCD", options)):
            options[i] = f"({label}) {str(option).strip()}"
        options = " ".join(options)
        # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    elif data_name == "sat_math":
        options = example["options"].strip()
        assert "A" == options[0]
        options = "(" + options
        for ch in "BCD":
            if f" {ch}) " in options:
                options = regex.sub(f" {ch}\) ", f" ({ch}) ", options)
        # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    elif "aqua" in data_name:
        options = example["options"]
        choice = "(" + "(".join(options)
        choice = choice.replace("(", " (").replace(")", ") ").strip()
        choice = "\nAnswer Choices: " + choice
        question = example["question"].strip() + choice
    elif data_name == "gaokao_math_qa":
        options_dict = example["options"]
        options = []
        for key in options_dict:
            options.append(f"({key}) {options_dict[key]}")
        options = " ".join(options)
        question = f"{example['question'].strip()}\n选项: {options}"
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    # assert question != ""
    # Yes or No question
    _, gt_ans = parse_ground_truth(example, data_name)
    if isinstance(gt_ans, str):
        gt_lower = gt_ans.lower()
        if gt_lower in ["true", "false"]:
            question += " (True or False)"
        if gt_lower in ["yes", "no"]:
            question += " (Yes or No)"
    return question.strip()


def run_execute(executor, result, prompt_type, data_name, execute=False):
    if not result or result == "error":
        return None, None
    report = None

    if "program_only" in prompt_type:
        prediction = extract_program_output(result)
    elif prompt_type in ["pot", "pal"] and execute:
        code = extract_program(result)
        prediction, report = executor.apply(code)
    else:
        prediction = extract_answer(result, data_name)

    # prediction = strip_string(prediction, skip_unit=data_name == "carp_en")
    prediction = strip_string(prediction, skip_unit=data_name in STRIP_EXCEPTIONS)
    return prediction, report










# trajcectory.py

"""
trajcectory:
[
    {"role": "rationale", "content": "..."},
    {"role": "program", "content": "..."},
    {"role": "output", "content": "..."},
    {"role": "rationale", "content": "..."},
    ...
]
"""

def text_to_trajectory(traj_str: str) -> None:
    """
    """
    # parse the above interleaved string of raionale, program, output, raionale, program, output, ...
    # output a list of dict
    trajectory = []
    cur_role = "rationale"
    cur_content = ""

    # print(traj_str)
    for i, line in enumerate(traj_str.split("\n")):
        if line == "```python": # program begin
            assert cur_role == "rationale"
            if cur_content:
                trajectory.append({"role": cur_role, "content": cur_content})
                cur_content = ""
            cur_role = "program"
        elif cur_role == "program" and line == "```": # program end
            assert cur_content
            trajectory.append({"role": cur_role, "content": cur_content}) 
            cur_content = ""
            cur_role = "output"
        elif cur_role == "output" and line.startswith("```output"): # output begin
            assert cur_content == ""
        elif cur_role == "output" and line == "```": # output end
            trajectory.append({"role": cur_role, "content": cur_content})
            cur_content = ""
            cur_role = "rationale"
        else: # content
            cur_content += line
            if i < len(traj_str.split("\n")) - 1:
                cur_content += "\n"
    # the last content
    if cur_content:
        trajectory.append({"role": cur_role, "content": cur_content})
    return trajectory


def trajectory_to_text(trajectory: list) -> str:
    text = ""
    for item in trajectory:
        content = item["content"]
        if item["role"] == "program":
            content = f"```python\n{content}```\n"
        elif item["role"] == "output":
            content = f"```output\n{content}```\n"
        text += content
    return text


def is_execution_success(output):
    error_key_words = ["error", "exception", "no algorithms", "no algorithms", "cannot", "nan", "..."]
    success = all([k not in output.lower() for k in error_key_words])
    return success


def extract_program(text:str=None, trajectory:list=None, last_only=False) -> str:
    assert text is not None or trajectory is not None, "Either text or trajectory should be provided."
    if trajectory is None:
        try:
            trajectory = text_to_trajectory(text)
        except:
            return "raise ValueError('Invalid trajectory')"

    program_list = []
    import_lines = []
    for i, item in enumerate(trajectory):
        if item["role"] == "program":
            cur_program = item["content"]
            if i < len(trajectory) - 1:
                assert trajectory[i+1]["role"] == "output"
                output = trajectory[i+1]["content"].strip()
                if is_execution_success(output):
                    program_list.append(cur_program)
                else:
                    # extract import lines only
                    for line in cur_program.split("\n"):
                        if line.startswith("import") or line.startswith("from"):
                            import_lines.append(line)
            else:
                program_list.append(cur_program)
    # add import lines to the first program
    if len(program_list) == 0:
        program_list.append("")
    if len(import_lines) > 0:
        program_list[0] = "\n".join(import_lines) + "\n" + program_list[0]
    for i, program in enumerate(program_list[:-1]):
        program_list[i] = "\n".join([line for line in program.split("\n") if not line.strip().startswith("print(")])

    if last_only:
        program = program_list[-1]
    else:
        program = "\n".join(program_list)
    return program


def extract_program_output(pred_str, last_only=True):
    """
    extract output between ```output\n...\n```, use regex, there might be multiple outputs, each output may have multiple lines
    """
    outputs = re.findall(r"```output\n(.*?)\n```", pred_str, re.DOTALL)
    if last_only:
        return outputs[-1] if len(outputs) > 0 else ""
    else:
        return outputs










# grader.py

def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)

def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()










# evaluate.py

def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=16) as pool:
        future = pool.map(math_equal_process, params, timeout=10)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0]
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    return samples, result_json










# python_executor.py

class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []
    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r'(\s|^)?input\(', code_piece):
        # regex.search(r'(\s|^)?os.', code_piece):
            raise RuntimeError()
        exec(code_piece, self._global_vars)

        # TODO: use: https://github.com/shroominic/codebox-api
        # @high safe exec in sandbox
        # byte_code = compile_restricted(
        #     code_piece,
        #     filename='<inline code>',
        #     mode='exec'
        # )
        # print("global vars:", self._global_vars)
        # _print_ = PrintCollector
        # exec(byte_code, {'__builtins__': utility_builtins}, None)
        
    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)
    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']

class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        'datetime': datetime.datetime, 
        'timedelta': dateutil.relativedelta.relativedelta,
        'relativedelta': dateutil.relativedelta.relativedelta
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()

class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {'dict': CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 5,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.pool = Pool(multiprocess.cpu_count())
        self.timeout_length = timeout_length

    def process_generation_to_code(self, gens: str):
        return [g.strip().split('\n') for g in gens]

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout = None,
        runtime = None,
        answer_symbol = None,
        answer_expr = None,
        timeout_length = 10,
        auto_mode=False
    ):
        try:
            if auto_mode:
                if "print(" in code[-1]:
                    program_io = io.StringIO()
                    with redirect_stdout(program_io):
                        timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                    program_io.seek(0)
                    result = program_io.read()
                else:
                    print(code)
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code[:-1]))
                    result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            else:
                if get_answer_from_stdout:
                    program_io = io.StringIO()
                    with redirect_stdout(program_io):
                        timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                    program_io.seek(0)
                    result = program_io.read()
                elif answer_symbol:
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                    result = runtime._global_vars[answer_symbol]
                elif answer_expr:
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                    result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
                else:
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code[:-1]))
                    result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            report = "Done"
            str(result)
            pickle.dumps(result) # serialization check
        except:
            result = ''
            report = traceback.format_exc().split('\n')[-2]
        return result, report

    def apply(self, code):
        return self.batch_apply([code])[0]

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        # with ProcessPool(max_workers=min(len(all_code_snippets), os.cpu_count())) as pool:
        with ProcessPool(max_workers=min(len(all_code_snippets), 1)) as pool:
            executor = partial(
                self.execute,
                get_answer_from_stdout=self.get_answer_from_stdout,
                runtime=self.runtime,
                answer_symbol=self.answer_symbol,
                answer_expr=self.answer_expr,
                timeout_length=self.timeout_length, # this timeout not work
                auto_mode=True
            )
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            if len(all_code_snippets) > 100:  
                progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")  
            else:  
                progress_bar = None 

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    all_exec_results.append(("", "Timeout Error"))
                    timeout_cnt += 1
                except Exception as error:
                    print(error)
                    exit()
                if progress_bar is not None:
                    progress_bar.update(1) 
            
            if progress_bar is not None:
                progress_bar.close() 

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # post processing
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results