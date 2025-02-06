import re
import numpy as np
import sympy
from sympy.core.sympify import SympifyError
from sympy.parsing.latex import parse_latex

import signal

INVALID_ANSWER = "[invalidanswer]"


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def normalize_numeric(s):
    if s is None:
        return None
    for unit in [
        "eV",
        " \\mathrm{~kg} \\cdot \\mathrm{m} / \\mathrm{s}",
        " kg m/s",
        "kg*m/s",
        "kg",
        "m/s",
        "m / s",
        "m s^{-1}",
        "\\text{ m/s}",
        " \\mathrm{m/s}",
        " \\text{ m/s}",
        "g/mole",
        "g/mol",
        "\\mathrm{~g}",
        "\\mathrm{~g} / \\mathrm{mol}",
        "W",
        "erg/s",
        "years",
        "year",
        "cm",
    ]:
        s = s.replace(unit, "")
        s = s.strip()
    for maybe_unit in ["m", "s", "cm"]:
        s = s.replace("\\mathrm{" + maybe_unit + "}", "")
        s = s.replace("\\mathrm{~" + maybe_unit + "}", "")
        s = s.strip()
    s = s.strip("$")
    try:
        return float(eval(s))
    except:
        try:
            expr = parse_latex(s)
            if expr.is_number:
                return float(expr)
            return INVALID_ANSWER
        except:
            return INVALID_ANSWER


def numeric_equality(n1, n2, threshold=0.01):
    if n1 is None or n2 is None:
        return False
    if np.isclose(n1, 0) or np.isclose(n2, 0) or np.isclose(n1 - n2, 0):
        return np.abs(n1 - n2) < threshold * (n1 + n2) / 2
    else:
        return np.isclose(n1, n2)


def normalize_symbolic_equation(s):
    if not isinstance(s, str):
        return INVALID_ANSWER
    if s.startswith("\\["):
        s = s[2:]
    if s.endswith("\\]"):
        s = s[:-2]
    s = s.replace("\\left(", "(")
    s = s.replace("\\right)", ")")
    s = s.replace("\\\\", "\\")
    if s.startswith("$") or s.endswith("$"):
        s = s.strip("$")
    try:
        maybe_expression = parse_latex(s)
        if not isinstance(maybe_expression, sympy.core.relational.Equality):
            # we have equation, not expression
            return INVALID_ANSWER
        else:
            return maybe_expression
    except:
        return INVALID_ANSWER


class SymbolicMathMixin:
    """
    Methods useful for parsing mathematical expressions from text and determining equivalence of expressions.
    """

    SUBSTITUTIONS = [  # used for text normalize
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
    ]
    REMOVED_EXPRESSIONS = [  # used for text normalizer
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
    ]

    def normalize_tex(self, final_answer: str) -> str:
        """
        Normalizes a string representing a mathematical expression.
        Used as a preprocessing step before parsing methods.

        Copied character for character from appendix D of Lewkowycz et al. (2022)
        """
        final_answer = final_answer.split("=")[-1]

        for before, after in self.SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in self.REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

        # Normalize shorthand TeX:
        #  \fracab -> \frac{a}{b}
        #  \frac{abc}{bef} -> \frac{abc}{bef}
        #  \fracabc -> \frac{a}{b}c
        #  \sqrta -> \sqrt{a}
        #  \sqrtab -> sqrt{a}b
        final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
        final_answer = final_answer.replace("$", "")

        # Normalize 100,000 -> 100000
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")

        return final_answer

    def parse_tex(self, text: str, time_limit: int = 5) -> sympy.Basic:
        """
        Wrapper around `sympy.parse_text` that outputs a SymPy expression.
        Typically, you want to apply `normalize_text` as a preprocessing step.
        """
        try:
            with timeout(seconds=time_limit):
                parsed = parse_latex(text)
        except (
                # general error handling: there is a long tail of possible sympy/other
                # errors we would like to catch
                Exception
        ) as e:
            print(f"failed to parse {text} with exception {e}")
            return None

        return parsed

    def is_exp_equiv(self, x1: sympy.Basic, x2: sympy.Basic, time_limit=5) -> bool:
        """
        Determines whether two sympy expressions are equal.
        """
        try:
            with timeout(seconds=time_limit):
                try:
                    diff = x1 - x2
                except (SympifyError, ValueError, TypeError) as e:
                    print(
                        f"Couldn't subtract {x1} and {x2} with exception {e}"
                    )
                    return False

                try:
                    if sympy.simplify(diff) == 0:
                        return True
                    else:
                        return False
                except (SympifyError, ValueError, TypeError) as e:
                    print(f"Failed to simplify {x1}-{x2} with {e}")
                    return False
        except TimeoutError as e:
            print(f"Timed out comparing {x1} and {x2}")
            return False
        except Exception as e:
            print(f"failed on unrecognized exception {e}")
            return False

    def is_tex_equiv(self, x1: str, x2: str, time_limit=5) -> bool:
        """
        Determines whether two (ideally normalized using `normalize_text`) TeX expressions are equal.

        Does so by first checking for string exact-match, then falls back on sympy-equivalence,
        following the (Lewkowycz et al. 2022) methodology.
        """
        if x1 == x2:
            # don't resort to sympy if we have full string match, post-normalization 
            return True
        else:
            return False
        parsed_x2 = self.parse_tex(x2)
        if not parsed_x2:
            # if our reference fails to parse into a Sympy object, 
            # we forgo parsing + checking our generated answer.
            return False
        return self.is_exp_equiv(self.parse_tex(x1), parsed_x2, time_limit=time_limit)
