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

# NOTE: we temporarily disable sympy and latex2sympy to run save only
# from sympy import simplify, N
# from sympy.parsing.sympy_parser import parse_expr
# from sympy.parsing.latex import parse_latex
# from latex2sympy2 import latex2sympy





# get_examples
def get_examples():
    examples = {}
    examples["gsm8k"] = [
        (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
        ),
        (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
        ),
        (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
        ),
        (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
        ),
        (
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
        ),
        (
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
        ),
        (
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
        ),
        (
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
        ),
    ]
    examples["gsm8k-pal"] = [
        (
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            '```python\ndef solution():\n    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""\n    money_initial = 23\n    bagels = 5\n    bagel_cost = 3\n    money_spent = bagels * bagel_cost\n    money_left = money_initial - money_spent\n    result = money_left\n    return result\n```',
        ),
        (
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            '```python\ndef solution():\n    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""\n    golf_balls_initial = 58\n    golf_balls_lost_tuesday = 23\n    golf_balls_lost_wednesday = 2\n    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n    result = golf_balls_left\n    return result\n```',
        ),
        (
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            '```python\ndef solution():\n    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""\n    computers_initial = 9\n    computers_per_day = 5\n    num_days = 4  # 4 days between monday and thursday\n    computers_added = computers_per_day * num_days\n    computers_total = computers_initial + computers_added\n    result = computers_total\n    return result\n```',
        ),
    ]
    examples["gsm8k-tora"] = [
        (
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "```python\ndef money_left():\n    money_initial = 23\n    bagels = 5\n    bagel_cost = 3\n    money_spent = bagels * bagel_cost\n    remaining_money = money_initial - money_spent\n    return remaining_money\n \nremaining_money = money_left()\nprint(remaining_money)\n```\n```output\n8\n```\nOlivia has $\\boxed{8}$ dollars left.",
        ),
        (
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "```python\ndef remaining_golf_balls():\n    golf_balls_initial = 58\n    golf_balls_lost_tuesday = 23\n    golf_balls_lost_wednesday = 2\n    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n    remaining_golf_balls = golf_balls_left\n    return remaining_golf_balls\n\nanswer = remaining_golf_balls() \nprint(answer)\n```\n```output\n33\n```\nMichael had $\\boxed{33}$ golf balls at the end of Wednesday.",
        ),
        (
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "```python\ndef total_computers():\n    computers_initial = 9\n    computers_per_day = 5\n    num_days = 4  # 4 days between monday and thursday\n    computers_added = computers_per_day * num_days\n    computers_total = computers_initial + computers_added\n    return computers_total\n\ntotal_computers = total_computers()\nprint(total_computers)\n```\n```output\n29\n```\nThere're $\\boxed{29}$ computers in the server room.",
        ),
    ]

    examples["math"] = [
        (
            "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
            "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}}",
        ),
        (
            "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
            "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in  $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.\nThe answer is 63\\pi",
        ),
        (
            "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
            "Let's think step by step\nIf $(x,y)$ lies on the circle,\nso does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),\nso we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence,\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$\nso the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$\nThe answer is \\sqrt{2}",
        ),
        (
            "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
            "Let's think step by step\nThe condition $f(f(x))$ means that $f$ is the inverse of itself,\nso its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$,\nif we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$,\nand therefore $-d=a$ and $a+d=\\boxed{0}$.\nThe answer is 0",
        ),
        (
            "Expand $(2z^2 + 5z - 6)(3z^3 - 2z + 1)$.",
            "Let's think step by step\n$$\\begin{array}{crrrrrrr}\n& & & 3z^3 & & -2z & + 1 & \\\\\n\\times & & & & 2z^2 & +5z & -6 \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& & & -18z^3 & & +12z & -6 & \\\\\n& & +15z^4 & & -10z^2 & +5z & & \\\\\n+ & 6z^5 & & -4z^3 & +2z^2 & & & \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& 6z^5 & +15z^4 & -22z^3 & - 8z^2 &+17z & -6 &\n\\end{array}$$\nThe answer is 6z^5+15z^4-22z^3-8z^2+17z-6",
        ),
    ]

    examples["math_pal"] = [
        (
            "Display the final result in LaTeX.\n\n Find the coefficient of $x^3$ when $3(x^2 - x^3+x) +3(x +2x^3- 3x^2 + 3x^5+x^3) -5(1+x-4x^3 - x^2)$ is simplifie.",
            "```python\nfrom sympy import symbols, simplify\n\ndef solution():\n    x = symbols('x')\n    expr = 3*(x**2 - x**3 + x) + 3*(x + 2*x**3 - 3*x**2 + 3*x**5 + x**3) - 5*(1 + x - 4*x**3 - x**2)\n    simplified_expr = simplify(expr)\n\n    x3_coefficient = simplified_expr.as_coefficients_dict()[x**3]\n    result = x3_coefficient\n    return result\n```",
        ),
        (
            "The surface area of a sphere with radius $r$ is $4\\pi r^2$. Including the area of its circular base, what is the total surface area of a hemisphere with radius 6 cm? Express your answer in terms of $\\pi$.",
            "```python\nimport math\n\ndef solution():\n    radius = 6\n\n    # Surface area of the hemisphere\n    hemisphere_area = 2 * math.pi * radius**2\n\n    # Area of the circular base\n    base_area = math.pi * radius**2\n\n    # Total surface area\n    total_surface_area = hemisphere_area + base_area\n\n    # Formatting the result in LaTeX\n    result = r'{}\\\\pi'.format(total_surface_area / math.pi)\n    return result\n```",
        ),
        (
            "Monica tosses a fair 6-sided die.  If the roll is a prime number, then she wins that amount of dollars (so that, for example, if she rolls 3, then she wins 3 dollars).  If the roll is composite, she wins nothing. Otherwise, she loses 3 dollars. What is the expected value of her winnings on one die toss? Express your answer as a dollar value to the nearest cent.",
            '```python\ndef solution():\n    # Probabilities of each outcome\n    prime_prob = 1 / 6\n    composite_prob = 1 / 3\n    otherwise_prob = 1 / 6\n\n    # Expected value of each outcome\n    prime_expected_value = (2 * prime_prob) + (3 * prime_prob) + (5 * prime_prob)\n    composite_expected_value = 0 * composite_prob\n    otherwise_expected_value = -3 * otherwise_prob\n\n    # Total expected value\n    total_expected_value = prime_expected_value + composite_expected_value + otherwise_expected_value\n\n    # Dollar value to the nearest cent\n    result = "{:.2f}".format(total_expected_value)\n    return result\n```',
        ),
        (
            "Given $\\mathbf{a} = \\begin{pmatrix} -7 \\\\ 0 \\\\ 1 \\end{pmatrix}$ and $\\mathbf{b} = \\begin{pmatrix} 4 \\\\ 2 \\\\ -1 \\end{pmatrix},$ find $\\mathbf{a} - 3 \\mathbf{b}.$",
            "```python\nimport numpy as np\n\ndef solution()\n    a = np.array([-7, 0, 1])\n    b = np.array([4, 2, -1])\n\n    result = a - 3 * b\n\n    result = r'\\begin{{pmatrix}} {} \\\\ {} \\\\ {} \\end{{pmatrix}}'.format(result[0], result[1], result[2])\n    return result\n```",
        ),
    ]
    examples["math-tora"] = [
        (
            "Convert the point $(0, -3 \\sqrt{3}, 3)$ in rectangular coordinates to spherical coordinates.  Enter your answer in the form $(\\rho,\\theta,\\phi),$ where $\\rho > 0,$ $0 \\le \\theta < 2 \\pi,$ and $0 \\le \\phi \\le \\pi.$",
            "To convert the point $(x, y, z)$ in rectangular coordinates to spherical coordinates $(\\rho, \\theta, \\phi)$, we use the following formulas:\n1. $\\rho = \\sqrt{x^2 + y^2 + z^2}$\n2. $\\theta = \\arctan{\\frac{y}{x}}$\n3. $\\phi = \\arccos{\\frac{z}{\\rho}}$\n\nLet's write a function:\n\n```python\nfrom sympy import sqrt, atan2, acos, pi\n\ndef rectangular_to_spherical():\n    x, y, z = 0, -3*sqrt(3), 3\n    rho = sqrt(x**2 + y**2 + z**2)\n    theta = atan2(y, x)\n    phi = acos(z/rho)\n    return rho, theta, phi\n\nspherical_coordinates = rectangular_to_spherical()\nprint(spherical_coordinates)\n```\n```output\n(6, -pi/2, pi/3)\n```\n\nThe required form of the answer is $(\\rho, \\theta, \\phi)$, where $\\rho > 0$, $0 \\le \\theta < 2\\pi$, and $0 \\le \\phi \\le \\pi$. Since our initial output has $\\theta = -\\pi/2$, which doesn't meet the condition for $\\theta$, we add $2\\pi$ to adjust its value.\n\nHence, the spherical coordinates for the point $(0, -3\\sqrt{3}, 3)$ are $\\boxed{(6, \\frac{3\\pi}{2}, \\frac{\\pi}{3})}$.",
        ),
        (
            "Compute $1011_2 + 101_2 - 1100_2 + 1101_2$.  Express your answer in base 2.",
            'Let\'s write a single python function for binary computation:\n\n```python\ndef binary_sum_diff():\n    # Convert binary to decimal\n    num1 = int("1011", 2)\n    num2 = int("101", 2)\n    num3 = int("1100", 2)\n    num4 = int("1101", 2)\n\n    # Perform the operations\n    result = num1 + num2 - num3 + num4\n\n    # Convert the result back to binary\n    result_binary = format(result, "b")\n\n    return result_binary\n\nresult = binary_sum_diff()\nprint(result)\n```\n```output\n10001\n```\nThe answer expressed in base 2 is $\\boxed{10001_2}$.',
        ),
        (
            "Find the matrix that corresponds to projecting onto the vector $\\begin{pmatrix} 2 \\\\ -3 \\end{pmatrix}.$",
            "```python\nfrom sympy import Matrix, simplify\n\ndef projection_matrix():\n    v = Matrix([2, -3])\n\n    # Calculate the projection matrix\n    P = (v * v.T) / (v.T * v)[0]\n\n    # Simplify the elements\n    P = P.applyfunc(simplify)\n\n    return P\n\nresult = projection_matrix()\nprint(result)\n```\n```output\nMatrix([[4/13, -6/13], [-6/13, 9/13]])\n```\n\nThe matrix is $\\boxed{\\begin{pmatrix} \\frac{4}{13} & -\\frac{6}{13} \\ -\\frac{6}{13} & \\frac{9}{13} \\end{pmatrix}}.$",
        ),
        (
            "Solve the inequality\n\\[\\frac{x^2 - 25}{x + 5} < 0.\\]",
            "Let's use `sympy` to solve the inequality:\n\n```python\nfrom sympy import symbols, simplify\nfrom sympy.solvers.inequalities import solve_univariate_inequality\nfrom sympy.core.relational import StrictLessThan\n\ndef solve_inequality():\n    x = symbols('x')\n    expression = (x**2 - 25) / (x + 5)\n    inequality = StrictLessThan(expression, 0)\n    solution = solve_univariate_inequality(inequality, x, relational=False)\n    simplified_solution = simplify(solution)\n    \n    return simplified_solution\n\nresult = solve_inequality()\nprint(result)\n```\n```output\nUnion(Interval.open(-oo, -5), Interval.Lopen(-5, 5))\n```\nThe answer is $\\boxed{(-\\infty,-5)\\cup(-5,5)}$",
        ),
        (
            'In the figure, triangles $ABC$ and $BCD$ are equilateral triangles. What is the value of $AD \\div BC$ when expressed in simplest radical form?\n\n[asy]\ndraw((0,0)--(5,8.7)--(10,0)--cycle);\ndraw((10,0)--(15,8.7)--(5,8.7));\nlabel("$A$",(0,0),SW);\nlabel("$B$",(5,8.7),N);\nlabel("$C$",(10,0),SE);\nlabel("$D$",(15,8.7),NE);\n[/asy]',
            "```python\nfrom sympy import Rational, sqrt, simplify\n\ndef ad_divided_by_bc():\n\n    x = Rational(1, 1)  # Side length of equilateral triangles\n\n    ad_squared = 2 * x**2 * (1 + Rational(1, 2))  # Using the law of cosines with cos(2*pi/3) = -1/2\n    ad = sqrt(ad_squared)\n\n    bc = x # BC is the side length of the equilateral triangles\n\n    simplified_ratio = simplify(ad / bc)\n\n    return simplified_ratio\n\nresult = ad_divided_by_bc()\nprint(result)\n```\n```output\nsqrt(3)\n```\nThe value of $AD \\div BC$ is $\\boxed{\\sqrt{3}}$.",
        ),
    ]
    examples["carp_en"] = [
        (
            "If $3 a ^ { m + 2 } b$ and $\\frac { 1 } { 2 } ab ^ { n - 1 }$ are similar terms, then $m + n$ is equal to.",
            "Let's think step by step\n$3 a ^ { m + 2 } b$ and $\\frac { 1 } { 2 } ab ^ { n - 1 }$ are like terms. We can obtain $m + 2 = 1$ and $n - 1 = 1$. Solving for $m$ and $n$, we get $m = - 1$ and $n = 2$. Therefore, $m + n = - 1 + 2 = 1$.\nThe answer is: 1",
        ),
        (
            "The solution to the equation $y - \\frac { y - 1 } { 2 } = - \\frac { y + 2 } { 5 }$ is ____ ?",
            "Let's think step by step\nTo eliminate the denominator, we have $10 y - 5 ( y - 1 ) = - 2 ( y + 2 )$. Expanding the brackets gives $10 y - 5 y + 5 = - 2 y - 4$. Rearranging terms gives $10 y - 5 y + 2 y = - 4 - 5$, which simplifies to $7 y = - 9$. Dividing both sides by 7 gives $y = - \\frac { 9 } { 7 }$.\nThe answer is: y = - \\frac { 9 } { 7 }",
        ),
        (
            "If $( m + 4 ) ^ 2 + | n - 3 | = 0$, then $\\frac { 1 } { 2 } m - n$ = ____?",
            "Let's think step by step\n$\\because ( m + 4 ) ^ 2 + | n - 3 | = 0$, $\\therefore m + 4 = 0$, $n - 3 = 0$, which means $m = - 4$, $n = 3$. Then the original expression equals $- 2 - 3 = - 5$.\nThe answer is: - 5",
        ),
        (
            "Given a quadratic equation in one variable $x$, $x ^ 2 + x + m = 0$, with one root being $x = 1$, what is the other root of this equation?",
            "Let's think step by step\nSuppose the quadratic equation in one variable about $x$ is $x ^ 2 + x + m = 0$, and $\\alpha$ is another real root of the equation. Since one real root of the quadratic equation in one variable about $x$ is $1$, we have $\\alpha + 1 = - 1$. Therefore, $\\alpha = - 2$.\nThe answer is: - 2",
        ),
        (
            "The parabola $y = - 5 { x } ^ 2 + 1$ is translated $2$ units upward and $1$ unit to the left, resulting in the parabola _____.",
            "Let's think step by step\nThe parabola $y = - 5 { x } ^ 2 + 1$ is first shifted upward by 2 units, resulting in $y = - 5 { x } ^ 2 + 3$. Then it is shifted left by 1 unit, resulting in $y = - 5 {( x + 1 )} ^ 2 + 3$.\nThe answer is: y = - 5 ( x + 1 )  ^ { 2 } + 3",
        ),
        (
            "If the radical $\\sqrt { x - 8 }$ is defined, then the range of real numbers for $x$ is ____ ?",
            "Let's think step by step\nSince the radical $\\sqrt { x - 8 }$ is defined, therefore $x - 8 \\ge 0$, which implies $x \\ge 8$.\nThe answer is: x \\ge 8",
        ),
        (
            "If $a ^ { m } \\times a ^ { 2 } = a ^ { 7 }$, then the value of $m$ is ____?",
            "Let's think step by step\nAccording to the multiplication rule of powers with the same base: when multiplying powers with the same base, keep the base the same and add the exponents. We have $m + 2 = 7$, so solving for $m$ gives $m = 5$.\nThe answer is: 5",
        ),
        (
            "If line segment $a$ and $b$ satisfy $\\frac { a } { b } = \\frac { 5 } { 2 }$, then the value of $\\frac { a - b } { b }$ is ____?",
            "Let's think step by step\n$\\because \\frac { a } { b } = \\frac { 5 } { 2 }$, $\\therefore$ we can assume $a = 5 k$, then $b = 2 k$, $\\therefore \\frac { a - b } { b } = \\frac { 5 k - 2 k } { 2 k } = \\frac { 3 } { 2 }$.\nThe answer is: \\frac { 3 } { 2 }",
        ),
    ]

    examples["minerva_math"] = [
        (
            "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "The expressions inside each square root must be non-negative.\nTherefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$.\nAlso, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.\nTherefore, the domain of the expression is $\\boxed{[2,5)}$.",
        ),
        (
            "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
        ),
        (
            "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{align*}\n30n&=480\\\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}",
        ),
        (
            "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$",
        ),
    ]

    examples["aqua"] = [
        (
            "John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?\nAnswer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64",
            "If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. The answer is (A).",
        ),
        (
            "If a / b = 3/4 and 8a + 5b = 22,then find the value of a.\nAnswer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2",
            "a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2. The answer is (B).",
        ),
        (
            "A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?\nAnswer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km",
            "The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. The answer is (E).",
        ),
        (
            "How many keystrokes are needed to type the numbers from 1 to 500?\nAnswer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788",
            "There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392. The answer is (B).",
        ),
    ]
    examples["sat_math"] = [
        (
            "If $\frac{x-1}{3}=k$ and $k=3$, what is the value of $x$ ? \nAnswer Choices: (A) 2 (B) 4 (C) 9 (D) 10",
            "If k = 3, then x - 1 = 3 * 3, therfore, x - 1 = 9 and x = 10. The answer is D",
        ),
        (
            "For $i=\\sqrt{-1}$, what is the sum $(7+3 i)+(-8+9 i)$ ? \nAnswer Choices: (A) $-1+12 i$ (B) $-1-6 i$ (C) $15+12 i$ (D) $15-6 i$ 3",
            "For (7+3 i)+(-8+9 i), the real part is 7 + (-8) = -1, the imageinary part is 3 i + 9 i = 12 i. The answer is A",
        ),
        (
            "On Saturday afternoon, Armand sent $m$ text messages each hour for 5 hours, and Tyrone sent $p$ text messages each hour for 4 hours. Which of the following represents the total number of messages sent by Armand and Tyrone on Saturday afternoon?\nAnswer Choices: (A) $9 m p$ (B) $20 m p$ (C) $5 m+4 p$ (D) $4 m+5 p$",
            "Armand texts m messages each hour for 5 hours, which leads to 5m messages. Tyrone texts p messages each hour for 4 hours, which leds to 4p messages. The total is 5m + 4p. The answer is C.",
        ),
        (
            "$$\begin{array}{r}3 x+4 y=-23 \\2 y-x=-19\\end{array}$$What is the solution $(x, y)$ to the system of equations above?\nAnswer Choices: (A) $(-5,-2)$ (B) $(3,-8)$ (C) $(4,-6)$ (D) $(9,-6)$",
            "By solving this equation, we found that x = 3 and y = -8. The answer is B.",
        ),
    ]
    examples["mmlu_mathematics"] = [
        (
            "Simplify and write the result with a rational denominator: $$\\sqrt{\\sqrt[3]{\\sqrt{\frac{1}{729}}}}$$\nAnswer Choices: (A) \\frac{3\\sqrt{3}}{3} (B) \\frac{1}{3} (C) \\sqrt{3} (D) \\frac{\\sqrt{3}}{3}",
            "Factoring $729=3^6$ and combining the roots $\frac{1}{2}\frac{1}{3}\frac{1}{2}=\frac{1}{12}$, we get that $\\sqrt{\\sqrt[3]{\\sqrt{\frac{1}{729}}}}=\\left(\frac{1}{3^6}\right)^{\frac{1}{12}}=\frac{1}{3^{\frac{1}{2}}}=\frac{3}{\\sqrt{3}}$. The answer is (D).",
        ),
        (
            "Five thousand dollars compounded annually at an $x\\%$ interest rate takes six years to double. At the same interest rate, how many years will it take $\\$300$ to grow to $\\$9600$?\nAnswer Choices:(A) 12 (B) 1 (C) 30 (D) 5",
            "To go from $\\$300$ to $\\$9600$, the value must go up by a factor of $9600/300=32=2^5$. Since at this interest rate it takes six years for it to double, it will take $5*6=30$ years to grow to $\\$9600$. The answer is (C).",
        ),
        (
            "Ten students take a biology test and receive the following scores: 45, 55, 50, 70, 65, 80, 40, 90, 70, 85. What is the mean of the students’ test scores?\nAnswer Choices: (A) 55 (B) 60 (C) 62 (D) 65",
            "There are 10 students and the sum of their scores is $45 + 55 + 50 + 70 + 65 + 80 + 40 + 90 + 70 + 85 = 650$, the mean is $650/10=65$. The answer is (D).",
        ),
        (
            "The variable $x$ varies directly as the square of $y$, and $y$ varies directly as the cube of $z$. If $x$ equals $-16$ when $z$ equals 2, what is the value of $x$ when $z$ equals $\frac{1}{2}$?\nAnswer Choices: (A) -1 (B) 16 (C) -\frac{1}{256} (D) \\frac{1}{16}",
            "We know that $x \\propto y^2$ and $y \\propto z^3$, so $x = k z^6$ for some constant $k$. Plugging in for $x=-16$ and $z=2$, the constant value is $k=\frac{x}{z^6}=\frac{-16}{64}=-\frac{1}{4}$. So, when $z=\frac{1}{2}$, the value of $x$ is $x=kz^6=-\frac{1}{4}\frac{1}{2^6}=-\frac{1}{256}$. The answer is (C).",
        ),
        (
            "Joe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.)\nAnswer Choices: (A) 3 (B) 15 (C) 6 (D) 5",
            "The least common multiple of 2, 3 and 5 is 30, so during a 7 minute dance, all the three lights will come on at the same time $2*7+1=15$ times. The answer is (B).",
        ),
    ]
    examples["mmlu_physics"] = [
        (
            "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?\nAnswer Choices: (A) 10 W (B) 30 W (C) 60 W (D) 240 W",
            "Rate of energy usage is known as power; in an dissipative electrical circuit, power is given by voltage times current. So in our case, the power is 120 V times 2 amps, or 240 W. The answer is (D).",
        ),
        (
            "A point charge, Q = +1 mC, is fixed at the origin. How much work is required to move a charge, Q = +8 µC, from the point (0, 4 meters) to the point (3 meters, 0)?\nAnswer Choices: (A) 3.5 J (B) 6.0 J (C) 22.5 J (D) 40 J",
            "To calculate the work required to move a charge from one location to another in a fixed electric field, it is enough to calculate the potential difference between the two locations. Here, the potential only depends on the distance between the charges; it’s $k q_1 q_2 / r$, where $k$ is Coulomb’s constant. Plugging in values $q_1 = $ 1 mC, $q_2 = 8 \\mu$ C, gives the answer as 5.992 J, which rounds to 6 J. The answer is (B).",
        ),
        (
            "Which of the following conditions will ensure that angular momentum is conserved? I. Conservation of linear momentum II. Zero net external force III. Zero net external torque.\nAnswer Choices: (A) I and II only (B) I and III only (C) II and III only (D) III only",
            "Torque is defined as the change in angular momentum; if there is zero external torque, angular momentum is conserved. The answer is (D).",
        ),
        (
            "A photocell of work function ϕ = 2eV is connected to a resistor in series. Light of frequency f = 1 × 10^15 Hz hits a metal plate of the photocell. If the power of the light is P = 100 W, what is the current through the resistor?\nAnswer Choices: (A) 2:00 AM (B) 6:00 AM (C) 12:00 AM (D) 24 A",
            "The only answer above which has units of current is D, 24 A. The answer is (D).",
        ),
        (
            "A pipe full of air is closed at one end. A standing wave is produced in the pipe, causing the pipe to sound a note. Which of the following is a correct statement about the wave’s properties at the closed end of the pipe?\nAnswer Choices: (A) The pressure is at a node, but the particle displacement is at an antinode. (B) The pressure is at an antinode, but the particle displacement is at a node. (C) The pressure and the particle displacement are both at nodes. (D) The pressure and the particle displacement are both at antinodes.",
            "At the closed end of the pipe, the particles cannot have any net displacement because the pipe closure stops them. So the particle displacement is at a node. This closure also causes the pressure to be maximal, i.e. an antinode. The answer is (B).",
        ),
    ]
    examples["mmlu_chemistry"] = [
        (
            "Which of the following is considered an acid anhydride?\nAnswer Choices: (A) HCl (B) H2SO3 (C) SO2 (D) Al(NO3)3",
            "An acid anhydride is a compound that is derived by removing water from an acid. The chemical formula for water is H2O, which means that we need to determine which of these options, when combined with H2O, forms an acid. SO2, or Sulfur dioxide, when combined with H2O, makes H2SO4, or sulfuric acid. The answer is (C).",
        ),
        (
            "Which of the following is expected to be a polar molecule?\nAnswer Choices: (A) PCl4F (B) BF3 (C) CO2 (D) Si(CH3)4",
            "A polar molecule is one that has a slightly positive charge on one end of the molecule and a slightly negative charge on the other end. Boron trifluoride (BF3) has Boron as the center atom and three fluorine atoms attached to it; it is trigonal planar and symmetric, so it is nonpolar. Carbon Dioxide (CO2) has Carbon as the central atom with double bonds to two Oxygen atoms - this is also symmetrical and therefore nonpolar. The same is the case for tetramethyl silane (SI(CH3)4), which is a Silicon atom surrounded by four methyl groups. The structure of PCL4F is that Phosphorus is the central atom, attached to four chlorines and one fluorine atom. This is asymmetrical, and therefore has a net dipole and is expected to be a polar molecule. The answer is (A).",
        ),
        (
            "From the solubility rules, which of the following is true?\nAnswer Choices: (A) All chlorides, bromides, and iodides are soluble (B) All sulfates are soluble (C) All hydroxides are soluble (D) All ammonium-containing compounds are soluble",
            "The chlorides, bromides, and iodides of lead, silver, and mercury are not soluble in water. This rules out (A). The sulfates of lead, barium, and calcium are not soluble in water, which rules out (B). The hydroxides of any metal besides sodium, potassium, ammonium, calcium, and barium are insoluble. This rules out (C). Typically ammonium ions indicate a soluble ionic substance. The answer is (D).",
        ),
        (
            "A new compound is synthesized and found to be a monoprotic acid with a molar mass of 248 g/mol. When 0.0050 mol of this acid are dissolved in 0.500 L of water, the pH is measured as 3.89. What is the pKa of this acid?\nAnswer Choices: (A) 3.89 (B) 7.78 (C) 5.78 (D) 2.33",
            "Recall that $[A] = [H^{+}]$. Here, this is equal to $$10^{-3.89}$. Then we have $K_{a} = $frac{[H^{+}][A^{-}]}{[HA]} = \\frac{10^{-3.89} \\cdot 10^{-3.89}}{10^{-2}}. The resulting exponent is $-3.89 + (-3.89) - (-2) = 5.78$, therefore $K_a = 10^{-5.78}$. The $pK_a$ is the negative log of $K_a$, which is equal to $5.78$. The answer is (C).",
        ),
        (
            "A solution contains 2.00 mole of acetic acid, CH3COOH, and 1.00 mole of calcium acetate, Ca(CH3COO)2. The solution is able to resist the addition of a small amount of strong acid or strong base with only minor changes in the pH of the solution. Larger quantities of strong acid or strong base can cause a significant change in pH. How many moles of nitric acid, HNO3, may be added before the pH begins to change significantly?\nAnswer Choices: (A) 0.500 mole (B) 1.00 mole (C) 2.00 mole (D) 3.00 mole",
            "We would like to compute the buffer capacity of this solution. First we write the equation for the ionization of the weak acid, in this case of acetic acid. $CH_{3}COOH (aq) + H_{2}O \rightarrow H_{3}O^{+} + CH3COO^{-}$. The conjugate base is therefore the acetate ion. The added strong acid, Nitric acid, will react with the conjugate base. Therefore the maximum amount of acid that can be added will be equal to the amount of acetate ion, or 2 moles. The answer is (C).",
        ),
    ]
    examples["mmlu_biology"] = [
        (
            "In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell?\nAnswer Choices: (A) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (B) Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane (C) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (D) Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane",
            "Protein synthesis starts at the ribosome, so we can eliminate (A) and (C). The ribosome is often in the endoplasmic reticulum and moves from there to the Golgi apparatus, where it is modified and packaged into a vesicle. The vesicle then floats to the plasma membrane and is secreted. The answer is (D).",
        ),
        (
            "A mutation in a bacterial enzyme changed a previously polar amino acid into a nonpolar amino acid. This amino acid was located at a site distant from the enzyme’s active site. How might this mutation alter the enzyme’s substrate specificity?\nAnswer Choices: (A) By changing the enzyme’s pH optimum (B) By changing the enzyme’s location in the cell (C) By changing the shape of the protein (D) An amino acid change away from the active site cannot alter the enzyme’s substrate specificity.",
            "A change in an amino acid leads to a change in the primary structure of the protein. A change in the primary structure may lead to a change in the secondary and the tertiary structure of the protein. A change in the tertiary structure means a change in the shape of the protein, so (C) has to be correct. Since the change does not affect the active site of the enzyme, we do not expect the activity of the enzyme to be affected. The answer is (C).",
        ),
        (
            "Which of the following is not a way to form recombinant DNA?\nAnswer Choices: (A) Translation (B) Conjugation (C) Specialized transduction (D) Transformation",
            "The introduction of foreign DNA or RNA into bacteria or eukaryotic cells is a common technique in molecular biology and scientific research. There are multiple ways foreign DNA can be introduced into cells including transformation, transduction, conjugation, and transfection. In contrast, (A) is not a way to form DNA: during translation the ribosomes synthesize proteins from RNA. The answer is (A).",
        ),
        (
            "Homologous structures are often cited as evidence for the process of natural selection. All of the following are examples of homologous structures EXCEPT\nAnswer Choices: (A) the wings of a bird and the wings of a bat (B) the flippers of a whale and the arms of a man (C) the pectoral fins of a porpoise and the flippers of a seal (D) the forelegs of an insect and the forelimbs of a dog",
            "Homologous structures are similar physical features in organisms that share a common ancestor ​​but different functions. Comparisons (B) and (C) are clearly homologous because they share a common ancestor and the structures serve different purposes. Bat wings and birg wings are also homologous, while they are both wings, the forelimbs serve different purposes. Insects and dogs are very far ancestors since one is vertebrate while the other is invertebrate and the forelimbs serve the same purpose, so they are not homologous. The answer is (D).",
        ),
        (
            "Which of the following is not known to be involved in the control of cell division?\nAnswer Choices: (A) Cyclins (B) Protein kinases (C) Checkpoints (D) Fibroblast cells",
            "Normal cells move through the cell cycle in a regulated way. At the checkpoint stage, they use information about their own internal state and cues from the environment around them to decide whether to proceed with cell division. Cues like these act by changing the activity of core cell cycle regulators inside the cell. The most common regulators are cyclins and cyclin-dependent kinases. Fibroblast cells do not play any role in cell division. The answer is (D).",
        ),
    ]
    examples["mmlu_computer"] = [
        (
            "Which of the following is an example of the use of a device on the Internet of Things (IoT) ?\nAnswer Choices: (A) A car alerts a driver that it is about to hit an object. (B) A hiker uses a G P S watch to keep track of her position. (C) A refrigerator orders milk from an online delivery service when the milk in the refrigerator is almost gone. (D) A runner uses a watch with optical sensors to monitor his heart rate.",
            "The term Internet of Things (IoT) refers to common devices which are connected to the internet, enabling new functionality. Choice A is incorrect because it does not describe an internet connected device. In choice B, the watch is only described as having GPS functionality but no internet connectivity. Choice C describes a common device (a refrigerator) which has internet connectivity enabling new functionality (online ordering). Choice D does not mention internet connectivity for the watch, only optical sensors. The answer is (C).",
        ),
        (
            "Many Web browsers allow users to open anonymous windows. During a browsing session in an anonymous window, the browser does not record a browsing history or a list of downloaded files. When the anonymous window is exited, cookies created during the session are deleted. Which of the following statements about browsing sessions in an anonymous window is true?\nAnswer Choices: (A) The activities of a user browsing in an anonymous window will not be visible to people who monitor the user's network, such as the system administrator. (B) Items placed in a Web store's shopping cart for future purchase during the anonymous browsing session will not be saved on the user's computer. (C) A user will not be able to log in to e-mail or social media accounts during the anonymous browsing session. (D) A user browsing in an anonymous window will be protected from viruses launched from any web sites visited or files downloaded.",
            "Choice A is incorrect as it only describes network traffic, which an anonymous browser does not change. Choice B is correct as it correctly describes how an anonymous browser will prevent saving data on the user’s computer after the session is ended. Choice C is incorrect because an anonymous browser will not prevent logging in to email or social media accounts. Choice D is incorrect because an anonymous browser in itself performs no virus protection. The answer is (B).",
        ),
        (
            'What is the output of "abc"[::-1] in Python 3? \nAnswer Choices: (A) Error (B) abc (C) cba (D) c',
            'We know that the slicing operator [::-1] takes all of the elements in the string in reverse order, so we reverse the order of the string "abc", resulting in "cba". The answer is (C).',
        ),
        (
            'In the program below, the initial value of X is 5 and the initial value of Y is 10.\nIF (X < 0){\n DISPLAY ("Foxtrot")\n} ELSE {\n IF (X > Y){\n  DISPLAY ("Hotel")\n } ELSE {\n  IF (Y > 0){\n   DISPLAY ("November")\n  } ELSE {\n   DISPLAY ("Yankee")\n  }\n }\n}\nWhat is displayed as a result of running the program?\nAnswer Choices: (A) Foxtrot (B) Hotel (C) November (D) Yankee',
            'Because X has the value 5, the first conditional IF (X < 0) is false, so we move to the first ELSE clause. Because X is 5 and Y is 10, the second conditional IF (X > Y) is false, so we move to the following ELSE clause. Since Y is 10, the conditional IF (Y > 0) is true, so the command DISPLAY ("November") is executed. The answer is (C).',
        ),
        (
            "A list of numbers has n elements, indexed from 1 to n. The following algorithm is intended to display the number of elements in the list that have a value greater than 100. The algorithm uses the variables count and position. Steps 3 and 4 are missing.\n Step 1: Set count to 0 and position to 1.\n Step 2: If the value of the element at index position is greater than 100, increase the value of count by 1.\n Step 3: (missing step)\n Step 4: (missing step)\n Step 5: Display the value of count.\nWhich of the following could be used to replace steps 3 and 4 so that the algorithm works as intended?\nAnswer Choices: (A) Step 3: Increase the value of position by 1.\n  Step 4: Repeat steps 2 and 3 until the value of count is greater than 100.\n(B) Step 3: Increase the value of position by 1.\n  Step 4: Repeat steps 2 and 3 until the value of position is greater than n.\n(C) Step 3: Repeat step 2 until the value of count is greater than 100.\n  Step 4: Increase the value of position by 1.\n(D) Step 3: Repeat step 2 until the value of position is greater than n.\n  Step 4: Increase the value of count by 1.",
            "Choice A is incorrect, because its Step 4 has an incorrect termination condition, stopping when count is greater than 100. We need to stop after inspecting all elements in the list. Choice B is correct because it correctly increments both count and position, and correctly repeats these steps and terminates when all elements in the list have been inspected. Choice C is incorrect because it incorrectly increments the variable count until its value is greater than 100, regardless of the elements in the list. Choice D is incorrect because its step 3 does not increment the value of position, so it will repeat forever. The answer is (B).",
        ),
    ]
    # mammoth
    examples["mmlu_stem"] = [
        (
            "Simplify and write the result with a rational denominator: $$\\sqrt{\\sqrt[3]{\\sqrt{\frac{1}{729}}}}$$\nAnswer Choices: (A) \\frac{3\\sqrt{3}}{3} (B) \\frac{1}{3} (C) \\sqrt{3} (D) \\frac{\\sqrt{3}}{3}",
            "Factoring $729=3^6$ and combining the roots $\\frac{1}{2}\\frac{1}{3}\\frac{1}{2}=\\frac{1}{12}$, we get that $\\sqrt{\\sqrt[3]{\\sqrt{\frac{1}{729}}}}=\\left(\frac{1}{3^6}\right)^{\frac{1}{12}}=\frac{1}{3^{\frac{1}{2}}}=\frac{3}{\\sqrt{3}}$. The answer is (D).",
        ),
        (
            "In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell?\nAnswer Choices: (A) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (B) Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane (C) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (D) Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane",
            "Protein synthesis starts at the ribosome, so we can eliminate (A) and (C). The ribosome is often in the endoplasmic reticulum and moves from there to the Golgi apparatus, where it is modified and packaged into a vesicle. The vesicle then floats to the plasma membrane and is secreted. The answer is (D).",
        ),
        (
            "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?\nAnswer Choices: (A) 10 W (B) 30 W (C) 60 W (D) 240 W",
            "Rate of energy usage is known as power; in an dissipative electrical circuit, power is given by voltage times current. So in our case, the power is 120 V times 2 amps, or 240 W. The answer is (D).",
        ),
        (
            "Which of the following is considered an acid anhydride?\nAnswer Choices: (A) HCl (B) H2SO3 (C) SO2 (D) Al(NO3)3",
            "An acid anhydride is a compound that is derived by removing water from an acid. The chemical formula for water is H2O, which means that we need to determine which of these options, when combined with H2O, forms an acid. SO2, or Sulfur dioxide, when combined with H2O, makes H2SO4, or sulfuric acid. The answer is (C).",
        ),
        (
            'What is the output of "abc"[::-1] in Python 3? \nAnswer Choices: (A) Error (B) abc (C) cba (D) c',
            'We know that the slicing operator [::-1] takes all of the elements in the string in reverse order, so we reverse the order of the string "abc", resulting in "cba". The answer is (C).',
        ),
    ]
    examples["gaokao"] = [
        (
            "已知 $\\alpha, \\beta, \\gamma$ 是互不相同的锐角, 则在 $\\sin \\alpha \\cos \\beta, \\sin \\beta \\cos \\gamma, \\sin \\gamma \\cos \\alpha$ 三个值中, 大于 $\\frac{1}{2}$ 的个数的最大值是 ( )\n从以下选项中选择:\n(A) 0\n(B) 1\n(C) 2\n(D) 3",
            "1. 如果 $\\alpha, \\beta, \\gamma$ 均小于 $60^\\circ$，那么他们的正弦值都小于 $\\frac{1}{2}$，因此三个值中不可能有大于 $\\frac{1}{2}$ 的值。\n2. 如果有一个角大于 $60^\\circ$，假设为 $\\alpha$，那么对应的正弦值大于 $\\frac{1}{2}$。此时，由于三角形内角和为 $180^\\circ$，所以 $\\beta + \\gamma < 120^\\circ$。这意味着 $\\beta, \\gamma$ 的余弦值均大于 $\\frac{1}{2}$，所以此时 $\\sin \\alpha \\cos \\beta > \\frac{1}{2}, \\sin \\beta \\cos \\gamma > \\frac{1}{2}$。\n3. 如果有两个角大于 $60^\\circ$，例如 $\\alpha$ 和 $\\beta$，那么由于三角形内角和为 $180^\\circ$，我们可以得到 $\\gamma < 60^\\circ$，此时 $\\sin \\gamma < \\frac{1}{2}$。由于 $\\alpha$ 和 $\\beta$ 的余弦值都小于 $\\frac{1}{2}$，因此三个值中不可能有大于 $\\frac{1}{2}$ 的值。\n4. 如果三个角都大于 $60^\\circ$，显然不符合题意。\n综上所述，当有一个角大于 $60^\\circ$ 时，大于 $\\frac{1}{2}$ 的个数的最大值是 2。\n答案是 C",
        ),
        (
            "正方体 $A B C D-A_{1} B_{1} C_{1} D_{1}$ 中, $B B_{1}$ 与平面 $A C D_{1}$ 所成角的余弦值为 ( )\n从以下选项中选择:\n(A) $\\frac{\\sqrt{2}}{3}$\n(B) $\\frac{\\sqrt{3}}{3}$\n(C) $\\frac{2}{3}$\n(D) $\\frac{\\sqrt{6}}{3}$",
            "设上下底面的中心分别为 $\\mathrm{O}_{1}, \\mathrm{O}$, 设正方体的棱长等于 1 , 则 $O_{1} O$ 与平面 $A C D_{1}$ 所成角就是 $B B_{1}$ 与平面 $A C D_{1}$ 所成角, 即 $\\angle O_{1} O D_{1}$, 直角三角形 $\\mathrm{OO}_{1} \\mathrm{D}_{1}$ 中, $\\cos \\angle \\mathrm{O}_{1} \\mathrm{OD}_{1}=\\frac{\\mathrm{O}_{1} \\mathrm{O}}{\\mathrm{OD}_{1}}=\\frac{\\frac{1}{\\sqrt{6}}}{2}=\\frac{\\sqrt{6}}{3}$. \n答案是 C",
        ),
        (
            "设函数 $f(x)=\\left\\{\\begin{array}{ll}1+\\log _{2}(2-x), & x<1 \\ 2^{x-1}, & x \\geqslant 1,\\end{array}\\right.$ 则 $f(-2)+f\\left(\\log _{2} 12\\right)=$ ( )\n从以下选项中选择:\n(A) 3\n(B) 6\n(C) 9\n(D) 12",
            "首先，我们可以根据定义计算 $f(-2)$ 和 $f(\\log_2 12)$：\n$f(-2)=1+\\log_2(2-(-2))=1+\\log_2 4=3$\n$f(\\log_2 12)=2^{\\log_2 12-1}=6$\n因此，$f(-2)+f(\\log_2 12)=3+6=9$。\n答案是 C",
        ),
        (
            "已知函数 $f(x)=a x^{3}-3 x^{2}+1$, 若 $f(x)$ 存在唯一的零点 $x_{0}$, 且 $x_{0}>$ 0 , 则实数 $\\mathrm{a}$ 的取值范围是 ( )\n从以下选项中选择:\n(A) $(1,+\\infty)$\n(B) $(2,+\\infty)$\n(C) $(-\\infty,-1)$\n(D) $(-\\infty,-2)$",
            "首先，我们可以通过求出函数的导函数 $f'(x)$ 来判断函数在 $x>0$ 区间内的单调性。在这里，我们求出导函数 $f'(x)$ 为 $f'(x)=3ax^2-6x$。\n然后，我们需要求出导函数 $f'(x)$ 的零点，以确定函数 $f(x)$ 在 $x>0$ 区间内的单调性。导函数 $f'(x)$ 的零点为 $x=0$ 和 $x=\\frac{2}{\\sqrt{a}}$。注意到 $x>0$，所以我们得到 $a<0$。此外，由于函数 $f(x)$ 在 $x=0$ 处的函数值为 $1$，因此不能有 $a=\\frac{4}{3}$。\n综上所述，当 $a$ 的取值范围为 $a<-\\frac{4}{3}$ 时，函数 $f(x)$ 在 $x>0$ 区间内是单调递减的，此时存在唯一的零点 $x_0$。因此，答案为 $(-\\infty,-2)$。\n答案是 D",
        ),
        (
            "设 $\\left\\{a_{n}\\right\\}$ 是公差不为 0 的无穷等差数列, 则“ $\\left\\{a_{n}\\right\\}$ 为递增数列”是“存在正整数 $N_{0}$, 当 $n>N_{0}$ 时, $a_{n}>0$ ”的 ( )\n从以下选项中选择:\n(A) 充分而不必要条件\n(B) 必要而不充分条件\n(C) 充分必要条件\n(D) 既不充分也不必要条件",
            "首先，我们可以通过举例来判断该条件是充分还是必要条件。如果一个数列递增，那么它的公差一定大于 0，也就是存在正整数 $N_{0}$，当 $n>N_{0}$ 时，$a_{n}>0$。因此，“ $\\left\\{a_{n}\\right\\}$ 为递增数列”是“存在正整数 $N_{0}$, 当 $n>N_{0}$ 时, $a_{n}>0$ ”的必要条件。\n接下来，我们需要判断是否充分。也就是说，如果存在正整数 $N_{0}$，当 $n>N_{0}$ 时，$a_{n}>0$，那么能否得出“ $\\left\\{a_{n}\\right\\}$ 为递增数列”这一结论。答案是肯定的。因为如果 $a_{n}>0$，那么 $a_{n+1}-a_{n}>0$，即公差大于 0，因此该数列是递增的。因此，该条件是充分条件。\n综上所述，选项为 (C) 充分必要条件。\n答案是 C",
        ),
    ]

    return examples


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

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    # print("Saved to", save_path)


EXAMPLES = get_examples()


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
    "direct": ("### Instruction:\n{input}\n\n### Response:\n", "{output}", "\n\n"),
    "o1": (
        "### Instruction:\nReturn your final response within \\boxed{{}}. {input}\n\n### Think:\n",
        "{output}",
        "\n\n",
    ),
    "r1": (
        "Please reason step by step, and put your final answer within \\boxed{{}}. {input}\n\n<think>\n",
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

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
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

    print(result_json)
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