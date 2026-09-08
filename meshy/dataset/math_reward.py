"""Miles ``--rm-type math`` boxed-answer reward.

Copied from ``miles.rollout.rm_hub.math_utils``.  The original implementation
comes from Agentica DeepScaleR's math reward utilities.
"""

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


def mathd_normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        match = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if match is not None:
            answer = match.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(string: str) -> str:
    def _fix_fracs(value: str) -> str:
        substrings = value.split("\\frac")
        new_value = substrings[0]
        for substring in substrings[1:]:
            new_value += "\\frac"
            if substring[0] == "{":
                new_value += substring
                continue
            try:
                assert len(substring) >= 2
            except Exception:
                return value
            numerator, denominator = substring[0], substring[1]
            if denominator != "{":
                new_value += f"{{{numerator}}}{{{denominator}}}" + substring[2:]
            else:
                new_value += f"{{{numerator}}}" + denominator + substring[2:]
        return new_value

    def _fix_a_slash_b(value: str) -> str:
        if len(value.split("/")) != 2:
            return value
        numerator, denominator = value.split("/")
        try:
            numerator = int(numerator)
            denominator = int(denominator)
            assert value == f"{numerator}/{denominator}"
            return f"\\frac{{{numerator}}}{{{denominator}}}"
        except Exception:
            return value

    def _remove_right_units(value: str) -> str:
        if "\\text{ " not in value:
            return value
        parts = value.split("\\text{ ")
        assert len(parts) == 2
        return parts[0]

    def _fix_sqrt(value: str) -> str:
        if "\\sqrt" not in value:
            return value
        parts = value.split("\\sqrt")
        new_value = parts[0]
        for part in parts[1:]:
            if part[0] != "{":
                new_value += "\\sqrt{" + part[0] + "}" + part[1:]
            else:
                new_value += "\\sqrt" + part
        return new_value

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace(r"\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    return sympy_parser.parse_expr(
        expr.replace("^", "**"),
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    replacements = {
        "√": "sqrt",
        "π": "pi",
        "∞": "inf",
        "∪": "U",
        "·": "*",
        "×": "*",
    }
    for old, new in replacements.items():
        expr = expr.replace(old, new)
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except Exception:
        return False


def _is_int(value: float) -> bool:
    try:
        return abs(value - int(round(value))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(value: str) -> bool:
    try:
        value = float(_strip_properly_formatted_commas(value))
        return abs(value - int(round(value))) <= 1e-7
    except Exception:
        return False


def _str_to_int(value: str) -> int:
    return int(float(value.replace(",", "")))


def _inject_implicit_mixed_number(step: str) -> str:
    return re.compile(r"([0-9]) +([0-9])").sub(r"\1+\2", step)


def _strip_properly_formatted_commas(expr: str) -> str:
    pattern = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = pattern.sub(r"\1\3\4", expr)
        if next_expr == expr:
            return next_expr
        expr = next_expr


def _normalize(expr: str | None) -> str | None:
    if expr is None:
        return None
    match = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if match is not None:
        expr = match.group("text")
    expr = expr.replace("\\%", "%").replace("\\$", "$")
    expr = expr.replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")
    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def count_unknown_letters_in_expr(expr: str) -> int:
    expr = expr.replace("sqrt", "").replace("frac", "")
    return len({character for character in expr if character.isalpha()})


def should_allow_eval(expr: str) -> bool:
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    if any(bad_string in expr for bad_string in BAD_SUBSTRINGS):
        return False
    return not any(re.search(bad_regex, expr) is not None for bad_regex in BAD_REGEXES)


def are_equal_under_sympy(
    ground_truth_normalized: str, given_normalized: str
) -> bool:
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            return sympy.simplify(_sympy_parse(expr)) == 0
    except Exception:
        pass
    return False


def split_tuple(expr: str) -> list[str]:
    expr = _strip_properly_formatted_commas(expr)
    if not expr:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(character not in expr[1:-1] for character in TUPLE_CHARS)
    ):
        return [element.strip() for element in expr[1:-1].split(",")]
    return [expr]


def last_boxed_only_string(string: str) -> str | None:
    index = string.rfind("\\boxed")
    if index < 0:
        index = string.rfind("\\fbox")
        if index < 0:
            return None
    right_brace_index = None
    open_braces = 0
    for current in range(index, len(string)):
        if string[current] == "{":
            open_braces += 1
        elif string[current] == "}":
            open_braces -= 1
            if open_braces == 0:
                right_brace_index = current
                break
    if right_brace_index is None:
        return None
    return string[index : right_brace_index + 1]


def remove_boxed(value: str | None) -> str | None:
    prefix = "\\boxed{"
    try:
        assert value is not None
        assert value[: len(prefix)] == prefix
        assert value[-1] == "}"
        return value[len(prefix) : -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the answer from the final LaTeX ``\\boxed{}`` command."""
    return remove_boxed(last_boxed_only_string(solution))


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    if not given_normalized:
        return False

    ground_truth_elements = split_tuple(ground_truth_normalized)
    given_elements = split_tuple(given_normalized)
    if len(ground_truth_elements) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        return False
    if len(ground_truth_elements) != len(given_elements):
        return False
    for ground_truth_element, given_element in zip(
        ground_truth_elements, given_elements, strict=False
    ):
        if _is_frac(ground_truth_element) and _is_frac(given_element):
            is_correct = ground_truth_element == given_element
        elif _str_is_int(ground_truth_element) != _str_is_int(given_element):
            is_correct = False
        else:
            is_correct = are_equal_under_sympy(
                ground_truth_element, given_element
            )
        if not is_correct:
            return False
    return True


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    return mathd_normalize_answer(ground_truth) == mathd_normalize_answer(
        given_answer
    )


def extract_answer(passage: str) -> str | None:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade_answer_verl(solution_str: str, ground_truth: object) -> bool:
    """Match Miles' binary ``--rm-type math`` answer checker."""
    if not ground_truth:
        return False
    ground_truth = str(ground_truth)
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(
        given_answer, ground_truth
    )


def math_reward(solution_str: str, ground_truth: object) -> float:
    """Return the scalar reward used by Miles' ``--rm-type math`` route."""
    return float(grade_answer_verl(solution_str, ground_truth))
