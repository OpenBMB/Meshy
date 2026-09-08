import pytest

from meshy.dataset.math_reward import grade_answer_verl, math_reward


@pytest.mark.parametrize(
    ("response", "ground_truth", "expected"),
    [
        (r"\boxed{42}", "42", 1.0),
        (r"\boxed{wrong}", "42", 0.0),
        ("42", "42", 0.0),
        (r"first \boxed{0}, finally \boxed{42}", "42", 1.0),
        (r"\boxed{\frac{1}{2}}", "0.5", 1.0),
        # Miles intentionally requires strict form matching when only one side
        # normalizes to an integer; it does not accept this simplification.
        (r"\boxed{2+2}", "4", 0.0),
        (r"\boxed{42}", r"\boxed{42}", 1.0),
        (r"\boxed{42}", "", 0.0),
    ],
)
def test_miles_math_reward(response, ground_truth, expected):
    assert math_reward(response, ground_truth) == expected


def test_grade_answer_verl_returns_boolean():
    assert grade_answer_verl(r"\boxed{42}", "42")
