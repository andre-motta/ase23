from simpleCalculator import add, subtract, multiply, divide
import pytest


def test_simpleCalculator_add():
    x = 5
    y = 6
    z = add(x, y)
    assert z == 11, "test failed"


def test_simpleCalculator_subtract():
    x = 10
    y = 6
    z = subtract(x, y)
    assert z == 4, "test failed"


def test_simpleCalculator_multiply():
    x = 5
    y = 6
    z = multiply(x, y)
    assert z == 30, "test failed"


def test_simpleCalculator_divide():
    x = 30
    y = 6
    z = divide(x, y)
    assert z == 5, "test failed"
