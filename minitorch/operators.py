"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List


def add(x: float, y: float) -> float:
    """Adds two floats and returns the result."""
    return x + y


def mul(x: float, y: float) -> float:
    """Multiply two floats and returns the result."""
    return x * y


def id(x: float) -> float:
    """Returns the input float."""
    return x


def neg(x: float) -> float:
    """Negate a float and returns the result."""
    return -x


def lt(x: float, y: float) -> bool:
    """Returns True if x is less than y, False otherwise."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Returns True if x is equal to y, False otherwise."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of two floats."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Returns True if x is close to y by 1e-2, False otherwise."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Returns the sigmoid of a float by calculating 1/(1+e^-x)"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Returns the relu of a float by returning x if x > 0 else 0"""
    return x if x > 0 else 0


def log(x: float) -> float:
    """Returns the log of a float by calculating log(x)"""
    return math.log(x)


def exp(x: float) -> float:
    """Returns the exp of a float by calculating e^x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the inverse of a float by calculating 1/x"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Returns the derivative of ln(x) multiplied by y with respect to x"""
    return y / x


def inv_back(x: float, y: float) -> float:
    """Returns the derivative of 1/x with respect to x, or -y/(x^2)"""
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Returns the derivative of relu(x) multiplied by y with respect to x"""
    return y if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(func: Callable[[float], float], iter: Iterable[float]) -> Iterable[float]:
    """Takes in func, a callable function that takes in a float and returns a float, and iter which is an iterable collection of floats, and returns the result of calling func on every element in iter"""
    return [func(x) for x in iter]


def zipWith(
    func: Callable[[float, float], float],
    iter1: Iterable[float],
    iter2: Iterable[float],
) -> Iterable[float]:
    """Takes in func, a callable that takes in 2 floats and returns a float, along with two iterables of floats, and returns an array full of func on each list"""
    return [func(item1, item2) for (item1, item2) in zip(iter1, iter2)]


def reduce(func: Callable[[float, float], float], itrble: Iterable[float]) -> float:
    """Takes in func, a callable that takes in 2 floats and returns a float, along with an iterable of floats and returns the result of computing func on each one of the elements"""
    if list(itrble) == []:
        return 0
    itr = iter(itrble)
    res = next(itr)
    for elem in itr:
        res = func(res, elem)
    return res


def negList(ls: List[float]) -> List[float]:
    """Takes in a list of floats and returns the list of each float's negative"""
    return list(map(neg, ls))


def addLists(ls1: List[float], ls2: List[float]) -> List[float]:
    """Takes in two lists of floats and returns a list of the elements from each list summed"""
    return list(zipWith(add, ls1, ls2))


def sum(ls: List[float]) -> float:
    """Takes in a list of floatsand sums the result of each element in the list"""
    return reduce(add, ls)


def prod(ls: List[float]) -> float:
    """Takes in a list of floats and computes the product of each element in the list"""
    return reduce(mul, ls)
