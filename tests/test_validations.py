from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, get_type_hints

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from xclim.core.validations import (
    safe_isinstance,
    validate_arg,
    validate_args,
    validate_iterable,
    validate_mapping,
    validate_ndarray,
    validate_non_generic,
    validate_parameters,
    validate_sequence,
    validate_tuple,
    validate_type,
)


def sample_func_long(
    a: int,
    b: list[int],
    c: dict[str, int],
    d: tuple[int, str],
    e: tuple[int, ...],
    f: int | None,
    g: Literal["x", "y"],
    h: int | str,
    i: list[int | str],
    j: Annotated[int, "positive"],
    k: Callable[[int], str],
    m: set[float],
    n: frozenset[int],
    o: Sequence[int],
    p: Mapping[str, list[int]],
    q: dict[str, list[tuple[int, float | None]]],
    r: pd.DataFrame,
    s: pd.Series,
    t: np.ndarray,
    u: npt.NDArray[np.int64],
    v,
):
    return a, b, c, d, e, f, g, h, i, j, k, m, n, o, p, q, r, s, t, u


def sample_func_short(a: int):
    return a


def sample_func_kwargs(a: int, **kwargs):
    return {"a": a, **kwargs}


@pytest.fixture
def series_ind():
    return pd.Series([1, 2, 3, 4], name="value")


@pytest.fixture
def df_ind():
    return pd.DataFrame({"value1": [1, 2, 3, 4], "value2": [1, 1, 2, 2]})


@validate_parameters
def add(x: int, y: int) -> int:
    return x + y


@validate_parameters
def mulc(x: int, y: int = 2) -> int:
    return x * y


PARAMETERS = inspect.signature(sample_func_long).parameters
TYPE_HINTS = get_type_hints(sample_func_long)


def test_validate_non_generic():
    assert validate_non_generic(5, int)
    assert not validate_non_generic("5", int)
    assert not validate_non_generic(5, 123)


def test_validate_mapping():
    assert not validate_mapping([], dict, (str, int))
    assert validate_mapping({"a": 1}, dict, ())
    assert validate_mapping({"a": 1}, dict, (str, int))
    assert not validate_mapping({"a": "1"}, dict, (str, int))


def test_validate_iterable():
    assert not validate_iterable((1, 2), list, (int,))
    assert validate_iterable([1, 2], list, ())
    assert validate_iterable([1, 2], list, (int,))
    assert not validate_iterable([1, "2"], list, (int,))


def test_validate_sequence():
    assert not validate_sequence("123", (int,))
    assert validate_sequence([1, 2], ())
    assert validate_sequence([1, 2], (int,))
    assert not validate_sequence([1, "2"], (int,))


def test_validate_tuple():
    assert not validate_tuple("123", (int,))
    assert validate_tuple((1, 2), ())
    assert validate_tuple((1, 2), (int, Ellipsis))
    assert not validate_tuple((1, "2"), (int, Ellipsis))
    assert validate_tuple((1, "2"), (int, str))
    assert not validate_tuple((1,), (int, str))
    assert not validate_tuple((1, 2), (int, str))


def test_validate_ndarray():
    arr = np.array([1, 2, 3])

    assert not validate_ndarray([1, 2, 3], (None, np.int64))
    assert validate_ndarray(arr, ())
    assert validate_ndarray(arr, (np.int64,))
    assert validate_ndarray(arr, (None, Any))
    assert validate_ndarray(arr, (None, None))
    assert validate_ndarray(arr, (None, np.int64))
    assert not validate_ndarray(arr, (None, np.float64))
    assert not validate_ndarray(arr, (None, object()))


def test_safe_isinstance():
    assert safe_isinstance(5, int)
    assert not safe_isinstance(5, list[int])


@pytest.mark.parametrize(
    "value,expected",
    [
        # Simple types
        (123, Any),
        ("abc", str),
        (5, int),
        (int, type[int]),
        # Annotated / Literal
        (10, Annotated[int, "meta"]),
        ("x", Literal["x", "y"]),
        # Union / Optional
        (5, int | str),
        ("hello", int | str),
        (None, int | None),
        # Callable
        (lambda x: x, Callable[[int], int]),
        # Mappings
        ({"a": 1}, dict[str, int]),
        ({"a": 1}, dict),
        ({}, dict[str, int]),
        # Iterables
        ([1, 2, 3], list[int]),
        ([1, 2, 3], list),
        ([1, 2, 3], Sequence[int]),
        ([1, 2, 3], Sequence),
        ([1], tuple[int, ...]),
        ((1, 2, 3), tuple[int, ...]),
        ((1, 2, 3), tuple),
        ((1, "x"), tuple[int, str]),
        # Sets/ frozensets
        ({1.0, 2.0}, set[float]),
        (frozenset({1, 2}), frozenset[int]),
        # NumPy arrays
        (np.array([1, 2, 3]), np.ndarray),
        (np.array([1, 2, 3], dtype=np.int64), npt.NDArray[np.int64]),
        (np.array([1, 2, 3]), npt.NDArray[np.int64]),
        (np.array([1.0, 2.0]), npt.NDArray[Any]),
        (np.array([1, 2, 3]), npt.NDArray[Any]),
        # pandas objects
        (pd.DataFrame({"a": [1, 2]}), pd.DataFrame),
        (pd.Series([1, 2]), pd.Series),
    ],
)
def test_validate_type_valid(value, expected):
    assert validate_type(value, expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        # Simple types
        ("5", int),
        ("z", Literal["x", "y"]),
        (5.5, int | str),
        (5, str),
        # Mappings
        ({1: "a"}, dict[str, int]),
        ({"a": "b"}, dict[str, int]),
        ([("a", 1)], dict[str, int]),
        # Sequence
        ([1, "x"], list[int]),
        ((1,), tuple[int, str]),
        ((1, 2), tuple[int, str]),
        ((1, "x"), tuple[int, int]),
        ((1, 2), list[int]),
        ([1, "x"], Sequence[int]),
        ("abc", Sequence[str]),
        # NumPy arrays
        (np.array([1.0, 2.0]), npt.NDArray[np.int64]),
        (np.array([1, 2, 3]), npt.NDArray[object]),
        # pandas objects
        (pd.Series([1, 2]), pd.DataFrame),
        # Others
        (5, list[int]),
        (5, 123),
        (123, Callable[[int], int]),
        ("abc", Sequence[int]),
    ],
)
def test_validate_type_invalid(value, expected):
    assert not validate_type(value, expected)


@pytest.mark.parametrize(
    "key,value",
    [
        ("a", 5),
        ("b", [1, 2, 3]),
        ("c", {"x": 1, "y": 2}),
        ("d", (1, "hello")),
        ("e", (1, 2, 3)),
        ("f", None),
        ("f", 10),
        ("g", "x"),
        ("h", 42),
        ("h", "hello"),
        ("i", [1, "a", 3]),
        ("j", 99),
        ("k", lambda x: str(x)),
        ("m", {1.0, 2.5}),
        ("n", frozenset({1, 2, 3})),
        ("o", [1, 2, 3]),
        ("p", {"a": [1, 2], "b": [3]}),
        ("q", {"a": [(1, 2.0), (3, None)]}),
        ("r", pd.DataFrame({"a": [1, 2, 3]})),
        ("s", pd.Series([1, 2, 3])),
        ("t", np.array([1, 2, 3])),
        ("u", np.array([1, 2, 3], dtype=np.int64)),
        ("v", 1),
    ],
)
def test_validate_arg_valid(key, value):
    validate_arg(
        key=key,
        value=value,
        func_name="sample_func_long",
        parameters=PARAMETERS,
        type_hints=TYPE_HINTS,
        has_arguments=False,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("a", "not int"),
        ("b", [1, "wrong"]),
        ("c", {"x": "wrong"}),
        ("d", (1, 2)),
        ("e", ("wrong",)),
        ("f", "not optional int"),
        ("g", "z"),
        ("h", 3.14),
        ("i", [1, 2.0]),
        ("g", "z"),
        ("j", "not int"),
        ("k", "not callable"),
        ("m", {1, 2}),
        ("n", {1, 2}),
        ("o", ["x", "y"]),
        ("p", {"a": ["wrong"]}),
        ("q", {"a": [(1, "wrong")]}),
        ("d", (1, 2)),
        ("e", ("a",)),
        ("r", {"a": [1, 2, 3]}),
        ("s", [1, 2, 3]),
        ("t", [1, 2, 3]),
        ("u", np.array([1, 2, 3], dtype=float)),
    ],
)
def test_validate_arg_invalid_type(key, value):
    with pytest.raises(TypeError):
        validate_arg(
            key=key,
            value=value,
            func_name="sample_func_long",
            parameters=PARAMETERS,
            type_hints=TYPE_HINTS,
            has_arguments=False,
        )


def test_validate_arg_invalid_parameter_name():
    with pytest.raises(ValueError):
        validate_arg(
            key="unknown",
            value=123,
            func_name="sample_func_long",
            parameters=PARAMETERS,
            type_hints=TYPE_HINTS,
            has_arguments=False,
        )


def test_has_arguments_skips_validation():
    validate_arg(
        key="unknown",
        value="anything",
        func_name="sample_func_long",
        parameters=PARAMETERS,
        type_hints=TYPE_HINTS,
        has_arguments=True,
    )


def test_validate_args_passing_required_only():
    validate_args(sample_func_short, kwargs={"a": 2})


def test_validate_args_passing_with_extra_kwargs():
    validate_args(sample_func_kwargs, kwargs={"a": 2, "extra": 123})


def test_validate_args_passing_with_args_and_kwargs():
    validate_args(sample_func_kwargs, args=2, kwargs={"extra": 123})


def test_validate_args_invalid_param():
    with pytest.raises(ValueError, match="is not a valid parameter of function"):
        validate_args(sample_func_short, kwargs={"a": 2, "extra": 123})


def test_validate_args_missing_required_param():
    with pytest.raises(TypeError, match="is missing for function"):
        validate_args(sample_func_kwargs, kwargs={"extra": 123})


def test_validate_args_too_many_args():
    with pytest.raises(TypeError, match="Too many positional arguments for function"):
        validate_args(sample_func_short, args=(1, 2))


def test_validate_args_passing_args_only():
    validate_args(
        sample_func_long,
        args=(
            5,
            [1, 2, 3],
            {"x": 1, "y": 2},
            (1, "hello"),
            (1, 2, 3),
            10,
            "x",
            42,
            [1, "a", 3],
            99,
            lambda x: str(x),
            {1.0, 2.5},
            frozenset({1, 2, 3}),
            [1, 2, 3],
            {"a": [1, 2], "b": [3]},
            {"a": [(1, 2.0), (3, None)]},
            pd.DataFrame({"a": [1, 2, 3]}),
            pd.Series([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3], dtype=np.int64),
            1,
        ),
    )


def test_validate_args_passing_kwargs_only():
    validate_args(
        sample_func_long,
        kwargs={
            "a": 5,
            "b": [1, 2, 3],
            "c": {"x": 1, "y": 2},
            "d": (1, "hello"),
            "e": (1, 2, 3),
            "f": 10,
            "g": "x",
            "h": 42,
            "i": [1, "a", 3],
            "j": 99,
            "k": lambda x: str(x),
            "m": {1.0, 2.5},
            "n": frozenset({1, 2, 3}),
            "o": [1, 2, 3],
            "p": {"a": [1, 2], "b": [3]},
            "q": {"a": [(1, 2.0), (3, None)]},
            "r": pd.DataFrame({"a": [1, 2, 3]}),
            "s": pd.Series([1, 2, 3]),
            "t": np.array([1, 2, 3]),
            "u": np.array([1, 2, 3], dtype=np.int64),
            "v": 1,
        },
    )


def test_validate_parameters_pass_pos():
    assert add(2, 3) == 5


def test_validate_parameters_pass_kw():
    assert add(x=2, y=3) == 5


def test_validate_parameters_error_pos():
    with pytest.raises(TypeError, match="does not match expected type"):
        add(2, "3")


def test_validate_parameters_error_kw():
    with pytest.raises(TypeError, match="does not match expected type"):
        add(x=2, y="3")


def test_validate_parameters_error_few():
    with pytest.raises(TypeError, match="Required parameter"):
        add(2)  # pylint: disable=E1120


def test_validate_parameters_error_many():
    with pytest.raises(TypeError, match="Too many positional arguments"):
        add(2, 3, 4)  # pylint: disable=E1121


def test_validate_parameters_error_invalid_kw():
    with pytest.raises(ValueError, match="not a valid parameter"):
        add(x=2, z=3)  # pylint: disable=E1120


def test_validate_parameters_pass_default():
    assert mulc(2) == 4
