"""Module containing base QC which call multiple QC functions and could be applied on a DataBundle."""

from __future__ import annotations

import collections.abc as abc
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import wraps
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
import numpy.typing as npt
import pandas as pd


def validate_non_generic(value: Any, expected: Any) -> bool:
    """
    Validate a non-generic type (str, int, float, etc.).

    Parameters
    ----------
    value : Any
        The value to validate.
    expected : Any
        The expected type.

    Returns
    -------
    bool
        True if `value` matches `expected`, False otherwise.
    """
    if isinstance(expected, type):
        return isinstance(value, expected)
    return False


def validate_mapping(value: Mapping[Any, Any], origin: type, args: tuple[Any, ...]) -> bool:
    """
    Validate a mapping type (dict, Mapping).

    Parameters
    ----------
    value : Mapping[Any, Any]
        The value to validate.
    origin : type
        The mapping type (e.g., dict).
    args : tuple[Any, ...]
        Expected key and value types.

    Returns
    -------
    bool
        True if `value` matches the mapping type and key/value types, False otherwise.
    """
    if not isinstance(value, origin):
        return False
    if not args:
        return True
    key_type, val_type = args
    return all(validate_type(k, key_type) and validate_type(v, val_type) for k, v in value.items())


def validate_iterable(value: Iterable[Any], origin: type, args: tuple[Any, ...]) -> bool:
    """
    Validate an iterable type (list, set, frozenset).

    Parameters
    ----------
    value : Any
        The value to validate.
    origin : type
        The iterable type.
    args : tuple[Any, ...]
        Expected element types.

    Returns
    -------
    bool
        True if all elements match the expected type, False otherwise.
    """
    if not isinstance(value, origin):
        return False
    if not args:
        return True
    elem_type = args[0]
    return all(validate_type(v, elem_type) for v in value)


def validate_sequence(value: Any, args: tuple[Any, ...]) -> bool:
    """
    Validate a generic sequence type (e.g., Sequence[int]).

    Parameters
    ----------
    value : Any
        The value to validate.
    args : tuple[Any, ...]
        Expected element types.

    Returns
    -------
    bool
        True if all elements match the expected type, False otherwise.
    """
    if not isinstance(value, abc.Sequence) or isinstance(value, (str, bytes)):
        return False
    if not args:
        return True
    elem_type = args[0]
    return all(validate_type(v, elem_type) for v in value)


def validate_tuple(value: Any, args: tuple[Any, ...]) -> bool:
    """
    Validate a tuple type (fixed-length or homogeneous).

    Parameters
    ----------
    value : Any
        The value to validate.
    args : tuple[Any, ...]
        Expected element types.

    Returns
    -------
    bool
        True if the tuple matches the expected types and length, False otherwise.
    """
    if not isinstance(value, abc.Sequence) or isinstance(value, (str, bytes)):
        return False
    if not args:
        return True
    if len(args) == 2 and args[1] is Ellipsis:
        return all(validate_type(v, args[0]) for v in value)
    if len(args) != len(value):
        return False
    return all(validate_type(v, t) for v, t in zip(value, args, strict=False))


def validate_ndarray(value: Any, args: tuple[Any, ...]) -> bool:
    """
    Validate a numpy ndarray type, optionally checking dtype.

    Parameters
    ----------
    value : Any
        The value to validate.
    args : tuple[Any, ...]
        Expected dtype (first argument may be `Any` or unspecified).

    Returns
    -------
    bool
        True if `value` is an ndarray and matches expected dtype, False otherwise.
    """
    if not isinstance(value, np.ndarray):
        return False

    if not args:
        return True

    if len(args) < 2:
        return True

    expected_dtype = args[1]

    inner = get_args(expected_dtype)
    if inner:
        expected_dtype = inner[0]

    if expected_dtype in (Any, None):
        return True

    try:
        return np.issubdtype(value.dtype, expected_dtype)
    except TypeError:
        return False


def safe_isinstance(value: Any, origin: Any) -> bool:
    """
    Safely check if value is an instance of a type, avoiding TypeError for weird generics.

    Parameters
    ----------
    value : Any
        Value to check.
    origin : Any
        Type or generic to check against.

    Returns
    -------
    bool
        True if `value` is an instance of `origin`, False otherwise.
    """
    try:
        return isinstance(value, origin)
    except TypeError:
        return False


def validate_type(value: Any, expected: Any) -> bool:
    """
    Recursively validate that a value matches the expected type hint.

    Parameters
    ----------
    value : Any
        The value to validate.
    expected : Any
        The expected value type for validation.

    Returns
    -------
    bool
        - True if type of `value` does match `expected`.
        - False if type of `value` does not match `expected`.
    """
    if expected is Any:
        return True

    origin = get_origin(expected)
    args = get_args(expected)

    if origin is Annotated:
        return validate_type(value, args[0])

    if origin is Literal:
        return value in args

    if origin in (Union, UnionType):
        return any(validate_type(value, t) for t in args)

    if origin is abc.Callable:
        return callable(value)

    if origin is tuple:
        return validate_tuple(value, args)

    if origin in (np.ndarray, npt.NDArray):
        return validate_ndarray(value, args)

    if isinstance(expected, type) and issubclass(expected, (pd.DataFrame, pd.Series)):
        return isinstance(value, expected)

    if isinstance(origin, type):
        if issubclass(origin, abc.Mapping):
            return validate_mapping(value, origin, args)
        if issubclass(origin, (list, set, frozenset)):
            return validate_iterable(value, origin, args)
        if issubclass(origin, abc.Sequence):
            return validate_sequence(value, args)

    if origin is None:
        return validate_non_generic(value, expected)

    return safe_isinstance(value, origin)


def validate_arg(
    key: str,
    value: Any,
    func_name: str,
    parameters: Mapping[str, inspect.Parameter],
    type_hints: Mapping[str, Any],
    has_arguments: bool,
) -> None:
    """
    Validate argument against a function's signature, taking decorators into account.

    Parameters
    ----------
    key : str
        The name of the argument to validate.
    value : Any
        The value of the argument to validate.
    func_name : str
        The name of the function (used in error message).
    parameters : Mapping[str, inspect.Parameter]
        A mapping of parameter names to `inspect.Parameter` objects,
        typically from `inspect.signature(func).parameters`.
    type_hints : Mapping[str, type]
        A mapping of parameter names to expected types,
        typically from `typing.get_type_hints(func)`.
    has_arguments : bool
        Whether the function accepts arbitrary arguments.
    """
    if has_arguments:
        return

    if key not in parameters:
        raise ValueError(f"Parameter '{key}' is not a valid parameter of function '{func_name}'.")

    expected = type_hints.get(key)
    if not expected or expected is inspect._empty:
        return

    if not validate_type(value, expected):
        raise TypeError(
            f"Parameter '{key}' does not match {expected!r}. Got value {value!r} of type {type(value).__name__}."
        )


def validate_args(
    func: Callable[..., Any],
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> None:
    """
    Validate positional and keyword arguments against a function's signature.

    This function checks that:
    - All provided keyword arguments correspond to valid parameters of the given function.
    - All required parameters of the function are present in the provided keyword arguments.

    Parameters
    ----------
    func : Callable[..., Any]
        The function whose signature is used for validation.
    args : Sequence[Any], optional
        Sequence of arguments intended to be passed to `func`.
    kwargs : Mapping[str, Any], optional
        Dictionary of keyword arguments intended to be passed to `func`.

    Raises
    ------
    ValueError
        If `kwargs` contains a key that is not a parameter of `func`.
    TypeError
        If a required parameter of `func` is missing from `kwargs`.
    """
    args = args or ()
    if not isinstance(args, (list, tuple)):
        args = (args,)

    kwargs = kwargs or {}

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    positional_params = [
        p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    has_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

    if len(args) > len(positional_params) and not has_args:
        raise TypeError(f"Too many positional arguments for function '{func.__name__}'.")

    bound_args = [positional_params[i].name for i in range(min(len(args), len(positional_params)))]

    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    type_hints = get_type_hints(func)

    for i, arg in enumerate(args):
        validate_arg(bound_args[i], arg, func.__name__, sig.parameters, type_hints, has_args)

    for key, value in kwargs.items():
        validate_arg(key, value, func.__name__, sig.parameters, type_hints, has_kwargs)

    for param in params:
        if (
            param.default is inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and param.name not in kwargs
            and param.name not in bound_args
        ):
            raise TypeError(f"Required parameter '{param.name}' is missing for function '{func.__name__}'.")


def validate_parameters(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that validates function arguments against type hints before calling the function.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to wrap.

    Returns
    -------
    Callable[..., Any]
        The wrapped function with argument validation.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # numpydoc ignore=GL08
        validate_args(func, args, kwargs)
        return func(*args, **kwargs)

    return wrapper
