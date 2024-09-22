"""
Mathematical results related to the analemma TODO
"""

import sympy as sp
from sympy.abc import alpha, psi, theta, sigma, i, d, iota, delta
from galgebra import mv
from galgebra.ga import Ga
from typing import Tuple
from functools import lru_cache

from analemma.algebra import util

_cache_max_size = 50


@lru_cache(maxsize=_cache_max_size)
def _space_algebra(symbol: str) -> Tuple[mv.Mv]:
    """
    TODO
    """

    coords = sp.symbols("1 2 3", real=True)
    G3 = Ga(symbol, g=[1, 1, 1], coords=coords)

    return G3


_base_symbol: str = "e"


@lru_cache(maxsize=_cache_max_size)
def scalar_element(value: float, base_symbol: str = _base_symbol) -> mv.Mv:
    """
    TODO
    """
    return _space_algebra(base_symbol).mv(value, "scalar")


@lru_cache(maxsize=_cache_max_size)
def base(base_symbol: str = _base_symbol) -> Tuple[mv.Mv]:
    """
    TODO
    """
    return _space_algebra(base_symbol).mv()


@lru_cache(maxsize=_cache_max_size)
def planet(
    axis_tilt_symbol: sp.Symbol = alpha,
    rotation_symbol: sp.Symbol = psi,
    base_symbol: str = _base_symbol,
) -> Tuple[mv.Mv]:
    """
    TODO
    """
    (e1, e2, e3) = base(base_symbol)
    e1_prime = util.rotate(e1, axis_tilt_symbol, e1 ^ e3).trigsimp()

    f1 = util.rotate(e1_prime, rotation_symbol, e1_prime ^ e2).trigsimp().trigsimp()
    f2 = util.rotate(e2, rotation_symbol, e1_prime ^ e2).trigsimp().trigsimp()
    f3 = util.rotate(e3, axis_tilt_symbol, e1 ^ e3).trigsimp().trigsimp()

    return (f1, f2, f3)


@lru_cache(maxsize=_cache_max_size)
def surface(
    latitude_symbol: sp.Symbol = theta,
    base_frame: Tuple[mv.Mv] = None,
    base_symbol: str = _base_symbol,
) -> Tuple[mv.Mv]:
    """
    TODO note it is 90 minus latitude
    """
    f1, f2, f3 = planet() if base_frame is None else base_frame

    n1 = util.rotate(f1, latitude_symbol, f3 ^ f1).trigsimp().trigsimp()
    n2 = f2

    # n3 needs a little love
    raw_n3 = util.rotate(f3, theta, f3 ^ f1).obj.trigsimp()
    sympy_n3 = sp.expand(
        sp.expand_trig(raw_n3)
    )  # galgebra's Mv doesn't have expand_trig as a method
    n3 = mv.Mv(sympy_n3, ga=_space_algebra(base_symbol))

    return n1, n2, n3


@lru_cache(maxsize=_cache_max_size)
def sunray(
    orbit_symbol: sp.Symbol = sigma,
    base_symbol: str = _base_symbol,
):
    """
    TODO
    """
    (e1, e2, e3) = base(base_symbol)
    return util.rotate(e1, orbit_symbol, e1 ^ e2).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def dial(
    incl_symbol: sp.Symbol = i,
    decl_symbol: sp.Symbol = d,
    base_symbol: str = "u",
):
    """
    TODO
    """
    u1, u2, u3 = base(base_symbol)
    m1 = util.rotate(
        util.rotate(u1, incl_symbol, u1 ^ u3), decl_symbol, u1 ^ u2
    ).trigsimp()
    m2 = util.rotate(util.rotate(u2, incl_symbol, u1 ^ u3), decl_symbol, u1 ^ u2)
    m3 = util.rotate(
        util.rotate(u3, incl_symbol, u1 ^ u3), decl_symbol, u1 ^ u2
    ).trigsimp()
    return m1, m2, m3


@lru_cache(maxsize=_cache_max_size)
def gnomon(
    incl_symbol: sp.Symbol = iota,
    decl_symbol: sp.Symbol = delta,
    base_symbol: str = "u",
):
    """
    TODO
    """
    u1, u2, u3 = base(base_symbol)
    return util.rotate(
        util.rotate(u3, incl_symbol, u1 ^ u3), decl_symbol, u1 ^ u2
    ).trigsimp()
