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


@lru_cache(maxsize=_cache_max_size)
def base(base_symbol: str) -> Tuple[mv.Mv]:
    """
    TODO
    """
    return _space_algebra(base_symbol).mv()


@lru_cache(maxsize=_cache_max_size)
def base_bivec(base_symbol: str) -> Tuple[mv.Mv]:
    """
    TODO
    """
    v1, v2, v3 = base(base_symbol)
    return v1 ^ v2, v1 ^ v3, v2 ^ v3


_fixed_stars_symbol: str = "e"


@lru_cache(maxsize=_cache_max_size)
def scalar_element(value: float, base_symbol: str = _fixed_stars_symbol) -> mv.Mv:
    """
    TODO
    """
    return _space_algebra(base_symbol).mv(value, "scalar")


@lru_cache(maxsize=_cache_max_size)
def planet(
    axis_tilt_symbol: sp.Symbol = alpha,
    rotation_symbol: sp.Symbol = psi,
) -> Tuple[mv.Mv]:
    """
    TODO
    """
    (e1, e2, e3) = base(_fixed_stars_symbol)
    e1_prime = util.rotate(e1, axis_tilt_symbol, e1 ^ e3).trigsimp()

    f1 = util.rotate(e1_prime, rotation_symbol, e1_prime ^ e2).trigsimp().trigsimp()
    f2 = util.rotate(e2, rotation_symbol, e1_prime ^ e2).trigsimp().trigsimp()
    f3 = util.rotate(e3, axis_tilt_symbol, e1 ^ e3).trigsimp().trigsimp()

    return (f1, f2, f3)


@lru_cache(maxsize=_cache_max_size)
def surface(latitude_symbol: sp.Symbol = theta) -> Tuple[mv.Mv]:
    """
    TODO note it is 90 minus latitude
    """
    f1, f2, f3 = planet()

    n1 = util.rotate(f1, latitude_symbol, f3 ^ f1).trigsimp().trigsimp()
    n2 = f2

    # n3 needs a little love
    raw_n3 = util.rotate(f3, theta, f3 ^ f1).obj.trigsimp()
    sympy_n3 = sp.expand(
        sp.expand_trig(raw_n3)
    )  # galgebra's Mv doesn't have expand_trig as a method
    n3 = mv.Mv(
        sympy_n3, ga=_space_algebra(_fixed_stars_symbol)
    )  # TODO replace with f1.Ga.mv()?

    return n1, n2, n3


@lru_cache(maxsize=_cache_max_size)
def surface_bivec(latitude_symbol: sp.Symbol = theta) -> Tuple[mv.Mv]:
    """
    TODO note it is 90 minus latitude
    """
    n1, n2, n3 = surface(latitude_symbol)
    n12 = mv.Mv(sp.trigsimp(sp.expand_trig((n1 ^ n2).obj)), ga=n1.Ga)
    n13 = mv.Mv(sp.trigsimp(sp.expand_trig((n1 ^ n3).obj)), ga=n1.Ga)
    n23 = mv.Mv(sp.trigsimp(sp.expand_trig((n2 ^ n3).obj)), ga=n1.Ga)
    return n12, n13, n23


@lru_cache(maxsize=_cache_max_size)
def meridian_plane():
    """
    TODO btw theta cancels
    """
    n1, _, n3 = surface()
    return (n1 ^ n3).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def sunray(
    orbit_symbol: sp.Symbol = sigma,
):
    """
    TODO
    """
    (e1, e2, e3) = base(_fixed_stars_symbol)
    return util.rotate(e1, orbit_symbol, e1 ^ e2).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def dial(
    incl_symbol: sp.Symbol = i,
    decl_symbol: sp.Symbol = d,
):
    """
    TODO
    """
    n1, n2, n3 = base("n")
    m1 = util.rotate(
        util.rotate(n1, incl_symbol, n1 ^ n3), decl_symbol, n1 ^ n2
    ).trigsimp()
    m2 = util.rotate(util.rotate(n2, incl_symbol, n1 ^ n3), decl_symbol, n1 ^ n2)
    m3 = util.rotate(
        util.rotate(n3, incl_symbol, n1 ^ n3), decl_symbol, n1 ^ n2
    ).trigsimp()
    return m1, m2, m3


@lru_cache(maxsize=_cache_max_size)
def dialface(
    incl_symbol: sp.Symbol = i,
    decl_symbol: sp.Symbol = d,
):
    """
    TODO
    """
    m1, m2, m3 = dial()
    return (m1 ^ m2).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def _gnomon(
    zero_decl: bool = True,
    incl_symbol: sp.Symbol = iota,
    decl_symbol: sp.Symbol = delta,
):
    """
    TODO
    """
    n1, n2, n3 = base("n")
    g_incl = util.rotate(n3, incl_symbol, n1 ^ n3).trigsimp()
    if zero_decl:
        return g_incl
    return util.rotate(g_incl, decl_symbol, n1 ^ n2).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def gnomon(
    base_symbol: str = "n",
    zero_decl: bool = True,
    incl_symbol: sp.Symbol = iota,
    decl_symbol: sp.Symbol = delta,
):
    if base_symbol == "n":
        return _gnomon(zero_decl, incl_symbol, decl_symbol)
    elif base_symbol == "e":
        gn = _gnomon(zero_decl, incl_symbol, decl_symbol)
        return util.project_vector(gn, target_frame=base("n"), render_frame=surface())
    else:
        raise Exception("TODO")
