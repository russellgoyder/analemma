"""
TODO
"""

import sympy as sp
from sympy import sin, cos, tan
from sympy.abc import alpha, sigma, psi, iota, theta, mu, i, d, D
from galgebra import mv
from functools import lru_cache

from analemma.algebra import frame, util

_cache_max_size = 50

_Xi = sp.Symbol(r"\Xi")
_Psi = sp.Symbol(r"\Psi")


def shadow_bivector(
    sunray: mv.Mv = frame.sunray(), gnomon: mv.Mv = frame.gnomon("e")
) -> mv.Mv:
    """
    TODO
    """
    return (sunray ^ gnomon).trigsimp()


def shadow_bivector_explicit(
    hour_angle_symbol: sp.Symbol = mu,
    shadow_plane_magnitude_symbol: sp.Symbol = _Xi,
    gnomon_incl_symbol: sp.Symbol = iota,
):
    """
    TODO
    """
    n12, n13, n23 = frame.base_bivec("n")
    mu = hour_angle_symbol
    Xi = shadow_plane_magnitude_symbol
    iota = gnomon_incl_symbol
    return mv.Mv(
        sin(Xi) * (-sin(mu) * (sin(iota) * n12 + cos(iota) * n23) + cos(mu) * n13),
        ga=n12.Ga,
    )


def shadow_bivector_magnitude_angle_cos(sunray: mv.Mv, gnomon: mv.Mv) -> mv.Mv:
    """
    TODO return type
    """
    return sp.trigsimp((sunray | gnomon).obj)


def hour_angle_sincos(shadow_plane: mv.Mv, meridian_plane: mv.Mv, zero_decl=True):
    """
    TODO
    """
    sinXi_sin_mu_delta = (
        cos(alpha) * sin(psi) * cos(sigma) - cos(psi) * sin(sigma)
        if zero_decl
        else None
    )
    sinXi_cos_mu_delta = sp.trigsimp(-(shadow_plane | meridian_plane).obj)
    return sinXi_sin_mu_delta, sinXi_cos_mu_delta


@lru_cache(maxsize=_cache_max_size)
def hour_angle_tan(shadow_plane: mv.Mv, meridian_plane: mv.Mv, zero_decl=True):
    """
    TODO
    """
    if not zero_decl:
        raise Exception(
            "A nice factorization for sin(mu) remains elusive for non-zero gnomon declination"
        )
    sinXi_sin_mu, sinXi_cos_mu = hour_angle_sincos(shadow_plane, meridian_plane)
    mu_ratio = sinXi_sin_mu / sp.collect(sinXi_cos_mu, cos(iota - theta))
    return sp.simplify(mu_ratio.subs(sin(sigma), tan(sigma) * cos(sigma)))


@lru_cache(maxsize=_cache_max_size)
def dialface_shadowbivector_intersection(
    dialface: mv.Mv, shadow_bivector: mv.Mv
) -> mv.Mv:
    """
    TODO
    """
    n1, n2, n3 = frame.base("n")
    return ((n1 ^ n2 ^ n3) * dialface * shadow_bivector).get_grade(1).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def dialface_shadowbivector_angle_cos(dialface: mv.Mv, shadow_bivector: mv.Mv) -> mv.Mv:
    """
    TODO
    """
    return sp.trigsimp(((dialface | shadow_bivector) / sin(_Xi)).obj)


@lru_cache(maxsize=_cache_max_size)
def unit_shadow(dialface: mv.Mv, shadow_bivector: mv.Mv) -> mv.Mv:
    """
    TODO
    """
    u = dialface_shadowbivector_intersection(dialface, shadow_bivector)
    return util.update_coeffs(u / sin(_Xi) / sin(_Psi), sp.factor)


@lru_cache(maxsize=_cache_max_size)
def unit_noon_shadow(dialface: mv.Mv, shadow_bivector: mv.Mv) -> mv.Mv:
    """
    TODO
    """
    w_hat = unit_shadow(dialface, shadow_bivector)

    cosPsi = dialface_shadowbivector_angle_cos(dialface, shadow_bivector)
    sinPsi_noon = sp.sqrt(1 - cosPsi**2).subs(mu, 0)
    return w_hat.subs(sin(_Psi), sinPsi_noon).subs(mu, 0)


@lru_cache(maxsize=_cache_max_size)
def noon_angle_sincos(dialface: mv.Mv, shadow_bivector: mv.Mv) -> mv.Mv:
    """
    TODO
    """
    w_hat = unit_shadow(dialface, shadow_bivector)
    noon = unit_noon_shadow(dialface, shadow_bivector)
    # TODO bug here if other names are chosen for angles
    cos_zeta = sp.collect(
        sp.collect(sp.trigsimp((w_hat | noon).obj), sin(mu) * sin(i) * sin(d)), cos(mu)
    )

    sin_zeta = -(w_hat ^ noon) | frame.dialface()
    return sin_zeta, cos_zeta


@lru_cache(maxsize=_cache_max_size)
def noon_angle_tan(dialface: mv.Mv, shadow_bivector: mv.Mv) -> mv.Mv:
    """
    TODO
    """
    sin_zeta, cos_zeta = noon_angle_sincos(dialface, shadow_bivector)
    return (sin_zeta / cos_zeta).subs(sin(mu), tan(mu) * cos(mu))


_mu_s = sp.Symbol(r"\mu_s")
_Xi_s = sp.Symbol(r"\Xi_s")


@lru_cache(maxsize=_cache_max_size)
def shadow_triangle_solution(
    latitude_symbol: sp.Symbol = theta,
    dial_incl_symbol: sp.Symbol = i,
    dial_decl_symbol: sp.Symbol = d,
    gnomon_incl_symbol: sp.Symbol = iota,
    hour_angle_symbol: sp.Symbol = _mu_s,
    shadow_plane_magnitude_symbol: sp.Symbol = _Xi_s,
):
    """
    TODO
    """
    theta = latitude_symbol
    i = dial_incl_symbol
    d = dial_decl_symbol
    iota = gnomon_incl_symbol
    mu_s = hour_angle_symbol
    Xi_s = shadow_plane_magnitude_symbol

    numerator = sin(i) * sin(iota) * cos(d) + cos(i) * cos(iota)
    denominator = sin(Xi_s) * (
        (sin(i) * cos(d) * cos(theta) - cos(i) * sin(theta)) * cos(mu_s)
        - sin(i) * sin(d) * sin(mu_s)
    ) + sin(alpha) * cos(sigma) * (sin(i) * sin(theta) * cos(d) + cos(i) * cos(theta))
    return numerator / denominator


@lru_cache(maxsize=_cache_max_size)
def gnomon_shadow_angle_sincos(
    dial_incl_symbol: sp.Symbol = i,
    dial_decl_symbol: sp.Symbol = d,
    gnomon_incl_symbol: sp.Symbol = iota,
    hour_angle_symbol: sp.Symbol = mu,
    dialface_shadowbivector_angle_symbol=_Psi,  # TODO private really if appears in interface here?
):
    """
    TODO
    """
    i = dial_incl_symbol
    d = dial_decl_symbol
    iota = gnomon_incl_symbol
    mu = hour_angle_symbol
    Psi = dialface_shadowbivector_angle_symbol

    sin_beta = (sin(i) * sin(iota) * cos(d) + cos(i) * cos(iota)) / sin(Psi)
    cos_beta = (
        (sin(i) * cos(d) * cos(iota) - sin(iota) * cos(i)) * cos(mu)
        - sin(d) * sin(i) * sin(mu)
    ) / sin(Psi)
    return sin_beta, cos_beta


def shadow_length(
    denom_symbol: sp.Symbol = D,
    shadow_plane_magnitude_symbol: sp.Symbol = _Xi,
    dialface_shadowbivector_angle_symbol=_Psi,  # TODO private really if appears in interface here?
):
    """
    TODO
    """
    D = denom_symbol
    Xi = shadow_plane_magnitude_symbol
    Psi = dialface_shadowbivector_angle_symbol

    return sin(Xi) * sin(Psi) / D


def shadow_coords_xy(
    dial_incl_symbol: sp.Symbol = i,
    dial_decl_symbol: sp.Symbol = d,
    gnomon_incl_symbol: sp.Symbol = iota,
    hour_angle_symbol: sp.Symbol = mu,
    shadow_plane_magnitude_symbol: sp.Symbol = _Xi,
):
    """
    TODO
    """
    i = dial_incl_symbol
    d = dial_decl_symbol
    iota = gnomon_incl_symbol
    mu = hour_angle_symbol
    Xi = shadow_plane_magnitude_symbol

    x = sin(Xi) * (cos(mu) * cos(d) - sin(mu) * sin(d) * cos(iota)) / D
    y = -sin(Xi) * (
        (
            cos(mu) * sin(d) * cos(i)
            + sin(mu) * (sin(i) * sin(iota) + sin(mu) * cos(d) * cos(i) * cos(iota))
        )
        / D
    )
    return x, y


def shadow_coords_xy_explicit():
    """
    TODO
    """

    x, y = shadow_coords_xy()

    S = shadow_bivector()
    M = frame.meridian_plane()
    sinXi_sin_mu, sinXi_cos_mu = hour_angle_sincos(S, M)

    Xi = sp.Symbol(r"\Xi")

    def _subs_numer(term):
        term = (
            (term / sin(Xi))
            .subs(sin(mu), sinXi_sin_mu)
            .subs(cos(mu), sinXi_cos_mu)
            .simplify()
        )
        term = term.subs(sin(sigma), tan(sigma) * cos(sigma)).simplify()
        return sp.collect(sp.collect(sp.collect(term, cos(psi)), sin(psi)), tan(sigma))

    mu_s = sp.Symbol(r"\mu_s")
    Xi_s = sp.Symbol(r"\Xi_s")
    Xi = sp.Symbol(r"\Xi")
    D_soln = shadow_triangle_solution()

    D_explicit = sp.simplify(
        D_soln.subs(mu_s, mu)
        .subs(Xi_s, Xi)
        .subs(sin(mu), sinXi_sin_mu / sin(Xi))
        .subs(cos(mu), sinXi_cos_mu / sin(Xi))
        .subs(iota, theta)
    )
    D_simple = sp.denom(D_explicit).subs(sin(sigma), tan(sigma) * cos(sigma)).simplify()

    xe = _subs_numer(x * D) / D_simple
    ye = _subs_numer(y * D) / D_simple
    return xe, ye


def sunray_dialface_angle_sin():
    """
    TODO
    """
    return (
        -sin(alpha) * sin(i) * sin(theta) * cos(d) * cos(sigma)
        - sin(alpha) * cos(i) * cos(sigma) * cos(theta)
        + sin(d) * sin(i) * sin(psi) * cos(alpha) * cos(sigma)
        - sin(d) * sin(i) * sin(sigma) * cos(psi)
        - sin(i) * sin(psi) * sin(sigma) * cos(d) * cos(theta)
        - sin(i) * cos(alpha) * cos(d) * cos(psi) * cos(sigma) * cos(theta)
        + sin(psi) * sin(sigma) * sin(theta) * cos(i)
        + sin(theta) * cos(alpha) * cos(i) * cos(psi) * cos(sigma)
    )
