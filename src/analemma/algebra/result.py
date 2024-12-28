"""
Alegbraic results for the geometry of sundials and the analemma
"""

import sympy as sp
from sympy import sin, cos, tan
from sympy.abc import alpha, sigma, psi, iota, theta, mu, i, d, D
from galgebra import mv
from functools import lru_cache
from typing import Tuple, TypeVar

from analemma.algebra import frame, util

_cache_max_size = 50

Xi = sp.Symbol(r"\Xi")
Psi = sp.Symbol(r"\Psi")

Expression = TypeVar("SympyExpression")


def shadow_bivector(
    sunray: mv.Mv = frame.sunray(), gnomon: mv.Mv = frame.gnomon("e")
) -> mv.Mv:
    r"""
    Form the bivector encoding the plane containing the sun ray, the gnomon, and (therefore) the shadow

    Given the `sunray` $s$ and `gnomon` $g$, form $S = s\wedge g$.

    See also [Setup and Definitions](../nb/sundial_setup.md).

    Parameters:
        sunray: The vector encoding the sun ray
        gnomon: The vector encoding the gnomon

    Returns:
        The shadow bivector $S$
    """
    return (sunray ^ gnomon).trigsimp()


def shadow_bivector_explicit(
    hour_angle_symbol: sp.Symbol = mu,
    shadow_plane_magnitude_symbol: sp.Symbol = Xi,
    gnomon_incl_symbol: sp.Symbol = iota,
) -> mv.Mv:
    r"""
    An explicit expression for the bivector encoding the plane containing the sun ray, the gnomon, and (therefore) the shadow

    See [The Shadow Plane](../nb/hour_angle.md#the-shadow-plane).

    Parameters:
        hour_angle_symbol: The hour angle, $\mu$
        shadow_plane_magnitude_symbol: The magnitude of the shadow bivector (see [angle between sun ray and gnomon][analemma.algebra.result.shadow_bivector_magnitude_angle_cos])
        gnomon_incl_symbol: The gnomon inclination, $\iota$

    Returns:
        An explicit expression for the shadow bivector $S$ under the assumption of zero gnomon declination ($\delta=0$)
    """
    n12, n13, n23 = frame.base_bivec("n")
    mu = hour_angle_symbol
    Xi = shadow_plane_magnitude_symbol
    iota = gnomon_incl_symbol
    return mv.Mv(
        sin(Xi) * (-sin(mu) * (sin(iota) * n12 + cos(iota) * n23) + cos(mu) * n13),
        ga=n12.Ga,
    )


def shadow_bivector_magnitude_angle_cos(sunray: mv.Mv, gnomon: mv.Mv) -> Expression:
    r"""
    Cosine of the angle between sun ray and gnomon, $\cos(\Xi)$

    Given the `sunray` $s$ and `gnomon` $g$, form $\cos(\Xi) = s\cdot g$. Note also that the magnitude of the shadow
    bivector $S$ is $\sin(\Xi)$ because $S^2 = -\sin^2(\Xi)$.

    See [The Shadow Plane](../nb/hour_angle.md#the-shadow-plane) and [Normalization](../nb/shadow_angle.md#normalization).

    Parameters:
        sunray: The vector encoding the sun ray
        gnomon: The vector encoding the gnomon

    Returns:
        $s\cdot g = \cos(\Xi)$
    """
    return sp.trigsimp((sunray | gnomon).obj)


def hour_angle_sincos(
    shadow_plane: mv.Mv, meridian_plane: mv.Mv, zero_decl: bool = True
) -> Tuple[Expression]:
    r"""
    Explicit expressions for the sine and cosine of the hour angle $\mu$

    The hour angle is measured between the meridian plane and the [shadow plane][analemma.algebra.result.shadow_bivector]
    (containing sun ray and gnomon). It is given by

    $$\cos(\mu) = \frac{-S\cdot M}{\sqrt{-S^2}\sqrt{-M^2}} = \frac{-S\cdot M}{\sin(\Xi)}$$

    given that $M^2 = -1$. $\sin(\mu) = \sqrt{1-\cos^2(\mu)}$ can be factorized when the gnomon declination $\delta =
    0$.

    See [The Hour Angle](../nb/hour_angle.md#the-hour-angle_1)

    Parameters:
        shadow_plane: The bivector $S = s\wedge g$ for sun ray $s$ and gnomon $g$
        meridian_plane: The bivector $M = n_1\wedge n_3$ (see [analemma.algebra.frame.meridian_plane][])
        zero_decl: Flag controlling assumption of zero gnomon declination. If False, None is returned for $\sin(\mu)$

    Returns:
        A 2-tuple of explicit expressions for `(`$\sin(\mu)$`, `$\cos(\mu)$`)`
    """
    sinXi_sin_mu_delta = (
        cos(alpha) * sin(psi) * cos(sigma) - cos(psi) * sin(sigma)
        if zero_decl
        else None
    )
    sinXi_cos_mu_delta = sp.trigsimp(-(shadow_plane | meridian_plane).obj)
    return sinXi_sin_mu_delta, sinXi_cos_mu_delta


@lru_cache(maxsize=_cache_max_size)
def hour_angle_tan(
    shadow_plane: mv.Mv, meridian_plane: mv.Mv, zero_decl: bool = True
) -> Expression:
    r"""
    Explicit expression for the tangent of the hour angle $\mu$

    The hour angle is measured between the meridian plane and the [shadow
    plane][analemma.algebra.result.shadow_bivector] (containing sun ray and gnomon).

    $\tan(\mu)$ is formed as the ratio of $\sin(\Xi)\sin(\mu)$ and $\sin(\Xi)\cos(\mu)$ as returned by
    [hour_angle_sincos][analemma.algebra.result.hour_angle_sincos] in which the
    [shadow plane magnitude][analemma.algebra.result.shadow_bivector_magnitude_angle_cos] factor of $\sin(\Xi)$ cancels.

    See [The Hour Angle](../nb/hour_angle.md#the-hour-angle_1).

    Parameters:
        shadow_plane: The bivector $S = s\wedge g$ for sun ray $s$ and gnomon $g$
        meridian_plane: The bivector $M = n_1\wedge n_3$ (see [analemma.algebra.frame.meridian_plane][])
        zero_decl: Flag controlling assumption of zero gnomon declination. Currently, supplying False reuslts in an error

    Returns:
        An explicit expression for $\tan(\mu)$
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
    r"""
    Form a vector along the intersection of the shadow plane and dial face

    Given the `dialface` $G$ and `shadow_bivector` $S$, this is given by

    $$u = \langle I G S \rangle_1$$

    where $I = e_1 \wedge e_2 \wedge e_3$ is the pseudo-scalar.

    $u$ has length $\sin(\Xi)\sin(Psi)$
    (see [Normalization](../nb/shadow_angle.md#normalization) and
    [dialface_shadowbivector_angle_cos][analemma.algebra.result.dialface_shadowbivector_angle_cos]).

    See [Intersecting the Shadow Plane and Dial Face](../nb/shadow_angle.md#intersecting-the-shadow-plane-and-dial-face).

    Parameters:
        dialface: Unit bivector encoding the face of the sundial see [analemma.algebra.frame.dialface][])
        shadow_bivector: The bivector $S = s\wedge g$ for sun ray $s$ and gnomon $g$

    Returns:
        The vector $u$ along the intersection between the shadow plane and dial face
    """
    n1, n2, n3 = frame.base("n")
    return ((n1 ^ n2 ^ n3) * dialface * shadow_bivector).get_grade(1).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def dialface_shadowbivector_angle_cos(
    dialface: mv.Mv, shadow_bivector: mv.Mv
) -> Expression:
    r"""
    Form the cosine of the angle $\Psi$ between the dial face and shadow plane

    The cosine of the angle between $S$ and $G$ is given by

    $$\cos(\Psi) = \frac{S\cdot G}{\sqrt{-S^2}\sqrt{-G^2}}$$

    Given that $S^2 = -\sin^2(\Xi)$ and $G^2 = -1$, we have

    $$ \cos(\Psi) = \frac{S\cdot G}{\sin(\Xi)}$$ as implemented by this function

    See [Normalization](../nb/shadow_angle.md#normalization)

    Parameters:
        dialface: Unit bivector encoding the face of the sundial see [analemma.algebra.frame.dialface][])
        shadow_bivector: The bivector $S = s\wedge g$ for sun ray $s$ and gnomon $g$

    Returns:
        Cosine of angle $\Psi$ between dial face and shadow plane
    """
    return sp.trigsimp(((dialface | shadow_bivector) / sin(Xi)).obj)


@lru_cache(maxsize=_cache_max_size)
def unit_shadow(dialface: mv.Mv, shadow_bivector: mv.Mv) -> mv.Mv:
    r"""
    Form $\hat{w}$, a unit vector parallel to the shadow

    The [Normalization](../nb/shadow_angle.md#normalization) section of the [shadow angle notebook](../nb/shadow_angle.md)
    shows that the length of $u$, the vector formed as the
    [intersection of the shadow plane and dial face][analemma.algebra.result.dialface_shadowbivector_intersection]
    is $\sin(\Xi)\sin(\psi)$. Here, we divide $u$ by its length to form a unit vector parallel to the shadow:

    $$\hat{w} = \frac{w}{L} = \frac{u}{|u|} = \frac{u}{\sin(\Xi)\sin(\psi)}$$

    See [Normalization](../nb/shadow_angle.md#normalization)

    Parameters:
        dialface: Unit bivector encoding the face of the sundial see [analemma.algebra.frame.dialface][])
        shadow_bivector: The bivector $S = s\wedge g$ for sun ray $s$ and gnomon $g$

    Returns:
        Unit vector parallel to the shadow, $\hat{w}$.
    """
    u = dialface_shadowbivector_intersection(dialface, shadow_bivector)
    return util.update_coeffs(u / sin(Xi) / sin(Psi), sp.factor)


@lru_cache(maxsize=_cache_max_size)
def unit_noon_shadow(dialface: mv.Mv, shadow_bivector: mv.Mv) -> mv.Mv:
    r"""
    Evalute the unit shadow vector at noon

    Noon occurs when the hour angle $\mu$ is zero, which is when the shadow plane is parallel to the meridian plane.

    See [The Shadow Angle Relative to Noon](../nb/shadow_angle.md#the-shadow-angle-relative-to-noon)

    Parameters:
        dialface: Unit bivector encoding the face of the sundial see [analemma.algebra.frame.dialface][])
        shadow_bivector: The bivector $S = s\wedge g$ for sun ray $s$ and gnomon $g$

    Returns:
        The unit shadow vector at noon when $\mu = 0$
    """
    w_hat = unit_shadow(dialface, shadow_bivector)

    cosPsi = dialface_shadowbivector_angle_cos(dialface, shadow_bivector)
    sinPsi_noon = sp.sqrt(1 - cosPsi**2).subs(mu, 0)
    return w_hat.subs(sin(Psi), sinPsi_noon).subs(mu, 0)


@lru_cache(maxsize=_cache_max_size)
def noon_angle_sincos(dialface: mv.Mv, shadow_bivector: mv.Mv) -> Tuple[mv.Mv]:
    r"""
    The sine and cosine of the angle $\zeta$ between the shadow at an arbitrary time and the shadow at noon

    $\cos(\zeta)$ is calculated as $\hat{w}(\mu) \cdot \hat{w}(0)$ where $\hat{w}(\mu)$ is the unit shadow at
    an arbitrary time, or equivalently hour angle, and $\hat{w}(0)$ is the unit shadow at noon, when $\mu = 0$.

    Instead of attempting to factorize $\sqrt{1-\cos^2(\zeta)}$, we can use the fact that the shadow lives in
    the plane of the dial face, so $\hat{w}(\mu) \wedge \hat{w}(0) = \sin(\zeta) G$, and $G^2 = -1$.

    See [The Shadow Angle Relative to Noon](../nb/shadow_angle.md#the-shadow-angle-relative-to-noon)

    Parameters:
        dialface: Unit bivector encoding the face of the sundial see [analemma.algebra.frame.dialface][])
        shadow_bivector: The bivector $S = s\wedge g$ for sun ray $s$ and gnomon $g$

    Returns:
        The 2-tuple `(`$\sin(\zeta)$`, `$\cos(\zeta)$`)`, where $\zeta$ is the noon shadow angle
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
    r"""
    The tangent of the angle $\zeta$ between the shadow at an arbitrary time and the shadow at noon

    Formed as the ratio of $\sin(\zeta)$ and $\cos(\zeta)$ as returned by
    [noon_angle_sincos][analemma.algebra.result.noon_angle_sincos]

    See [The Shadow Angle Relative to Noon](../nb/shadow_angle.md#the-shadow-angle-relative-to-noon)

    Parameters:
        dialface: Unit bivector encoding the face of the sundial see [analemma.algebra.frame.dialface][])
        shadow_bivector: The bivector $S = s\wedge g$ for sun ray $s$ and gnomon $g$

    Returns:
        $\tan(\zeta), where $\zeta$ is the noon shadow angle
    """
    sin_zeta, cos_zeta = noon_angle_sincos(dialface, shadow_bivector)
    return (sin_zeta / cos_zeta).subs(sin(mu), tan(mu) * cos(mu))


mu_s = sp.Symbol(r"\mu_s")
Xi_s = sp.Symbol(r"\Xi_s")


@lru_cache(maxsize=_cache_max_size)
def shadow_triangle_solution(
    latitude_symbol: sp.Symbol = theta,
    dial_incl_symbol: sp.Symbol = i,
    dial_decl_symbol: sp.Symbol = d,
    gnomon_incl_symbol: sp.Symbol = iota,
    hour_angle_symbol: sp.Symbol = mu_s,
    shadow_plane_magnitude_symbol: sp.Symbol = Xi_s,
) -> Expression:
    r"""
    The scale factor $\lambda$ on the unit sun ray $s$ that solves the shadow triangle

    Given the triangle $w(\lambda) = g + \lambda s$ where $g$ is the gnomon and $s$ is the sun ray,
    enforce that $w(\lambda)$ lies in the dial face by solving $w(\lambda) \wedge G = 0$ for $\lambda$.
    The resulting solution is given by this function.

    See [Solving the Shadow Triangle](../nb/shadow_length.md#solving-the-shadow-triangle)

    ![](https://raw.githubusercontent.com/russellgoyder/sundial-latex/main/figs/ShadowTriangle.png "The shadow triangle.").

    Parameters:
        latitude_symbol: $90^\circ$ minus the latitude at which the frame is embedded
        dial_incl_symbol: The inclination angle of the dial face relative to the surface frame
        dial_decl_symbol: The declination angle of the dial face relative to the surface frame
        gnomon_incl_symbol: The inclination angle of the gnomon relative to the surface frame
        hour_angle_symbol: The [hour angle][analemma.algebra.result.hour_angle_tan] of the sun
        shadow_plane_magnitude_symbol: The angle between the sun ray and gnomon in the [shadow bivector][analemma.algebra.result.shadow_bivector]

    Returns:
        An expression for $\lambda$ that ensures that $w(\lambda) = g + \lambda s$ is the shadow cast by the gnomon on the dial face
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
    dialface_shadowbivector_angle_symbol: sp.Symbol = Psi,
) -> Tuple[Expression]:
    r"""
    The sine and cosine of the angle $\beta$ between the gnomon and the shadow

    $\cos(\beta)$ is readily available via $g \cdot \hat{w}$ and $\sin(\beta)$ is related to $g \wedge \hat{w}$.
    This function provides explicit forms for these results.

    See [Angle between Gnomon and Shadow](../nb/shadow_length.md#angle-between-gnomon-and-shadow) and
    [shadow_triangle_solution][analemma.algebra.result.shadow_triangle_solution]

    Parameters:
        dial_incl_symbol: The inclination angle of the dial face relative to the surface frame
        dial_decl_symbol: The declination angle of the dial face relative to the surface frame
        gnomon_incl_symbol: The inclination angle of the gnomon relative to the surface frame
        hour_angle_symbol: The [hour angle][analemma.algebra.result.hour_angle_tan] of the sun
        dialface_shadowbivector_angle_symbol: The angle between the dial face and shadow bivector (see [dialface_shadowbivector_angle_cos][analemma.algebra.result.dialface_shadowbivector_angle_cos])

    Returns:
        The 2-tuple `(`$\sin(\beta)$`, `$\cos(\beta)$`)`, where $\beta$ is the gnomon-shadow angle
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
    shadow_plane_magnitude_symbol: sp.Symbol = Xi,
    dialface_shadowbivector_angle_symbol: sp.Symbol = Psi,
) -> Expression:
    r"""
    An expression for $L$, the length of the shadow

    $N$, the numerator of $\lambda$ in the solution to the shadow triangle is equal to
    $\sin(\beta)$, which given the form of $\sin(\beta)$ from
    [gnomon_shadow_angle_sincos][analemma.algebra.result.gnomon_shadow_angle_sincos] means that

    $$\sin(\beta) = \frac{N}{\sin(\Psi)}$$

    and so, given the form of $L$ given in
    [Solving the Shadow Triangle](../nb/shadow_length.md#solving-the-shadow-triangle), we have

    $$L = \frac{\sin(\Xi)\sin(\Psi)}{D}$$

    See also [The Shadow Vector](../nb/shadow_length.md#the-shadow-vector)

    Parameters:
        denom_symbol: The denominator $D$ of the shadow triangle solution $\lambda = \frac{N}{D}$
        shadow_plane_magnitude_symbol: The magnitude angle of the shadow bivector (see [angle between sun ray and gnomon][analemma.algebra.result.shadow_bivector_magnitude_angle_cos])
        dialface_shadowbivector_angle_symbol: The angle between the dial face and shadow bivector (see [dialface_shadowbivector_angle_cos][analemma.algebra.result.dialface_shadowbivector_angle_cos])

    Returns:
        $L = \sin(\Xi)\sin(\Psi)/D$
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
    shadow_plane_magnitude_symbol: sp.Symbol = Xi,
) -> Tuple[Expression]:
    r"""
    The Cartesian coordinates of the analemma on the face of a sundial

    Given the shadow vector $w = L\hat{w}$ (constructed by scaling the
    [unit shadow][analemma.algebra.result.unit_shadow] by its [length][analemma.algebra.result.shadow_length]),
    the coordinates of the tip of the shadow on the dial face can be found via

    $x = m_1 \cdot w$

    $y = m_2 \cdot w$

    where $m_1$ and $m_2$ are the first two vectors in the [dial frame][analemma.algebra.frame.dial].

    $z = 0$ because the shadow is parallel to the face of the dial by definition.

    This function provides $x$ and $y$ in closed form.

    See [The Analemma](../nb/analemma.md)

    Parameters:
        dial_incl_symbol: The inclination angle of the dial face relative to the surface frame
        dial_decl_symbol: The declination angle of the dial face relative to the surface frame
        gnomon_incl_symbol: The inclination angle of the gnomon relative to the surface frame
        hour_angle_symbol: The [hour angle][analemma.algebra.result.hour_angle_tan] of the sun
        shadow_plane_magnitude_symbol: The angle between the sun ray and gnomon in the [shadow bivector][analemma.algebra.result.shadow_bivector]

    Returns:
        2-tuple of `(`$x$`, `$y$`)`
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
            + sin(mu) * (sin(i) * sin(iota) + cos(d) * cos(i) * cos(iota))
        )
        / D
    )
    return x, y


def shadow_coords_xy_explicit() -> Tuple[Expression]:
    r"""
    Explicit expressions for the Cartesian coordinates of the analemma in terms of fundamental angles

    [shadow_coords_xy][analemma.algebra.result.shadow_coords_xy] expresses the analemma in terms of a mixture
    of the angles found in [][analemma.algebra.frame] and the following derived quantities:

     1. $\mu$, the [hour angle][analemma.algebra.result.hour_angle_tan]
     1. $\Xi$, the [shadow bivector magnitude angle][analemma.algebra.result.shadow_bivector_magnitude_angle_cos]
     1. $D$, the denominator of $\lambda$, the [solution][analemma.algebra.result.shadow_triangle_solution] to the shadow triangle problem

    This function eliminates these quantities in the expressions for $x$ and $y$, the coordinates of the analemma,
    leaving only the fundamental angles that determine the location, orientation and geometry of the sundial
    as a function of date and time.

    See [The Analemma](../nb/analemma.md)

    Returns:
        2-tuple of `(`$x$`, `$y$`)` in terms of fundamental angles
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


def sunray_dialface_angle_sin() -> Expression:
    r"""
    The sine of the angle $\gamma$ between the sun ray and the dial face

    It can be important (when plotting analemmas for example) to determine an effective sunrise and sunset
    for a given sundial, defined as the points in the sun's journey across the sky when the angle between
    the sun ray $s$ and the dial face $G$ is zero (or $\pi$). This angle is given by

    $\sin(\gamma) = -(G \wedge s) \cdot I$

    where $I = e_1 \wedge e_2 \wedge e_3$.

    This function returns an explicit expression for $\sin(\gamma)$ in terms of fundamental angles.
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


def gnomon_dialface_angle_sin(
    dial_incl_symbol: sp.Symbol = i,
    dial_decl_symbol: sp.Symbol = d,
    gnomon_incl_symbol: sp.Symbol = iota,
) -> Expression:
    r"""
    The sine of the angle between the gnomon and the dial face

    In Rohr's 1996 book *SUNDIALS History, Theory and Practice*, he refers to the substyle, which is the projection of
    the style onto the dial face. Here, we allow the gnomon to have an independent inclination $\iota$ which matches a
    style when $\iota = \theta$, the ($90^\circ$ minus) latitude angle.

    We can readily project the gnomon $g$ onto the dial face $G$. The vector $\bar{b} = g\cdot G$ is parallel to $G$ (ie
    lies in the dial face) and is rotated with the orientation of $G$ (anticlockwise as seen from above the dial) by
    $\pi/2$ relative to the subgnomon $b = R \bar{b} \tilde{R}$ where $R = \exp(\frac{\pi}{4} \, G)$ undoes that $90^\circ$
    rotation (notice the angle in $R$ is positive in contrast to typical [rotations][analemma.algebra.util.rotate]).

    Let $A$ denote the angle between the (unit length) gnomon $g$ and the subgnomon $b$ (or
    equivalently the dial face $G$). Then, $\cos(A) = \sqrt{b^2}$, and $1-\cos^2(A)$ factorizes nicely to give $\sin(A)$
    as returned by this function.

    See also [Comparison with Rohr's Book](../nb/rohr_comparison.md) and
    [analemma.tests.test_results.test_gnomon_dialface_angle_pythagoras_identity][].

    Parameters:
        dial_incl_symbol: The inclination angle of the dial face relative to the surface frame
        dial_decl_symbol: The declination angle of the dial face relative to the surface frame
        gnomon_incl_symbol: The inclination angle of the gnomon relative to the surface frame

    Returns:
        $\sin(A)$, the angle between gnomon and subgnomon (or, equivalently, the dial face)
    """
    i = dial_incl_symbol
    d = dial_decl_symbol
    iota = gnomon_incl_symbol
    return cos(i) * cos(iota) + sin(i) * sin(iota) * cos(d)
