"""
Key vectors, bivectors, and frames defining the location, orientation and geometry of a sundial.

See [Setup and Definitions](../nb/sundial_setup.md) for the algebraic form taken by these objects.
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
def space_algebra(symbol: str) -> Ga:
    """
    The Geometric Algebra (GA) of 3-dimensional Euclidean space

    Parameters:
        symbol: Symbol to use for the basis vectors, eg 'e' gives $e_1, e_2, e_3$
            The vectors are one-based because zero is reserved for time, but relativitic effects are negligible in this analysis.

    Returns:
        A [galgebra.ga.Ga](https://galgebra.readthedocs.io/en/latest/) object representing the Geometric Algebra
    """

    coords = sp.symbols("1 2 3", real=True)
    G3 = Ga(symbol, g=[1, 1, 1], coords=coords)

    return G3


@lru_cache(maxsize=_cache_max_size)
def base(base_symbol: str) -> Tuple[mv.Mv]:
    """
    The set of basis vectors from the Geometric Algebra (GA) with basis vectors identified by `base_symbol`

    See also [Fixed Stars and Earth Frames](../nb/sundial_setup.md#fixed-stars-and-earth-frames)

    ![Earth's orientation and orbit](https://raw.githubusercontent.com/russellgoyder/sundial-latex/main/figs/MainArena.png "Earth's orientation and orbit.").

    Parameters:
        base_symbol: Symbol to identify the basis vectors, eg supplying '$e$' gives $e_1, e_2, e_3$

    Returns:
        A 3-tuple of galgebra.mv.Mv objects each representing a basis vector
    """
    return space_algebra(base_symbol).mv()


@lru_cache(maxsize=_cache_max_size)
def base_bivec(base_symbol: str) -> Tuple[mv.Mv]:
    r"""
    The set of basis bivectors from the Geometric Algebra (GA) with basis vectors identified by `base_symbol`

    Parameters:
        base_symbol: Symbol to identify the basis bivectors, eg supplying 'e' gives $e_1\wedge e_2,\; e_1\wedge e_3,\; e_2\wedge e_3$

    Returns:
        A 3-tuple of galgebra.mv.Mv objects each representing a basis bivector

    """
    v1, v2, v3 = base(base_symbol)
    return v1 ^ v2, v1 ^ v3, v2 ^ v3


_fixed_stars_symbol: str = "e"


@lru_cache(maxsize=_cache_max_size)
def scalar_element(value: float, base_symbol: str = _fixed_stars_symbol) -> mv.Mv:
    """
    The unit scalar element of the Geometric Algebra (GA) scaled by `value`

    Parameters:
        value: The numerical value of the returned scalar
        base_symbol: Symbol to identify the basis vectors of the GA

    Returns:
        The scalar element of the GA with value `value`
    """
    return space_algebra(base_symbol).mv(value, "scalar")


@lru_cache(maxsize=_cache_max_size)
def planet(
    axis_tilt_symbol: sp.Symbol = alpha,
    rotation_symbol: sp.Symbol = psi,
) -> Tuple[mv.Mv]:
    r"""
    A vector frame embedded in a planet

    This frame accounts for the tilt of the planet's axis of rotation relative to the plane of its orbit around a star
    (via axis_tilt_symbol), and encodes the rotation itself via `rotation_symbol`. The frame is calculated by forming
    the two rotors

    $R_\alpha = \exp(-e_1 \wedge e_3\frac{1}{2}\alpha)$

    $R_\psi = \exp(-R_\alpha e_1 \tilde{R}_\alpha \wedge e_2\frac{1}{2}\psi)$

    and applying them together as $R_\psi R_\alpha$ to the fixed $\{e_i\}$ frame from [analemma.algebra.frame.base][].

    ![Earth's orientation and orbit](https://raw.githubusercontent.com/russellgoyder/sundial-latex/main/figs/MainArena.png "Earth's orientation and orbit.").

    See also [Fixed Stars and Earth Frames](../nb/sundial_setup.md#fixed-stars-and-earth-frames)

    Parameters:
        axis_tilt_symbol: Symbol denoting the angle of tilt of the planet's axis of rotation
        rotation_symbol: Symbol denoting the planet's angle of rotation, starting when $f_1$ is parallel to $e_1$

    Returns:
        Tuple of 3 basis vectors (as galgebra.mv.Mv objects) embedded in the planet with $f_3$ parallel to the axis of rotation
    """
    (e1, e2, e3) = base(_fixed_stars_symbol)
    e1_prime = util.rotate(e1, axis_tilt_symbol, e1 ^ e3).trigsimp()

    f1 = util.rotate(e1_prime, rotation_symbol, e1_prime ^ e2).trigsimp().trigsimp()
    f2 = util.rotate(e2, rotation_symbol, e1_prime ^ e2).trigsimp().trigsimp()
    f3 = util.rotate(e3, axis_tilt_symbol, e1 ^ e3).trigsimp().trigsimp()

    return (f1, f2, f3)


@lru_cache(maxsize=_cache_max_size)
def surface(latitude_symbol: sp.Symbol = theta) -> Tuple[mv.Mv]:
    r"""
    A vector frame embedded in the surface of a planet

    The frame is formed by rotating the frame given by [analemma.algebra.frame.planet][] by an angle `latitude_symbol`
    in the $f_1 \wedge f_3$ plane, using the rotor $R_\theta = \exp(-f_3 \wedge f_1\frac{1}{2}\theta)$.

    See also [Surface Frame](../nb/sundial_setup.md#surface-frame)

    ![](https://raw.githubusercontent.com/russellgoyder/sundial-latex/main/figs/SurfaceFrame.png "Frame embedded in Earth's surface.").

    Parameters:
        latitude_symbol: $90^\circ$ minus the latitude at which the frame is embedded

    Returns:
        Tuple of 3 basis vectors (as galgebra.mv.Mv objects) at the surface of the planet with $n_3$ pointing overhead

    Note:
        `latitude_symbol` is only directly related to latitude. It is $90^\circ$ minus the latitude at which the frame is embedded.
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
        sympy_n3, ga=space_algebra(_fixed_stars_symbol)
    )  # TODO replace with f1.Ga.mv()?

    return n1, n2, n3


@lru_cache(maxsize=_cache_max_size)
def surface_bivec(latitude_symbol: sp.Symbol = theta) -> Tuple[mv.Mv]:
    r"""
    Frame of bivectors contructed from the [surface][analemma.algebra.frame.surface] vector frame

    Given the vector frame $n_1, n_2, n_3$, the bivector frame is formed as:

    $n_1\wedge n_2,\; n_1\wedge n_3,\; n_2\wedge n_3$

    See also [Setup and Definitions](../nb/sundial_setup.md)

    Parameters:
        latitude_symbol: $90^\circ$ minus the latitude at which the frame is embedded

    Returns:
        Tuple of 3 basis bivectors (as galgebra.mv.Mv objects) at the surface of the planet with $n_3$ pointing overhead

    Note:
        `latitude_symbol` is only directly related to latitude. It is $90^\circ$ minus the latitude at which the frame is embedded.
    """
    n1, n2, n3 = surface(latitude_symbol)
    n12 = mv.Mv(sp.trigsimp(sp.expand_trig((n1 ^ n2).obj)), ga=n1.Ga)
    n13 = mv.Mv(sp.trigsimp(sp.expand_trig((n1 ^ n3).obj)), ga=n1.Ga)
    n23 = mv.Mv(sp.trigsimp(sp.expand_trig((n2 ^ n3).obj)), ga=n1.Ga)
    return n12, n13, n23


@lru_cache(maxsize=_cache_max_size)
def meridian_plane() -> mv.Mv:
    r"""
    The meridian plane encoded as a bivector

    The meridian plane contains a line of longitude and is perpendicular to all lines of latitude. It is therefore
    parallel to North-South lines and is formed as $n_1\wedge n_3$. Note that while $n_1$ and $n_3$ are both functions
    of latitude, this dependency cancels in the meridian plane and it depends on the
    [planet][analemma.algebra.frame.planet] parameters (axis tilt and rotation angles) only. See also
    [Orbit Rotor and Meridian Plane](../nb/sundial_setup.md#orbit-rotor-and-meridian-plane)

    Returns:
        Bivector representing the meridian plane
    """
    n1, _, n3 = surface()
    return (n1 ^ n3).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def sunray(
    orbit_symbol: sp.Symbol = sigma,
) -> mv.Mv:
    r"""
    A vector parallel to rays of sun light traveling toward the center of the planet

    This vector is formed by rotating $e_1$ by the orbit angle using the rotor

    $R_\sigma = \exp(-e_1 \wedge e_2 \frac{1}{2}\sigma)$

    to give

    $s = R_\sigma e_1 \tilde{R}_\sigma = \cos\sigma \, e_1 + \sin\sigma e_2$

    See also [Orbit Rotor and Meridian Plane](../nb/sundial_setup.md#orbit-rotor-and-meridian-plane)

    Parameters:
        orbit_symbol: The planet's orbit angle

    Returns:
        The vector $s$ parallel to sun rays traveling directly toward a planet from its star
    """
    (e1, e2, _) = base(_fixed_stars_symbol)
    return util.rotate(e1, orbit_symbol, e1 ^ e2).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def dial(
    incl_symbol: sp.Symbol = i,
    decl_symbol: sp.Symbol = d,
) -> Tuple[mv.Mv]:
    r"""
    The dial frame

    This vector frame is aligned with the face of the sundial such that $m_3$ is perpendicular pointing up. It is formed
    by applying the rotor $R_dR_i$ to the [surface][analemma.algebra.frame.surface] frame, where

    $R_i = \exp( - n_1 \wedge n_3 \frac{1}{2} i )$

    and

    $R_d = \exp( - n_1 \wedge n_2 \frac{1}{2} d )$

    See also [Dial Face and Gnomon](../nb/sundial_setup.md#dial-face-and-gnomon)

    ![](https://raw.githubusercontent.com/russellgoyder/sundial-latex/main/figs/DialFrame.png "Frame embedded in the sundial's face.").

    Parameters:
        incl_symbol: The inclination angle of the dial face relative to the surface frame
        decl_symbol: The declination angle of the dial face relative to the surface frame

    Returns:
        A 3-tuple of vectors forming the dial frame
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
def dialface() -> mv.Mv:
    r"""
    The dial face as a unit bivector

    Given $m_1$ and $m_2$ from the [dial frame][analemma.algebra.frame.dial], form:

    $m_1 \wedge m_2$

    See also [Dial Face and Gnomon](../nb/sundial_setup.md#dial-face-and-gnomon)

    Returns:
        A bivector representing the face of the sundial
    """
    m1, m2, _ = dial()
    return (m1 ^ m2).trigsimp()


@lru_cache(maxsize=_cache_max_size)
def _form_gnomon(
    zero_decl: bool = True,
    incl_symbol: sp.Symbol = iota,
    decl_symbol: sp.Symbol = delta,
):
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
) -> mv.Mv:
    r"""
    Form the gnomon on the sundial

    The gnomon (silent g) is the part of a sundial that casts the shadow. Here we assume it is a rod or stick of unit
    length, and calculate it by rotating the [surface][analemma.algebra.frame.surface] vector $n_3$ (pointing overhead)
    by two angles via $g = R_\delta R_\iota n_3 \tilde{$_\iota}\tilde{R_\delta}$ where

    $R_\iota = \exp( - n_1 \wedge n_3 \frac{1}{2} \iota )$

    and

    $R_\delta = \exp( - n_1 \wedge n_2 \frac{1}{2} \delta )$

    See also [Dial Face and Gnomon](../nb/sundial_setup.md#dial-face-and-gnomon)

    ![](https://raw.githubusercontent.com/russellgoyder/sundial-latex/main/figs/Gnomon.png "The gnomon.").

    Parameters:
        base_symbol: Symbol identifing the basis on which to project the gnomon

            * '`e`' selects the [frame of the fixed stars][analemma.algebra.frame.base]
            * '`n`' selects the [surface frame][analemma.algebra.frame.surface]
        zero_decl: Indicates whether declination is assumed to be zero, which gives simpler results
        incl_symbol: The inclination angle of the gnomon relative to the surface frame
        decl_symbol: The declination angle of the gnomon relative to the surface frame

    Returns:
        A vector representing the gnomon
    """
    if base_symbol == "n":
        return _form_gnomon(zero_decl, incl_symbol, decl_symbol)
    elif base_symbol == "e":
        gn = _form_gnomon(zero_decl, incl_symbol, decl_symbol)
        return util.update_coeffs(
            util.project_vector(gn, target_frame=base("n"), render_frame=surface())
        )
    else:
        raise Exception("Base symbol must be either 'n' or 'e'")
