"""
Tests relating to content of [algebra.result](result_ref.md)
"""

import sympy as sp
from sympy import sin, cos
from sympy.abc import mu, iota, theta, alpha, psi, sigma, i, d
from analemma.algebra import frame, util, result


def test_3frame_orthonormality():
    """
    Ensure that each frame of three vectors is orthonormal
    """

    def _check_orthnormal(three_frame) -> None:
        v1, v2, v3 = three_frame
        # more elegant to ensure that the geometric product of the vectors is equal to the pseudoscalar:
        # assert (v1*v2*v3).trigsimp().obj.equals((e1^e2^e3).obj)
        # but this is significantly faster and equivalent
        assert v1 | v1 == v2 | v2 == v3 | v3 == frame.scalar_element(1)
        assert v1 | v2 == v1 | v3 == v2 | v3 == frame.scalar_element(0)

    _check_orthnormal(frame.base("e"))
    # _check_orthnormal(frame.dial()) # need to project/move between GAs
    _check_orthnormal(frame.planet())
    _check_orthnormal(frame.surface())


def test_dialface_orientation():
    r"""
    Ensure that the dial face formed from rotated vectors matches the corresponding rotated surface bivector

    The dial face $G$ is formed as $m_1\wedge m_2$ in [analemma.algebra.frame.dialface][]. Check that this
    is consistent with $R n_1 \wedge n_2 \tilde{R}$ where

    $R = \exp(-n_1\wedge n_2\frac{1}{2}d) \, \exp(-n_1\wedge n_3\frac{1}{2}i)$
    """
    Gn = frame.dialface()
    # this is formed as m1^m2
    # check that we get the same result by rotating n1^n2
    nn1, nn2, nn3 = frame.base("n")
    assert (
        Gn == util.rotate(util.rotate(nn1 ^ nn2, i, nn1 ^ nn3), d, nn1 ^ nn2).trigsimp()
    )


def test_shadow_bivector_magnitude():
    r"""
    Checks related to the magnitude of $S^2$.

    Check that $S^2 = (s\wedge g)^2 = (s\cdot g)^2 - s^2 g^2$ where $s^2 = g^2 = 1$

    and ensure that

    $\sqrt{-S^2} = \sqrt{1 - (s\cdot g)^2} = \sqrt{1 - \cos^2(\Xi}) = \sin(\Xi)$
    """
    s = frame.sunray()
    g = frame.gnomon("e")
    sg_squared = (s | g) * (s | g)

    S = result.shadow_bivector(s, g)
    S_norm_check = (S | S) - (sg_squared - 1)
    assert S_norm_check.obj.equals(0)

    sinXi = sp.sqrt(
        1
        - (
            (cos(alpha) * cos(sigma) * cos(psi) + sin(sigma) * sin(psi))
            * sin(iota - theta)
            + sin(alpha) * cos(sigma) * cos(iota - theta)
        )
        ** 2
    )
    sg_check = sinXi**2 - (1 - sg_squared)
    assert sg_check.obj.equals(0)


def test_hour_angle_pythagoras_identity():
    r"""
    Check that $\sin^2(\Xi)\sin^2(\mu) + \sin^2(\Xi)\cos^2(\mu) = \sin^2(\Xi)$
    """

    s = frame.sunray()
    g = frame.gnomon("e")
    S = result.shadow_bivector(s, g)

    M = frame.meridian_plane()

    sinXi_sin_mu, sinXi_cos_mu = result.hour_angle_sincos(S, M)

    sinXi = sp.sqrt(
        1
        - (
            (cos(alpha) * cos(sigma) * cos(psi) + sin(sigma) * sin(psi))
            * sin(iota - theta)
            + sin(alpha) * cos(sigma) * cos(iota - theta)
        )
        ** 2
    )
    Xi_check = sp.trigsimp(
        sp.expand(sinXi_sin_mu**2) + sp.expand(sinXi_cos_mu**2) - sp.expand(sinXi**2)
    )
    assert Xi_check.equals(0)


def test_shadow_plane_consistency():
    r"""
    Compare two forms of the shadow plane $S$

    Ensure that the [explicit form][analemma.algebra.result.shadow_bivector_explicit]
    is equal to the [form derived as $s \wedge g$][analemma.algebra.result.shadow_bivector].
    """
    S_explicit = result.shadow_bivector_explicit()
    on_e_bivecs = util.project_bivector(
        S_explicit, frame.base_bivec("n"), frame.surface_bivec()
    )

    s = frame.sunray()
    g = frame.gnomon("e")
    S = result.shadow_bivector(s, g)

    sinXi_sin_mu, sinXi_cos_mu = result.hour_angle_sincos(S, frame.meridian_plane())
    Xi = sp.Symbol(r"\Xi")

    should_be_zero = S - on_e_bivecs.subs(sin(mu), sinXi_sin_mu / sin(Xi)).subs(
        cos(mu), sinXi_cos_mu / sin(Xi)
    )
    assert should_be_zero.obj.equals(0)


def test_shadow_bivector_magnitude_angle_cos():
    r"""
    Compare two different ways of calculating $\cos(\Xi)$

    Ensure that $s\cdot g$ as computed via symbolic algebra is equal to an independent derivation
    """

    s = frame.sunray()
    gn = frame.gnomon("n", zero_decl=True)
    g = util.project_vector(
        gn, target_frame=frame.base("n"), render_frame=frame.surface()
    )
    cosXi = result.shadow_bivector_magnitude_angle_cos(s, g)
    should_be = -(
        sin(alpha) * cos(iota - theta) + sin(iota - theta) * cos(alpha) * cos(psi)
    ) * cos(sigma) - sin(psi) * sin(sigma) * sin(iota - theta)
    assert sp.simplify(cosXi - should_be).equals(0)


def test_shadowplane_dialface_angle():
    r"""
    Compare two ways of calculating $\cos(\Psi)$

    Ensure that

    $\frac{S\cdot G}{\sqrt{-S^2}\sqrt{-G^2}} = \frac{S\cdot G}{\sin(\Xi)}$
    """

    # check (force sympy to give up the positive square root in each factor in the denominator)
    def _norm(B):
        return sp.powdenest(sp.sqrt(sp.trigsimp((-B | B).obj)), force=True)

    Sn = result.shadow_bivector_explicit()
    Gn = frame.dialface()
    Xi = sp.Symbol(r"\Xi")
    SuGu_check = sp.trigsimp((Sn | Gn).obj) / _norm(Sn) / _norm(Gn) - (Sn | Gn) / sin(
        Xi
    )
    assert SuGu_check.obj.equals(0)


def test_shadowplane_dialface_intersection_length():
    r"""
    A check on the length of $u$, the vector formed as the intersection of shadow bivector and dial face

    Ensure that $u^2 = \sin^2(\Xi) - (S\cdot G)^2$
    """
    Sn = result.shadow_bivector_explicit()
    Gn = frame.dialface()
    u = result.dialface_shadowbivector_intersection(Gn, Sn)
    Xi = sp.Symbol(r"\Xi")
    u2_check = (u | u) - (sin(Xi) ** 2 - (Sn | Gn) ** 2)
    assert u2_check.obj.equals(0)


def test_unit_shadow_normalization():
    r"""
    Check the normalization of $u$, the vector formed as the intersection of shadow bivector and dial face

    Check that

    $\hat{w}^2 = \left(\frac{u}{\sin(\Xi)\sin(\Psi)}\right)^2 = 1$

    by showing that

    $\frac{u^2}{\sin^2(\Xi)} - \sin^2\Psi = 0$
    """
    Sn = result.shadow_bivector_explicit()
    Gn = frame.dialface()
    cosPsi = result.dialface_shadowbivector_angle_cos(Gn, Sn)
    sinPsiSquared = 1 - sp.expand(cosPsi**2)
    u = result.dialface_shadowbivector_intersection(Gn, Sn)
    Xi = sp.Symbol(r"\Xi")
    u_over_sinXi_squared = sp.trigsimp(((u | u) / sin(Xi) ** 2).obj)
    what_check = sp.trigsimp(u_over_sinXi_squared - sinPsiSquared)
    assert what_check.equals(0)


def test_noon_shadow_angle_pythagoras_identity():
    r"""
    Check that the expressions for $\sin(\zeta)$ and $\cos(\zeta)$ obey the Pythagorean identity

    We have $\sin^2(\zeta) + \cos^2(\zeta) = 1$ if the numerator of the left hand side is equal to the denominator,
    and we have an explicit expression for $\cos(\Psi) = \sqrt{1 - \sin^2(\Psi)}$.
    """
    Sn = result.shadow_bivector_explicit()
    Gn = frame.dialface()
    sin_zeta, cos_zeta = result.noon_angle_sincos(Gn, Sn)
    cosPsi = result.dialface_shadowbivector_angle_cos(Gn, Sn)
    term = sp.simplify(sp.expand(sp.trigsimp(sin_zeta.obj) ** 2 + cos_zeta**2))
    zeta_check = sp.trigsimp(sp.numer(term) - (1 - cosPsi**2))
    assert zeta_check.equals(0)


def test_shadow_triangle_solution():
    r"""
    Check that the shadow triangle solution $\lambda$ solves the problem $g + \lambda s = 0$
    """
    Gn = frame.dialface()
    G = util.project_bivector(Gn, frame.base_bivec("n"), frame.surface_bivec())

    e1, e2, e3 = frame.base("e")
    g = frame.gnomon("e", zero_decl=True)
    p = sp.Symbol("p")
    s = frame.sunray()

    triangle_condition = ((g + p * s) ^ G) * (e1 ^ e2 ^ e3)

    p_sympy = sp.solve(sp.trigsimp(triangle_condition.obj), p)[0]
    p_result = result.shadow_triangle_solution()

    assert (sp.numer(p_sympy) - sp.numer(p_result)).equals(0)

    mu_s = sp.Symbol(r"\mu_s")
    Xi_s = sp.Symbol(r"\Xi_s")
    Xi = sp.Symbol(r"\Xi")

    S = result.shadow_bivector(s, g)
    M = frame.meridian_plane()
    sinXi_sin_mu, sinXi_cos_mu = result.hour_angle_sincos(S, M)

    denom_explicit = sp.simplify(
        sp.denom(p_result)
        .subs(mu_s, mu)
        .subs(Xi_s, Xi)
        .subs(sin(mu), sinXi_sin_mu / sin(Xi))
        .subs(cos(mu), sinXi_cos_mu / sin(Xi))
        .subs(iota, theta)
    )
    assert sp.simplify(sp.denom(p_sympy) - denom_explicit).equals(0)


def test_beta_pythagoras_identity():
    r"""
    Check that the expressions for $\sin(\beta)$ and $\cos(\beta)$ obey the Pythagorean identity
    """
    sin_beta, cos_beta = result.gnomon_shadow_angle_sincos()

    Gn = frame.dialface()
    Sn = result.shadow_bivector_explicit()
    cosPsi = result.dialface_shadowbivector_angle_cos(Gn, Sn)

    beta_check = sp.trigsimp(
        sp.expand(sp.numer(sin_beta) ** 2 + sp.numer(cos_beta) ** 2) - (1 - cosPsi**2)
    )
    assert beta_check.equals(0)


def test_sunray_shadow_projection():
    r"""
    Check that $\lambda + \cos(\Xi)$ is equal to the projection of $w$ onto $s$
    """

    gn = frame.gnomon("n", zero_decl=True)
    p = sp.Symbol("p")

    s = frame.sunray()
    sn = util.project_vector(s, frame.surface(), frame.base("n"))

    g = util.project_vector(
        gn, target_frame=frame.base("n"), render_frame=frame.surface()
    )
    cosXi = result.shadow_bivector_magnitude_angle_cos(s, g)

    wn = gn + p * sn
    ws_check = (sn | wn) - (p + cosXi)
    assert ws_check.obj.equals(0)


def test_gnomon_shadow_projection():
    r"""
    Check that $1 + \lambda\cos(\Xi)$ is equal to the projection of $w$ onto $g$
    """

    gn = frame.gnomon("n", zero_decl=True)
    p = sp.Symbol("p")

    s = frame.sunray()
    sn = util.project_vector(s, frame.surface(), frame.base("n"))

    g = util.project_vector(
        gn, target_frame=frame.base("n"), render_frame=frame.surface()
    )
    cosXi = result.shadow_bivector_magnitude_angle_cos(s, g)

    wn = gn + p * sn
    wg_check = (gn | wn) - (1 + p * cosXi)
    assert wg_check.obj.equals(0)


def test_gnomon_dialface_angle_pythagoras_identity():
    r"""
    [analemma.algebra.result.gnomon_dialface_angle_sin][] gives $\sin(A)$, the between the gnomon and the subgnomon (or,
    equivalently, the dial face). The subgnomon $b$ has length $\cos(A)$, and this test ensures that $b^2 = 1 -
    \sin^2(A)$.
    """
    sinA = result.gnomon_dialface_angle_sin()

    gn = frame.gnomon()

    Gn = frame.dialface()
    bn = util.update_coeffs(
        util.rotate(gn | Gn, -sp.pi / 2, Gn).trigsimp()
    )  # subgnomon on n-basis (surface frame)
    bm = util.project_vector(
        bn, frame.dial(), frame.base("m")
    )  # project onto m-basis (dial frame)

    assert sp.trigsimp(sp.expand_trig((bm | bm).obj) - (1 - sinA**2)).equals(0)
