"""
Utilities for working with the algebra of sundials and analemma
"""

import sympy as sp
from galgebra import mv
from sympy import sin, cos
from typing import Tuple, Callable


def rotate(mv: mv.Mv, angle: float, bivec: mv.Mv) -> mv.Mv:
    r"""
    Rotate a multivector

    Forms the rotor $R$ that rotates `mv` by `angle` in the plane `bivec`. Denote:

    * `mv` by $M$
    * `angle` by $\theta$
    * `bivec` by $B$

    Then $R = \exp(-B\theta/2) = \cos(\theta/2) - B\sin(\theta/2)$

    and this the function returns $R\,M\,\tilde{R}$.

    Used extensively in [analemma.algebra.frame][].

    Parameters:
        mv: Multivector to be rotated
        angle: Angle of rotation
        bivec: Unit bivector defining the plane of rotation

    Returns:
        The multivector `mv` rotated by `angle` in plane `bivector`
    """
    rotor = (cos(angle / 2) - (bivec) * sin(angle / 2)).trigsimp()
    return rotor * mv * rotor.rev()


def project_vector(
    vec: mv.Mv,
    target_frame: Tuple[mv.Mv],
    render_frame: Tuple[mv.Mv],
    simp: Callable[[mv.Mv], mv.Mv] = lambda mv: sp.trigsimp(mv),
) -> mv.Mv:
    r"""
    Project a vector onto a basis in one Geometric Algebra (GA), and express that basis in terms of another GA

    Given a vector `vec` as $v$ and the `target_frame` $a_1, a_2, a_3$, form $v\cdot a_i$ for $i = 1,2,3$ and return the
    vector

    $\sum_i f(v\cdot a_i) \, b_i$

    where the frame $b_1, b_2, b_3$ is given by `render_frame` and $f()$ is the simplification function in `simp`. For
    example, it is often useful to work in terms of the [surface frame][analemma.algebra.frame.surface] $n_1, n_2, n_3$,
    but sometimes we need vectors in that space expressed (rendered) in terms of the fixed stars frame $e_1, e_2, e_3$.

    Parameters:
        vec: Vector to be projected
        target_frame: Frame on which to evaluate the components of `vec`
        render_frame: Frame to which the components will be multiplied
        simp: Simplification function $f()$

    Returns:
        Projected, simplified and rendered vector
    """
    coeffs = [simp((vec | b).obj) for b in target_frame]
    return sum([c * b for c, b in zip(coeffs, render_frame)])


def project_bivector(
    bivec: mv.Mv,
    target_frame: Tuple[mv.Mv],
    render_frame: Tuple[mv.Mv],
    simp: Callable[[mv.Mv], mv.Mv] = lambda mv: sp.trigsimp(sp.expand_trig(mv)),
) -> mv.Mv:
    r"""
    Project a bivector onto a basis in one Geometric Algebra (GA), and express that basis in terms of another GA

    Given a bivector `bivec` as $B$ and the `target_frame` $A_1, A_2, A_3$, form $-B\cdot A_i$ for $i = 1,2,3$ and return the
    bivector

    $\sum_i f(-B\cdot A_i) \, B_i$

    where the frame $B_1, B_2, B_3$ is given by `render_frame` and $f()$ is the simplification function in `simp`. For
    example, it is often useful to work in terms of the [surface frame][analemma.algebra.frame.surface_bivec]

    $n_1\wedge n_2,\, n_1\wedge n_3,\, n_2\wedge n_3$,

    but sometimes we need vectors in that space expressed (rendered) in terms of the fixed stars frame

    $e_1\wedge e_2,\, e_1\wedge e_3,\, e_2\wedge e_3$.

    Parameters:
        bivec: Bivector to be projected
        target_frame: Bivector frame on which to evaluate the components of `vec`
        render_frame: Bivector frame to which the components will be multiplied
        simp: Simplification function $f()$

    Returns:
        Projected, simplified and rendered bivector
    """
    coeffs = [simp((-bivec | b).obj) for b in target_frame]
    return sum([c * b for c, b in zip(coeffs, render_frame)])


def update_coeffs(vec: mv.Mv, simp: Callable[[mv.Mv], mv.Mv] = sp.trigsimp) -> mv.Mv:
    """
    Apply an algebraic manipulation to the components of a vector

    There is a behaviour, perhaps a bug, in [galgebra.ga.Ga](https://galgebra.readthedocs.io/en/latest/) whereby the
    internal sympy objects holding the coefficients of a multiector remain in the state they were before the latest
    manipulation, and out of sync with what is observed when the multivector is rendered. This can result in subsequent
    simplifications failing to achieve the desired goal.

    This function exists to address this behaviour by applying the manipulation (typically a simplification) in `simp`
    to the coefficients of `vec` individual and reforming the vector with the results.

    Parameters:
        vec: Vector to be updated
        simp: Algebraic manipulation to apply to the coefficients of `vec`

    Returns:
        The result of applying `simp` to each component of `vec` individually then recombining with the basis vectors in `vec`
    """
    return sum([simp(coeff) * vec for coeff, vec in zip(vec.get_coefs(1), vec.Ga.mv())])
