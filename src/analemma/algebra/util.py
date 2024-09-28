"""
TODO
"""

import sympy as sp
from galgebra import mv
from sympy import sin, cos
from typing import Tuple


def rotate(mv: mv.Mv, angle: float, bivec: mv.Mv) -> mv.Mv:
    """
    TODO
    """
    rotor = (cos(angle / 2) - (bivec) * sin(angle / 2)).trigsimp()
    return rotor * mv * rotor.rev()


def project_vector(
    vec: mv.Mv,
    target_frame: Tuple[mv.Mv],
    render_frame: Tuple[mv.Mv],
    simp=lambda mv: sp.trigsimp(mv),
) -> mv.Mv:
    """
    TODO needs UX work/clarity?
    """
    coeffs = [simp((vec | b).obj) for b in target_frame]
    return sum([c * b for c, b in zip(coeffs, render_frame)])


def project_bivector(
    bivec: mv.Mv,
    target_frame: Tuple[mv.Mv],
    render_frame: Tuple[mv.Mv],
    simp=lambda mv: sp.trigsimp(sp.expand_trig(mv)),
) -> mv.Mv:
    """
    TODO
    """
    coeffs = [simp((-bivec | b).obj) for b in target_frame]
    return sum([c * b for c, b in zip(coeffs, render_frame)])


def update_coeffs(vec: mv.Mv, simp=sp.trigsimp) -> mv.Mv:
    """
    TODO generalize beyond vec
    """
    return sum([simp(coeff) * vec for coeff, vec in zip(vec.get_coefs(1), vec.Ga.mv())])
