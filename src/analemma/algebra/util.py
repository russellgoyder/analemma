"""
TODO
"""

from galgebra import mv
from sympy import sin, cos


def rotate(mv: mv.Mv, angle: float, bivec: mv.Mv) -> mv.Mv:
    """
    TODO
    """
    rotor = (cos(angle / 2) - (bivec) * sin(angle / 2)).trigsimp()
    return rotor * mv * rotor.rev()
