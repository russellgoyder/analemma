"""
TODO
"""

import sympy as sp
from galgebra.printer import latex
from IPython.display import Math
from galgebra import mv
from typing import Tuple, List


def multivector(mv: mv.Mv) -> Math:
    r"""
    Render a GAlgebra multivector in a manner compatible with mkdocs

    Multivector mv on its own in a cell appears fine in the notebook but when nbconverted
    to Markdown comes out as \begin{equation*} x \end{equation*} instead of eg $\displaystyle x$
    and consequently isn't rendered in HTML after running mkdocs.
    """
    return Math(latex(mv))


def _extract_base_symbol(mv: mv.Mv) -> str:
    # small hack. From https://github.com/pygae/galgebra/blob/master/galgebra/metric.py#L732
    # it looks like we lose the character passed into galgebra.ga.Ga to denote the basis vectors
    # but can extract it as follows if frame._space_algebra is used to form the GA
    return str(mv.Ga.r_symbols[0]).split("_")[0]


def _frame_lines(symbol: str, frame: Tuple[mv.Mv]) -> List[str]:
    lines = []
    for n, vec in enumerate(frame):
        lines.append(f"{symbol}_{n} &= {latex(vec)}")
    return lines


def frame(symbol: str, frame: Tuple[mv.Mv]):
    """
    TODO
    """

    beginning = r"\begin{align} "

    lines = _frame_lines(symbol, frame)
    middle = r"\nonumber \\ ".join(lines)

    end = r"\nonumber \end{align}"

    return Math(beginning + middle + end)


def expression(lhs: str, rhs: sp.Symbol) -> Math:
    """
    TODO
    """
    # would like to use mv.Fmt but it seems to give LaTeX output which doesn't
    # render after eg jupyter nbconvert --to Markdown
    return Math(rf"""
        \begin{{equation}}
            {lhs} = {latex(rhs)} \nonumber
        \end{{equation}}
        """)


def expressions(lhss: Tuple[str], rhss: Tuple[sp.Symbol]) -> Math:
    """
    TODO
    """
    equations = []
    for lhs, rhs in zip(lhss, rhss):
        equations.append(
            rf"""
        \begin{{equation}}
            {lhs} = {latex(rhs)} \nonumber
        \end{{equation}}
        """
        )
    return Math(r"\\".join(equations))


def _vec_lines(vec: mv.Mv, func=lambda x: x) -> List[str]:
    symbol = _extract_base_symbol(vec)
    coeffs = vec.get_coefs(1)
    lines = []
    for n, coeff in enumerate(coeffs):
        lines.append(rf"\left({latex(func(coeff))}\right) {symbol}_{n+1}")
    return lines


def _bivec_lines(bivec: mv.Mv, func=lambda x: x) -> List[str]:
    symbol = _extract_base_symbol(bivec)
    coeffs = bivec.get_coefs(2)
    return [
        rf"{latex(func(coeffs[0]))} {symbol}_1 \wedge {symbol}_2",
        rf"{latex(func(coeffs[1]))} {symbol}_1 \wedge {symbol}_3",
        rf"{latex(func(coeffs[2]))} {symbol}_2 \wedge {symbol}_3",
    ]


def _lines(mv: mv.Mv, func=lambda x: x) -> List[str]:
    if len(mv.grades) != 1:
        raise Exception(
            f"Discovered a non-blade multivector when preparing to render, with grades {mv.grades}"
        )
    grade = mv.grades[0]
    if grade == 1:
        return _vec_lines(mv, func)
    elif grade == 2:
        return _bivec_lines(mv, func)
    else:
        raise Exception(
            "Unsupported line rendering for pure-blade multivector of grade {grade}"
        )


def _beginning_string(lhs: sp.Symbol) -> str:
    return rf"\begin{{align}} {lhs} & = "


_middle_string: str = r" \nonumber \\ &"
_end_string: str = r"& \nonumber \end{align}"


def align(lhs: sp.Symbol, mv: mv.Mv, func=lambda x: x) -> Math:
    """
    TODO
    """
    return Math(
        _beginning_string(lhs) + _middle_string.join(_lines(mv, func)) + _end_string
    )


def simplification(
    lhs: str, stages: Tuple[mv.Mv], as_lines: Tuple[bool] = None
) -> Math:
    """
    TODO
    """

    if as_lines is None:
        as_lines = [False] * len(stages)

    stage_middles = []
    for do_lines, stage in zip(as_lines, stages):
        if do_lines:
            lines = _lines(stage)
            stage_middles.append(_middle_string.join(lines))
        else:
            stage_middles.append(latex(stage))

    middle = (_middle_string + " = ").join(stage_middles)

    return Math(_beginning_string(lhs) + middle + _end_string)
