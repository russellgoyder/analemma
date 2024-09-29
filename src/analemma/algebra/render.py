"""
Functionality for pretty-printing multivectors in Jupyter notebooks
"""

import sympy as sp
from galgebra.printer import latex
from IPython.display import Math
from galgebra import mv
from typing import Tuple, List, Callable


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
        lines.append(f"{symbol}_{n+1} &= {latex(vec)}")
    return lines


def frame(symbol: str, frame: Tuple[mv.Mv]) -> Math:
    r"""
    Render a vector frame

    For example,

    $f_1 = \cos(\alpha)\cos(\psi) e_1 + \sin(\psi) e_2 + \sin(\alpha)\cos(\psi) e_3$

    $f_2 = -\cos(\alpha)\sin(\psi) e_1 + \cos(\psi) e_2 - \sin(\alpha)\sin(\psi) e_3$

    $f_3 = -\sin(\alpha) e_1 + \cos(\alpha) e_3$

    Parameters:
        symbol: Symbol to use for each vector in the frame
        frame: Tuple of vectors to be rendered

    Returns:
        IPython.display.Math object holding the rendered LaTeX output
    """

    beginning = r"\begin{align} "

    lines = _frame_lines(symbol, frame)
    middle = r"\nonumber \\ ".join(lines)

    end = r"\nonumber \end{align}"

    return Math(beginning + middle + end)


def expression(lhs: str, rhs: sp.Symbol) -> Math:
    r"""
    Render an expression

    For example,

    $f_1\wedge f_2 = \cos(\alpha) \, e_1 \wedge e_2 - \sin(\alpha) \, e_2 \wedge 3_3$

    Parameters:
        lhs: String to put on the left hand side of the rendered equality
        rhs: Expression to be placed on the right hand side of the rendered equality

    Returns:
        IPython.display.Math object holding the rendered LaTeX output
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
    Render multiple expressions

    Parameters:
        lhss: Tuple of strings to put on the left hand side of each rendered equality
        rhss: Tuple of expressions to be placed on the right hand side of each rendered equality

    Returns:
        IPython.display.Math object holding the rendered LaTeX output
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


def align(
    lhs: sp.Symbol, mv: mv.Mv, func: Callable[[mv.Mv], mv.Mv] = lambda x: x
) -> Math:
    """
    Render a vector or a bivector with each component on a separate line

    Parameters:
        lhs: String to put on the left hand side of the rendered equality
        mv: Multivector whose components are to be rendered on the right hand side, one per line
        func: Manipulation to be performed on each coefficient of the given vector or bivector

    Returns:
        IPython.display.Math object holding the rendered LaTeX output
    """
    return Math(
        _beginning_string(lhs) + _middle_string.join(_lines(mv, func)) + _end_string
    )


def simplification(
    lhs: str, stages: Tuple[mv.Mv], as_lines: Tuple[bool] = None
) -> Math:
    """
    Render the simplification of a vector or a bivector

    Parameters:
        lhs: String to put on the left hand side of the rendered equality
        stages: Tuple of equivalent multivectors in different forms
        as_lines: Flag to indicate whether to render each stage in multiline format

    Returns:
        IPython.display.Math object holding the rendered LaTeX output
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
