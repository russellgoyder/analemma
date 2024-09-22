"""
TODO
"""

import sympy as sp
from galgebra.printer import latex
from IPython.display import Math
from galgebra import mv
from typing import Tuple


def multivector(mv: mv.Mv) -> Math:
    r"""
    Render a GAlgebra multivector in a manner compatible with mkdocs

    Multivector mv on its own in a cell appears fine in the notebook but when nbconverted
    to Markdown comes out as \begin{equation*} x \end{equation*} instead of eg $\displaystyle x$
    and consequently isn't rendered in HTML after running mkdocs.
    """
    return Math(latex(mv))


def frame(frame: Tuple[mv.Mv], symbol: str):
    """
    TODO
    """
    return Math(rf"""
            \begin{{align}}
            {symbol}_1 &= {latex(frame[0])} \nonumber \\
            {symbol}_2 &= {latex(frame[1])} \nonumber \\
            {symbol}_3 &= {latex(frame[2])} \nonumber
            \end{{align}}
            """)


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


def vector(lhs: sp.Symbol, vec: Tuple[mv.Mv], symbol="e", func=lambda x: x) -> Math:
    """
    TODO
    """
    # would like to use .Fmt(3) but having rendering issues
    coeffs = vec.get_coefs(1)
    return Math(rf"""
            \begin{{align}}
            {lhs} & = {latex(func(coeffs[0]))} {symbol}_1 \nonumber \\
            & + {latex(func(coeffs[1]))} {symbol}_2 \nonumber \\
            & + {latex(func(coeffs[2]))} {symbol}_3 \nonumber
            \end{{align}}
            """)
