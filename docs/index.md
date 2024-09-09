
# Sundials, Orbits and Analemmas

The `analemma` package performs sundial calculations, allowing for very general geometry that covers all common types of dial. Highlights include

 * exact parametric expressions for the analemma on any type of sundial
 * orbits and the equation of time for any planet
 * a plotting module to draw the analemma

The analemma is the path traced by the shadow on a sundial (or the sun in the sky) when observed at the same time each day throughout one year. The exact formulae for the analemma are derived from first principles in [The Sundial Problem from a New Angle](https://russellgoyder.github.io/sundial-latex/) and reproduced using [SymPy](https://www.sympy.org/en/index.html) and [GAlgebra](https://github.com/pygae/galgebra) in two Jupyter notebooks.

 * [Sundial Calculations](sundial.md) works in terms of several angles describing the sundial and its planet, two of which vary with time. $\sigma_t$ measures the progress of the planet around its orbit and $\psi_t$ measures the rotation of the planet on its axis.
 * [The Equation of Time](equation_of_time.md) relates $\sigma_t$ and $\psi_t$ to time by calculating the planet's orbit.

The results of these derivations are implemented in two modules in the `analemma` package.  

 * [analemma.geometry][] implements the results from [Sundial Calculations](sundial.md)
 * [analemma.orbit][] implements the results from [The Equation of Time](equation_of_time.md)

Finally, [analemma.plot][] provides functionality for plotting the analemma which is demonstrated in [Analemma Plots](sundial_plots.md).

