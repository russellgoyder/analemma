
# Sundials, Orbits and Analemmas

The `analemma` package performs sundial calculations, allowing for very general geometry that covers all common types of dial. Highlights include

 * exact parametric expressions for the analemma on any type of sundial
 * orbits and the equation of time for any planet
 * a plotting module to draw the analemma

## Install

```bash
pip install analemma
```

## Usage

```python
import matplotlib.pyplot as plt
from analemma import orbit, plot as ap

earth = orbit.PlanetParameters.earth()
vertical_dial = ap.DialParameters.vertical(location='Cambridge, UK')

fig, ax = plt.subplots()
ax.axis("equal")

ap.plot_analemma(ax, earth, vertical_dial)
```

See [Analemma Plots](sundial_plots.md) for complete examples.

## Background

The analemma is the path traced by the shadow on a sundial (or the sun in the sky) when observed at the same time each day throughout one year. In this package, the analemma is expressed in an exact parametric expression whose derivation was first presented in 2006. See the [project homepage](https://russellgoyder.github.io/sundial-latex/) for more information.

The full derivation is reproduced here using [SymPy](https://www.sympy.org/en/index.html) and [GAlgebra](https://github.com/pygae/galgebra) in two Jupyter notebooks.

 * [Sundial Calculations](sundial.md) works in terms of several angles describing the sundial and its planet, two of which vary with time. $\sigma_t$ measures the progress of the planet around its orbit and $\psi_t$ measures the rotation of the planet on its axis.
 * [The Equation of Time](equation_of_time.md) relates $\sigma_t$ and $\psi_t$ to time by calculating the planet's orbit.

The results of these derivations are implemented in two modules in the `analemma` package.  

 * [analemma.geometry][] implements the results from [Sundial Calculations](sundial.md)
 * [analemma.orbit][] implements the results from [The Equation of Time](equation_of_time.md)

Finally, [analemma.plot][] provides functionality for plotting the analemma which is demonstrated in [Analemma Plots](sundial_plots.md).

|  |  |
| ------: | --------- |
| Project Homepage | https://russellgoyder.github.io/sundial-latex/ |
| Documentation | https://analemma.readthedocs.io/en/stable/ |
| Repository | https://github.com/russellgoyder/sundial |
| Issue Tracker | https://github.com/russellgoyder/sundial/issues |