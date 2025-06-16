
[![Python package CICD](https://github.com/russellgoyder/analemma/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/russellgoyder/analemma/actions/workflows/publish-to-pypi.yml) [![Documentation Status](https://readthedocs.org/projects/analemma/badge/?version=stable)](https://analemma.readthedocs.io/en/stable/?badge=stable)

<style>
  .center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  }
  .quote-text {
      text-align: center;
      margin: 0 auto;
      font-size: 1.5em;
  }
  .source-text{
    text-align: right;
    margin-right: 10%;
    font-size: 0.9em;
  }
</style>

<img src="res/analemma_logo_dark_mode_red.svg" height="200" class="center"/>

<div class="quote-text">
  <p><em>They deciphered my analemma!</em></p>
</div>
<div class="source-text">
  <p>- Fraa Orolo in <em>Anathem</em> by Neal Stephenson (2008)</p>
</div>
<br>

The analemma is the beautiful double-loop path traced by the shadow on a sundial (or the sun in the sky) when observed at the same time each day throughout one year. This package calculates and plots the analemma for all types of sundial anywhere on earth, or any planet.

```bash
pip install analemma
```

|  |  |
| ------: | --------- |
| PyPI | [https://pypi.org/project/analemma/](https://pypi.org/project/analemma/) |
| Documentation | [https://analemma.readthedocs.io/en/stable/](https://analemma.readthedocs.io/en/stable/) |
| Source | [https://github.com/russellgoyder/analemma](https://github.com/russellgoyder/analemma) |
| Issue Tracker | [https://github.com/russellgoyder/analemma/issues](https://github.com/russellgoyder/analemma/issues) |
| Discussions | [https://github.com/russellgoyder/analemma/discussions](https://github.com/russellgoyder/analemma/discussions) |
| Changelog | [https://analemma.readthedocs.io/en/stable/changelog](https://analemma.readthedocs.io/en/stable/changelog) |

<p float="left">
  <img src="docs/paper/figs/MainArena.png" height="250"/>
  <img src="docs/paper/figs/ArbitraryDial.png" height="250" /> 
</p>

## Resources

The `analemma` package implements the mathematics of sundial and orbits for an arbitrary sundial geometry and orientation placed anywhere on any planet. The analemma expressed in an exact parametric expression which reduces to standard results for the special cases of common dial types. Despite sundials being thousands of years old, I cannot find an expression for the analemma quite as general as this in the literature.

### Reference Article

The analemma is derived from first principles in [The sundial problem from a new angle - Russell Goyder.pdf](<docs/paper/The sundial problem from a new angle - Russell Goyder.pdf>) - the latest version of a paper [originally published](https://iopscience.iop.org/article/10.1088/0143-0807/27/2/023) in the [European Journal of Physics](https://iopscience.iop.org/journal/0143-0807) in 2006. The LaTeX source and figures for the paper are in the [analemma repository](https://github.com/russellgoyder/analemma/tree/main/docs/paper).

### Jupyter Notebooks

You can view and run Jupyter notebooks which reproduce all derivations and calculations in the article using symbolic algebra.

 * [Sundial geometry and analemma derivation](https://analemma.readthedocs.io/en/stable/#the-mathematics-of-sundials-and-the-analemma)
 * [Example plots of the analemma](https://analemma.readthedocs.io/en/stable/nb/sundial_plots/)
 * [The Equation of Time](https://analemma.readthedocs.io/en/stable/nb/equation_of_time/)
 * [Orbits](https://analemma.readthedocs.io/en/stable/nb/orbit_analysis/)


I used geometric algebra for this work and have a [cheat sheet](https://russellgoyder.ca/geometric-algebra-cheat-sheet/) which could you useful if you familiar with the topic.

### Presentations

I presented this work in [part 1](https://youtu.be/gO77fMRwn5E) and [part 2](https://youtu.be/w6-A_uJp15M) of a [metauni seminar](https://metauni.org/anythingatall/).

### Building the LaTeX source

To build, run `latexmk -pdf` in `docs/paper`. To clean up, `latexmk -C` and if you want, `find figs -name 'figure?.eps-converted-to.pdf' | xargs rm`. Note - fixed a typo in figure 3 (DialFrame.fig) using inkscape which couldn't save as fig, so I saved it as DialFrame.svg.

---

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

&copy; [Russell Goyder](https://russellgoyder.ca)
