#!/bin/bash

git checkout docs/sundial.ipynb
rm -rf docs/*_files
rm -f docs/sundial.md docs/equation_of_time.md docs/sundial_plots.md docs/orbit_analysis.md
find . -name .DS_Store -delete
