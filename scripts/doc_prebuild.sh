#!/bin/bash

for x in $(ls docs/nb/*.ipynb); do python scripts/dollar_dollar.py $x $x; done
find docs/nb -type 'f' -name '*.ipynb' -not -name "*-checkpoint.ipynb" | xargs jupyter nbconvert --to Markdown
