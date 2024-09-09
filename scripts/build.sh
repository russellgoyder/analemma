#!/bin/bash

python scripts/dollar_dollar.py docs/sundial.ipynb docs/sundial.ipynb
find docs -type 'f' -name '*.ipynb' -not -name "*-checkpoint.ipynb" | xargs jupyter nbconvert --to Markdown
mkdocs serve
