#!/bin/bash

python scripts/dollar_dollar.py docs/nb/sundial.ipynb docs/nb/sundial.ipynb
find docs/nb -type 'f' -name '*.ipynb' -not -name "*-checkpoint.ipynb" | xargs jupyter nbconvert --to Markdown
mkdocs serve
