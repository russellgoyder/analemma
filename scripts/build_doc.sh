#!/bin/bash

n=$(git status docs/nb --porcelain=1 | wc -l)
if [ $n -ne 0 ]
then 
    echo Error: Cannot build doc with uncommitted change to notebooks 1>&2
    exit 1
fi

for x in $(ls docs/nb/*.ipynb); do python scripts/dollar_dollar.py $x $x; done

find docs/nb -type 'f' -name '*.ipynb' -not -name "*-checkpoint.ipynb" | xargs jupyter nbconvert --to Markdown
mkdocs serve
