#!/bin/bash

n=$(git status docs/nb --porcelain=1 | wc -l)
if [ $n -ne 0 ]
then 
    echo Error: Cannot build doc with uncommitted change to notebooks 1>&2
    exit 1
fi

./scripts/doc_prebuild.sh
mkdocs serve
