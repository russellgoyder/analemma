#!/bin/bash

git checkout docs/nb/*.ipynb # undo scripts/dollar_dollar.py
rm -rf docs/nb/*_files
rm -f docs/nb/*.md
find . -name .DS_Store -delete
