#!/bin/bash

magick analemma_logo.svg \
      \( -clone 0 -resize 32x32 \) \
      \( -clone 0 -resize 48x48 \) \
      \( -clone 0 -resize 96x96 \) \
      \( -clone 0 -resize 192x192 \) \
      -delete 0 -alpha on -colors 256 favicon.ico
