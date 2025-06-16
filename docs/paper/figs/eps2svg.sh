
# note not DialFrame.svg because I saved it as svg after fixing a typo using Inkscape
for x in SurfaceFrame Gnomon ShadowTriangle EquationOfTime HorizontalDialZoom1 ArbitraryDial ArcticDial
do
    inkscape --export-filename=$x.svg $x.eps
done
