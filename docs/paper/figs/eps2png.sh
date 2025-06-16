
for x in MainArena SurfaceFrame DialFrame Gnomon ShadowTriangle EquationOfTime HorizontalDialZoom1 ArbitraryDial ArcticDial
do
    inkscape -b FFFFFF --export-filename=$x.png $x.eps
done
