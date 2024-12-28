
# Changelog

## v0.3.1

Fixed image links in API reference.

## v0.3

Improved test coverage.

Added plots for common dial types.

Enabled Binder for notebooks.

## v0.2

Integated [Skyfield](https://rhodesmill.org/skyfield/) for season events and apsides.

## v0.1

Generalized `hour_offset` to float in order to plot analemma at any time of day.

Made better use of type annotations to improve reference documentation.

Moved detailed orbit analysis into [separate notebook](nb/orbit_analysis.md) to make [Equation of Time](nb/equation_of_time.md) more prominent.

Refactored long notebook presenting the analemma derivation into a series of more consumable notebooks each loading algebraic results from the a new subpackage.

Made the test suite part of the package to improve transparency.