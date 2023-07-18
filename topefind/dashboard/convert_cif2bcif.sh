#!/bin/bash

# This script requires:
# - node: https://nodejs.org/en
# - molstar: you can install it with npm install molstar

# Serving .bcif files brings for this use casa gives a x10 reduction of the total asset size.
# .mmtf would have been similarly performant with direct conversion from biotite, but mmtf is currently
# unsupported by pdbe-molstar.

for file in *.cif; do
    outputFile="${file%.cif}.bcif"
    node node_modules/molstar/lib/commonjs/servers/model/preprocess -i "$file" -ob "$outputFile"
done