#!/bin/sh
# Compile the Verilog-A models the ngspice harness loads through OSDI into datasets/osdi/
# (git-ignored, like the other datasets). Needs OpenVAF-reloaded and its source tree,
# whose integration_tests/ hold the reference .va models:
#   OPENVAF_DIR=/path/to/OpenVAF tests/build_osdi.sh
set -eu
: "${OPENVAF_DIR:?set OPENVAF_DIR to an OpenVAF(-reloaded) checkout}"
OPENVAF="${OPENVAF:-$OPENVAF_DIR/target/release/openvaf-r}"
out="$(cd "$(dirname "$0")/.." && pwd)/datasets/osdi"
mkdir -p "$out"
for m in BSIM4/bsim4 BSIMCMG/bsimcmg; do
  "$OPENVAF" "$OPENVAF_DIR/integration_tests/$m.va" -o "$out/$(basename $m).osdi"
done
