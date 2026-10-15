#!/bin/sh
# Compile the Verilog-A models the ngspice harness loads through OSDI into datasets/osdi/
# (git-ignored, like the other datasets), plus IHP SG13G2's PSP103 when datasets/pdk/ihp
# holds that PDK. Needs OpenVAF-reloaded and its source tree,
# whose integration_tests/ hold the reference .va models:
#   OPENVAF_DIR=/path/to/OpenVAF tests/build_osdi.sh
set -eu
: "${OPENVAF_DIR:?set OPENVAF_DIR to an OpenVAF(-reloaded) checkout}"
OPENVAF="${OPENVAF:-$OPENVAF_DIR/target/release/openvaf-r}"
out="$(cd "$(dirname "$0")/.." && pwd)/datasets/osdi"
mkdir -p "$out"
# bsim4.va (Cogenda's port) differs from ngspice's b4set.c in three defaults: wk2 and wgamma1 are
# never set to 0 (the "wk1 = 0" line is duplicated where wk2 belongs), so a card that gives k2 or
# gamma1 without its W term bins against the -12345789 "not given" marker (QA3 BUG-1: 3-5x Id on
# most sky130 bins); and TNOM defaults to 25 C where ngspice uses 27 (BUG-2). Patched copy:
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
perl -0pe 's/(if \(!BSIM4wk1Given\)\s*BSIM4wk1 = 0\.0;\s*)if \(!BSIM4wk1Given\)\s*BSIM4wk1 = 0\.0;/$1if (!BSIM4wk2Given) BSIM4wk2 = 0.0;\n            if (!BSIM4wgamma1Given) BSIM4wgamma1 = 0.0;/ or die "bsim4.va: wk2 default patch no longer applies\n";
           s/`define DEFAULT_TNOM 25\b/`define DEFAULT_TNOM 27/ or die "bsim4.va: TNOM patch no longer applies\n"' \
  "$OPENVAF_DIR/integration_tests/BSIM4/bsim4.va" > "$tmp/bsim4.va"
"$OPENVAF" "$tmp/bsim4.va" -o "$out/bsim4.osdi"
"$OPENVAF" "$OPENVAF_DIR/integration_tests/BSIMCMG/bsimcmg.va" -o "$out/bsimcmg.osdi"
# IHP SG13G2's own PSP103 (module psp103va), when its PDK files are in datasets/pdk/ihp
ihp="$out/../pdk/ihp/ihp-sg13g2/libs.tech/verilog-a/psp103/psp103.va"
[ -f "$ihp" ] && "$OPENVAF" "$ihp" -o "$out/psp103.osdi"
