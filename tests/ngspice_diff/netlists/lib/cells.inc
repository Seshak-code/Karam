* Cells included from inside a .lib section (path relative to the library)
.subckt inv in out vdd
MP out in vdd vdd pm W=4u L=1u
MN out in 0 0 nm W=2u L=1u
.ends
