# tensors/ — SoA layout of the linear elements

`physics_tensors.h` defines `TensorizedBlock`: structure-of-arrays tensors for resistors, capacitors,
inductors and independent V/I sources, plus `mos1Params()`, `diodeModel()` and `bjtModel()`, which turn a
parser-resolved `.model` card into the parameters the device physics evaluates.

`physics_tensors.cpp` implements `tensorizeNetlist()`: one pass over the global block and every subcircuit
instance (in that order; the transient source refresh in `stampAllElements` walks the same order) that
flattens the linear elements into those tensors with global node numbers.

Nonlinear devices (MOSFET, diode, BJT, OSDI) have no SoA tensors: `stampAllElements` stamps them by walking
the netlist (`forEachMosfet`, `forEachJunction`, `forEachOsdi`), one device definition per kind for the
stamp, the KCL audit, history, AC and noise. GPU device evaluation is Phase 4.
