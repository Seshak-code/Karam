  std::string auditPhysicalResiduals(const TensorNetlist &netlist, const std::vector<double> &v_candidate) {
    std::string report;
    auto getV = [&](int nodeIdx) {
      if (nodeIdx <= 0 || nodeIdx > (int)v_candidate.size()) return 0.0;
      return v_candidate[nodeIdx - 1];
    };

    int n = netlist.numGlobalNodes;
    std::vector<double> node_residuals(n, 0.0);
    auto addRes = [&](int nodeIdx, double val) {
      if (nodeIdx > 0 && nodeIdx <= n) node_residuals[nodeIdx - 1] += val;
    };

    // AUDIT 1: Voltage Sources
    for (const auto &v : netlist.globalBlock.voltageSources) {
      double v1 = getV(v.nodePositive);
      double v2 = getV(v.nodeNegative);
      double r_int = 1e-3; 
      double i_src = (v.voltage_V - (v1 - v2)) / r_int;
      addRes(v.nodePositive, i_src);
      addRes(v.nodeNegative, -i_src);
      if (std::abs(i_src) > 100.0) {
        report += "WARN: VSource (Nodes " + std::to_string(v.nodePositive) + "," + std::to_string(v.nodeNegative) + ") I=" + std::to_string(i_src) + " A\n";
      }
    }

    // AUDIT 2: Current Sources
    for (const auto &cs : netlist.globalBlock.currentSources) {
      addRes(cs.nodePositive, -cs.current_A);
      addRes(cs.nodeNegative, cs.current_A);
    }

    // AUDIT 3: Resistors
    for (const auto &r : netlist.globalBlock.resistors) {
       double v1 = getV(r.nodeTerminal1);
       double v2 = getV(r.nodeTerminal2);
       double i_r = (v1 - v2) / r.resistance_ohms;
       addRes(r.nodeTerminal1, -i_r);
       addRes(r.nodeTerminal2, i_r);
    }

    // AUDIT 4: Diodes
    for (const auto &d : netlist.globalBlock.diodes) {
       double v_a = getV(d.anode);
       double v_c = getV(d.cathode);
       DiodeParams<double> params = {d.saturationCurrent_I_S_A, d.emissionCoefficient_N, d.thermalVoltage_V_T_V};
       double i_d = diode_current_clamped(v_a - v_c, params, 30.0);
       addRes(d.anode, -i_d);
       addRes(d.cathode, i_d);
    }

    // AUDIT 5: BJTs
    for (const auto &q : netlist.globalBlock.bjts) {
        double v_c = getV(q.nodeCollector);
        double v_b = getV(q.base);
        double v_e = getV(q.emitter);
        
        BJTParams<double> params;
        params.Is = q.saturationCurrent_I_S_A;
        params.BetaF = q.betaF;
        params.BetaR = q.betaR;
        params.Vt = q.thermalVoltage_V_T_V;
        
        BJTCurrents currents = bjt_ebers_moll(v_c, v_b, v_e, q.isNPN, params);
        
        addRes(q.nodeCollector, currents.Ic);
        addRes(q.base, currents.Ib);
        addRes(q.emitter, currents.Ie);

        if (std::abs(currents.Ic) > 10.0) {
            char buf[256];
            snprintf(buf, sizeof(buf), "WARN: BJT %s Ic=%e A\n      Params: Is=%e Vt=%e\n      Voltages: Vc=%e Vb=%e Ve=%e\n", 
                     q.instanceName.c_str(), currents.Ic, params.Is, params.Vt, v_c, v_b, v_e);
            report += std::string(buf);
        }
        if (std::abs(currents.Ib) > 10.0) report += "WARN: BJT " + q.instanceName + " Ib=" + std::to_string(currents.Ib) + " A\n";
        if (std::abs(currents.Ie) > 10.0) report += "WARN: BJT " + q.instanceName + " Ie=" + std::to_string(currents.Ie) + " A\n";
    }

    // AUDIT 6: KCL Check
    for (int i=0; i<n; ++i) {
        if (std::abs(node_residuals[i]) > 10.0) {
            report += "FAIL: Node " + std::to_string(i+1) + " KCL Resid=" + std::to_string(node_residuals[i]) + " A. V=" + std::to_string(getV(i+1)) + "\n";
        }
    }

    // Capture Partial Voltages
    if (report.length() > 0) {
        report += "Voltages: ";
        for(size_t i=0; i<v_candidate.size(); ++i) {
            if (i >= 10) { report += "..."; break; } 
            report += std::to_string(v_candidate[i]) + ", ";
        }
        report += "\n";
    }

    return report;
  }
