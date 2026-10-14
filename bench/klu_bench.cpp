// klu_bench — one solver on one Matrix Market matrix, as one "key=value ..." line.
//
//   klu_bench <matrix.mtx> <tree|klu> [reps]
//
// tree: TreeSPICE's contraction solver. plan = planContraction (ordering + tree,
//       once per pattern); fs = factor + solve without refinement (what every
//       Newton iteration pays); omega after the engine's default 2 refinement steps.
// klu:  SuiteSparse KLU (BTF + AMD, the circuit-simulation baseline). analyze +
//       factor once; fs = klu_refactor + klu_solve (KLU's per-iteration cost).
// b = A*1. omega = componentwise backward error max_i |b-Ax|_i / (|A||x|+|b|)_i.
// Timings are the minimum over `reps` repetitions. Run each solver in its own
// process (bench/klu_bench.py does) so peak RSS is per solver.
#include "math/contraction_solver.h"
#include <klu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

struct Triplets { int n = 0; std::vector<int> r, c; std::vector<double> v; };

Triplets readMatrixMarket(const char* path) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line) || line.rfind("%%MatrixMarket", 0) != 0) throw std::runtime_error("not Matrix Market");
    for (auto& ch : line) ch = (char)std::tolower(ch);
    if (line.find("coordinate") == std::string::npos || line.find("complex") != std::string::npos)
        throw std::runtime_error("need a real coordinate matrix");
    const bool pattern = line.find("pattern") != std::string::npos;
    const bool sym = line.find(" symmetric") != std::string::npos, skew = line.find("skew") != std::string::npos;
    while (std::getline(in, line) && (line.empty() || line[0] == '%')) {}
    int rows, cols; long nnz;
    std::istringstream(line) >> rows >> cols >> nnz;
    if (rows != cols) throw std::runtime_error("matrix is not square");
    Triplets t; t.n = rows;
    for (long k = 0; k < nnz; ++k) {
        int i, j; double v = 1.0;
        in >> i >> j;
        if (!pattern) in >> v;
        t.r.push_back(i - 1); t.c.push_back(j - 1); t.v.push_back(v);
        if ((sym || skew) && i != j) { t.r.push_back(j - 1); t.c.push_back(i - 1); t.v.push_back(skew ? -v : v); }
    }
    return t;
}

// Compressed rows (CSR) of t, or of its transpose (= CSC of t); duplicates summed.
void compress(const Triplets& t, bool transpose, std::vector<int>& ptr, std::vector<int>& idx, std::vector<double>& val) {
    const int n = t.n;
    std::vector<std::vector<std::pair<int, double>>> rows(n);
    for (size_t k = 0; k < t.v.size(); ++k) {
        int i = transpose ? t.c[k] : t.r[k], j = transpose ? t.r[k] : t.c[k];
        rows[i].push_back({j, t.v[k]});
    }
    ptr.assign(1, 0); idx.clear(); val.clear();
    for (auto& row : rows) {
        std::sort(row.begin(), row.end(), [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t k = 0; k < row.size(); ++k) {
            if (k > 0 && row[k].first == row[k - 1].first) { val.back() += row[k].second; continue; }
            idx.push_back(row[k].first); val.push_back(row[k].second);
        }
        ptr.push_back((int)idx.size());
    }
}

double omega(const Csr_matrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    double w = 0.0;
    for (int i = 0; i < A.rows; ++i) {
        double r = b[i], s = std::abs(b[i]);
        for (int k = A.row_pointer[i]; k < A.row_pointer[i + 1]; ++k) {
            double ax = A.values[k] * x[A.col_indices[k]];
            r -= ax; s += std::abs(ax);
        }
        if (!std::isfinite(r)) return INFINITY;
        if (s > 0.0) w = std::max(w, std::abs(r) / s);
    }
    return w;
}

template <class F> double timeMin(int reps, F&& f) {
    double best = INFINITY;
    for (int k = 0; k < reps; ++k) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

double peakRssMB() {
    rusage u{}; getrusage(RUSAGE_SELF, &u);
#ifdef __APPLE__
    return u.ru_maxrss / 1048576.0;  // bytes
#else
    return u.ru_maxrss / 1024.0;     // kilobytes
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <matrix.mtx> <tree|klu> [reps]\n", argv[0]); return 2; }
    const std::string solver = argv[2];
    const int reps = argc > 3 ? std::max(1, std::atoi(argv[3])) : 3;
    Triplets t;
    try { t = readMatrixMarket(argv[1]); } catch (const std::exception& e) { std::printf("error=\"%s\"\n", e.what()); return 1; }

    Csr_matrix A;
    compress(t, false, A.row_pointer, A.col_indices, A.values);
    A.rows = A.cols = t.n; A.nnz = (int)A.values.size();
    std::vector<double> ones(t.n, 1.0), b(t.n), x;
    matrixMultiplication(A, ones.data(), b.data());
    std::printf("n=%d nnz=%d solver=%s", t.n, A.nnz, solver.c_str());

    if (solver == "tree") {
        ContractionPlan plan;
        double tPlan = timeMin(1, [&] { plan = planContraction(A); });
        SolverResult res;
        double tFs = timeMin(reps, [&] { res = solveContraction(A, b, plan, 0); });
        res = solveContraction(A, b, plan, 2);
        std::printf(" ok=%d plan=%.6g fs=%.6g flops=%.6g peak_front=%.0f fronts=%zu omega=%.3g",
                    res.converged ? 1 : 0, tPlan, tFs, plan.flopEstimate, std::sqrt(plan.peakEntries),
                    plan.fronts.size(), res.converged ? omega(A, res.solution, b) : INFINITY);
    } else if (solver == "klu") {
        std::vector<int> Ap, Ai; std::vector<double> Ax;
        compress(t, true, Ap, Ai, Ax);  // CSC
        klu_common c; klu_defaults(&c);
        klu_symbolic* S = nullptr; klu_numeric* N = nullptr;
        double tAn = timeMin(1, [&] { S = klu_analyze(t.n, Ap.data(), Ai.data(), &c); });
        double tF = S ? timeMin(1, [&] { N = klu_factor(Ap.data(), Ai.data(), Ax.data(), S, &c); }) : 0.0;
        if (!S || !N) { std::printf(" ok=0 status=%d\n", c.status); return 1; }
        double tFs = timeMin(reps, [&] {
            klu_refactor(Ap.data(), Ai.data(), Ax.data(), S, N, &c);
            x = b; klu_solve(S, N, t.n, 1, x.data(), &c);
        });
        klu_flops(S, N, &c);
        std::printf(" ok=%d analyze=%.6g factor=%.6g fs=%.6g flops=%.6g lu_nnz=%d omega=%.3g",
                    c.status == KLU_OK ? 1 : 0, tAn, tF, tFs, c.flops, N->lnz + N->unz, omega(A, x, b));
        klu_free_numeric(&N, &c); klu_free_symbolic(&S, &c);
    } else { std::printf(" error=\"unknown solver\"\n"); return 2; }
    std::printf(" rss_mb=%.0f\n", peakRssMB());
    return 0;
}
