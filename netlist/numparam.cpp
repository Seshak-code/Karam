#include "numparam.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace acutesim { namespace numparam {
namespace {

std::string upper(std::string s) {
    for (auto &c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}
bool identStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
bool identChar(char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '.'; }
std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
std::string num(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}

// ---------------------------------------------------------------- expressions
class Parser {
public:
    Parser(const std::string &s, const std::function<bool(const std::string &, double &)> &look)
        : s_(s), look_(look) {}
    double run() {
        const double v = ternary();
        ws();
        if (p_ != s_.size()) fail("unexpected '" + s_.substr(p_) + "'");
        if (!std::isfinite(v)) fail("the result is not a finite number");  // ln(0), sqrt(-1), 1e400 (ngspice errors)
        return v;
    }

private:
    const std::string &s_;
    const std::function<bool(const std::string &, double &)> &look_;
    size_t p_ = 0;

    [[noreturn]] void fail(const std::string &why) { throw std::runtime_error("expression '" + s_ + "': " + why); }
    void ws() { while (p_ < s_.size() && std::isspace((unsigned char)s_[p_])) ++p_; }
    bool eat(const char *op) {
        ws();
        const size_t n = std::char_traits<char>::length(op);
        if (s_.compare(p_, n, op) != 0) return false;
        p_ += n;
        return true;
    }
    double ternary() {
        const double c = orExpr();
        if (!eat("?")) return c;
        const double a = ternary();
        if (!eat(":")) fail("'?' without ':'");
        const double b = ternary();
        return c != 0.0 ? a : b;
    }
    double orExpr() {
        double v = andExpr();
        while (eat("||")) { const double r = andExpr(); v = (v != 0.0 || r != 0.0); }
        return v;
    }
    double andExpr() {
        double v = cmp();
        while (eat("&&")) { const double r = cmp(); v = (v != 0.0 && r != 0.0); }
        return v;
    }
    double cmp() {
        double v = add();
        for (;;) {
            if (eat("==")) v = v == add();
            else if (eat("!=") || eat("<>")) v = v != add();
            else if (eat("<=")) v = v <= add();
            else if (eat(">=")) v = v >= add();
            else if (eat("<")) v = v < add();
            else if (eat(">")) v = v > add();
            else return v;
        }
    }
    double add() {
        double v = mul();
        for (;;) {
            if (eat("+")) v += mul();
            else if (eat("-")) v -= mul();
            else return v;
        }
    }
    double mul() {
        double v = unary();
        for (;;) {
            ws();
            if (s_.compare(p_, 2, "**") == 0) return v;  // power, handled in unary's pow()
            if (eat("*")) v *= unary();
            else if (eat("/")) v /= nonzero(unary());
            else if (eat("%")) v = std::fmod(v, nonzero(unary()));
            else return v;
        }
    }
    double nonzero(double d) { if (d == 0.0) fail("division by zero"); return d; }
    double unary() {
        if (eat("-")) return -unary();
        if (eat("+")) return unary();
        if (eat("!")) return unary() == 0.0;
        double b = primary();  // ** and ^ are left-associative in ngspice (2**3**2 = 64),
        while (eat("**") || eat("^")) b = std::pow(b, signedPrimary());  // binding tighter than unary -
        return b;
    }
    double signedPrimary() {  // an exponent: 2**-1
        if (eat("-")) return -signedPrimary();
        if (eat("+")) return signedPrimary();
        return primary();
    }
    double number() {
        const char *start = s_.c_str() + p_;
        char *end = nullptr;
        double v = std::strtod(start, &end);
        p_ += (size_t)(end - start);
        std::string suf;  // SI scale factor, then unit letters (ignored)
        while (p_ < s_.size() && std::isalpha((unsigned char)s_[p_])) suf += (char)std::toupper((unsigned char)s_[p_++]);
        if (suf.rfind("MEG", 0) == 0) return v * 1e6;  // MIL is milli here, as ngspice's numparam
        if (suf.empty()) return v;
        switch (suf[0]) {
            case 'T': return v * 1e12;
            case 'G': return v * 1e9;
            case 'K': return v * 1e3;
            case 'M': return v * 1e-3;
            case 'U': return v * 1e-6;
            case 'N': return v * 1e-9;
            case 'P': return v * 1e-12;
            case 'F': return v * 1e-15;
            case 'A': return v * 1e-18;
            default: return v;  // unit letters only (e.g. 10V)
        }
    }
    double call(const std::string &f, const std::vector<double> &a) {
        auto need = [&](size_t n) { if (a.size() != n) fail(f + "() takes " + std::to_string(n) + " argument(s)"); };
        auto need2 = [&](size_t lo, size_t hi) { if (a.size() < lo || a.size() > hi) fail(f + "(): wrong argument count"); };
        if (f == "MIN" || f == "MAX") { need(2); return f == "MIN" ? std::min(a[0], a[1]) : std::max(a[0], a[1]); }
        // Statistical functions return their nominal value: no Monte Carlo yet.
        // ponytail: agauss/gauss/aunif/unif/limit are deterministic (nominal); a corner
        // that switches them on (e.g. sky130 mismatch with mc_mm_switch=1) is not random.
        if (f == "AGAUSS" || f == "GAUSS" || f == "AUNIF" || f == "UNIF" || f == "LIMIT") { need2(2, 3); return a[0]; }
        if (f == "POW" || f == "PWR") { need(2); return std::pow(f == "PWR" ? std::abs(a[0]) : a[0], a[1]); }
        if (f == "ATAN2") { need(2); return std::atan2(a[0], a[1]); }
        if (f == "HYPOT") { need(2); return std::hypot(a[0], a[1]); }
        need(1);
        const double x = a[0];
        if (f == "SQRT") return std::sqrt(x);
        if (f == "EXP") return std::exp(x);
        if (f == "LN" || f == "LOG") return std::log(x);
        if (f == "LOG10") return std::log10(x);
        if (f == "ABS") return std::abs(x);
        if (f == "FLOOR") return std::floor(x);
        if (f == "CEIL") return std::ceil(x);
        if (f == "INT") return std::trunc(x);
        if (f == "NINT") return std::nearbyint(x);  // half to even: nint(2.5) = 2
        if (f == "SQR") return x * x;
        if (f == "SGN" || f == "SIGN") return (x > 0) - (x < 0);
        if (f == "SIN") return std::sin(x);
        if (f == "COS") return std::cos(x);
        if (f == "TAN") return std::tan(x);
        if (f == "ASIN") return std::asin(x);
        if (f == "ACOS") return std::acos(x);
        if (f == "ATAN") return std::atan(x);
        if (f == "SINH") return std::sinh(x);
        if (f == "COSH") return std::cosh(x);
        if (f == "TANH") return std::tanh(x);
        if (f == "ASINH") return std::asinh(x);
        if (f == "ACOSH") return std::acosh(x);
        if (f == "ATANH") return std::atanh(x);
        fail("function " + f + "() is not supported yet");
    }
    double primary() {
        ws();
        if (p_ >= s_.size()) fail("unexpected end");
        if (eat("(")) {
            const double v = ternary();
            if (!eat(")")) fail("missing ')'");
            return v;
        }
        if (eat("{")) {  // {expr} nested inside a quoted expression
            const double v = ternary();
            if (!eat("}")) fail("missing '}'");
            return v;
        }
        const char c = s_[p_];
        if (std::isdigit((unsigned char)c) || c == '.') return number();
        if (!identStart(c)) fail(std::string("unexpected '") + c + "'");
        size_t b = p_;
        while (p_ < s_.size() && identChar(s_[p_])) ++p_;
        const std::string name = upper(s_.substr(b, p_ - b));
        if (eat("(")) {
            std::vector<double> args;
            if (!eat(")")) {
                do args.push_back(ternary()); while (eat(","));
                if (!eat(")")) fail("missing ')' after " + name + " arguments");
            }
            return call(name, args);
        }
        if (name == "PI") return M_PI;
        if (name == "TEMPER") return 27.0;  // circuit temperature in C: .temp is not supported, always 27
        double v;
        if (look_(name, v)) return v;
        fail("unknown parameter " + name);
    }
};

// ---------------------------------------------------------------- scopes
// Parameters by upper-case name: raw expressions, evaluated lazily (forward
// references work; a cycle is an error), then cached. Parent: the enclosing scope.
struct Scope {
    std::map<std::string, std::string> raw;
    std::map<std::string, int> line;  // where each raw parameter was defined
    std::map<std::string, double> val;
    std::set<std::string> busy;
    Scope *parent = nullptr;

    bool lookup(const std::string &n, double &v) {
        if (auto it = val.find(n); it != val.end()) { v = it->second; return true; }
        if (auto it = raw.find(n); it != raw.end()) {
            if (!busy.insert(n).second) throw std::runtime_error("parameter " + n + " depends on itself");
            v = evaluate(it->second);
            busy.erase(n);
            val[n] = v;
            return true;
        }
        return parent && parent->lookup(n, v);
    }
    bool has(const std::string &n) const {
        return val.count(n) || raw.count(n) || (parent && parent->has(n));
    }
    double evaluate(std::string e) {
        e = trim(e);
        if (e.size() >= 2 && ((e.front() == '{' && e.back() == '}') || (e.front() == '\'' && e.back() == '\'')))
            e = e.substr(1, e.size() - 2);
        return Parser(e, [this](const std::string &n, double &v) { return lookup(n, v); }).run();
    }
};

// name = value pairs of text (".param a=1 b='x*2' c = {y}", subckt/X parameters):
// a value runs to the next top-level "name =" (quotes, braces and parens may hold
// spaces). first: offset where the first pair starts (text.size() if none).
std::vector<std::pair<std::string, std::string>> pairs(const std::string &t, size_t *first = nullptr) {
    std::vector<size_t> starts;
    int depth = 0;
    bool q = false;
    for (size_t i = 0; i < t.size(); ++i) {
        const char c = t[i];
        if (c == '\'') q = !q;
        if (q) continue;
        if (c == '{' || c == '(') ++depth;
        if (c == '}' || c == ')') --depth;
        if (depth || !identStart(c) || (i && identChar(t[i - 1]))) continue;
        size_t j = i;
        while (j < t.size() && identChar(t[j])) ++j;
        size_t k = j;
        while (k < t.size() && t[k] == ' ') ++k;
        if (k < t.size() && t[k] == '=' && (k + 1 >= t.size() || t[k + 1] != '=')) starts.push_back(i);
        i = j - 1;
    }
    if (first) *first = starts.empty() ? t.size() : starts[0];
    std::vector<std::pair<std::string, std::string>> out;
    for (size_t s = 0; s < starts.size(); ++s) {
        const size_t eq = t.find('=', starts[s]);
        const size_t end = s + 1 < starts.size() ? starts[s + 1] : t.size();
        std::string v = trim(t.substr(eq + 1, end - eq - 1));
        // "r=3k PARAMS: q=2": a params: keyword between pairs ends up after the previous value
        if (v.size() > 7 && upper(v.substr(v.size() - 7)) == "PARAMS:" && std::isspace((unsigned char)v[v.size() - 8]))
            v = trim(v.substr(0, v.size() - 7));
        out.push_back({upper(trim(t.substr(starts[s], eq - starts[s]))), v});
    }
    return out;
}
// The words before the first "name =" pair, without a "params:" keyword.
std::vector<std::string> headWords(const std::string &t) {
    size_t first;
    pairs(t, &first);
    std::vector<std::string> w;
    std::istringstream in(t.substr(0, first));
    for (std::string x; in >> x;)
        if (upper(x) != "PARAMS:") w.push_back(x);
    return w;
}
std::vector<std::string> words(const std::string &t) {
    std::istringstream in(t);
    std::vector<std::string> w;
    for (std::string x; in >> x;) w.push_back(x);
    return w;
}

// {expr} and 'expr' -> number; name=<param in scope> -> name=number.
std::string substitute(const std::string &line, Scope &sc) {
    std::string out;
    for (size_t i = 0; i < line.size();) {
        const char c = line[i];
        if (c == '{' || c == '\'') {
            const char close = c == '{' ? '}' : '\'';
            int depth = 0;
            size_t j = i;
            for (; j < line.size(); ++j) {
                if (c == '{' && line[j] == '{') ++depth;
                if (line[j] == close && (c == '\'' ? j > i : --depth == 0)) break;
            }
            if (j >= line.size()) throw std::runtime_error(std::string("unbalanced ") + c + " in: " + line);
            out += num(sc.evaluate(line.substr(i, j - i + 1)));
            i = j + 1;
            continue;
        }
        if (c == '=' ) {
            out += c;
            size_t j = i + 1;
            while (j < line.size() && line[j] == ' ') out += line[j++];
            size_t k = j;
            if (k < line.size() && identStart(line[k])) {
                while (k < line.size() && identChar(line[k])) ++k;
                const std::string id = upper(line.substr(j, k - j));
                double v;
                if ((k == line.size() || line[k] == ' ' || line[k] == ')' || line[k] == ',') && sc.has(id) && sc.lookup(id, v)) {
                    out += num(v);
                    i = k;
                    continue;
                }
            }
            i = j;
            continue;
        }
        out += c;
        ++i;
    }
    return out;
}

using Lines = std::vector<std::pair<int, std::string>>;

struct Subckt {
    std::string name;
    std::vector<std::string> ports;
    std::vector<std::pair<std::string, std::string>> defaults;
    Lines body;
};

class Expander {
public:
    explicit Expander(int &errLine) : errLine_(errLine) {}

    Lines run(const Lines &in) {
        Lines top;
        for (size_t i = 0; i < in.size(); ++i) {  // collect .subckt bodies; the rest is the top level
            errLine_ = in[i].first;
            const auto w = words(in[i].second);
            if (in[i].first == 1 || w.empty() || upper(w[0]) != ".SUBCKT") { top.push_back(in[i]); continue; }
            const std::string hdr = in[i].second.substr(in[i].second.find(w[0]) + w[0].size());
            const auto hw = headWords(hdr);
            if (hw.empty()) throw std::runtime_error(".subckt needs a name");
            Subckt s{upper(hw[0]), {hw.begin() + 1, hw.end()}, pairs(hdr), {}};
            if (subckts_.count(s.name)) throw std::runtime_error("subcircuit " + s.name + " defined twice");
            for (++i; i < in.size(); ++i) {
                const auto bw = words(in[i].second);
                const std::string k = bw.empty() ? "" : upper(bw[0]);
                if (k == ".ENDS") break;
                if (k == ".SUBCKT") { errLine_ = in[i].first; throw std::runtime_error("nested .subckt definitions are not supported (missing .ends for " + s.name + "?)"); }
                s.body.push_back(in[i]);
            }
            if (i == in.size()) throw std::runtime_error(".subckt " + s.name + " has no .ends");
            subckts_[s.name] = std::move(s);
        }
        for (const auto &[n, l] : top) collectParams(l, global_, n);
        Lines out = walk(top, global_, "", nullptr);
        Lines params;  // the evaluated global parameters, for the parser's {NAME} / bare-name values
        for (const auto &[name, e] : global_.raw) {
            errLine_ = global_.line[name];
            double v;
            global_.lookup(name, v);
            params.push_back({0, ".param " + name + "=" + num(v)});
        }
        size_t end = out.size();  // definitions go before .end (the parser stops there)
        for (size_t i = 0; i < out.size(); ++i)
            if (const auto w = words(out[i].second); i && !w.empty() && upper(w[0]) == ".END") { end = i; break; }
        out.insert(out.begin() + (long)std::min<size_t>(1, out.size()), params.begin(), params.end());
        end += params.size();
        out.insert(out.begin() + (long)end, defs_.begin(), defs_.end());
        return out;
    }

private:
    int &errLine_;
    std::map<std::string, Subckt> subckts_;
    std::map<std::string, std::string> specialized_;  // key -> emitted subckt name
    Scope global_;
    Lines defs_;  // specialized .subckt definitions and their hoisted .model cards

    static bool isParam(const std::string &l) {
        const auto w = words(l);
        return !w.empty() && (upper(w[0]) == ".PARAM" || upper(w[0]) == ".PARAMS");
    }
    void collectParams(const std::string &l, Scope &sc, int n = 0) {
        if (!isParam(l)) return;
        const auto w = words(l);
        for (auto &[k, v] : pairs(l.substr(l.find(w[0]) + w[0].size()))) { sc.raw[k] = v; sc.line[k] = n; }
    }

    // Lines of one scope: .if resolved, .param consumed, X cards specialized,
    // expressions substituted. localModels: renames for hoisted local .model cards.
    Lines walk(const Lines &in, Scope &sc, const std::string &prefix, std::map<std::string, std::string> *localModels) {
        Lines out;
        std::vector<std::pair<bool, bool>> ifs;  // (this branch active, some branch taken)
        std::vector<bool> elses;                 // per open .if: its .else seen
        auto active = [&] { for (auto &f : ifs) if (!f.first) return false; return true; };
        for (const auto &[n, raw] : in) {
            errLine_ = n;
            if (n == 1 && prefix.empty()) { out.push_back({n, raw}); continue; }
            const auto w = words(raw);
            if (w.empty()) continue;
            const std::string k = upper(w[0]);
            const std::string rest = raw.substr(raw.find(w[0]) + w[0].size());
            if (k == ".IF") {
                const bool on = active() && sc.evaluate(rest) != 0.0;
                ifs.push_back({on, on});
                elses.push_back(false);
                continue;
            }
            if (k == ".ELSEIF" || k == ".ELSE") {
                if (ifs.empty()) throw std::runtime_error(k + " without .if");
                if (elses.back()) throw std::runtime_error(k + " after .else");
                elses.back() = k == ".ELSE";
                const auto f = ifs.back();  // a copy: pop_back ends the element's lifetime
                ifs.pop_back();
                const bool outer = active();
                const bool on = outer && !f.second && (k == ".ELSE" || sc.evaluate(rest) != 0.0);
                ifs.push_back({on, f.second || on});
                continue;
            }
            if (k == ".ENDIF") {
                if (ifs.empty()) throw std::runtime_error(".endif without .if");
                ifs.pop_back();
                elses.pop_back();
                continue;
            }
            if (!active()) continue;
            if (k == ".PARAM" || k == ".PARAMS") { if (!prefix.empty()) collectParams(raw, sc); continue; }  // top level: collected
            if (k == ".OSDI" || k == ".INCLUDE" || k == ".INC" || k == ".LIB" || k == ".TITLE") { out.push_back({n, raw}); continue; }
            std::string line = raw;
            if (localModels)  // names of this subcircuit's own models -> hoisted names
                for (const auto &[from, to] : *localModels) line = renameWord(line, from, to);
            line = substitute(line, sc);
            if (k == ".MODEL" && localModels) {
                defs_.push_back({n, line});
                continue;
            }
            if (k[0] == 'X') line = instance(line, sc, n);
            out.push_back({n, line});
        }
        if (!ifs.empty()) throw std::runtime_error(".if without .endif");
        return out;
    }

    static std::string renameWord(const std::string &line, const std::string &from, const std::string &to) {
        std::string out;
        const std::string up = upper(line);
        for (size_t i = 0; i < line.size();) {
            if (up.compare(i, from.size(), from) == 0 && (i == 0 || !identChar(line[i - 1])) &&
                (i + from.size() == line.size() || !identChar(line[i + from.size()]))) {
                out += to;
                i += from.size();
            } else out += line[i++];
        }
        return out;
    }

    // X<name> <nodes...> <subckt> [params:] [k=v ...] -> X<name> <nodes...> <specialized subckt>
    std::string instance(const std::string &line, Scope &caller, int n) {
        const auto hw = headWords(line);
        if (hw.size() < 2) throw std::runtime_error(hw.empty() ? "empty X card" : hw[0] + ": no subcircuit name");
        const std::string name = upper(hw.back());
        auto it = subckts_.find(name);
        if (it == subckts_.end()) return line;  // the parser reports the unknown subcircuit
        const Subckt &s = it->second;
        Scope sc;
        sc.parent = &global_;
        for (const auto &[k, v] : s.defaults) sc.raw[k] = v;
        // The body's own .param values are defaults too: an X card overrides them
        // (ngspice; sky130 declares l, w, ... that way). Overridden values win
        // because a scope reads val before raw.
        Scope body;
        for (const auto &[bn, bl] : s.body) collectParams(bl, body);
        std::string given;
        for (const auto &[k, v] : pairs(line)) {
            if (k == "M" && !sc.raw.count(k) && !body.raw.count(k))  // ngspice scales every device inside
                throw std::runtime_error(hw[0] + ": m= on a subcircuit instance is not supported yet");
            if (!sc.raw.count(k) && !body.raw.count(k)) throw std::runtime_error(hw[0] + ": " + s.name + " has no parameter " + k);
            sc.raw.erase(k);
            sc.val[k] = caller.evaluate(v);
            if (body.raw.count(k)) given += " " + k + "=" + num(sc.val[k]);
        }
        std::string key = s.name + given;
        for (const auto &[k, v] : s.defaults) { double x; sc.lookup(k, x); key += " " + k + "=" + num(x); }
        // Generated names use '|', which SPICE names never hold (sky130's hold "__"): the parser's
        // messages map them back to the user's names (SPICEParser::parse, fail). Only a key with no
        // parameters keeps the plain name: body-.param subckts (sky130) differ by `given` alone
        auto [spec, isNew] = specialized_.emplace(key, key == s.name ? s.name : s.name + "|" + std::to_string(specialized_.size() + 1));
        if (isNew) {
            for (const auto &[bn, bl] : s.body) collectParams(bl, sc);  // body .params (any order)
            std::map<std::string, std::string> models;
            for (const auto &[bn, bl] : s.body)
                if (const auto w = words(bl); w.size() > 1 && upper(w[0]) == ".MODEL") {
                    const std::string m = upper(w[1]);
                    models[m] = spec->second + "|" + m;
                    if (const size_t dot = m.rfind('.'); dot != std::string::npos)  // binned family NAME.N:
                        models[m.substr(0, dot)] = spec->second + "|" + m.substr(0, dot);  // devices name NAME
                }
            std::string hdr = ".subckt " + spec->second;
            for (const auto &p : s.ports) hdr += " " + p;
            Lines body = walk(s.body, sc, spec->second, &models);
            errLine_ = n;
            defs_.push_back({s.body.empty() ? n : s.body.front().first, hdr});
            defs_.insert(defs_.end(), body.begin(), body.end());
            defs_.push_back({s.body.empty() ? n : s.body.back().first, ".ends"});
        }
        std::string out;
        for (size_t i = 0; i + 1 < hw.size(); ++i) out += hw[i] + " ";
        return out + spec->second;
    }
};

}  // namespace

double eval(const std::string &expr, const std::function<bool(const std::string &, double &)> &lookup) {
    return Parser(expr, lookup).run();
}

std::vector<std::pair<int, std::string>> expand(const std::vector<std::pair<int, std::string>> &lines, int &errLine) {
    return Expander(errLine).run(lines);
}

}}  // namespace acutesim::numparam
