/* rad-calib-sim -- replay a --draft-calib-log and print what each drafting policy would have bought.
 *
 *   rad-calib-sim LOG [--cost a,b,c,..] [--draft-ms-round MS] [--static-T T1,T2,..] [--margin M]
 *                     [--kmax K] [--interval N] [--T t] [--autofit] [--grid-report]
 *
 * LOG: "R step seq T depth slot since acc alpha top1 ent adaedl tau0 n q[8] p[8]" per judged child of a
 * reached row, "S step ms rows acc" per verify step (core/sched/mtp_calib.h writes it). The cost table
 * is the verify step's ms for 0,1,2.. draft nodes (the --draft-cost-table of the adaptive rule;
 * default 12.3 ms at 3 nodes, +1 ms a node), --draft-ms-round the serial head's ms a round.
 *
 * Policies (a chain: the first child of each depth):
 *   chainK       a fixed chain of K nodes: tokens = 1 + min(accepted, K). EXACT for K <= the log's
 *                chain length (the same draws, truncated).
 *   adaptive     the EVICT-style rule of --draft-adaptive (abi/rad_mtp_adapt.h) deciding the chain
 *                length from a per-depth probability p_d, with p_d from
 *                  static      softmax(v / T_d), the --draft-calib list
 *                  online-T    the same with the per-depth temperature the online controller
 *                              has fitted by then (the controller learns causally from the log)
 *                  online-map  the controller's binned isotonic map of alpha against the static p
 *                  oracle      the row's exact label alpha (an upper bound)
 *                A depth the log never reached has no law to read: its p_d is the running mean of
 *                the policy's predictions at that depth (approximation, stated in the output).
 * Output: tokens/step, ms/step, tokens/ms, mean chain length and the calibration error of the
 * predictors per depth. */
#include "mtp_calib.h"
#include "rad_mtp_adapt.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace rad;

namespace {

struct Ev {
    int    depth = 1, slot = 0, since = 0, acc = 0, n = 0;
    double alpha = 0, tau0 = 1, q[8] = {}, p[8] = {};
    float  T = 1.0f;
};
struct StepRec {
    long long step = 0;
    double    ms = 0;
    int       rows = 0, acc = 0;
    std::vector<Ev> ev;   /* of the one sequence the step carried (batch 1) */
};

std::vector<float> parse_list(const char* s) {
    std::vector<float> v;
    while (s && *s) {
        char* e = nullptr;
        v.push_back((float)std::strtod(s, &e));
        if (e == s) break;
        s = *e == ',' ? e + 1 : e;
    }
    return v;
}

bool load(const char* path, std::vector<StepRec>* out) {
    std::ifstream f(path);
    if (!f) return false;
    std::map<long long, StepRec> by;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string tag;
        is >> tag;
        if (tag == "R") {
            long long step; unsigned long long seq; Ev e;
            double T;
            is >> step >> seq >> T >> e.depth >> e.slot >> e.since >> e.acc >> e.alpha;
            double top1, ent, ada;
            is >> top1 >> ent >> ada >> e.tau0 >> e.n;
            for (int k = 0; k < 8; ++k) is >> e.q[k];
            for (int k = 0; k < 8; ++k) is >> e.p[k];
            if (!is) continue;
            e.T = (float)T;
            StepRec& s = by[step];
            s.step = step;
            s.ev.push_back(e);
        } else if (tag == "S") {
            long long step;
            is >> step;
            StepRec& t = by[step];
            t.step = step;
            is >> t.ms >> t.rows >> t.acc;
        }
    }
    for (auto& kv : by) if (!kv.second.ev.empty()) out->push_back(kv.second);
    return true;
}

RadCalibFeat feat_of(const Ev& e) {
    RadMtpCoupleState s{};
    s.count = e.n;
    for (int k = 0; k < e.n; ++k) s.q[k] = e.q[k];
    RadCalibFeat f;
    rad_calib_features(s, &f);
    return f;
}

struct Acc {
    double tokens = 0, ms = 0, k = 0;
    long long steps = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: rad-calib-sim LOG [options] (see the file header)\n"); return 2; }
    std::string cost_s;
    double round_ms = 0.5, margin = 1.0;
    std::vector<float> static_T;
    int kmax = 0, interval = 256;
    bool autofit = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--cost") cost_s = val();
        else if (a == "--draft-ms-round") round_ms = std::atof(val());
        else if (a == "--static-T") static_T = parse_list(val());
        else if (a == "--margin") margin = std::atof(val());
        else if (a == "--kmax") kmax = std::atoi(val());
        else if (a == "--interval") interval = std::atoi(val());
        else if (a == "--autofit") autofit = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    std::vector<StepRec> steps;
    if (!load(argv[1], &steps) || steps.empty()) { std::fprintf(stderr, "cannot read a log with rows: %s\n", argv[1]); return 1; }
    RadMtpAdapt ad;
    rad_mtp_adapt_defaults(&ad);
    ad.on = 1;
    ad.margin = (float)margin;
    if (!cost_s.empty() && !rad_mtp_adapt_parse(("on=1;margin=" + std::to_string(margin) + ";cost=" + cost_s).c_str(), &ad)) {
        std::fprintf(stderr, "bad --cost\n");
        return 2;
    }
    int obs_max = 0;
    for (const StepRec& s : steps)
        for (const Ev& e : s.ev) obs_max = std::max(obs_max, e.depth);
    if (kmax <= 0) kmax = obs_max;
    std::printf("log %s: %zu steps with rows, chain length seen %d, cost C(k): ", argv[1], steps.size(), obs_max);
    for (int k = 0; k <= kmax + 1; ++k) std::printf("%.2f ", (double)rad_mtp_adapt_cost(&ad, k));
    std::printf("ms, head %.2f ms a round\n", round_ms);

    auto make_ctl = [&](MtpCalibCtl* c) {
        MtpCalibCfg cc;
        cc.mode = 1;
        cc.autofit = autofit;
        cc.ece = true;
        cc.interval = interval;
        cc.static_T = static_T;
        c->configure(cc);
    };
    MtpCalibCtl ctl;
    make_ctl(&ctl);

    enum { FIX3, FIX4, FIXK, ADS, ADT, ADM, ORA, NP };
    const char* names[NP] = { "chain3", "chain4", "chainK(max)", "adaptive static", "adaptive online-T",
                              "adaptive online-map", "adaptive oracle" };
    Acc acc[NP];
    double meanp[NP][16] = {}, meann[NP][16] = {};
    const int D = std::min(kmax, 15);
    auto cost_of = [&](int k, bool all_rounds) {
        return (double)rad_mtp_adapt_cost(&ad, k) + round_ms * (all_rounds ? kmax : k);
    };
    for (const StepRec& st : steps) {
        /* the chain: slot-0 events by depth */
        const Ev* ch[16] = {};
        for (const Ev& e : st.ev)
            if (e.slot == 0 && e.depth >= 1 && e.depth <= 15) ch[e.depth] = &e;
        int m = 0;
        while (m + 1 <= 15 && ch[m + 1]) ++m;
        int accepted = 0;
        while (accepted + 1 <= m && ch[accepted + 1]->acc) ++accepted;
        auto add = [&](int p, int k, bool all_rounds) {
            Acc& a = acc[p];
            a.tokens += 1 + std::min(accepted, k);
            a.ms += cost_of(k, all_rounds);
            a.k += k;
            ++a.steps;
        };
        for (int p : { FIX3, FIX4, FIXK }) {
            const int k = p == FIX3 ? 3 : p == FIX4 ? 4 : kmax;
            if (k > kmax) continue;
            add(p, k, false);
        }
        const float T0 = st.ev.empty() ? 1.0f : st.ev[0].T;
        for (int p : { ADS, ADT, ADM, ORA }) {
            double E = 1.0, cum = 1.0;
            int k = 0;
            for (int d = 1; d <= D; ++d) {
                double pd;
                if (ch[d]) {
                    const RadCalibFeat f = feat_of(*ch[d]);
                    pd = p == ADS ? ctl.predict_static(d, 0, f)
                       : p == ADT ? ctl.predict_live(T0, d, 0, f)
                       : p == ADM ? ctl.predict_map(T0, d, 0, f) : ch[d]->alpha;
                    meanp[p][d] = meanp[p][d] * 0.99 + pd;
                    meann[p][d] = meann[p][d] * 0.99 + 1;
                } else {
                    pd = meann[p][d] > 0 ? meanp[p][d] / meann[p][d] : 0.5;
                }
                const double pc = cum * pd;
                if (!rad_mtp_adapt_admit(&ad, k, (float)E, (float)pc)) break;
                cum = pc; E += pc; ++k;
            }
            add(p, k, true);
        }
        /* the controller learns from the step afterwards (causal) */
        for (const Ev& e : st.ev) {
            const RadCalibFeat f = feat_of(e);
            ctl.observe_event(e.T, (float)(1.0 / e.tau0), e.depth, e.slot, e.alpha, f);
            if (e.slot == 0) ctl.observe_overlap(e.T, (float)(1.0 / e.tau0), e.depth, e.q, e.p, e.n, e.tau0);
        }
        ctl.step_boundary();
    }
    std::printf("%-22s %10s %10s %12s %8s\n", "policy", "tok/step", "ms/step", "tok/ms", "mean k");
    double best = 0;
    for (int p = 0; p < NP; ++p) if (acc[p].steps) best = std::max(best, acc[p].tokens / acc[p].ms);
    for (int p = 0; p < NP; ++p) {
        if (!acc[p].steps) continue;
        const Acc& a = acc[p];
        std::printf("%-22s %10.4f %10.3f %12.5f %8.2f   (%+.2f%% vs chain3)\n", names[p], a.tokens / a.steps,
                    a.ms / a.steps, a.tokens / a.ms, a.k / a.steps,
                    acc[FIX3].steps ? 100.0 * ((a.tokens / a.ms) / (acc[FIX3].tokens / acc[FIX3].ms) - 1.0) : 0.0);
    }
    std::printf("\ncalibration of the first child per depth class (Brier / ECE) -- static | live T | learned map:\n");
    for (int c = 0; c < ctl.classes(); ++c)
        for (int d = 0; d < kCalDepths; ++d)
            std::printf("  T%.2f depth%s%d: brier %.4f | %.4f | %.4f   ece %.4f | %.4f | %.4f   live T %.2f tau %.2f\n",
                        (double)ctl.class_T(c), d == kCalDepths - 1 ? ">=" : "=", d + 1, ctl.brier_static(c, d),
                        ctl.brier_live(c, d), ctl.brier_map(c, d), ctl.ece_static(c, d), ctl.ece_live(c, d),
                        ctl.ece_map(c, d), (double)ctl.live_cal(c, d), (double)ctl.live_tau(c, d));
    return 0;
}
