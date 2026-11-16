/* mtp_calib.h -- online, lossless acceptance calibration of the DARTree controller and the
 * automatic per-depth draft temperature (docs/DESIGN-online-calib.md).
 *
 * Under --draft-coupling the acceptance of a reached row is exactly alpha = sum min(P, Q) with Q
 * the law the draft was drawn from (abi/rad_mtp_calib.h). The verify exports P on Q's support, the
 * host derives alpha per judged child, and this class turns the labels into
 *
 *   (a) a calibration map from the draft-side features of the row to alpha: per (request class,
 *       depth class, slot) binned EMA statistics with isotonic (PAV) smoothing and a prior equal to
 *       the static prediction (--draft-calib's softmax(v / T_d) slot probability, the identity on
 *       that feature). The device grower can only apply a per-depth temperature, so the live
 *       output of the map is the per-depth calibration TEMPERATURE that minimises the Brier error
 *       of the slot probabilities against the labels (Brier is exact here: the labels carry no
 *       0/1 noise); the full map feeds the ECE report and the replay tool;
 *   (b) a per-depth draft temperature tau_d: running sums of the overlap sum min(P, Q_tau) over a
 *       small grid, every grid point scored on the SAME logged rows (Q_tau is rebuilt from the
 *       stored Q, no exploration cost), the live tau_d switching to the argmax with hysteresis.
 *
 * LOSSLESSNESS. Both outputs are words in the next draws (RadMtpCoupleDraw::dpad_). They choose
 * which Q a step draws from and the calibrated slot priorities of its best-first rule, and read
 * past statistics only: the state stores the Q that was sampled and the verify reads it, so what
 * the controller learns, however it moves between steps, cannot reach the output law
 * (tests/mtp_rrs_test.cpp, "a draft law that changes every step").
 *
 * Host only, no device access: everything here runs on rows the engine already read back.
 */
#pragma once

#include "rad_mtp_calib.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

constexpr int kCalDepths = RAD_MTP_DRAW_DEPTHS;   /* depth classes 1, 2, 3+ */
constexpr int kCalSlots  = 3;                     /* slots 0, 1, 2+ */
constexpr int kCalBins   = 16;
constexpr int kCalCodes  = 15;                    /* calibration temperature codes 1..15 */

inline int cal_depth_class(int depth) { return std::min(std::max(depth, 1), kCalDepths) - 1; }
inline int cal_slot_class(int slot) { return std::min(std::max(slot, 0), kCalSlots - 1); }

/* The probability of the j-th largest entry of a law under calibration temperature T:
 * softmax(log q / T) = q^(1/T) renormalised (the grower's recal of its slot priorities). `slot` is
 * q sorted descending. */
inline double cal_slot_prob(const double* slot, int n, int j, double T, double lift = 0.0) {
    if (j >= n) return 0.0;
    if (lift > 0.0) return lift + (1.0 - lift) * cal_slot_prob(slot, n, j, T, 0.0);
    if (T == 1.0) return slot[j];
    const double e = 1.0 / T, top = std::log(slot[0]);
    double sum = 0, pj = 0;
    for (int k = 0; k < n; ++k) {
        const double w = slot[k] > 0 ? std::exp(e * (std::log(slot[k]) - top)) : 0.0;
        sum += w;
        if (k == j) pj = w;
    }
    return sum > 0 ? pj / sum : 0.0;
}

/* Binned EMA statistics of alpha against a feature x in [0, 1], isotonic by pool-adjacent-violators
 * at query time. The prior is the identity (alpha = x, "the static map is right") worth `prior_n`
 * pseudo-observations a bin, so a bin with no data predicts the static value. */
struct CalMap {
    double n[kCalBins] = {}, sa[kCalBins] = {}, sx[kCalBins] = {};
    static int bin_of(double x) {
        const int b = (int)(x * kCalBins);
        return b < 0 ? 0 : b >= kCalBins ? kCalBins - 1 : b;
    }
    void decay(double d) { for (int b = 0; b < kCalBins; ++b) { n[b] *= d; sa[b] *= d; sx[b] *= d; } }
    void clear(double keep) { decay(keep); }
    void observe(double x, double alpha, double decay_ = 1.0) {
        (void)decay_;
        const int b = bin_of(x);
        n[b] += 1; sa[b] += alpha; sx[b] += x;
    }
    /* The isotonic fit of all bins; est[b] non-decreasing in b. */
    void fit(double prior_n, double* est) const {
        double v[kCalBins], w[kCalBins];
        int blk[kCalBins], nb = 0;
        for (int b = 0; b < kCalBins; ++b) {
            const double mid = (b + 0.5) / kCalBins;
            const double xm = n[b] > 0 ? sx[b] / n[b] : mid;
            v[b] = (sa[b] + prior_n * xm) / (n[b] + prior_n);
            w[b] = n[b] + prior_n;
        }
        double bv[kCalBins], bw[kCalBins];
        for (int b = 0; b < kCalBins; ++b) {
            bv[nb] = v[b]; bw[nb] = w[b]; blk[nb] = 1; ++nb;
            while (nb > 1 && bv[nb - 2] > bv[nb - 1]) {
                const double W = bw[nb - 2] + bw[nb - 1];
                bv[nb - 2] = (bv[nb - 2] * bw[nb - 2] + bv[nb - 1] * bw[nb - 1]) / W;
                bw[nb - 2] = W;
                blk[nb - 2] += blk[nb - 1];
                --nb;
            }
        }
        int b = 0;
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < blk[i]; ++j) est[b++] = bv[i];
    }
    double predict(double x, double prior_n) const {
        double est[kCalBins];
        fit(prior_n, est);
        return est[bin_of(x)];
    }
};

/* Calibration error of a predictor against labels: bins of the prediction. */
struct CalEce {
    double n[kCalBins] = {}, sp[kCalBins] = {}, sa[kCalBins] = {};
    double brier = 0, total = 0;
    void add(double pred, double alpha) {
        const int b = CalMap::bin_of(pred);
        n[b] += 1; sp[b] += pred; sa[b] += alpha;
        brier += (pred - alpha) * (pred - alpha);
        total += 1;
    }
    double ece() const {
        if (!(total > 0)) return 0.0;
        double e = 0;
        for (int b = 0; b < kCalBins; ++b)
            if (n[b] > 0) e += std::fabs(sp[b] - sa[b]);
        return e / total;
    }
    double mean_brier() const { return total > 0 ? brier / total : 0.0; }
};

struct MtpCalibCfg {
    int    mode = 0;               /* 0 static, 1 online, 2 off (--draft-calib) */
    bool   autofit = false;        /* --draft-coupling-temp-autofit */
    bool   ece = false;            /* --draft-calib-ece */
    std::string log_path;          /* --draft-calib-log */
    int    interval = 256;         /* verify steps between refits */
    std::vector<float> static_T;   /* --draft-calib's list (prior) */
    std::vector<float> tau_grid = { 0.05f, 0.6f, 0.75f, 0.9f, 1.0f, 1.15f };
    double decay = 0.9995;         /* per observation */
    double hyst_tau = 0.005;       /* relative overlap gain a switch needs */
    double hyst_cal = 0.01;        /* relative Brier gain a switch needs */
    double min_n = 60;             /* effective observations before a depth may move */
    double prior_n = 4;            /* map pseudo-observations a bin */
    int    max_classes = 8;
};

class MtpCalibCtl {
public:
    ~MtpCalibCtl() { close_log(); }

    /* False when nothing is asked for: the engine then stages no word at all. */
    bool configure(const MtpCalibCfg& c) {
        cfg_ = c;
        enabled_ = c.mode == 1 || c.autofit || c.ece || !c.log_path.empty();
        cls_.clear();
        step_ = 0;
        close_log();
        if (!c.log_path.empty()) {
            log_ = std::fopen(c.log_path.c_str(), "w");
            if (!log_) return false;
            static char buf[1 << 20];
            std::setvbuf(log_, buf, _IOFBF, sizeof buf);
            std::fprintf(log_, "# rad calib log v1: R step seq T depth slot since acc alpha top1 ent adaedl tau0 "
                               "n q[8] p[8] | S step ms rows acc\n");
        }
        return true;
    }
    bool enabled() const { return enabled_; }
    bool online() const { return cfg_.mode == 1; }
    bool autofit() const { return cfg_.autofit; }
    const MtpCalibCfg& cfg() const { return cfg_; }

    /* The word for a draw of a request at temperature `T` whose static draft scale is `dscale`
     * (0 = 1). 0 when disabled. */
    uint32_t word(float T, float dscale) {
        if (!enabled_) return 0;
        Cls& c = cls(T, dscale);
        uint32_t w = RAD_MTP_DRAW_EXPORT_BIT;
        for (int d = 0; d < kCalDepths; ++d) {
            unsigned b = 0;
            if (cfg_.autofit && c.tau_set[d]) b |= c.tau_live[d];
            if (cfg_.mode == 1 && c.cal_live[d]) b |= (unsigned)c.cal_live[d] << 4;
            w |= b << (8 * d);
        }
        return w;
    }

    /* One judged row. `qs`: the state the engine staged (Q, draft order, nkids); `pq`: P's mass on
     * qs.ids (the export); `judged`: how many of the row's children the verify evaluated (the
     * accepted one and the rejected ones before it; all of them when none was accepted), `accepted`
     * the index of the accepted child or -1. `since`: tokens since the request's last rejection. */
    void observe_row(float T, uint64_t seq, const RadMtpCoupleState& qs, const double* pq, int judged,
                     int accepted, int since) {
        if (!enabled_ || qs.count < 1) return;
        double alpha[8];
        const int m = rad_calib_labels(qs, pq, alpha);
        if (judged > m) judged = m;
        RadCalibFeat f;
        rad_calib_features(qs, &f);
        const int depth = qs.round > 0 ? qs.round : 1;
        const int dc = cal_depth_class(depth);
        const double tau0 = rad_calib_tau_of(qs.draw, depth);
        Cls& c = cls(T, qs.draw.dscale);
        for (int t = 0; t < judged; ++t) {
            const bool acc = t == accepted;
            observe_child(c, dc, t, alpha[t], f);
            if (log_)
                log_row(seq, T, depth, t, since, acc, alpha[t], f, tau0, qs, pq);
        }
        if (judged >= 1 && cfg_.autofit) observe_tau(c, dc, qs, pq, tau0);
        ++rows_;
        row_alpha_sum_ += alpha[0];
    }

    /* The same updates from a label already in hand (the replay tool reads them from a log). */
    void observe_event(float T, float dscale, int depth, int slot, double alpha, const RadCalibFeat& f) {
        if (!enabled_) return;
        observe_child(cls(T, dscale), cal_depth_class(depth), slot, alpha, f);
        ++rows_;
    }
    void observe_overlap(float T, float dscale, int depth, const double* q, const double* pq, int n,
                         double tau0) {
        if (!enabled_ || !cfg_.autofit) return;
        RadMtpCoupleState qs{};
        qs.count = n;
        for (int k = 0; k < n; ++k) qs.q[k] = q[k];
        observe_tau(cls(T, dscale), cal_depth_class(depth), qs, pq, tau0);
    }
    /* What the three predictors say about slot `slot` of a row with features `f` at `depth`: the
     * static map, the live temperature, the learned binned map. */
    double predict_static(int depth, int slot, const RadCalibFeat& f) const {
        return cal_slot_prob(f.slot, f.n, slot, static_of(depth));
    }
    double predict_live(float T, int depth, int slot, const RadCalibFeat& f) const {
        const int dc = cal_depth_class(depth);
        const Cls* c = find(T);
        if (c && c->cal_live[dc])
            return cal_slot_prob(f.slot, f.n, slot, rad_mtp_cal_of_code(c->cal_live[dc]),
                                 rad_mtp_cal_lift_of_code(c->cal_live[dc]));
        return cal_slot_prob(f.slot, f.n, slot, static_of(depth));
    }
    double predict_map(float T, int depth, int slot, const RadCalibFeat& f) const {
        const Cls* c = find(T);
        const double ps = predict_static(depth, slot, f);
        return c ? c->map[cal_depth_class(depth)][cal_slot_class(slot)].predict(ps, cfg_.prior_n) : ps;
    }
    /* Observe, then refit every `interval` observed steps: the replay tool's step boundary. */
    void step_boundary() { ++step_; if (cfg_.interval > 0 && step_ % cfg_.interval == 0) refit(); }

    /* After a request's verify: `accepted` of `n_spec` drafts; keeps its since-rejection run. */
    int since_of(uint64_t seq) const {
        const auto it = run_.find(seq);
        return it == run_.end() ? 0 : it->second;
    }
    void request_step(uint64_t seq, int accepted, int n_spec) {
        if (run_.size() > 4096) run_.clear();
        int& r = run_[seq];
        r = accepted < n_spec ? 0 : r + accepted;
    }

    /* One verify step done: refit every `interval` steps; the step's record for the log. */
    void end_step(double ms, int rows, int accepted) {
        ++step_;
        if (log_) std::fprintf(log_, "S %lld %.4f %d %d\n", (long long)step_, ms, rows, accepted);
        if (cfg_.interval > 0 && step_ % cfg_.interval == 0) refit();
    }
    int64_t steps() const { return step_; }

    void refit() {
        for (Cls& c : cls_)
            for (int d = 0; d < kCalDepths; ++d) {
                if (cfg_.mode == 1) refit_cal(c, d);
                if (cfg_.autofit) refit_tau(c, d);
            }
    }
    void flush() { if (log_) std::fflush(log_); }
    void close_log() { if (log_) { std::fclose(log_); log_ = nullptr; } }

    /* ---- introspection for the stats line and the tests ---- */
    struct ClsView { int key; double T; };
    int classes() const { return (int)cls_.size(); }
    float live_tau(int cls_i, int depth_class) const {
        const Cls& c = cls_[(size_t)cls_i];
        return c.tau_set[depth_class] ? rad_mtp_tau_of_code(c.tau_live[depth_class]) : c.base_tau;
    }
    float live_cal(int cls_i, int depth_class) const {
        const Cls& c = cls_[(size_t)cls_i];
        return c.cal_live[depth_class] ? rad_mtp_cal_of_code(c.cal_live[depth_class])
                                       : static_of(depth_class + 1);
    }
    float live_lift(int cls_i, int depth_class) const {
        const Cls& c = cls_[(size_t)cls_i];
        return rad_mtp_cal_lift_of_code(c.cal_live[depth_class]);
    }
    float class_T(int cls_i) const { return cls_[(size_t)cls_i].T; }
    double ece_static(int cls_i, int d) const { return cls_[(size_t)cls_i].e_static[d].ece(); }
    double ece_live(int cls_i, int d) const { return cls_[(size_t)cls_i].e_live[d].ece(); }
    double ece_map(int cls_i, int d) const { return cls_[(size_t)cls_i].e_map[d].ece(); }
    double brier_static(int cls_i, int d) const { return cls_[(size_t)cls_i].e_static[d].mean_brier(); }
    double brier_live(int cls_i, int d) const { return cls_[(size_t)cls_i].e_live[d].mean_brier(); }
    double brier_map(int cls_i, int d) const { return cls_[(size_t)cls_i].e_map[d].mean_brier(); }
    double map_predict(int cls_i, int d, int slot, double x) const {
        return cls_[(size_t)cls_i].map[d][cal_slot_class(slot)].predict(x, cfg_.prior_n);
    }
    uint64_t rows() const { return rows_; }

    /* The periodic stats suffix: "calib T0.7[d1 tau .75 cal 1.2 ece s.045/l.021] ..." */
    std::string stats() const {
        std::string o;
        char b[160];
        for (size_t i = 0; i < cls_.size(); ++i) {
            const Cls& c = cls_[i];
            std::snprintf(b, sizeof b, " T%.2f:", c.T);
            o += b;
            for (int d = 0; d < kCalDepths; ++d) {
                std::snprintf(b, sizeof b, " d%d%s tau %.2f cal %.2f lift %.2f", d + 1, d == kCalDepths - 1 ? "+" : "",
                              (double)live_tau((int)i, d), (double)live_cal((int)i, d), (double)live_lift((int)i, d));
                o += b;
                if (cfg_.ece) {
                    std::snprintf(b, sizeof b, " ece %.3f/%.3f/%.3f", ece_static((int)i, d),
                                  ece_live((int)i, d), ece_map((int)i, d));
                    o += b;
                }
                o += ";";
            }
        }
        return o;
    }

private:
    struct Cls {
        int   key = 0;
        float T = 1.0f, base_tau = 1.0f;
        /* tau fit */
        std::vector<unsigned> grid;                 /* codes */
        std::vector<double>   ov[kCalDepths];       /* running overlap per grid point */
        double                ovn[kCalDepths] = {};
        unsigned char         tau_live[kCalDepths] = {};
        bool                  tau_set[kCalDepths] = {};
        /* calibration fit: index 0 the static temperature, 1..15 the code table */
        double                br[kCalDepths][kCalCodes + 1] = {};
        double                brn[kCalDepths] = {};
        unsigned char         cal_live[kCalDepths] = {};
        CalMap                map[kCalDepths][kCalSlots];
        CalEce                e_static[kCalDepths], e_live[kCalDepths], e_map[kCalDepths];
    };

    Cls& cls(float T, float dscale) {
        const int key = (int)std::lround(std::min(std::max((double)T, 0.0), 4.0) * 20.0);
        for (Cls& c : cls_) if (c.key == key) return c;
        if ((int)cls_.size() >= cfg_.max_classes) return cls_.back();
        cls_.emplace_back();
        Cls& c = cls_.back();
        c.key = key;
        c.T = T;
        c.base_tau = dscale > 0.0f ? 1.0f / dscale : 1.0f;
        for (float g : cfg_.tau_grid) c.grid.push_back(rad_mtp_tau_code_of(g));
        c.grid.push_back(rad_mtp_tau_code_of(c.base_tau));
        std::sort(c.grid.begin(), c.grid.end());
        c.grid.erase(std::unique(c.grid.begin(), c.grid.end()), c.grid.end());
        for (int d = 0; d < kCalDepths; ++d) c.ov[d].assign(c.grid.size(), 0.0);
        return c;
    }

    const Cls* find(float T) const {
        const int key = (int)std::lround(std::min(std::max((double)T, 0.0), 4.0) * 20.0);
        for (const Cls& c : cls_) if (c.key == key) return &c;
        return nullptr;
    }

    float static_of(int depth) const {
        if (cfg_.mode == 2 || cfg_.static_T.empty()) return 1.0f;
        const size_t i = std::min<size_t>((size_t)std::max(depth, 1) - 1, cfg_.static_T.size() - 1);
        return cfg_.static_T[i] > 0.05f ? cfg_.static_T[i] : 1.0f;
    }

    void observe_child(Cls& c, int dc, int t, double alpha, const RadCalibFeat& f) {
        const double Ts = static_of(dc + 1);
        const double ps = cal_slot_prob(f.slot, f.n, t, Ts);
        const double pl = c.cal_live[dc]
            ? cal_slot_prob(f.slot, f.n, t, rad_mtp_cal_of_code(c.cal_live[dc]), rad_mtp_cal_lift_of_code(c.cal_live[dc]))
            : ps;
        CalMap& m = c.map[dc][cal_slot_class(t)];
        c.e_static[dc].add(ps, alpha);
        c.e_live[dc].add(pl, alpha);
        c.e_map[dc].add(m.predict(ps, cfg_.prior_n), alpha);   /* predicted BEFORE it learns this row */
        m.observe(ps, alpha);
        if (cfg_.mode != 1) return;
        const double dk = cfg_.decay;
        for (int k = 0; k <= kCalCodes; ++k) {
            const double pk = k == 0 ? ps : cal_slot_prob(f.slot, f.n, t, rad_mtp_cal_of_code((unsigned)k),
                                                         rad_mtp_cal_lift_of_code((unsigned)k));
            const double e = pk - alpha;
            c.br[dc][k] = c.br[dc][k] * dk + e * e;
        }
        c.brn[dc] = c.brn[dc] * dk + 1.0;
        for (int s = 0; s < kCalSlots; ++s) c.map[dc][s].decay(dk);
    }

    void observe_tau(Cls& c, int dc, const RadMtpCoupleState& qs, const double* pq, double tau0) {
        double qa[8], qg[8];
        const int n = std::min(qs.count, 8);
        for (int k = 0; k < n; ++k) qa[k] = qs.q[k];
        const double dk = cfg_.decay;
        for (size_t g = 0; g < c.grid.size(); ++g) {
            rad_calib_q_at(qa, n, tau0, (double)rad_mtp_tau_of_code(c.grid[g]), qg);
            c.ov[dc][g] = c.ov[dc][g] * dk + rad_calib_overlap(qg, pq, n);
        }
        c.ovn[dc] = c.ovn[dc] * dk + 1.0;
    }

    void refit_cal(Cls& c, int d) {
        if (c.brn[d] < cfg_.min_n) return;
        const int cur = c.cal_live[d];   /* 0 = static */
        int best = cur;
        for (int k = 0; k <= kCalCodes; ++k)
            if (c.br[d][k] < c.br[d][best]) best = k;
        if (best != cur && c.br[d][best] < c.br[d][cur] * (1.0 - cfg_.hyst_cal)) {
            c.cal_live[d] = (unsigned char)best;
            ++switches_;
        }
    }

    void refit_tau(Cls& c, int d) {
        if (c.ovn[d] < cfg_.min_n) return;
        const unsigned cur_code = c.tau_set[d] ? c.tau_live[d] : rad_mtp_tau_code_of(c.base_tau);
        size_t cur = 0, best = 0;
        for (size_t g = 0; g < c.grid.size(); ++g) {
            if (c.grid[g] == cur_code) cur = g;
            if (c.ov[d][g] > c.ov[d][best]) best = g;
        }
        if (best != cur && c.ov[d][best] > c.ov[d][cur] * (1.0 + cfg_.hyst_tau)) {
            c.tau_live[d] = (unsigned char)c.grid[best];
            c.tau_set[d] = true;
            ++switches_;
            /* the calibration statistics of this depth were taken against another Q */
            for (int k = 0; k <= kCalCodes; ++k) c.br[d][k] *= 0.2;
            c.brn[d] *= 0.2;
            for (int s = 0; s < kCalSlots; ++s) c.map[d][s].clear(0.2);
        }
    }

    void log_row(uint64_t seq, float T, int depth, int slot, int since, bool acc, double alpha, const RadCalibFeat& f,
                 double tau0, const RadMtpCoupleState& qs, const double* pq) {
        std::fprintf(log_, "R %lld %llu %.2f %d %d %d %d %.6f %.5f %.5f %.5f %.3f %d", (long long)step_ + 1,
                     (unsigned long long)seq, (double)T,
                     depth, slot, since, acc ? 1 : 0, alpha, f.top1, f.ent, f.adaedl, tau0, qs.count);
        for (int k = 0; k < 8; ++k) std::fprintf(log_, " %.6g", k < qs.count ? qs.q[k] : 0.0);
        for (int k = 0; k < 8; ++k) std::fprintf(log_, " %.6g", k < qs.count ? pq[k] : 0.0);
        std::fputc('\n', log_);
    }

    MtpCalibCfg cfg_;
    bool        enabled_ = false;
    std::vector<Cls> cls_;
    std::unordered_map<uint64_t, int> run_;
    int64_t     step_ = 0;
    uint64_t    rows_ = 0, switches_ = 0;
    double      row_alpha_sum_ = 0;
    std::FILE*  log_ = nullptr;
};

}  // namespace rad
