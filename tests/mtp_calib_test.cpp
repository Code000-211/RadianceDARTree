/* tests/mtp_calib_test.cpp -- the online acceptance calibration (core/sched/mtp_calib.h).
 *
 * A synthetic world gives rows whose draft law Q and target law P are known, so the exact label
 * alpha = sum min(P, Q) is known too, and the controller must
 *
 *   - learn a map from the draft-side features to alpha (binned isotonic, prior = the static
 *     prediction) that converges to the true alpha(feature) and is monotone;
 *   - fit the per-depth calibration temperature so the slot probabilities the adaptive rule reads
 *     are better calibrated than the static ones (lower ECE and Brier), and leave a right static
 *     prior alone (hysteresis);
 *   - fit the per-depth draft temperature to the overlap sum min(P, Q_tau) scored on the same
 *     rows, up to the one-hot limit, and keep a draft law that is already best;
 *   - write a replay log, and stage no word at all when nothing is switched on.
 */
#include "rad_test.h"

#include "sched/mtp_calib.h"
#include "sched/mtp_couple_plan.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace rad;

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9e3779b97f4a7c15ull + 1) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double u() { return ((next() >> 11) + 0.5) * 0x1p-53; }
    double normal() { return std::sqrt(-2 * std::log(u())) * std::cos(6.283185307179586 * u()); }
};

/* A row: head scores l[0..V), the draft law Q = softmax(l / tau0) over the top 8 and the target's
 * law P = softmax(l * g) over all V tokens (g > 1 sharper than the head's T1, g < 1 flatter). */
constexpr int V = 14;
struct Row {
    RadMtpCoupleState s{};
    double pq[8] = {};
};

Row make_row(Rng& r, double g, double tau0, int depth, double spread = 1.5) {
    double l[V];
    for (int i = 0; i < V; ++i) l[i] = spread * r.normal();
    int ord[V];
    for (int i = 0; i < V; ++i) ord[i] = i;
    std::sort(ord, ord + V, [&](int a, int b) { return l[a] > l[b]; });
    Row w;
    RadMtpCoupleState& s = w.s;
    s.status = 1; s.round = depth; s.draw.mode = 1;
    s.draw.dscale = (float)(1.0 / tau0);
    s.count = 8;
    double sum = 0, q[8];
    for (int k = 0; k < 8; ++k) { q[k] = std::exp(l[ord[k]] / tau0); sum += q[k]; }
    for (int k = 0; k < 8; ++k) { s.ids[k] = ord[k]; s.q[k] = q[k] / sum; }
    s.x = s.ids[0];
    double z = 0, p[V];
    for (int i = 0; i < V; ++i) { p[i] = std::exp(l[i] * g); z += p[i]; }
    for (int k = 0; k < 8; ++k) w.pq[k] = p[ord[k]] / z;
    return w;
}

MtpCalibCfg cfg_online(bool autofit) {
    MtpCalibCfg c;
    c.mode = 1;
    c.autofit = autofit;
    c.ece = true;
    c.interval = 32;
    c.min_n = 60;
    return c;
}

/* The map: alpha follows a known monotone curve of the feature with noise; the fit finds it. */
TEST(the_binned_isotonic_map_converges_to_the_true_alpha_of_its_feature) {
    CalMap m;
    for (int b = 0; b < kCalBins; ++b) {   /* no data: the prior, alpha = x */
        const double x = (b + 0.5) / kCalBins;
        CHECK(std::fabs(m.predict(x, 4.0) - x) < 1e-12);
    }
    Rng r(7);
    auto truth = [](double x) { return 0.12 + 0.8 * std::sqrt(x); };
    for (int i = 0; i < 40000; ++i) {
        const double x = r.u();
        m.observe(x, std::min(1.0, std::max(0.0, truth(x) + 0.15 * r.normal())));
    }
    double est[kCalBins], err0 = 0, err1 = 0;
    m.fit(4.0, est);
    for (int b = 0; b < kCalBins; ++b) {
        const double x = (b + 0.5) / kCalBins;
        if (b > 0) CHECK(est[b] >= est[b - 1] - 1e-12);   /* isotonic */
        err0 += std::fabs(x - truth(x));
        err1 += std::fabs(est[b] - truth(x));
        CHECK(std::fabs(est[b] - truth(x)) < 0.04);
    }
    std::printf("  map vs a true alpha(x): mean |error| prior %.4f -> learned %.4f\n", err0 / kCalBins, err1 / kCalBins);
    CHECK(err1 < 0.4 * err0);
    /* a non-monotone cloud is pooled monotone (PAV), never inverted */
    CalMap bad;
    for (int i = 0; i < 4000; ++i) { const double x = r.u(); bad.observe(x, x < 0.5 ? 0.8 : 0.3); }
    bad.fit(4.0, est);
    for (int b = 1; b < kCalBins; ++b) CHECK(est[b] >= est[b - 1] - 1e-12);
}

/* The controller on a head that is over-confident against its target (P flatter than Q): the
 * calibration temperature moves off the static 1, the ECE and Brier of the slot probabilities the
 * rule reads fall against the static map's, and the learned map is at least as good. */
TEST(the_online_calibration_reduces_the_error_of_an_overconfident_head) {
    MtpCalibCtl ctl;
    MtpCalibCfg cc = cfg_online(false);
    REQUIRE(ctl.configure(cc));
    Rng r(11);
    const int N = 12000;
    for (int i = 0; i < N; ++i) {
        const int depth = 1 + (int)(r.next() % 3);
        const Row w = make_row(r, 0.55, 1.0, depth);
        ctl.word(0.7f, 1.0f);   /* creates the class like a draw does */
        ctl.observe_row(0.7f, 1, w.s, w.pq, 1, -1, 0);
        if (i % 3 == 2) ctl.end_step(1.0, 4, 1);
    }
    REQUIRE_EQ(ctl.classes(), 1);
    for (int d = 0; d < kCalDepths; ++d) {
        std::printf("  depth class %d: live T %.2f, ECE static %.4f live %.4f map %.4f, Brier static %.4f live %.4f map %.4f\n",
                    d + 1, (double)ctl.live_cal(0, d), ctl.ece_static(0, d), ctl.ece_live(0, d), ctl.ece_map(0, d),
                    ctl.brier_static(0, d), ctl.brier_live(0, d), ctl.brier_map(0, d));
        CHECK(std::fabs(ctl.live_cal(0, d) - 1.0f) > 0.05f);   /* it moved off the static temperature */
        CHECK(ctl.brier_live(0, d) < ctl.brier_static(0, d));
        CHECK(ctl.ece_live(0, d) < ctl.ece_static(0, d));
        CHECK(ctl.brier_map(0, d) < ctl.brier_static(0, d));
    }
    /* the staged word carries the fitted codes, only for the online mode */
    const uint32_t w = ctl.word(0.7f, 1.0f);
    CHECK(w & RAD_MTP_DRAW_EXPORT_BIT);
    for (int d = 0; d < kCalDepths; ++d) CHECK((((w >> (8 * d)) & 0xff) >> 4) != 0);
    for (int d = 0; d < kCalDepths; ++d) CHECK((((w >> (8 * d)) & 0xff) & 15) == 0);   /* no tau codes: autofit off */
}

/* A head whose static map is already the fitted one keeps it: hysteresis must not chase noise. The
 * first controller finds the temperature the world wants; a second one that starts from it as its
 * static prior, on the same stream, never leaves it. */
TEST(a_calibrated_head_keeps_its_static_prior) {
    float fitted = 0;
    {
        MtpCalibCtl ctl;
        REQUIRE(ctl.configure(cfg_online(false)));
        Rng r(23);
        for (int i = 0; i < 8000; ++i) {
            const Row w = make_row(r, 0.8, 1.0, 1);
            ctl.word(1.0f, 1.0f);
            ctl.observe_row(1.0f, 1, w.s, w.pq, 1, -1, 0);
            if (i % 2) ctl.end_step(1.0, 4, 1);
        }
        fitted = ctl.live_cal(0, 0);
        CHECK(std::fabs(fitted - 1.0f) > 0.05f);
    }
    MtpCalibCtl ctl;
    MtpCalibCfg cc = cfg_online(false);
    cc.static_T = { fitted };
    REQUIRE(ctl.configure(cc));
    Rng r(23);
    int switches = 0;
    float last = fitted;
    for (int i = 0; i < 8000; ++i) {
        const Row w = make_row(r, 0.8, 1.0, 1);
        ctl.word(1.0f, 1.0f);
        ctl.observe_row(1.0f, 1, w.s, w.pq, 1, -1, 0);
        if (i % 2) ctl.end_step(1.0, 4, 1);
        if (std::fabs(ctl.live_cal(0, 0) - last) > 1e-6f) { ++switches; last = ctl.live_cal(0, 0); }
    }
    std::printf("  fitted T %.2f as the prior: %d switch(es)\n", (double)fitted, switches);
    CHECK_EQ(switches, 0);
}

/* The static list is the prior: a depth that starts at T_d = 1.3 begins there. */
TEST(the_static_list_is_the_prior_of_the_calibration) {
    MtpCalibCtl ctl;
    MtpCalibCfg cc = cfg_online(false);
    cc.static_T = { 1.3f, 1.1f };
    REQUIRE(ctl.configure(cc));
    ctl.word(1.0f, 1.0f);
    CHECK(std::fabs(ctl.live_cal(0, 0) - 1.3f) < 1e-6f);
    CHECK(std::fabs(ctl.live_cal(0, 1) - 1.1f) < 1e-6f);
    CHECK(std::fabs(ctl.live_cal(0, 2) - 1.1f) < 1e-6f);   /* the last repeats */
    /* and no word moves before the data says so */
    const uint32_t w = ctl.word(1.0f, 1.0f);
    for (int d = 0; d < kCalDepths; ++d) CHECK((((w >> (8 * d)) & 0xff) >> 4) == 0);
}

/* The draft temperature: the target is sharper than the head (g = 1 / 0.7), so the overlap peaks at
 * a draft temperature near 0.7 (the grid's 0.6 / 0.75); a head that already matches its target
 * keeps tau 1; a near-greedy target goes to the one-hot limit. The grid is scored on the SAME rows. */
TEST(the_draft_temperature_follows_the_measured_overlap) {
    struct Case { double g; const char* name; int want_lo, want_hi; };
    const Case cases[] = {
        { 1.0 / 0.7, "target sharper than head", 4, 7 },   /* codes 0.6 .. 0.8 */
        { 1.0,       "target == head",            0, 0 },   /* never leaves the base */
        { 25.0,      "near-greedy target",        1, 1 },   /* tau 0.05 */
    };
    for (const Case& c : cases) {
        MtpCalibCtl ctl;
        MtpCalibCfg cc = cfg_online(true);
        cc.mode = 2;   /* calibration off, the draft temperature only */
        REQUIRE(ctl.configure(cc));
        Rng r(31);
        for (int i = 0; i < 6000; ++i) {
            const Row w = make_row(r, c.g, 1.0, 1 + (int)(r.next() % 3));
            ctl.word(1.0f, 1.0f);
            ctl.observe_row(1.0f, 1, w.s, w.pq, 1, -1, 0);
            if (i % 2) ctl.end_step(1.0, 4, 1);
        }
        const uint32_t w = ctl.word(1.0f, 1.0f);
        const unsigned code = w & 15;
        std::printf("  %-26s -> tau code %u (%.2f), live tau %.2f\n", c.name, code, (double)rad_mtp_tau_of_code(code),
                    (double)ctl.live_tau(0, 0));
        CHECK(code >= (unsigned)c.want_lo && code <= (unsigned)c.want_hi);
        CHECK((w >> 4 & 15) == 0);   /* mode 2: no calibration code */
    }
}

/* Switched off, the controller stages nothing and a draw is what it was. */
TEST(with_everything_off_the_draw_carries_no_word) {
    MtpCalibCtl off;
    MtpCalibCfg cc;   /* static, no autofit, no ece, no log */
    CHECK(off.configure(cc));
    CHECK(!off.enabled());
    CHECK_EQ(off.word(0.7f, 1.0f), 0u);
    Request r;
    r.sp = SamplingParams{};
    r.sp.temp = 0.7f;
    r.output.assign(5, 0);
    MtpCoupleLaw law;
    const RadMtpCoupleDraw a = mtp_couple_draw_of(r, 1, 0, nullptr);
    const RadMtpCoupleDraw b = mtp_couple_draw_of(r, 1, 0, &law);
    law.calib = &off;
    const RadMtpCoupleDraw c = mtp_couple_draw_of(r, 1, 0, &law);
    CHECK(std::memcmp(&b, &c, sizeof b) == 0);   /* a disabled controller: the draw as it was */
    CHECK_EQ(a.dpad_, 0u);
    CHECK_EQ(c.dpad_, 0u);
    /* switched on, the word rides the draw and only dpad_ differs */
    MtpCalibCtl on;
    REQUIRE(on.configure(cfg_online(true)));
    law.calib = &on;
    const RadMtpCoupleDraw d = mtp_couple_draw_of(r, 1, 0, &law);
    CHECK(d.dpad_ & RAD_MTP_DRAW_EXPORT_BIT);
    RadMtpCoupleDraw d2 = d;
    d2.dpad_ = 0;
    CHECK(std::memcmp(&b, &d2, sizeof b) == 0);
    /* a greedy request never carries one */
    r.sp.top_k = 1;
    CHECK_EQ(mtp_couple_draw_of(r, 1, 0, &law).dpad_, 0u);
}

/* The replay log: one R per judged child and one S per step, in the documented columns. */
TEST(the_replay_log_has_a_row_per_judged_child_and_a_record_per_step) {
    const std::string path = radtest::tmpdir() + "/calib.log";
    {
        MtpCalibCtl ctl;
        MtpCalibCfg cc;
        cc.log_path = path;
        REQUIRE(ctl.configure(cc));
        CHECK(ctl.enabled());
        Rng r(5);
        for (int s = 0; s < 3; ++s) {
            ctl.word(1.0f, 1.0f);
            for (int d = 1; d <= 2; ++d) {
                const Row w = make_row(r, 0.8, 1.0, d);
                ctl.observe_row(1.0f, 9, w.s, w.pq, 1, d == 1 ? 0 : -1, ctl.since_of(9));
            }
            ctl.request_step(9, 1, 2);
            ctl.end_step(10.5, 3, 1);
        }
        ctl.close_log();
    }
    std::ifstream f(path);
    std::string line;
    int R = 0, S = 0;
    while (std::getline(f, line)) {
        if (line[0] == 'R') {
            ++R;
            std::istringstream is(line);
            std::string tag;
            long long step; unsigned long long seq; double T, alpha, top1, ent, ada, tau0;
            int depth, slot, since, acc, n;
            is >> tag >> step >> seq >> T >> depth >> slot >> since >> acc >> alpha >> top1 >> ent >> ada >> tau0 >> n;
            CHECK(is.good());
            CHECK(alpha >= 0 && alpha <= 1);
            CHECK_EQ(n, 8);
            CHECK(std::fabs(tau0 - 1.0) < 1e-3);
            double q[8], p[8], sq = 0;
            for (double& x : q) { is >> x; sq += x; }
            for (double& x : p) is >> x;
            CHECK(std::fabs(sq - 1.0) < 1e-4);
            double mn = 0;
            for (int k = 0; k < 8; ++k) mn += std::min(q[k], p[k]);
            CHECK(std::fabs(mn - alpha) < 1e-4);   /* the logged alpha is sum min(P, Q) of the logged columns */
        } else if (line[0] == 'S') {
            ++S;
        }
    }
    CHECK_EQ(R, 6);
    CHECK_EQ(S, 3);
}

/* A request's since-rejection run: grows with full accepts, resets on a rejection. */
TEST(the_tokens_since_the_last_rejection_are_tracked_per_request) {
    MtpCalibCtl ctl;
    REQUIRE(ctl.configure(cfg_online(false)));
    CHECK_EQ(ctl.since_of(4), 0);
    ctl.request_step(4, 3, 3);
    ctl.request_step(4, 3, 3);
    CHECK_EQ(ctl.since_of(4), 6);
    ctl.request_step(4, 1, 3);
    CHECK_EQ(ctl.since_of(4), 0);
    CHECK_EQ(ctl.since_of(5), 0);
}

/* Not a check: with RAD_CALIB_SYNTH_LOG=PATH it writes a replay log of a SYNTHETIC chain-4 world
 * (a head whose law is off from its target's by a depth-dependent factor, requests at T 0.7 and
 * 1.0) through the real log writer, for tools/rad_calib_sim. The numbers it yields describe the
 * synthetic head, not the model. */
TEST(a_synthetic_chain_log_for_the_replay_tool) {
    const char* path = std::getenv("RAD_CALIB_SYNTH_LOG");
    if (!path || !*path) return;
    MtpCalibCtl ctl;
    MtpCalibCfg cc;
    cc.log_path = path;
    REQUIRE(ctl.configure(cc));
    Rng r(99);
    for (const double T : { 0.7, 1.0 }) {
        for (int step = 0; step < 6000; ++step) {
            ctl.word((float)T, 1.0f);
            int acc = 0;
            for (int d = 1; d <= 4; ++d) {
                const double g = (0.95 - 0.12 * (d - 1)) / T;   /* the target's sharpness against the head's */
                const Row w = make_row(r, g, 1.0, d, 2.0);
                /* x ~ Q, accepted with the maximal coupling's min(1, P(x) / Q(x)) */
                double u = r.u();
                int k = 0;
                while (k < 7 && (u -= w.s.q[k]) > 0) ++k;
                const bool ok = r.u() < std::min(1.0, w.pq[k] / w.s.q[k]);
                ctl.observe_row((float)T, 1, w.s, w.pq, 1, ok ? 0 : -1, ctl.since_of(1));
                if (!ok) break;
                ++acc;
            }
            ctl.request_step(1, acc, 4);
            ctl.end_step(12.3 + 1.0 + 0.5 * 4, 5, acc);
        }
    }
    ctl.close_log();
}

}  // namespace

RAD_TEST_MAIN()
