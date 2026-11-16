/* rad_mtp_calib.h -- the exact acceptance label of a coupled draft row and the draft-side features
 * the online calibration (core/sched/mtp_calib.h) learns it from. Host code, plain functions, so
 * the engine, the replay tool and the tests share one definition.
 *
 * THE LABEL. Under maximal coupling the draft x ~ Q is accepted at a reached row with probability
 *
 *     alpha = sum_y min(P(y), Q(y))            (P: the target's truncated law at the row)
 *
 * averaged over the draw of x, and that number is a function of (P, Q) alone: observable for free
 * on every verified row that was reached, with no 0/1 noise. For a tree node with several drafted
 * children (recursive rejection sampling, abi/rad_mtp_rrs.h) child i is judged against the residual
 * P_i = norm(max(P_{i-1} - q_{i-1}, 0)) and the draft law q_i = q_{i-1} with child i-1 removed and
 * renormalised; its label is alpha_i = sum min(P_i, q_i), "P(child i accepted | row reached, the
 * earlier siblings rejected)" averaged over which token the draw put in slot i. The residual P_i
 * does not depend on the drawn tokens, q_i on the earlier ones only through the removed entries,
 * which the state records (ids[0 .. nkids) in draft order).
 *
 * P enters as its mass on Q's ids (the verify exports it, rad_mtp_couple_export_p); the rest of
 * P's mass (1 - sum) sits off Q's support and only normalises the residuals.
 */
#ifndef RAD_MTP_CALIB_H
#define RAD_MTP_CALIB_H

#include "rad_spechub.h"

#ifdef __cplusplus
#include <algorithm>
#include <cmath>

/* Labels of the children of state `s` (Q in draft order, s.nkids drafted children; a chain state
 * has one), given P's mass `pq[k]` on s.ids[k]. alpha[0 .. n) for n = max(1, nkids) capped by the
 * support; returns n. `a_real` (optional) gets each drafted child's own acceptance probability
 * min(1, P_i(x_i) / q_i(x_i)), the quantity the label averages over x_i. A degenerate residual (nothing left to judge) leaves the remaining labels 0. */
static inline int rad_calib_labels(const RadMtpCoupleState& s, const double* pq, double* alpha,
                                   double* a_real = nullptr) {
    int m = s.nkids >= 1 ? s.nkids : 1;
    if (m > s.count) m = s.count;
    double r[8], qr[8], sump = 0, so = 1.0;
    for (int k = 0; k < s.count; ++k) { r[k] = pq[k] > 0 ? pq[k] : 0; qr[k] = s.q[k]; sump += r[k]; }
    const double pout = sump < 1.0 ? 1.0 - sump : 0.0;
    int i = 0;
    for (; i < m; ++i) {
        double sq = 0;
        for (int k = 0; k < s.count; ++k) sq += qr[k];
        if (!(sq > 0)) break;
        double a = 0;
        for (int k = 0; k < s.count; ++k) a += std::min(r[k], qr[k] / sq);
        alpha[i] = a > 1.0 ? 1.0 : a;
        if (a_real) {
            const double qx = i < s.count ? qr[i] / sq : 0.0;
            a_real[i] = qx > 0 ? std::min(1.0, r[i] / qx) : 0.0;
        }
        double z = so * pout;
        for (int k = 0; k < s.count; ++k) {
            const double d = r[k] - qr[k] / sq;
            r[k] = d > 0 ? d : 0;
            z += r[k];
        }
        if (!(z > 0)) { ++i; break; }
        for (int k = 0; k < s.count; ++k) r[k] /= z;
        so /= z;
        if (i < s.count) qr[i] = 0;   /* the child's slot leaves the draft law */
    }
    for (int j = i; j < m; ++j) { alpha[j] = 0; if (a_real) a_real[j] = 0; }
    return m;
}

/* The cheap draft-side features of a law: the descending probabilities, top-1 mass, entropy (nats)
 * and the AdaEDL proxy 1 - sqrt(0.2 H) (clipped to [0, 1]). */
struct RadCalibFeat {
    double slot[8] = {};   /* q sorted descending, zero past the support */
    double top1 = 0, ent = 0, adaedl = 0;
    int    n = 0;
};

static inline void rad_calib_features(const RadMtpCoupleState& s, RadCalibFeat* f) {
    *f = RadCalibFeat{};
    f->n = s.count < 8 ? s.count : 8;
    for (int k = 0; k < f->n; ++k) f->slot[k] = s.q[k];
    std::sort(f->slot, f->slot + f->n, [](double a, double b) { return a > b; });
    f->top1 = f->slot[0];
    for (int k = 0; k < f->n; ++k) if (f->slot[k] > 0) f->ent -= f->slot[k] * std::log(f->slot[k]);
    const double a = 1.0 - std::sqrt(0.2 * f->ent);
    f->adaedl = a < 0 ? 0 : a > 1 ? 1 : a;
}

/* The law q^(tau0 / tau) renormalised (the same head scores at draft temperature tau, when q was
 * drawn at tau0) over the state's support. Valid for a state built without a top-k / top-p cut. */
static inline void rad_calib_q_at(const double* q, int n, double tau0, double tau, double* out) {
    const double e = tau0 / tau;
    double mx = 0;
    for (int k = 0; k < n; ++k) if (q[k] > mx) mx = q[k];
    double sum = 0;
    for (int k = 0; k < n; ++k) {
        out[k] = q[k] > 0 ? std::exp(e * (std::log(q[k]) - std::log(mx))) : 0.0;
        sum += out[k];
    }
    for (int k = 0; k < n; ++k) out[k] /= sum;
}

/* sum min(P, Q) over a support. */
static inline double rad_calib_overlap(const double* q, const double* pq, int n) {
    double a = 0;
    for (int k = 0; k < n; ++k) a += std::min(q[k], pq[k] > 0 ? pq[k] : 0.0);
    return a;
}

/* The temperature the draw (any depth's byte) applied to the head scores: 1 / dscale. */
static inline double rad_calib_tau_of(const RadMtpCoupleDraw& d, int depth) {
    const float ds = rad_mtp_draw_dscale_at(d, depth);
    return ds > 0.0f ? 1.0 / (double)ds : 1.0;
}

#endif /* __cplusplus */
#endif /* RAD_MTP_CALIB_H */
