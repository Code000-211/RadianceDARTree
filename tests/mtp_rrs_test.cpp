/* tests/mtp_rrs_test.cpp -- recursive rejection sampling over a draft tree's siblings
 * (--draft-coupling on with --draft-tree on, abi/rad_mtp_rrs.h), checked against its own law.
 *
 * The property: whatever the head's law Q and the tree's shape, the tokens a verify step commits
 * are a prefix of a sample of the TARGET's autoregressive law. The tests run the real shared code
 * (the propose row, the grower's coupled pass, the verify row) over a toy model with a known
 * target P(. | prefix) and a different head Q(. | prefix), many seeds, and compare the empirical
 * law of the emitted tokens with the exact one (chi-square, deterministic seeds):
 *
 *   one node, 2..4 children      the first emitted token ~ P (single-node recursion, exact)
 *   chain coupling               R = 3, no leaves: the legacy path, bit for bit
 *   stage-1 tree                 trunk chain R = 3 + 3 leaves at the root, drawn by the propose row
 *   stage-2 tree                 the grower's best-first tree (W = 2, R = 3, N = 7) with its coupled
 *                                pass, children created as a prefix of each node's draft order
 *   coupling + pruning           stage 1 and stage 2 with confidence pruning (--draft-tree-prune /
 *                                -rel) decided by the slot priorities (the j-th largest q) only: the
 *                                tree drops a suffix of a node's draft order and stays exact;
 *                                negative control: pruning that reads the drawn token's own q fails
 *   degenerate                   Q == P accepts the first child; disjoint supports reject all and
 *                                emit from P; greedy and unbound rows are not touched
 *   negative control             the with-replacement verify applied to without-replacement drafts
 *                                is NOT exact and the same test sees it
 *
 * The step is walked the way the core walks it: the verify row writes the sampler's token for each
 * visited node and ADV_ACCEPT's rule (the first child whose token is that token) follows it; after
 * the step, the sequence continues from the exact target so the first three tokens of every trial
 * have a known joint law.
 */
#include "rad_test.h"

#include "rad_mtp_tree_rrs.h"
#include "../plugin/spechub/spechub.h"

#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int V = 12;  /* the toy vocabulary: larger than the head's top 8, so the target has mass off Q's support */

uint64_t H(uint64_t a) { return rad_mtp_rrs_mix(a); }

/* The target's candidate plane at a prefix (ids and logits; some prefixes drop a token, as a
 * top-k / top-p would) and the head's shortlist (all V tokens, logits). The head is a noisy
 * version of the target, so acceptance is neither 0 nor 1. */
struct Law {
    int32_t pid[V];  float pval[V];  int pn = 0;      /* target candidates */
    double  p[V];                                      /* the exact target law over tokens */
    int32_t qid[8];  float qval[8];                    /* the head's shortlist */
};

Law law_at(uint64_t prefix_hash, double head_noise) {
    Law L;
    float lt[V], lq[V];
    for (int i = 0; i < V; ++i) {
        lt[i] = (float)((H(prefix_hash * 131 + (uint64_t)i * 977 + 1) >> 40) % 3000) / 1000.0f - 1.5f;
        const float nz = (float)((H(prefix_hash * 257 + (uint64_t)i * 331 + 7) >> 40) % 3000) / 1000.0f - 1.5f;
        lq[i] = 0.75f * lt[i] + (float)head_noise * nz;
    }
    const uint64_t h = H(prefix_hash + 99);
    const int drop = h % 3 == 0 ? (int)((h >> 8) % V) : -1;   /* a target token the filters removed */
    double z = 0;
    for (int i = 0; i < V; ++i) { L.p[i] = 0; if (i != drop) { L.pid[L.pn] = i; L.pval[L.pn++] = lt[i]; } }
    for (int j = 0; j < L.pn; ++j) z += std::exp((double)L.pval[j]);
    for (int j = 0; j < L.pn; ++j) L.p[L.pid[j]] = std::exp((double)L.pval[j]) / z;
    /* the head reports its 8 best (descending, ties to the lower id): the rest of the vocabulary is
     * off Q's support, where the target keeps its mass */
    int ord[V];
    for (int i = 0; i < V; ++i) ord[i] = i;
    std::stable_sort(ord, ord + V, [&](int a, int b) { return lq[a] > lq[b]; });
    for (int i = 0; i < 8; ++i) { L.qid[i] = ord[i]; L.qval[i] = lq[ord[i]]; }
    return L;
}

/* Confidence pruning of the drafted tree (the test's switch, read by both drafters). `abs` / `rel`
 * are the real rule's thresholds (abi/rad_mtp_tree.h), run by the real propose row and the real
 * coupled grower, which decide by slot priorities. `value_dep` is the NEGATIVE CONTROL: an
 * implementation that reads the drawn child's own q (it cuts a node's draft order at the first child
 * whose own q is below `vd_thr`, with everything after it), applied to the otherwise unpruned tree. */
struct Prune { double abs = 0, rel = 0; bool value_dep = false; double vd_thr = 0; };
const Prune* g_pr = nullptr;
double g_cut = 0, g_kids = 0;   /* bookkeeping: children cut / drafted, for the "it actually prunes" checks */

uint64_t child_hash(uint64_t ph, int tok) { return H(ph * 1000003ull + (uint64_t)tok + 17); }

/* A draw from the target law at the prefix: the ordinary pick. */
int sample_law(const Law& L, uint64_t seed, uint64_t pos, uint64_t key) {
    double u = rad_spechub_uniform(seed, pos, key);
    int last = 0;
    for (int i = 0; i < V; ++i) {
        if (L.p[i] <= 0) continue;
        last = i;
        u -= L.p[i];
        if (u < 0) return i;
    }
    return last;
}

RadMtpCoupleDraw make_draw(uint64_t seed, uint64_t base) {
    RadMtpCoupleDraw d{};
    d.seed = seed; d.base = base; d.seq = 1; d.card = 0; d.mode = 1;
    return d;
}

/* The verify row of the real plugin code over one candidate plane and one bound state. */
int verify(const RadMtpCoupleState& st, const Law& L, uint64_t seed, uint64_t pos, int fallback) {
    uint32_t params[24] = {};
    RadSampleParams* p = reinterpret_cast<RadSampleParams*>(params);
    p->seed = seed;
    p->pos = pos;
    int32_t tok = fallback;
    RadMtpCoupleVerifyCall c{};
    c.idx = L.pid; c.val = L.pval; c.params = params;
    c.state = reinterpret_cast<const uint32_t*>(&st);
    c.token = &tok;
    c.M = 1; c.n_cand = L.pn; c.vocab = 1000;
    c.ld[0] = L.pn; c.ld[1] = L.pn; c.ld[2] = 24; c.ld[3] = RAD_MTP_COUPLE_STATE_WORDS; c.ld[4] = 1;
    rad_mtp_couple_verify_row(c, 0);
    return tok;
}

/* ------------------------------------------------------------------ a toy drafted tree */

struct Tree {
    int      n = 0;                 /* nodes 1..n */
    int      tok[64] = {}, par[64] = {};
    uint64_t hash[64] = {};         /* the prefix hash of every node, node 0 the root */
    RadMtpCoupleState st[64] = {};  /* row k's coupled state (status 0: none) */
};

enum Kind { CHAIN, STAGE1, STAGE2, STAGE2B };

/* The negative control's pruning: at every node with several drafted children, cut the draft order
 * at the first child (past the first) whose OWN q, the law of the token it took, is below `thr`;
 * the node's children from there on, with their subtrees, are not in the tree and the state's
 * nkids shrinks. The count of children now depends on the value of the child it admits. */
void value_cut(Tree* t, double thr) {
    bool gone[64] = {};
    for (int p = 0; p <= t->n; ++p) {
        RadMtpCoupleState& s = t->st[p];
        if (s.status != 1 || s.nkids < 2) continue;
        int cut = -1;
        for (int i = 1; i < s.nkids; ++i) if (s.q[i] < thr) { cut = i; break; }
        if (cut < 0) continue;
        g_cut += s.nkids - cut;
        for (int j = p + 1; j <= t->n; ++j)
            if (t->par[j] == p)
                for (int i = cut; i < s.nkids; ++i) gone[j] = gone[j] || t->tok[j] == s.ids[i];
        s.nkids = cut;
    }
    for (int j = 1; j <= t->n; ++j)
        if (gone[j] || (t->par[j] >= 0 && gone[t->par[j]])) { gone[j] = true; t->par[j] = -99; }
}

/* The stage-1 / chain drafting, through the real propose row: round r over the head's law at the
 * chain prefix; the leaves of round 1 captured as the core does (columns 1.., not the pick, in
 * range, no repeats). */
void draft_chain(Tree* t, uint64_t seed, uint64_t base, int R, int L, double noise, uint64_t h0 = 1) {
    uint64_t h = h0;
    t->hash[0] = h;
    int n = 0;
    std::vector<int32_t> leaves;
    for (int r = 1; r <= R; ++r) {
        const Law law = law_at(h, noise);
        int32_t idx[8];
        float   val[8];
        for (int i = 0; i < 8; ++i) { idx[i] = law.qid[i]; val[i] = law.qval[i]; }
        const RadMtpCoupleDraw draw = make_draw(seed, base);
        const int32_t card = 0;
        RadMtpCoupleProposeCall c{};
        c.idx = idx; c.val = val; c.card = &card;
        c.draw = reinterpret_cast<const uint32_t*>(&draw);
        RadMtpCoupleState rows[8] = {};
        c.state = reinterpret_cast<uint32_t*>(rows);
        c.M = 1; c.vocab = 1000;
        c.ld_idx = 8; c.ld_val = 8; c.ld_draw = RAD_MTP_COUPLE_DRAW_WORDS;
        c.st_round = RAD_MTP_COUPLE_STATE_WORDS; c.st_row = 0;
        c.round = r;
        c.leaves = r == 1 ? L : 0;
        if (g_pr && !g_pr->value_dep) {
            RadMtpTreeCfg pc{};
            rad_mtp_tree_set_prune(&pc, g_pr->abs, g_pr->rel);
            c.prune = pc.prune; c.log_abs = pc.log_abs; c.log_rel = pc.log_rel;
        }
        rad_mtp_couple_propose_row(c, 0);
        t->st[r - 1] = rows[r - 1];
        const int pick = idx[0];
        if (r == 1)
            for (int i = 1; i < 8; ++i) {
                const int32_t id = idx[i];
                if (id < 0 || id >= 1000 || id == pick || (int)leaves.size() >= L) continue;
                if (std::find(leaves.begin(), leaves.end(), id) == leaves.end()) leaves.push_back(id);
            }
        ++n;
        t->tok[n] = pick; t->par[n] = n - 1;
        h = child_hash(h, pick);
        t->hash[n] = h;
    }
    for (int32_t id : leaves) {
        ++n;
        t->tok[n] = id; t->par[n] = 0;
        t->hash[n] = child_hash(t->hash[0], id);
    }
    t->n = n;
    if (L > 0) { g_kids += L; g_cut += L - (int)leaves.size(); }
    if (g_pr && g_pr->value_dep) value_cut(t, g_pr->vd_thr);
}

/* Stage 2: the grower's coupled passes over a head that reports the toy's Q for each frontier
 * row; node states land in the plane as the engine reads them back. */
void draft_grow(Tree* t, uint64_t seed, uint64_t base, int W, int R, int N, double noise,
                uint64_t h0 = 1) {
    RadMtpTreeCfg cfg{};
    cfg.rounds = R; cfg.nodes = N; cfg.width = W; cfg.block = 0; cfg.sel_pitch = 0; cfg.pitch = N;
    if (g_pr && !g_pr->value_dep) rad_mtp_tree_set_prune(&cfg, g_pr->abs, g_pr->rel);
    std::vector<int> st((size_t)rad_mtp_tree_state_words(N, W, R), 0);
    std::vector<RadMtpCoupleState> plane((size_t)N);
    const RadMtpCoupleDraw draw = make_draw(seed, base);
    RadMtpTreeCpl cpl{};
    cpl.draw = reinterpret_cast<const uint32_t*>(&draw);
    cpl.card = 0;
    cpl.state = reinterpret_cast<uint32_t*>(plane.data());
    cpl.st_node = RAD_MTP_COUPLE_STATE_WORDS;
    cpl.st_seq = 0;
    cpl.rows = N;
    cpl.vocab = 1000;
    int tok[64], par[64];
    uint64_t nodehash[64];
    nodehash[0] = h0;
    for (int round = 1; round <= R; ++round) {
        cfg.round = round;
        int nfront = round == 1 ? 1 : st[2];
        std::vector<int32_t> d((size_t)W * 8, -1);
        std::vector<float> v((size_t)W * 8, -INFINITY);
        for (int w = 0; w < nfront && w < W; ++w) {
            const int f = round == 1 ? 0 : st[4 + w];
            const Law law = law_at(nodehash[f], noise);
            for (int i = 0; i < 8; ++i) { d[(size_t)w * 8 + i] = law.qid[i]; v[(size_t)w * 8 + i] = law.qval[i]; }
        }
        const int rc = rad_mtp_tree_grow_seq_cpl(&cfg, st.data(), d.data(), v.data(), 0, 8, 1, 0,
                                                 nullptr, tok, par, nullptr, nullptr, nullptr, nullptr,
                                                 0, nullptr, nullptr, &cpl);
        CHECK_EQ(rc, 0);
        const int count = st[1];
        const int* nd = st.data() + 4 + W;
        for (int id = 1; id <= count; ++id)
            nodehash[id] = child_hash(nodehash[nd[6 * id + 1]], nd[6 * id + 0]);
    }
    t->n = N;
    for (int j = 1; j <= N; ++j) { t->tok[j] = tok[j - 1]; t->par[j] = par[j - 1]; }
    /* a pruned grower's inert entries (token -1): ADV_BTAKE's duplicate of node 1, a child of the
     * root the walk never takes and the verify's child list drops as a repeat */
    for (int j = 2; j <= N; ++j)
        if (t->tok[j] < 0) { t->tok[j] = t->tok[1]; t->par[j] = 0; g_cut += 1; }
    for (int j = 0; j <= N; ++j) t->hash[j] = j == 0 ? h0 : 0;
    for (int j = 1; j <= N; ++j) t->hash[j] = child_hash(t->hash[t->par[j]], t->tok[j]);
    for (int k = 0; k < N; ++k) t->st[k] = plane[(size_t)k];
    g_kids += N;
    if (g_pr && g_pr->value_dep) value_cut(t, g_pr->vd_thr);
}

/* One verify step over the drafted tree, walked as the core does. Returns the emitted tokens
 * (the accepted nodes' tokens and the final one), extended by direct target draws to `want`. */
int walk(const Tree& t, uint64_t seed, uint64_t n_out, double noise, int want, int* out,
         int* accepted) {
    int len = 0, cur = 0, depth = 0;
    uint64_t h = t.hash[0];
    *accepted = 0;
    for (;;) {
        const Law law = law_at(h, noise);
        const uint64_t pos = n_out + (uint64_t)depth;
        /* the row's children, in node order, without repeats */
        int32_t kid[64];
        int nk = 0;
        for (int j = cur + 1; j <= t.n; ++j) {
            if (t.par[j] != cur) continue;
            bool dup = false;
            for (int q = 0; q < nk; ++q) dup = dup || kid[q] == t.tok[j];
            if (!dup) kid[nk++] = t.tok[j];
        }
        int y = sample_law(law, seed, pos, 0);   /* the ordinary pick; an unbound row keeps it */
        const RadMtpCoupleState& c = t.st[cur];
        if (nk > 0 && c.status == 1) {
            const int kept = rad_mtp_rrs_prefix(c, kid, nk);
            if (kept > 0 && c.round == depth + 1 && c.x == kid[0]) {
                RadMtpCoupleState b = c;
                b.pos = rad_mtp_couple_verify_pos(n_out, depth + 1);
                if (c.nkids >= 2) b.nkids = kept;
                y = verify(b, law, seed, pos, y);
            }
        }
        out[len++] = y;
        int nxt = -1;
        for (int j = cur + 1; j <= t.n && nxt < 0; ++j)
            if (t.par[j] == cur && t.tok[j] == y) nxt = j;
        if (nxt < 0) break;
        cur = nxt; ++depth; ++*accepted;
        h = child_hash(h, y);
    }
    /* the steps after this one: exact draws from the target given what was committed */
    uint64_t ph = t.hash[0];
    for (int i = 0; i < len; ++i) ph = child_hash(ph, out[i]);
    while (len < want) {
        const Law law = law_at(ph, noise);
        const int y = sample_law(law, seed ^ 0xabcdef1234ull, n_out + (uint64_t)len, 1);
        out[len++] = y;
        ph = child_hash(ph, y);
    }
    return len;
}

/* Chi-square of the empirical first-`m`-token law against the target's, cells with too little
 * expected mass pooled; returns (chi2 - df) / sqrt(2 df). */
double zscore(const std::vector<double>& counts, const std::vector<double>& expect, double n) {
    double chi = 0, pool_c = 0, pool_e = 0;
    int cells = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        const double e = expect[i] * n;
        if (e < 20) { pool_c += counts[i]; pool_e += e; continue; }
        chi += (counts[i] - e) * (counts[i] - e) / e;
        ++cells;
    }
    if (pool_e >= 5) { chi += (pool_c - pool_e) * (pool_c - pool_e) / pool_e; ++cells; }
    const double df = cells - 1;
    return (chi - df) / std::sqrt(2.0 * df);
}

struct Result { double z; double mean_acc; double first_z; };

/* Run `trials` steps of one drafted-tree kind and compare the first three emitted tokens' joint
 * law (and the first token's marginal) with the target's. */
Result run_kind(Kind kind, int trials, double noise,
                const std::function<void(Tree*, int)>* hook = nullptr) {
    std::vector<double> cnt3((size_t)V * V * V, 0), cnt1((size_t)V, 0);
    double acc = 0;
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x1234567ull + (uint64_t)s * 7919ull;
        Tree t;
        if (kind == CHAIN)  draft_chain(&t, seed, 0, 3, 0, noise);
        else if (kind == STAGE1) draft_chain(&t, seed, 0, 3, 3, noise);
        else if (kind == STAGE2) draft_grow(&t, seed, 0, 2, 3, 7, noise);
        else                draft_grow(&t, seed, 0, 3, 4, 12, noise);
        if (hook) (*hook)(&t, s);
        int out[64], accepted = 0;
        walk(t, seed, 0, noise, 3, out, &accepted);
        ++cnt1[(size_t)out[0]];
        ++cnt3[(size_t)((out[0] * V + out[1]) * V + out[2])];
        acc += accepted;
    }
    std::vector<double> e1((size_t)V, 0), e3((size_t)V * V * V, 0);
    const uint64_t h0 = 1;
    const Law l0 = law_at(h0, noise);
    for (int a = 0; a < V; ++a) {
        e1[(size_t)a] = l0.p[a];
        const Law l1 = law_at(child_hash(h0, a), noise);
        for (int b = 0; b < V; ++b) {
            const Law l2 = law_at(child_hash(child_hash(h0, a), b), noise);
            for (int c = 0; c < V; ++c) e3[(size_t)((a * V + b) * V + c)] = l0.p[a] * l1.p[b] * l2.p[c];
        }
    }
    return { zscore(cnt3, e3, trials), acc / trials, zscore(cnt1, e1, trials) };
}

}  /* namespace */

/* The single-node recursion: m = 1..4 children drawn without replacement from Q, verified in draft
 * order, emit ~ P for every (P, Q) pair, including a P that lacks some of Q's tokens. */
TEST(one_node_with_several_children_emits_the_target_law) {
    const int trials = 400000;
    for (int m = 2; m <= 4; ++m)
        for (int pair = 0; pair < 6; ++pair) {
            const Law L = law_at(1000 + (uint64_t)pair, 1.0 + 0.4 * pair);
            std::vector<double> cnt((size_t)V, 0);
            for (int s = 0; s < trials; ++s) {
                RadMtpCoupleState st{};
                CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st));
                st.draw = make_draw(0x9e3779b97f4a7c15ull * (uint64_t)(s + 1), (uint64_t)s);
                st.status = 1; st.round = 1; st.pos = 5;
                rad_mtp_rrs_draw(&st, 0, m);
                CHECK(rad_mtp_couple_state_valid(st, 1000));
                CHECK_EQ(st.nkids, std::min(m, st.count));
                const int y = verify(st, L, st.draw.seed, 5, -1);
                CHECK(y >= 0 && y < V);
                ++cnt[(size_t)y];
            }
            std::vector<double> e(L.p, L.p + V);
            const double z = zscore(cnt, e, trials);
            if (z > 4.5) std::printf("  m=%d pair=%d z=%.2f\n", m, pair, z);
            CHECK(z < 4.5);
        }
}

/* The negative control: SpecInfer's with-replacement verify (q not removed or renormalised) on
 * drafts drawn without replacement is not exact, and the same statistic sees it. */
TEST(the_with_replacement_verify_is_not_exact_for_without_replacement_drafts) {
    const int trials = 400000, m = 3;
    const Law L = law_at(1002, 1.0);
    std::vector<double> cnt((size_t)V, 0);
    for (int s = 0; s < trials; ++s) {
        RadMtpCoupleState st{};
        rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st);
        st.draw = make_draw(0x9e3779b97f4a7c15ull * (uint64_t)(s + 1), (uint64_t)s);
        rad_mtp_rrs_draw(&st, 0, m);
        double r[V], q[V];
        for (int i = 0; i < V; ++i) { r[i] = L.p[i]; q[i] = 0; }
        for (int k = 0; k < st.count; ++k) q[st.ids[k]] = st.q[k];
        int out = -1;
        for (int i = 0; i < m && out < 0; ++i) {
            const int x = st.ids[i];
            const double u = rad_mtp_rrs_uaccept(st.draw.seed, 5, i);
            if (r[x] >= q[x] || u * q[x] < r[x]) { out = x; break; }
            double z = 0;
            for (int k = 0; k < V; ++k) { r[k] = std::max(r[k] - q[k], 0.0); z += r[k]; }
            for (int k = 0; k < V; ++k) r[k] /= z;   /* q never renormalised, the child never removed */
        }
        if (out < 0) {
            double u = rad_spechub_uniform(st.draw.seed, 5, RAD_SAMPLE_KEY_RESID);
            out = 0;
            for (int k = 0; k < V; ++k) { if (r[k] <= 0) continue; out = k; u -= r[k]; if (u < 0) break; }
        }
        ++cnt[(size_t)out];
    }
    std::vector<double> e(L.p, L.p + V);
    const double z = zscore(cnt, e, trials);
    CHECK(z > 20.0);
}

TEST(a_chain_stage1_and_stage2_tree_each_emit_the_target_law) {
    const int trials = 300000;
    const Result chain = run_kind(CHAIN, trials, 1.0);
    const Result s1    = run_kind(STAGE1, trials, 1.0);
    const Result s2    = run_kind(STAGE2, trials, 1.0);
    const Result s2b   = run_kind(STAGE2B, trials, 1.0);
    CHECK(s2b.z < 4.5);  CHECK(s2b.first_z < 4.5);
    std::printf("  chain z=%.2f acc=%.3f | stage1 z=%.2f acc=%.3f | stage2 z=%.2f acc=%.3f\n",
                chain.z, chain.mean_acc, s1.z, s1.mean_acc, s2.z, s2.mean_acc);
    CHECK(chain.z < 4.5);  CHECK(chain.first_z < 4.5);
    CHECK(s1.z < 4.5);     CHECK(s1.first_z < 4.5);
    CHECK(s2.z < 4.5);     CHECK(s2.first_z < 4.5);
    /* siblings are drafted to be accepted: a tree accepts at least as many drafts as the chain */
    CHECK(s1.mean_acc > chain.mean_acc);
    /* and a head far from the target changes the acceptance, never the law */
    const Result far = run_kind(STAGE2, 150000, 3.0);
    CHECK(far.z < 4.5);
}


/* Coupling with confidence pruning (DESIGN-coupling.md 3.5). The pruning rule reads the slot
 * priorities (the j-th largest q) only, so a node keeps a prefix of its draft order whatever tokens
 * were drawn into it, and the walk stays exact. The negative control cuts by the drawn child's own
 * q instead and the same statistic sees it. */
static Result run_pruned(Kind kind, const Prune& pr, int trials, double noise, double* cut_frac) {
    g_pr = &pr; g_cut = g_kids = 0;
    const Result r = run_kind(kind, trials, noise);
    g_pr = nullptr;
    *cut_frac = g_kids > 0 ? g_cut / g_kids : 0.0;
    return r;
}

TEST(coupling_with_confidence_pruning_emits_the_target_law) {
    const int trials = 300000;
    /* thresholds that cut some of the toy's children and keep others */
    const Prune a{ 0.12, 0.0 }, b{ 0.0, 0.17 }, ab{ 0.07, 0.4 };
    double cf;
    const Result s1a = run_pruned(STAGE1, a, trials, 1.0, &cf);
    std::printf("  stage1 abs.12   z=%.2f first=%.2f acc=%.3f cut=%.3f\n", s1a.z, s1a.first_z, s1a.mean_acc, cf);
    CHECK(s1a.z < 4.5); CHECK(s1a.first_z < 4.5); CHECK(cf > 0.1 && cf < 0.9);
    const Result s1b = run_pruned(STAGE1, b, trials, 1.0, &cf);
    std::printf("  stage1 rel.17  z=%.2f first=%.2f acc=%.3f cut=%.3f\n", s1b.z, s1b.first_z, s1b.mean_acc, cf);
    CHECK(s1b.z < 4.5); CHECK(s1b.first_z < 4.5); CHECK(cf > 0.1 && cf < 0.9);
    const Result s1h = run_pruned(STAGE1, Prune{ 0.99, 1.0 }, trials / 2, 1.0, &cf);
    CHECK(s1h.z < 4.5); CHECK(s1h.first_z < 4.5); CHECK(cf > 0.99);   /* the hard chain: all leaves cut */
    const Result s2a = run_pruned(STAGE2, ab, trials, 1.0, &cf);
    std::printf("  stage2 abs.07 rel.4 z=%.2f first=%.2f acc=%.3f cut=%.3f\n", s2a.z, s2a.first_z, s2a.mean_acc, cf);
    CHECK(s2a.z < 4.5); CHECK(s2a.first_z < 4.5); CHECK(cf > 0.05 && cf < 0.95);
    const Result s2b = run_pruned(STAGE2B, Prune{ 0.2, 0.17 }, trials / 2, 1.0, &cf);
    std::printf("  stage2b abs.2 rel.17 z=%.2f first=%.2f acc=%.3f cut=%.3f\n", s2b.z, s2b.first_z, s2b.mean_acc, cf);
    CHECK(s2b.z < 4.5); CHECK(s2b.first_z < 4.5); CHECK(cf > 0.01 && cf < 0.95);
    const Result s2h = run_pruned(STAGE2, Prune{ 0.99, 1.0 }, trials / 2, 1.0, &cf);
    CHECK(s2h.z < 4.5); CHECK(s2h.first_z < 4.5);
    /* and a far head: the law is the target's whatever the head says */
    const Result far = run_pruned(STAGE2, ab, 150000, 3.0, &cf);
    CHECK(far.z < 4.5); CHECK(far.first_z < 4.5);
}

TEST(pruning_that_reads_the_drawn_childs_own_q_is_not_exact) {
    const int trials = 300000;
    Prune v; v.value_dep = true; v.vd_thr = 0.09;
    double cf;
    const Result s1 = run_pruned(STAGE1, v, trials, 1.0, &cf);
    std::printf("  value-dependent stage1: z=%.1f first=%.1f cut=%.3f (known wrong)\n", s1.z, s1.first_z, cf);
    CHECK(cf > 0.05);
    CHECK(std::max(s1.z, s1.first_z) > 20.0);
    const Result s2 = run_pruned(STAGE2, v, trials, 1.0, &cf);
    std::printf("  value-dependent stage2: z=%.1f first=%.1f cut=%.3f (known wrong)\n", s2.z, s2.first_z, cf);
    CHECK(cf > 0.05);
    CHECK(std::max(s2.z, s2.first_z) > 20.0);
}

/* The slot rule itself: a prefix, a function of the multiset of q, monotone in the thresholds. */
TEST(the_slot_pruning_rule_reads_q_ranks_only) {
    for (int s = 0; s < 400; ++s) {
        const Law L = law_at(5000 + (uint64_t)s, 1.0);
        RadMtpCoupleState st{};
        CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st));
        RadMtpCoupleState sh = st;
        /* any permutation of the support (what a draw does) gives the same count */
        for (int i = 7; i > 0; --i) {
            const int j = (int)(H((uint64_t)s * 31 + (uint64_t)i) % (uint64_t)(i + 1));
            std::swap(sh.ids[i], sh.ids[j]); std::swap(sh.q[i], sh.q[j]);
        }
        RadMtpTreeCfg off{}, a{}, r{}, hard{};
        rad_mtp_tree_set_prune(&a, 0.1, 0.0);
        rad_mtp_tree_set_prune(&r, 0.0, 0.5);
        rad_mtp_tree_set_prune(&hard, 0.999, 1.0);
        CHECK_EQ(rad_mtp_rrs_slot_keep_count(st, &off, 4), 4);
        CHECK_EQ(rad_mtp_rrs_slot_keep_count(st, &hard, 4), 1);
        for (const RadMtpTreeCfg* c : { &a, &r }) {
            const int m = rad_mtp_rrs_slot_keep_count(st, c, 4);
            CHECK_EQ(m, rad_mtp_rrs_slot_keep_count(sh, c, 4));
            CHECK(m >= 1 && m <= 4);
            /* the first dropped slot's priority is below the rule, every kept one meets it */
            float ov[8];
            rad_mtp_rrs_slot_logq(st, ov);
            for (int j = 1; j < m; ++j) {
                if (c == &a) CHECK(std::exp((double)ov[j]) >= 0.1 * 0.999);
                else CHECK(std::exp((double)(ov[j] - ov[0])) >= 0.5 * 0.999);
            }
        }
        /* a drawn node: the kept children are the first m of the draft order */
        RadMtpCoupleState sd = st;
        sd.draw = make_draw(77 + (uint64_t)s, 0);
        const int m = rad_mtp_rrs_slot_keep_count(sd, &a, 4);
        rad_mtp_rrs_draw(&sd, 0, m);
        CHECK_EQ(sd.nkids, m);
    }
}

/* A whole generation: steps drafted at the committed prefix (base = the output length so far, the
 * engine's), each walked as the core walks it, until `want` tokens are out. The joint law of tokens
 * [1, 4) -- spanning the step boundaries, with token 0 and everything after marginalised -- and of
 * tokens [0, 3) must be the target's: this is what a draft or verify stream reused across steps
 * would break. */
Result run_steps(Kind kind, int trials, double noise) {
    const int want = 6;
    std::vector<double> cnt_a((size_t)V * V * V, 0), cnt_b((size_t)V * V * V, 0);
    double steps = 0;
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x7654321ull + (uint64_t)s * 104729ull;
        int out[64], len = 0;
        uint64_t h = 1;
        while (len < want) {
            Tree t;
            if (kind == CHAIN)       draft_chain(&t, seed, (uint64_t)len, 3, 0, noise, h);
            else if (kind == STAGE1) draft_chain(&t, seed, (uint64_t)len, 3, 3, noise, h);
            else if (kind == STAGE2) draft_grow(&t, seed, (uint64_t)len, 2, 3, 7, noise, h);
            else                     draft_grow(&t, seed, (uint64_t)len, 3, 4, 12, noise, h);
            int step_out[64], accepted = 0;
            const int n = walk(t, seed, (uint64_t)len, noise, 0, step_out, &accepted);
            for (int i = 0; i < n; ++i) { out[len++] = step_out[i]; h = child_hash(h, step_out[i]); }
            ++steps;
        }
        ++cnt_a[(size_t)((out[0] * V + out[1]) * V + out[2])];
        ++cnt_b[(size_t)((out[1] * V + out[2]) * V + out[3])];
    }
    /* expected: token 0 marginalised for (1,2,3) */
    std::vector<double> ea((size_t)V * V * V, 0), eb((size_t)V * V * V, 0);
    const uint64_t h0 = 1;
    const Law l0 = law_at(h0, noise);
    for (int a = 0; a < V; ++a) {
        const uint64_t ha = child_hash(h0, a);
        const Law l1 = law_at(ha, noise);
        for (int b = 0; b < V; ++b) {
            const uint64_t hb = child_hash(ha, b);
            const Law l2 = law_at(hb, noise);
            for (int c = 0; c < V; ++c) {
                const uint64_t hc = child_hash(hb, c);
                const Law l3 = law_at(hc, noise);
                ea[(size_t)((a * V + b) * V + c)] = l0.p[a] * l1.p[b] * l2.p[c];
                for (int d = 0; d < V; ++d)
                    eb[(size_t)((b * V + c) * V + d)] += l0.p[a] * l1.p[b] * l2.p[c] * l3.p[d];
            }
        }
    }
    return { zscore(cnt_a, ea, trials), steps / trials, zscore(cnt_b, eb, trials) };
}

TEST(a_generation_over_several_steps_keeps_the_target_law) {
    const int trials = 150000;
    const Kind kinds[] = { CHAIN, STAGE1, STAGE2, STAGE2B };
    const char* names[] = { "chain", "stage1", "stage2", "stage2b" };
    for (int k = 0; k < 4; ++k) {
        const Result r = run_steps(kinds[k], trials, 1.0);
        std::printf("  %-8s steps=%.3f  z[0..3)=%.2f  z[1..4)=%.2f\n", names[k], r.mean_acc, r.z, r.first_z);
        CHECK(r.z < 4.5);
        CHECK(r.first_z < 4.5);
    }
}

/* Which number of children is allowed (DESIGN-coupling.md 3.2). Three children are drawn; the node
 * keeps a prefix. A count that depends on the values drawn BEFORE a child (here x_1) is exact; one
 * that depends on the value of the child it admits (here: the second child is dropped, with the
 * rest, when it is even) is not, and the same statistic sees it. */
static double count_rule_z(int rule, int pair, int trials) {
    const Law L = law_at(2000 + (uint64_t)pair, 1.0 + 0.3 * pair);
    std::vector<double> cnt((size_t)V, 0);
    for (int s = 0; s < trials; ++s) {
        RadMtpCoupleState st{};
        rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st);
        st.draw = make_draw(0x9e3779b97f4a7c15ull * (uint64_t)(s + 1), (uint64_t)s);
        st.status = 1; st.round = 1; st.pos = 5;
        rad_mtp_rrs_draw(&st, 0, 4);
        int keep = 4;
        if (rule == 0) keep = 1 + (st.ids[0] % 3);                 /* a function of x_1 only */
        else if (rule == 1) keep = (st.ids[1] % 2 == 0) ? 1 : 4;   /* reads the child it admits */
        else if (rule == 2) keep = (st.ids[0] % 2 == 0) ? 2 : 4;   /* x_1 only */
        st.nkids = keep < st.count ? keep : st.count;
        const int y = verify(st, L, st.draw.seed, 5, -1);
        ++cnt[(size_t)y];
    }
    std::vector<double> e(L.p, L.p + V);
    return zscore(cnt, e, trials);
}

TEST(a_child_count_may_read_earlier_children_but_not_the_one_it_admits) {
    for (int pair = 0; pair < 4; ++pair) {
        CHECK(count_rule_z(0, pair, 300000) < 4.5);
        CHECK(count_rule_z(2, pair, 300000) < 4.5);
    }
    double worst = 0;
    for (int pair = 0; pair < 4; ++pair) worst = std::max(worst, count_rule_z(1, pair, 300000));
    std::printf("  value-dependent count (known wrong): worst z=%.1f\n", worst);
    CHECK(worst > 20.0);
}

/* Q == P: the first child is accepted whatever the seed; disjoint supports: nothing drafted is
 * accepted and the token is a draw from P. */
TEST(degenerate_laws) {
    const Law L = law_at(77, 1.0);
    /* P exactly the head's law: a plane equal to the shortlist */
    Law same = L;
    same.pn = 0;
    for (int i = 0; i < 8; ++i) { same.pid[same.pn] = L.qid[i]; same.pval[same.pn++] = L.qval[i]; }
    for (int s = 0; s < 2000; ++s) {
        RadMtpCoupleState st{};
        CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st));
        st.draw = make_draw(31ull * (uint64_t)(s + 1), 0);
        st.status = 1; st.round = 1; st.pos = 0;
        rad_mtp_rrs_draw(&st, 0, 3);
        CHECK_EQ(verify(st, same, st.draw.seed, 0, -7), st.ids[0]);
    }
    /* disjoint: the target lives on tokens the head never reports */
    Law dis{};
    dis.pn = 2; dis.pid[0] = 20; dis.pid[1] = 21; dis.pval[0] = 0.3f; dis.pval[1] = -0.4f;
    const double p20 = 1.0 / (1.0 + std::exp(-0.7));
    int n20 = 0;
    const int trials = 100000;
    for (int s = 0; s < trials; ++s) {
        RadMtpCoupleState st{};
        CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st));
        st.draw = make_draw(77ull * (uint64_t)(s + 1), 0);
        st.status = 1; st.round = 1; st.pos = 3;
        rad_mtp_rrs_draw(&st, 0, 4);
        const int y = verify(st, dis, st.draw.seed, 3, -7);
        CHECK(y == 20 || y == 21);
        n20 += y == 20;
    }
    CHECK(std::fabs((double)n20 / trials - p20) < 5 * std::sqrt(p20 * (1 - p20) / trials));
}

/* The draft order is a function of the draw alone: reproducible, a permutation of the support with
 * q carried along, the first child the head's trunk token, and the propose row leaves exactly the
 * drawn columns (holes after them) so the core's leaf capture sees them. */
TEST(the_draft_order_is_reproducible_and_the_wire_is_consistent) {
    const Law L = law_at(5, 1.0);
    RadMtpCoupleState a{}, b{};
    CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &a));
    b = a;
    a.draw = b.draw = make_draw(42, 7);
    a.status = b.status = 1;
    a.round = b.round = 1;
    rad_mtp_rrs_draw(&a, 0, 4);
    rad_mtp_rrs_draw(&b, 0, 4);
    CHECK(std::memcmp(&a, &b, sizeof a) == 0);
    double sum = 0;
    int seen = 0;
    int want = 0;
    for (int i = 0; i < 8; ++i) want |= 1 << L.qid[i];
    for (int i = 0; i < a.count; ++i) { sum += a.q[i]; seen |= 1 << a.ids[i]; }
    CHECK(std::fabs(sum - 1) < 1e-12);
    CHECK_EQ(seen, want);
    CHECK_EQ(a.x, a.ids[0]);
    CHECK_EQ(a.nkids, 4);
    CHECK(rad_mtp_couple_state_valid(a, 1000));
    /* the draw depends on the node and on the base */
    RadMtpCoupleState c = a;
    c.draw.base = 8;
    int o0[8], o1[8];
    rad_mtp_rrs_order(a, 0, o0);
    rad_mtp_rrs_order(c, 0, o1);
    bool differs = false;
    for (int i = 0; i < a.count; ++i) differs = differs || o0[i] != o1[i];
    rad_mtp_rrs_order(a, 1, o1);
    bool differs_node = false;
    for (int i = 0; i < a.count; ++i) differs_node = differs_node || o0[i] != o1[i];
    CHECK(differs || differs_node);
    /* the chain's rounds keep the legacy draw: leaves = 0 is today's inverse-CDF state */
    Tree t0, t1;
    draft_chain(&t0, 99, 4, 3, 0, 1.0);
    draft_chain(&t1, 99, 4, 3, 0, 1.0);
    for (int k = 0; k < 3; ++k) {
        CHECK_EQ(t0.st[k].nkids, 0);
        CHECK(std::memcmp(&t0.st[k], &t1.st[k], sizeof(RadMtpCoupleState)) == 0);
        CHECK_EQ(t0.st[k].x, t0.tok[k + 1]);
    }
    /* stage 1: leaves are the next drawn children, in draft order, never the trunk token */
    draft_chain(&t1, 99, 4, 3, 3, 1.0);
    CHECK_EQ(t1.st[0].nkids, 4);
    CHECK_EQ(t1.st[0].x, t1.tok[1]);
    for (int j = 0; j < 3; ++j) CHECK_EQ(t1.tok[4 + j], t1.st[0].ids[1 + j]);
    CHECK_EQ(t1.st[1].nkids, 0);
}

/* The coupled grower: an uncoupled draw (mode 0 or another card) leaves the pass exactly as
 * before; a coupled one creates each node's children as a prefix of its draft order. */
TEST(the_coupled_grower_creates_children_as_a_draft_order_prefix) {
    int multi = 0;
    for (int s = 0; s < 400; ++s) {
        Tree t;
        draft_grow(&t, 1000ull + (uint64_t)s, (uint64_t)s, 3, 3, 9, 1.0);
        for (int k = 0; k < 9; ++k) {
            int kids[16], nk = 0;
            for (int j = k + 1; j <= t.n; ++j)
                if (t.par[j] == k) kids[nk++] = t.tok[j];
            if (!nk) continue;
            CHECK_EQ(t.st[k].status, 1);
            CHECK(rad_mtp_couple_state_valid(t.st[k], 1000));
            /* a dry copy under the root is a repeat, not a drawn child */
            int distinct = 0;
            for (int q = 0; q < nk; ++q) {
                bool dup = false;
                for (int r = 0; r < q; ++r) dup = dup || kids[r] == kids[q];
                distinct += !dup;
            }
            CHECK_EQ(t.st[k].nkids, distinct);
            for (int q = 0; q < distinct; ++q) CHECK_EQ(t.st[k].ids[q], kids[q]);
            if (k > 0 && distinct >= 2) ++multi;
        }
    }
    CHECK(multi > 20);   /* the grower does give deeper nodes several children */
    /* uncoupled: the plain grower's output, word for word */
    RadMtpTreeCfg cfg{};
    cfg.rounds = 3; cfg.nodes = 7; cfg.width = 2; cfg.pitch = 7; cfg.round = 1;
    std::vector<int> sa((size_t)rad_mtp_tree_state_words(7, 2, 3), 0), sb = sa;
    int d[8] = { 0, 1, 2, 3, 4, 5, -1, -1 };
    float v[8] = { 0.5f, 0.1f, -0.3f, -0.8f, -1.2f, -2.f, -INFINITY, -INFINITY };
    int ta[7], pa[7], tb[7], pb[7];
    RadMtpCoupleDraw draw = make_draw(1, 2);
    draw.mode = 0;
    RadMtpCoupleState plane[7] = {};
    RadMtpTreeCpl cpl{};
    cpl.draw = reinterpret_cast<const uint32_t*>(&draw);
    cpl.state = reinterpret_cast<uint32_t*>(plane);
    cpl.st_node = RAD_MTP_COUPLE_STATE_WORDS; cpl.rows = 7; cpl.vocab = 1000;
    CHECK_EQ(rad_mtp_tree_grow_seq_cpl(&cfg, sa.data(), d, v, 0, 8, 1, 0, nullptr, ta, pa, nullptr,
                                       nullptr, nullptr, nullptr, 0, nullptr, nullptr, &cpl), 0);
    CHECK_EQ(rad_mtp_tree_grow_seq(&cfg, sb.data(), d, v, 0, 8, 1, 0, nullptr, tb, pb, nullptr,
                                   nullptr, nullptr, nullptr, 0, nullptr, nullptr), 0);
    CHECK(sa == sb);
    CHECK(plane[0].status == 0);
}

RAD_TEST_MAIN()

/* Dynamic rows (DESIGN-dynrows.md): the verify carries only the first n nodes of the drafted tree
 * (a prediction from earlier steps, the chain's form, an inert tail cut). A suffix of a tree in node
 * order is a tree, and a node's children it keeps are a prefix of its draft order, which the sampler
 * keeps by rad_mtp_rrs_prefix (nkids shrinks to what the verify still carries). Exact whatever n is,
 * as long as n does not read the values this step drew. */
TEST(a_verify_that_carries_a_prefix_of_the_tree_emits_the_target_law) {
    const int trials = 300000;
    const std::function<void(Tree*, int)> chain_form = [](Tree* t, int) { t->n = 3; };   /* R: drop the leaves */
    const std::function<void(Tree*, int)> fixed = [](Tree* t, int) { t->n = std::min(t->n, 4); };
    const std::function<void(Tree*, int)> by_hist = [](Tree* t, int s) {
        /* a count that is a function of the history (here the trial's own hash, not of this step's
         * draws): anything from the anchor alone to the whole tree */
        const int n = 1 + (int)(H((uint64_t)s * 977 + 13) % (uint64_t)t->n);
        t->n = std::min(t->n, n);
    };
    const struct { const char* name; Kind kind; const std::function<void(Tree*, int)>* hook; } cases[] = {
        { "stage1 chain form", STAGE1, &chain_form }, { "stage1 n<=4", STAGE1, &fixed },
        { "stage2 n<=4", STAGE2, &fixed },             { "stage2 by history", STAGE2, &by_hist },
        { "stage2b by history", STAGE2B, &by_hist },   { "stage2b n<=4", STAGE2B, &fixed },
    };
    for (const auto& c : cases) {
        const Result r = run_kind(c.kind, trials, 1.0, c.hook);
        std::printf("  %-18s z=%.2f first=%.2f acc=%.3f\n", c.name, r.z, r.first_z, r.mean_acc);
        CHECK(r.z < 4.5);
        CHECK(r.first_z < 4.5);
    }
    /* with pruning on too, and a head far from the target */
    double cf;
    g_pr = nullptr;
    const Prune ab{ 0.07, 0.4 };
    g_pr = &ab;
    const Result p2 = run_kind(STAGE2, trials, 1.0, &by_hist);
    const Result p1 = run_kind(STAGE1, trials, 1.0, &fixed);
    g_pr = nullptr;
    (void)cf;
    CHECK(p2.z < 4.5); CHECK(p2.first_z < 4.5);
    CHECK(p1.z < 4.5); CHECK(p1.first_z < 4.5);
    const Result far = run_kind(STAGE2, 150000, 3.0, &by_hist);
    CHECK(far.z < 4.5); CHECK(far.first_z < 4.5);
}

/* The negative control: a cut that reads the value of the child it drops (the first leaf of a
 * stage-1 tree with an even token, and everything after it) is not exact, and the same statistic
 * sees it. */
TEST(a_prefix_cut_that_reads_the_dropped_childs_value_is_not_exact) {
    const std::function<void(Tree*, int)> peek = [](Tree* t, int) {
        for (int j = 4; j <= t->n; ++j)             /* the leaves, behind the chain of 3 */
            if (t->tok[j] % 2 == 0) { t->n = j - 1; return; }
    };
    const Result r = run_kind(STAGE1, 400000, 1.0, &peek);
    std::printf("  value-dependent prefix cut (known wrong): z=%.1f first=%.1f\n", r.z, r.first_z);
    CHECK(std::max(r.z, r.first_z) > 20.0);
}

/* The chain's form over a tree that is not a chain (a step issued ahead predicted a chain, see
 * DESIGN-dynrows.md): the first n nodes in node order are verified as a chain whatever their real
 * parents. The sampler binds a row's coupled state only when its round is depth + 1 (nodes 1..k then
 * really are the chain) and its first drafted child is the next token; every other row takes the
 * ordinary pick. Exact for stages 2 and 2b, whose node order is creation order, not trunk first. */
/* What the core does: the verify is the chain of the first n nodes' tokens, and the sampler binds a
 * row only while the next node is really that row's child (Request::draft_chain, the leading run of
 * the arriving tree). `g_bind_all` is the negative control: bind by token alone. */
static bool g_bind_all = false;
static void chain_form_of(Tree* t) {
    int lead = 0;
    while (lead < t->n && t->par[lead + 1] == lead) ++lead;
    for (int j = 1; j <= t->n; ++j) t->par[j] = j - 1;
    if (!g_bind_all)
        for (int k = lead; k <= t->n; ++k) t->st[k] = RadMtpCoupleState{};
}

TEST(a_chain_form_verify_of_a_tree_that_is_not_a_chain_emits_the_target_law) {
    const int trials = 300000;
    g_bind_all = false;
    auto reparent = [](int n) {
        return std::function<void(Tree*, int)>([n](Tree* t, int) {
            t->n = std::min(t->n, n);
            chain_form_of(t);
        });
    };
    const auto c3 = reparent(3), c5 = reparent(5);
    const std::function<void(Tree*, int)> by_hist = [](Tree* t, int s) {
        const int n = 1 + (int)(H((uint64_t)s * 613 + 5) % (uint64_t)t->n);
        t->n = std::min(t->n, n);
        chain_form_of(t);
    };
    const struct { const char* name; Kind kind; const std::function<void(Tree*, int)>* hook; } cases[] = {
        { "stage2 chain n=3", STAGE2, &c3 },   { "stage2 chain n=5", STAGE2, &c5 },
        { "stage2b chain n=5", STAGE2B, &c5 }, { "stage2b chain by history", STAGE2B, &by_hist },
        { "stage1 chain n=5", STAGE1, &c5 },
    };
    const Prune ab{ 0.07, 0.4 };
    for (int pruned = 0; pruned < 2; ++pruned) {
        g_pr = pruned ? &ab : nullptr;
        for (const auto& c : cases) {
            const Result r = run_kind(c.kind, trials, 1.0, c.hook);
            std::printf("  %-26s pruned=%d z=%.2f first=%.2f acc=%.3f\n", c.name, pruned, r.z,
                        r.first_z, r.mean_acc);
            CHECK(r.z < 4.5);
            CHECK(r.first_z < 4.5);
        }
    }
    g_pr = nullptr;
}

/* Without the lead rule (a chain-form verify binds a row whenever the next token equals the state's
 * first child) the statistic sees the bias: the next token of a sibling can coincide with a child. */
TEST(binding_a_chain_form_row_by_token_alone_is_not_exact) {
    g_bind_all = true;
    const std::function<void(Tree*, int)> c5 = [](Tree* t, int) { t->n = std::min(t->n, 5); chain_form_of(t); };
    const Result r = run_kind(STAGE2, 300000, 1.0, &c5);
    g_bind_all = false;
    std::printf("  chain form bound by token alone (known wrong): z=%.1f\n", r.z);
    CHECK(r.z > 8.0);
}
