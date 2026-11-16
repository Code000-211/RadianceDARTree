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
#include "rad_mtp_traversal.h"
#include "../plugin/spechub/spechub.h"
#include "sched/ngram_tree.h"
#include "rad_mtp_calib.h"

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

double g_sharp = 1.0;   /* the laws' inverse temperature: a peaked target (T = 0.7 and a top-p) is the engine's regime */

Law law_at(uint64_t prefix_hash, double head_noise) {
    Law L;
    float lt[V], lq[V];
    for (int i = 0; i < V; ++i) {
        lt[i] = (float)g_sharp * ((float)((H(prefix_hash * 131 + (uint64_t)i * 977 + 1) >> 40) % 3000) / 1000.0f - 1.5f);
        const float nz = (float)((H(prefix_hash * 257 + (uint64_t)i * 331 + 7) >> 40) % 3000) / 1000.0f - 1.5f;
        lq[i] = 0.75f * lt[i] + (float)(head_noise * g_sharp) * nz;
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
struct Prune { double abs = 0, rel = 0; bool value_dep = false; double vd_thr = 0; RadMtpAdapt ad{}; };
const Prune* g_pr = nullptr;
int g_scheme = 0;   /* sibling scheme of the drafted trees: 0 Gumbel-top-k, m + 1 RheoSampling (--draft-siblings rheo) */
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

/* The draft law recipe under test (--draft-coupling-temp / -truncate): every draw carries it. The
 * default is the legacy law. */
struct LawRecipe { float scale = 0, top_p = 0; int top_k = 0; };
LawRecipe g_recipe;

/* The online word (RadMtpCoupleDraw::dpad_) per trial, cycled: a draft law that changes between
 * steps, as the online controller's does. Empty = none. */
std::vector<uint32_t> g_word_cycle;

RadMtpCoupleDraw make_draw(uint64_t seed, uint64_t base) {
    RadMtpCoupleDraw d{};
    d.seed = seed; d.base = base; d.seq = 1; d.card = 0; d.mode = 1;
    d.dscale = g_recipe.scale; d.dtop_p = g_recipe.top_p; d.dtop_k = g_recipe.top_k;
    if (!g_word_cycle.empty()) d.dpad_ = g_word_cycle[(size_t)(rad_mtp_rrs_mix(seed * 7 + 3) % g_word_cycle.size())];
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
    int      n_model = -1;          /* nodes past it are prompt-lookup grafts (-1: none) */
    int      tok[64] = {}, par[64] = {};
    uint64_t hash[64] = {};         /* the prefix hash of every node, node 0 the root */
    RadMtpCoupleState st[64] = {};  /* row k's coupled state (status 0: none) */
};

enum Kind { CHAIN, STAGE1, STAGE2, STAGE2B };

/* The negative control of the graft: the coupled row sees the lookup's nodes as children too. */
bool g_extras_visible = false;

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
        c.scheme = g_scheme;
        if (g_pr && !g_pr->value_dep) {
            RadMtpTreeCfg pc{};
            rad_mtp_tree_set_prune(&pc, g_pr->abs, g_pr->rel);
            c.prune = pc.prune; c.log_abs = pc.log_abs; c.log_rel = pc.log_rel;
            c.ad = g_pr->ad; c.max_rounds = R;
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
    if (g_pr && !g_pr->value_dep) { rad_mtp_tree_set_prune(&cfg, g_pr->abs, g_pr->rel); cfg.ad = g_pr->ad; }
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
    cpl.scheme = g_scheme;
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
            if (t.n_model >= 0 && j > t.n_model && !g_extras_visible) continue;   /* Request::draft_extra */
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
                if (c.nkids >= 2 || c.scheme) b.nkids = kept;
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


/* ------------------------------------------------------------------ traversal verification */

/* The walk's view of a drafted toy tree: the bound states exactly as `walk` binds them. */
struct TravTree {
    const Tree& t;
    double noise;
    uint64_t seed, n_out;
    Law laws[64];
    RadMtpCoupleState bound[64];
    bool has[64] = {};
    int depth[64] = {};
    TravTree(const Tree& tr, uint64_t sd, uint64_t no, double nz) : t(tr), noise(nz), seed(sd), n_out(no) {
        depth[0] = 0;
        for (int j = 0; j <= t.n; ++j) {
            if (j > 0) depth[j] = depth[t.par[j]] + 1;
            laws[j] = law_at(t.hash[j], noise);
            const RadMtpCoupleState& c = t.st[j];
            int32_t kid[64];
            int nk = 0;
            for (int q = j + 1; q <= t.n; ++q) {
                if (t.par[q] != j) continue;
                bool dup = false;
                for (int z = 0; z < nk; ++z) dup = dup || kid[z] == t.tok[q];
                if (!dup) kid[nk++] = t.tok[q];
            }
            if (nk > 0 && c.status == 1) {
                const int kept = rad_mtp_rrs_prefix(c, kid, nk);
                if (kept > 0 && c.round == depth[j] + 1 && c.x == kid[0]) {
                    bound[j] = c;
                    bound[j].pos = rad_mtp_couple_verify_pos(n_out, depth[j] + 1);
                    if (c.nkids >= 2 || c.scheme) bound[j].nkids = kept;
                    has[j] = true;
                }
            }
        }
    }
    const RadMtpCoupleState* state(int v) const { return has[v] ? &bound[v] : nullptr; }
    bool cand(int v, const int32_t** ids, const float** val, int64_t* n) const {
        *ids = laws[v].pid; *val = laws[v].pval; *n = laws[v].pn;
        return true;
    }
    int child(int v, int32_t token) const {
        for (int j = v + 1; j <= t.n; ++j)
            if (t.par[j] == v && t.tok[j] == token) return j;
        return -1;
    }
    int64_t vocab() const { return 1000; }
};

/* One step walked by traversal verification, then continued from the exact target. */
int walk_trav(const Tree& t, uint64_t seed, uint64_t n_out, double noise, int want, int* out,
              int* accepted, RadMtpTravOpt opt) {
    TravTree T(t, seed, n_out, noise);
    RadMtpTravOut o;
    int len = 0;
    *accepted = 0;
    if (rad_mtp_trav_walk(T, 0, seed, n_out, &o, opt)) {
        for (int i = 1; i < o.n; ++i) out[len++] = t.tok[o.node[i]];
        *accepted = o.n - 1;
        const int last = o.node[o.n - 1];
        out[len++] = o.has_y ? o.y : sample_law(T.laws[last], seed, n_out + (uint64_t)(o.n - 1), 0);
    } else {
        out[len++] = sample_law(T.laws[0], seed, n_out, 0);
    }
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

Result run_trav(Kind kind, int trials, double noise, RadMtpTravOpt opt) {
    std::vector<double> cnt3((size_t)V * V * V, 0), cnt1((size_t)V, 0);
    double acc = 0;
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x1234567ull + (uint64_t)s * 7919ull;
        Tree t;
        if (kind == CHAIN)  draft_chain(&t, seed, 0, 3, 0, noise);
        else if (kind == STAGE1) draft_chain(&t, seed, 0, 3, 3, noise);
        else if (kind == STAGE2) draft_grow(&t, seed, 0, 2, 3, 7, noise);
        else                draft_grow(&t, seed, 0, 3, 4, 12, noise);
        int out[64], accepted = 0;
        walk_trav(t, seed, 0, noise, 3, out, &accepted, opt);
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

/* The first FOUR emitted tokens' joint: the fourth is the final token of a three-deep accepted path,
 * drawn from the walk's residual law, which the three-token statistic never reads. */
double run_trav4(Kind kind, int trials, double noise) {
    std::vector<double> cnt((size_t)V * V * V * V, 0), ex((size_t)V * V * V * V, 0);
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x2468ace1ull + (uint64_t)s * 6151ull;
        Tree t;
        if (kind == CHAIN)  draft_chain(&t, seed, 0, 3, 0, noise);
        else if (kind == STAGE1) draft_chain(&t, seed, 0, 3, 3, noise);
        else if (kind == STAGE2) draft_grow(&t, seed, 0, 2, 3, 7, noise);
        else                draft_grow(&t, seed, 0, 3, 4, 12, noise);
        int out[64], accepted = 0;
        walk_trav(t, seed, 0, noise, 4, out, &accepted, RadMtpTravOpt{});
        ++cnt[(size_t)(((out[0] * V + out[1]) * V + out[2]) * V + out[3])];
    }
    const uint64_t h0 = 1;
    const Law l0 = law_at(h0, noise);
    for (int a = 0; a < V; ++a) {
        const uint64_t ha = child_hash(h0, a); const Law l1 = law_at(ha, noise);
        for (int b = 0; b < V; ++b) {
            const uint64_t hb = child_hash(ha, b); const Law l2 = law_at(hb, noise);
            for (int c = 0; c < V; ++c) {
                const Law l3 = law_at(child_hash(hb, c), noise);
                for (int d = 0; d < V; ++d)
                    ex[(size_t)(((a * V + b) * V + c) * V + d)] = l0.p[a] * l1.p[b] * l2.p[c] * l3.p[d];
            }
        }
    }
    return zscore(cnt, ex, trials);
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
Result run_steps(Kind kind, int trials, double noise, bool trav = false) {
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
            const int n = trav ? walk_trav(t, seed, (uint64_t)len, noise, 0, step_out, &accepted, RadMtpTravOpt{})
                               : walk(t, seed, (uint64_t)len, noise, 0, step_out, &accepted);
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

/* ==================================================================== --draft-adaptive
 * Cost-aware node selection and calibration (abi/rad_mtp_adapt.h, docs/DESIGN-adaptive.md). */

static RadMtpAdapt make_adapt(double margin, const char* extra = "") {
    RadMtpAdapt a;
    char buf[256];
    std::snprintf(buf, sizeof buf, "on=1;margin=%g;%s", margin, extra);
    CHECK(rad_mtp_adapt_parse(buf, &a));
    return a;
}

TEST(the_adaptive_cost_model_and_admission_rule) {
    RadMtpAdapt a;
    CHECK(rad_mtp_adapt_parse("", &a));
    CHECK(!a.on);
    CHECK(rad_mtp_adapt_admit(&a, 5, 3.0f, 1e-9f));   /* off admits everything */
    /* the profile default: 12.3 ms at 3 nodes (4 rows), +1.0 a node */
    CHECK(rad_mtp_adapt_parse("on=1", &a));
    CHECK(std::fabs(rad_mtp_adapt_cost(&a, 3) - 12.3f) < 1e-5f);
    CHECK(std::fabs(rad_mtp_adapt_cost(&a, 7) - 16.3f) < 1e-5f);
    /* a table: entries are the cost of 0,1,2.. nodes; a dip is raised, the tail extrapolates by the last slope */
    CHECK(rad_mtp_adapt_parse("on=1;cost=8.4,9.4,10.0,9.0,12.0", &a));
    CHECK_EQ(a.ncost, 5);
    CHECK(std::fabs(rad_mtp_adapt_cost(&a, 0) - 8.4f) < 1e-5f);
    CHECK(std::fabs(rad_mtp_adapt_cost(&a, 3) - 10.0f) < 1e-5f);
    CHECK(std::fabs(rad_mtp_adapt_cost(&a, 4) - 12.0f) < 1e-5f);
    CHECK(std::fabs(rad_mtp_adapt_cost(&a, 6) - 16.0f) < 1e-5f);
    for (int k = 0; k < 12; ++k) CHECK(rad_mtp_adapt_cost(&a, k + 1) >= rad_mtp_adapt_cost(&a, k));
    CHECK(!rad_mtp_adapt_parse("on=1;cost=1,x", &a));
    /* non-finite numbers and lists longer than the tables are refused, not silently cut */
    CHECK(!rad_mtp_adapt_parse("on=1;cost=1,nan", &a));
    CHECK(!rad_mtp_adapt_parse("on=1;cost=1,inf", &a));
    CHECK(!rad_mtp_adapt_parse("on=1;temp=nan", &a));
    CHECK(!rad_mtp_adapt_parse("on=1;cost=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25", &a));
    CHECK(rad_mtp_adapt_parse("on=1;cost=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24", &a));
    /* the rule is the ratio test: p * T(k) > margin * E * dC */
    CHECK(rad_mtp_adapt_parse("on=1;floor=1;draft=0;base=10;row=1;basek=0", &a));
    /* k=2, E=2: T=12, dC=1: break-even p = E*dC/T = 1/6 */
    CHECK(rad_mtp_adapt_admit(&a, 2, 2.0f, 0.17f));
    CHECK(!rad_mtp_adapt_admit(&a, 2, 2.0f, 0.16f));
    /* the floor is unconditional */
    CHECK(rad_mtp_adapt_parse("on=1;floor=3;draft=0;base=10;row=1;basek=0", &a));
    CHECK(rad_mtp_adapt_admit(&a, 2, 2.0f, 1e-6f));
    CHECK(!rad_mtp_adapt_admit(&a, 3, 2.0f, 1e-6f));
    /* a stricter margin admits a subset, and a likelier node never loses admission */
    const RadMtpAdapt lo = make_adapt(0.5), hi = make_adapt(2.0);
    for (int k = 1; k < 10; ++k)
        for (float p = 0.01f; p < 1.0f; p += 0.01f) {
            if (rad_mtp_adapt_admit(&hi, k, 1.0f + 0.3f * k, p))
                CHECK(rad_mtp_adapt_admit(&lo, k, 1.0f + 0.3f * k, p));
            if (rad_mtp_adapt_admit(&lo, k, 1.0f + 0.3f * k, p))
                CHECK(rad_mtp_adapt_admit(&lo, k, 1.0f + 0.3f * k, p + 0.005f));
        }
    /* a flat cost (a verify that does not grow with rows) admits whatever has mass */
    CHECK(rad_mtp_adapt_parse("on=1;cost=10,10,10,10,10,10;row=0", &a));
    CHECK(rad_mtp_adapt_admit(&a, 2, 3.0f, 0.01f));
}

TEST(the_calibration_is_a_monotone_map_of_the_row) {
    float lp[8], raw[8];
    double z = 0;
    const double w[8] = { 5, 3, 2, 1, 0.5, 0.2, 0.1, 0.05 };
    for (int i = 0; i < 8; ++i) z += w[i];
    for (int i = 0; i < 8; ++i) raw[i] = lp[i] = (float)std::log(w[i] / z);
    RadMtpAdapt a;
    CHECK(rad_mtp_adapt_parse("on=1;temp=1.5,1,0.8", &a));
    CHECK_EQ(a.ntemp, 3);
    CHECK(rad_mtp_adapt_temp(&a, 1) == 1.5f && rad_mtp_adapt_temp(&a, 2) == 1.0f);
    CHECK(rad_mtp_adapt_temp(&a, 3) == 0.8f && rad_mtp_adapt_temp(&a, 9) == 0.8f);   /* the last repeats */
    rad_mtp_adapt_recal(lp, 8, 1.5f);
    double s = 0;
    for (int i = 0; i < 8; ++i) { s += std::exp((double)lp[i]); if (i) CHECK(lp[i] <= lp[i - 1]); }
    CHECK(std::fabs(s - 1.0) < 1e-5);
    CHECK(std::exp((double)lp[0]) < std::exp((double)raw[0]));   /* T > 1 flattens the top */
    rad_mtp_adapt_recal(lp, 8, 1.0f / 1.5f);                     /* and back */
    for (int i = 0; i < 8; ++i) CHECK(std::fabs(lp[i] - raw[i]) < 1e-4f);
    RadMtpAdapt off;
    CHECK(rad_mtp_adapt_parse("", &off));
    CHECK(rad_mtp_adapt_temp(&off, 1) == 1.0f);
}

/* The online fit: a head that predicts 0.8 for tokens accepted 50% of the time is overconfident and
 * its temperature rises; an underconfident depth falls; a calibrated one stays. */
TEST(the_online_calibrator_follows_over_and_underconfidence) {
    RadMtpCalib c;
    uint64_t r = 12345;
    auto u = [&] { r = H(r); return (double)(r >> 11) / 9007199254740992.0; };
    for (int round = 0; round < 40; ++round) {
        for (int i = 0; i < 400; ++i) {
            c.observe(1, 0.8, u() < 0.5);    /* overconfident */
            c.observe(2, 0.3, u() < 0.6);    /* underconfident */
            c.observe(3, 0.5, u() < 0.5);    /* calibrated */
        }
        c.refit();
    }
    CHECK(c.temp[0] > 1.3f);
    CHECK(c.temp[1] < 0.8f);
    CHECK(std::fabs(c.temp[2] - 1.0f) < 0.2f);
    CHECK(c.temp[0] <= 3.0f && c.temp[1] >= 0.5f);
    CHECK(c.temp[5] == 1.0f);   /* an unobserved depth keeps its temperature */
}

/* Grow a tree over a fixed head with a given cfg (uncoupled); the grower's state for inspection. */
struct Grown { int active = 0, depth = 0, made = 0; int tok[64] = {}, par[64] = {}; };
static Grown grow_plain(const RadMtpTreeCfg& base, int W, int R, int N) {
    RadMtpTreeCfg cfg = base;
    cfg.rounds = R; cfg.nodes = N; cfg.width = W; cfg.block = 0; cfg.sel_pitch = 0; cfg.pitch = N;
    std::vector<int> st((size_t)rad_mtp_tree_state_words(N, W, R), 0);
    Grown g;
    uint64_t nodehash[64];
    nodehash[0] = 1;
    for (int round = 1; round <= R; ++round) {
        cfg.round = round;
        const int nfront = round == 1 ? 1 : st[2];
        std::vector<int32_t> d((size_t)W * 8, -1);
        std::vector<float> v((size_t)W * 8, -INFINITY);
        for (int w = 0; w < nfront && w < W; ++w) {
            const int f = round == 1 ? 0 : st[4 + w];
            const Law law = law_at(nodehash[f], 1.0);
            for (int i = 0; i < 8; ++i) { d[(size_t)w * 8 + i] = law.qid[i]; v[(size_t)w * 8 + i] = law.qval[i]; }
        }
        CHECK_EQ(rad_mtp_tree_grow_seq(&cfg, st.data(), d.data(), v.data(), 0, 8, 1, 0, nullptr, g.tok, g.par,
                                       nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr), 0);
        const int* nd = st.data() + 4 + W;
        for (int id = 1; id <= st[1]; ++id) nodehash[id] = child_hash(nodehash[nd[6 * id + 1]], nd[6 * id + 0]);
    }
    const int* nd = st.data() + 4 + W;
    g.made = st[1];
    for (int j = 1; j <= N; ++j)
        if (g.tok[j - 1] >= 0) { ++g.active; g.depth = std::max(g.depth, nd[6 * j + 2]); }
    return g;
}

TEST(the_adaptive_grower_is_off_bit_for_bit_and_shrinks_monotonically_in_the_margin) {
    RadMtpTreeCfg off{};
    const int N = 12, W = 3, R = 4;
    CHECK_EQ(grow_plain(off, W, R, N).active, N);                  /* the full budget, as before */
    /* a flat cost keeps everything */
    RadMtpTreeCfg flat{};
    CHECK(rad_mtp_adapt_parse("on=1;cost=10,10,10,10,10,10;row=0;draft=0", &flat.ad));
    CHECK_EQ(grow_plain(flat, W, R, N).active, N);
    int prev = N + 1, strictly = 0;
    for (double m : { 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 100.0 }) {
        RadMtpTreeCfg c{};
        c.ad = make_adapt(m);
        const int act = grow_plain(c, W, R, N).active;
        std::printf("  margin %-6g -> %d active node(s)\n", m, act);
        CHECK(act >= 1 && act <= N);
        CHECK(act <= prev);
        strictly += act < prev;
        prev = act;
    }
    CHECK(strictly >= 2);
    CHECK_EQ(prev, 1);   /* an impossibly strict margin degrades to the trunk's first node */
    /* the floor keeps nodes whatever the rule says */
    RadMtpTreeCfg fl{};
    fl.ad = make_adapt(100.0, "floor=3");
    CHECK_EQ(grow_plain(fl, W, R, N).active, 3);
    /* the chain (width 1): its depth is cut by confidence, never past the rounds */
    RadMtpTreeCfg ch{};
    ch.ad = make_adapt(1.0);
    const Grown cg = grow_plain(ch, 1, 5, 5);
    CHECK(cg.active >= 1 && cg.active <= 5 && cg.depth >= 1 && cg.depth <= cg.active);
    /* a sharper calibration (T < 1) never admits fewer nodes than a flatter one */
    RadMtpTreeCfg sharp{}, soft{};
    sharp.ad = make_adapt(1.0, "temp=0.6");
    soft.ad = make_adapt(1.0, "temp=2.0");
    CHECK(grow_plain(sharp, W, R, N).active >= grow_plain(soft, W, R, N).active);
}

/* The inert tail of an adaptive tree is what pruning's is: token -1, parent 0. */
TEST(an_adaptive_tree_ends_in_inert_entries) {
    RadMtpTreeCfg cfg{};
    cfg.ad = make_adapt(8.0);
    const int N = 10, W = 2, R = 4;
    const Grown g = grow_plain(cfg, W, R, N);
    CHECK(g.made >= 1 && g.made < N);
    for (int j = 1; j <= N; ++j) {
        if (j <= g.made) { CHECK(g.tok[j - 1] >= 0); CHECK(g.par[j - 1] < j); }
        else { CHECK_EQ(g.tok[j - 1], -1); CHECK_EQ(g.par[j - 1], 0); }
    }
}

/* Stage 1's leaves under the rule: a prefix of the descending slots, fewer under a stricter margin,
 * the same for any permutation of the support. */
TEST(the_stage1_leaf_rule_keeps_a_prefix) {
    for (int s = 0; s < 300; ++s) {
        const Law L = law_at(9000 + (uint64_t)s, 1.0);
        RadMtpCoupleState st{};
        CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st));
        RadMtpTreeCfg pc{};
        pc.rounds = 3;
        int prev = 8;
        for (double m : { 0.3, 1.0, 3.0, 10.0 }) {
            pc.ad = make_adapt(m);
            const int k = rad_mtp_rrs_slot_keep_count(st, &pc, 4);
            CHECK(k >= 1 && k <= 4);
            CHECK(k <= prev);
            prev = k;
            RadMtpCoupleState sd = st;
            sd.draw = make_draw(31 + (uint64_t)s, 0);
            rad_mtp_rrs_draw(&sd, 0, k);
            CHECK_EQ(sd.nkids, k);
            RadMtpCoupleState sh = st;
            for (int i = 7; i > 0; --i) {
                const int j = (int)(H((uint64_t)s * 17 + (uint64_t)i) % (uint64_t)(i + 1));
                std::swap(sh.ids[i], sh.ids[j]); std::swap(sh.q[i], sh.q[j]);
            }
            CHECK_EQ(rad_mtp_rrs_slot_keep_count(sh, &pc, 4), k);
        }
    }
}

/* LOSSLESS: coupling with adaptive shapes emits the target's law, in every drafter form, with and
 * without calibration, beside the confidence pruning, and with a far head. */
TEST(coupling_with_adaptive_selection_emits_the_target_law) {
    const int trials = 300000;
    struct Case { const char* name; Kind kind; Prune pr; int trials; double noise; };
    Prune m1, m2, m3, cal, sharp;
    m1.ad = make_adapt(1.0);
    m2.ad = make_adapt(2.5);
    m3.ad = make_adapt(0.6);
    Prune l1, l2;
    l1.ad = make_adapt(0.7);
    l2.ad = make_adapt(0.85, "temp=1.4");
    cal.ad = make_adapt(1.5, "temp=1.6,1.3,1.1");
    sharp.ad = make_adapt(1.0, "temp=0.7");
    Prune both = m2; both.abs = 0.05; both.rel = 0.3;   /* adaptive beside the confidence pruning */
    const Case cases[] = {
        { "stage1 m1",              STAGE1,  m1,    trials,     1.0 },
        { "stage1 m2.5",            STAGE1,  m2,    trials,     1.0 },
        { "stage1 cal",             STAGE1,  cal,   trials,     1.0 },
        { "stage1 m0.7",            STAGE1,  l1,    trials,     1.0 },
        { "stage1 m0.85 + calib",    STAGE1,  l2,    trials,     1.0 },
        { "stage2 m1",              STAGE2,  m1,    trials,     1.0 },
        { "stage2 m2.5",            STAGE2,  m2,    trials,     1.0 },
        { "stage2 m0.6",            STAGE2,  m3,    trials,     1.0 },
        { "stage2 cal",             STAGE2,  cal,   trials,     1.0 },
        { "stage2b m1",             STAGE2B, m1,    trials / 2, 1.0 },
        { "stage2b cal",            STAGE2B, cal,   trials / 2, 1.0 },
        { "stage2 prune+adapt",     STAGE2,  both,  trials,     1.0 },
        { "stage1 m2.5, far head",  STAGE1,  m2,    trials / 2, 3.0 },
        { "stage2 sharp, far head", STAGE2,  sharp, trials / 2, 3.0 },
    };
    for (const Case& c : cases) {
        double cf;
        const Result r = run_pruned(c.kind, c.pr, c.trials, c.noise, &cf);
        std::printf("  %-26s z=%.2f first=%.2f acc=%.3f cut=%.3f\n", c.name, r.z, r.first_z, r.mean_acc, cf);
        CHECK(r.z < 4.5);
        CHECK(r.first_z < 4.5);
    }
    /* the rule actually shapes the trees it is tested on (not a vacuous pass) */
    double cf;
    run_pruned(STAGE2, m2, 20000, 1.0, &cf);
    CHECK(cf > 0.05 && cf < 0.95);
    run_pruned(STAGE1, l1, 20000, 1.0, &cf);
    CHECK(cf > 0.05 && cf < 0.95);
}

/* Adaptive shapes through the dynamic-rows cuts (a suffix of the tree). */
TEST(adaptive_trees_cut_by_dynamic_rows_stay_exact) {
    const int trials = 250000;
    Prune m2;
    m2.ad = make_adapt(2.0);
    const std::function<void(Tree*, int)> c4 = [](Tree* t, int) { t->n = std::min(t->n, 4); };
    g_pr = &m2;
    for (Kind k : { STAGE2, STAGE2B, STAGE1 }) {
        const Result r = run_kind(k, trials, 1.0, &c4);
        std::printf("  kind=%d adaptive + rows cut at 4: z=%.2f first=%.2f acc=%.3f\n", (int)k, r.z,
                    r.first_z, r.mean_acc);
        CHECK(r.z < 4.5);
        CHECK(r.first_z < 4.5);
    }
    g_pr = nullptr;
}

/* ------------------------------------------------------------------ prompt-lookup grafts
 * (core/sched/ngram_tree.h, --draft-ngram-leaves). The graft's proposal is the target's own most
 * likely continuation of the root, damaged at one position by a history hash: a function of the
 * history, never of the coupled draw, as a lookup of the sequence's own text is. Extra nodes are
 * followed by the walk when the row's emitted token is theirs; the coupled row sees only the model's
 * children (n_model), so the emitted law stays the target's whatever the lookup proposes. */
static int argmax_tok(const Law& L) {
    int b = 0;
    for (int i = 1; i < V; ++i) if (L.p[i] > L.p[b]) b = i;
    return b;
}

static void graft(Tree* t, int s, int extra_room, int leaves, int how, double noise) {
    int32_t prop[6];
    uint64_t h = t->hash[0];
    for (int i = 0; i < 4; ++i) {
        const Law L = law_at(h, noise);
        int tk = argmax_tok(L);
        if (how == 1 && i == 0) tk = t->tok[1];                            /* the model's own first node */
        if (how == 2 && i == 0) tk = (int)(H((uint64_t)s * 31 + 7) % V);   /* any token of the toy */
        if (i == 2 && H((uint64_t)s * 977 + 3) % 3 == 0) tk = (tk + 1) % V;
        prop[i] = tk;
        h = child_hash(h, tk);
    }
    rad::NgramTreeCfg c;
    c.leaves = leaves; c.min_match = 4; c.max_nodes = std::min(t->n + extra_room, 60); c.max_depth = 6;
    int32_t tok[64], par[64];
    for (int j = 1; j <= t->n; ++j) { tok[j - 1] = t->tok[j]; par[j - 1] = t->par[j]; }
    /* the real nodes: a pruned grower's inert duplicates (a repeat of node 1 under the root) are free */
    int act = t->n;
    while (act > 1 && t->par[act] == 0 && t->tok[act] == t->tok[1]) --act;
    const rad::NgramTreeOut o = rad::ngram_tree_merge(tok, par, t->n, act, prop, 4, 8, c);
    if (o.added <= 0) return;
    t->n = o.n;
    t->n_model = o.n_model;
    for (int j = 1; j <= t->n; ++j) {
        t->tok[j] = tok[j - 1];
        t->par[j] = par[j - 1];
        t->hash[j] = child_hash(t->hash[t->par[j]], t->tok[j]);
    }
}

TEST(a_tree_with_prompt_lookup_nodes_emits_the_target_law) {
    const int trials = 300000;
    const std::function<void(Tree*, int)> app = [](Tree* t, int s) { graft(t, s, 3, 3, 0, 1.0); };
    const std::function<void(Tree*, int)> swap = [](Tree* t, int s) { graft(t, s, 0, 3, 0, 1.0); };
    const std::function<void(Tree*, int)> own = [](Tree* t, int s) { graft(t, s, 3, 3, 1, 1.0); };
    const std::function<void(Tree*, int)> any = [](Tree* t, int s) { graft(t, s, 2, 2, 2, 1.0); };
    const struct { const char* name; Kind kind; const std::function<void(Tree*, int)>* hook; } cases[] = {
        { "chain + graft", CHAIN, &app },         { "stage1 + graft", STAGE1, &app },
        { "stage1 swap", STAGE1, &swap },         { "stage1 own first", STAGE1, &own },
        { "stage2 + graft", STAGE2, &app },       { "stage2 swap", STAGE2, &swap },
        { "stage2 any token", STAGE2, &any },     { "stage2b swap", STAGE2B, &swap },
        { "stage2b own first", STAGE2B, &own },
    };
    const Prune ab{ 0.07, 0.4 };
    for (int pruned = 0; pruned < 2; ++pruned) {
        g_pr = pruned ? &ab : nullptr;
        for (const auto& c : cases) {
            const Result base = run_kind(c.kind, trials, 1.0, nullptr);
            const Result r = run_kind(c.kind, trials, 1.0, c.hook);
            std::printf("  %-18s pruned=%d z=%.2f first=%.2f acc=%.3f (without %.3f)\n", c.name, pruned,
                        r.z, r.first_z, r.mean_acc, base.mean_acc);
            CHECK(r.z < 4.5);
            CHECK(r.first_z < 4.5);
        }
    }
    g_pr = nullptr;
}


/* Traversal verification (--draft-verify traversal, abi/rad_mtp_traversal.h): the same trees,
 * decided bottom-up per path. Exact for every tree kind; accepts at least as much as the row-wise
 * recursion; each plausible mistake fails the statistic. */
TEST(traversal_verification_emits_the_target_law_and_accepts_more) {
    const int trials = 200000;
    const RadMtpTravOpt ok{};
    const Result chain = run_trav(CHAIN, trials, 1.0, ok);
    const Result s1    = run_trav(STAGE1, trials, 1.0, ok);
    const Result s2    = run_trav(STAGE2, trials, 1.0, ok);
    const Result s2b   = run_trav(STAGE2B, trials, 1.0, ok);
    const Result rchain = run_kind(CHAIN, trials, 1.0);
    const Result rs2    = run_kind(STAGE2, trials, 1.0);
    std::printf("  traversal chain z=%.2f acc=%.3f (rowwise %.3f) | stage1 z=%.2f acc=%.3f | "
                "stage2 z=%.2f acc=%.3f (rowwise %.3f) | stage2b z=%.2f acc=%.3f\n",
                chain.z, chain.mean_acc, rchain.mean_acc, s1.z, s1.mean_acc, s2.z, s2.mean_acc,
                rs2.mean_acc, s2b.z, s2b.mean_acc);
    CHECK(chain.z < 4.5); CHECK(chain.first_z < 4.5);
    CHECK(s1.z < 4.5);    CHECK(s1.first_z < 4.5);
    CHECK(s2.z < 4.5);    CHECK(s2.first_z < 4.5);
    CHECK(s2b.z < 4.5);   CHECK(s2b.first_z < 4.5);
    CHECK(chain.mean_acc > rchain.mean_acc);
    CHECK(s2.mean_acc > rs2.mean_acc);
    const Result far = run_trav(STAGE2, 100000, 3.0, ok);
    CHECK(far.z < 4.5);
}

TEST(traversal_verification_mistakes_are_not_exact) {
    for (int ctl = 1; ctl <= 3; ++ctl) {
        RadMtpTravOpt o; o.ctl = ctl;
        const Result r = run_trav(STAGE2, 100000, 1.0, o);
        std::printf("  negative control %d: z=%.2f first_z=%.2f\n", ctl, r.z, r.first_z);
        CHECK(r.z > 20.0 || r.first_z > 20.0);
    }
}

/* The review's additions: the walk's final token (the fourth of a three-deep path), a generation
 * over several steps (no uniform reused across steps), confidence pruning, and the value-dependent
 * pruning negative control, all through the traversal walk. */
TEST(traversal_verification_final_token_and_steps_are_exact) {
    const Kind kinds[] = { CHAIN, STAGE1, STAGE2, STAGE2B };
    const char* names[] = { "chain", "stage1", "stage2", "stage2b" };
    for (int k = 0; k < 4; ++k) {
        const double z4 = run_trav4(kinds[k], 300000, 1.0);
        const Result r = run_steps(kinds[k], 100000, 1.0, true);
        std::printf("  trav %-8s z4=%.2f steps=%.3f z[0..3)=%.2f z[1..4)=%.2f\n", names[k], z4, r.mean_acc, r.z, r.first_z);
        CHECK(z4 < 4.5);
        CHECK(r.z < 4.5);
        CHECK(r.first_z < 4.5);
    }
}

TEST(traversal_verification_with_confidence_pruning_is_exact) {
    const int trials = 200000;
    const Prune a{ 0.12, 0.0 }, b{ 0.0, 0.17 }, ab{ 0.07, 0.4 };
    struct Case { Kind k; Prune p; };
    const Case cases[] = { { STAGE1, a }, { STAGE1, b }, { STAGE2, ab }, { STAGE2B, Prune{ 0.2, 0.17 } },
                           { STAGE2, Prune{ 0.99, 1.0 } } };
    for (const Case& cs : cases) {
        g_pr = &cs.p; g_cut = g_kids = 0;
        const Result r = run_trav(cs.k, trials, 1.0, RadMtpTravOpt{});
        g_pr = nullptr;
        std::printf("  trav pruned kind=%d z=%.2f first=%.2f acc=%.3f cut=%.3f\n", (int)cs.k, r.z, r.first_z, r.mean_acc, g_kids > 0 ? g_cut / g_kids : 0.0);
        CHECK(r.z < 4.5); CHECK(r.first_z < 4.5);
        CHECK(g_cut > 0);
    }
    Prune v; v.value_dep = true; v.vd_thr = 0.09;
    for (Kind k : { STAGE1, STAGE2 }) {
        g_pr = &v; g_cut = g_kids = 0;
        const Result r = run_trav(k, trials, 1.0, RadMtpTravOpt{});
        g_pr = nullptr;
        std::printf("  trav value-dependent pruning kind=%d z=%.1f first=%.1f (known wrong)\n", (int)k, r.z, r.first_z);
        CHECK(g_cut > 0);
        CHECK(std::max(r.z, r.first_z) > 20.0);
    }
}

/* Peaked laws: the target's inverse temperature 3 (a few tokens carry the mass, the acceptance
 * ratios sit at or above one on the trunk), where a cap or rounding at p = 1 would show. */
TEST(traversal_verification_is_exact_on_peaked_laws) {
    const double old = g_sharp;
    for (double sh : { 3.0, 6.0 }) {
        g_sharp = sh;
        for (Kind k : { CHAIN, STAGE2, STAGE2B }) {
            const Result r = run_trav(k, 150000, 1.0, RadMtpTravOpt{});
            const double z4 = run_trav4(k, 200000, 1.0);
            std::printf("  trav sharp=%.0f kind=%d z=%.2f first=%.2f z4=%.2f acc=%.3f\n", sh, (int)k, r.z, r.first_z, z4, r.mean_acc);
            CHECK(r.z < 4.5); CHECK(r.first_z < 4.5); CHECK(z4 < 4.5);
        }
    }
    g_sharp = old;
}

/* A lookup node under a coupled row whose own child was displaced: the row keeps its bound state
 * (x = the displaced child's draw) and the lookup's one node hangs below it carrying the target's
 * most likely token. Hidden from the coupled row (Request::draft_extra) the verify is exact; read as
 * the row's child, the token that equals x selects the row's coupling by the draw's value and the
 * emitted law is no longer the target's. */
static void pin_under_row2(Tree* t, int) {
    const int d = argmax_tok(law_at(t->hash[2], 1.0));
    t->n = 3;                                   /* nodes 1, 2 and the lookup's: node 3 is replaced */
    t->tok[3] = d; t->par[3] = 2;
    t->hash[3] = child_hash(t->hash[2], d);
    t->n_model = 2;
}

TEST(a_lookup_node_under_a_coupled_row_stays_exact_only_while_hidden_from_it) {
    const std::function<void(Tree*, int)> pin = [](Tree* t, int s) { pin_under_row2(t, s); };
    for (Kind k : { CHAIN, STAGE1 }) {
        g_extras_visible = false;
        const Result ok = run_kind(k, 400000, 1.0, &pin);
        std::printf("  pinned lookup node, hidden, kind %d: z=%.2f first=%.2f\n", (int)k, ok.z, ok.first_z);
        CHECK(ok.z < 4.5);
        CHECK(ok.first_z < 4.5);
    }
    double worst = 0;
    g_extras_visible = true;
    for (Kind k : { CHAIN, STAGE1 }) {
        const Result r = run_kind(k, 400000, 1.0, &pin);
        std::printf("  pinned lookup node, visible to the coupled row (known wrong), kind %d: z=%.1f first=%.1f\n",
                    (int)k, r.z, r.first_z);
        worst = std::max(worst, std::max(r.z, r.first_z));
    }
    g_extras_visible = false;
    CHECK(worst > 8.0);
}

/* ------------------------------------------------------------------ a SuffixDecoding step
 *
 * --draft-suffix replaces the head's chain of a step by the sequence's own history (suffix.h): a
 * chain of up to 8 rows chosen from the committed text alone. The head had drafted (and DRAWN, under
 * --draft-coupling) a chain for the same positions, so the coupled states of that draft are still
 * in the sampler when the step is built. The step is exact because the scheduler marks it
 * draft_extra = 0 (no model node), which hides every proposed row from the coupled child list: no
 * row binds a state, each row's emitted token is the target's plain draw and is followed iff it
 * equals the proposal. Visible to the coupled rows, a proposal that happens to equal the head's
 * unused draw x would select the row's coupling by the VALUE of that draw, which is biased
 * (DESIGN-ngram.md point 3). Same statistic as the graft's: the first three tokens' joint law over
 * 400000 steps, a negative control that must fail. */
static void suffix_chain(Tree* t, int s, int how, int len, double noise) {
    const int orig[4] = { 0, t->tok[1], t->tok[2], t->tok[3] };
    uint64_t h = t->hash[0];
    for (int i = 0; i < len; ++i) {
        int tk = argmax_tok(law_at(h, noise));           /* a history that repeats the likely text */
        if (how == 1 && i < 3) tk = orig[i + 1];         /* the head's own draw, by chance of the text */
        if (H((uint64_t)s * 977 + (uint64_t)i * 31 + 5) % 4 == 0) tk = (tk + 1) % V;   /* a damaged copy */
        t->tok[i + 1] = tk;
        t->par[i + 1] = i;
        h = child_hash(h, tk);
        t->hash[i + 1] = h;
    }
    t->n = len;
    t->n_model = 0;                                      /* Request::draft_extra = 0 */
}

TEST(a_suffix_chain_beside_the_heads_stale_coupled_states_emits_the_target_law) {
    for (int how = 0; how < 2; ++how) {
        for (int len : { 3, 6, 8 }) {
            const std::function<void(Tree*, int)> h = [how, len](Tree* t, int s) { suffix_chain(t, s, how, len, 1.0); };
            g_extras_visible = false;
            const Result ok = run_kind(CHAIN, 400000, 1.0, &h);
            std::printf("  suffix chain, rows %d, how %d, hidden: z=%.2f first=%.2f acc=%.2f\n", len, how,
                        ok.z, ok.first_z, ok.mean_acc);
            CHECK(ok.z < 4.5);
            CHECK(ok.first_z < 4.5);
        }
    }
    /* the negative control: the same chains with the rows visible to the coupled verify */
    double worst = 0;
    g_extras_visible = true;
    for (int how = 0; how < 2; ++how) {
        const std::function<void(Tree*, int)> h = [how](Tree* t, int s) { suffix_chain(t, s, how, 6, 1.0); };
        const Result r = run_kind(CHAIN, 400000, 1.0, &h);
        std::printf("  suffix chain visible to the coupled rows (known wrong), how %d: z=%.1f first=%.1f\n",
                    how, r.z, r.first_z);
        worst = std::max(worst, std::max(r.z, r.first_z));
    }
    g_extras_visible = false;
    CHECK(worst > 8.0);
}

/* A suffix step is the plain rule: the row's draw is the target's and the proposal is followed iff
 * it equals it. A chain whose first row is never the likeliest token still emits the target's law. */
TEST(a_suffix_chain_with_a_wrong_first_row_stays_exact) {
    const std::function<void(Tree*, int)> wrong = [](Tree* t, int s) {
        suffix_chain(t, s, 0, 6, 1.0);
        const Law L0 = law_at(t->hash[0], 1.0);
        t->tok[1] = (argmax_tok(L0) + 1 + (int)(H((uint64_t)s) % (V - 1))) % V;   /* never the likeliest, any other */
        t->hash[1] = child_hash(t->hash[0], t->tok[1]);
        for (int i = 1; i < t->n; ++i) t->hash[i + 1] = child_hash(t->hash[i], t->tok[i + 1]);
    };
    g_extras_visible = false;
    const Result r = run_kind(CHAIN, 300000, 1.0, &wrong);
    std::printf("  suffix chain, first row wrong: z=%.2f first=%.2f\n", r.z, r.first_z);
    CHECK(r.z < 4.5);
    CHECK(r.first_z < 4.5);
}

/* ------------------------------------------------------------------ the draft law's recipe
 *
 * rad_mtp_couple_build_law: the draft temperature (dscale = 1 / T_draft) and the request's top-k /
 * top-p cut change WHICH Q the head drafts from, never the verify's exactness, because the state
 * stores the Q that was drawn from and the verify reads it. */

TEST(a_zero_recipe_is_the_legacy_law_bit_for_bit) {
    for (int s = 0; s < 200; ++s) {
        const Law L = law_at(9000 + (uint64_t)s, 1.0);
        RadMtpCoupleState a{}, b{};
        CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &a));
        RadMtpCoupleDraw d{};
        CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d, &b));
        CHECK(!std::memcmp(a.ids, b.ids, sizeof a.ids) && !std::memcmp(a.q, b.q, sizeof a.q));
        d.dscale = 1.0f;   /* an explicit T 1 is the same */
        RadMtpCoupleState c{};
        CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d, &c));
        CHECK(!std::memcmp(a.q, c.q, sizeof a.q));
    }
}

TEST(the_draft_temperature_scales_the_scores_before_the_softmax) {
    for (int s = 0; s < 200; ++s) {
        const Law L = law_at(9100 + (uint64_t)s, 1.0);
        for (float T : { 0.5f, 0.7f, 1.3f, 2.0f }) {
            RadMtpCoupleDraw d{};
            d.dscale = 1.0f / T;
            RadMtpCoupleState st{};
            CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d, &st));
            CHECK(rad_mtp_couple_state_valid([&] { RadMtpCoupleState v = st; v.draw.mode = 1; v.status = 1;
                                                  v.round = 1; v.x = v.ids[0]; return v; }(), 1000));
            double z = 0, w[8];
            for (int i = 0; i < 8; ++i) { w[i] = std::exp((double)L.qval[i] * (double)d.dscale); z += w[i]; }
            for (int k = 0; k < st.count; ++k) {
                int i = 0;
                while (L.qid[i] != st.ids[k]) ++i;
                CHECK(std::fabs(st.q[k] - w[i] / z) < 1e-6);
            }
            /* a sharper draft law (T < 1) has more mass on its mode */
            RadMtpCoupleState one{};
            rad_mtp_couple_build_q(L.qid, L.qval, 1000, &one);
            double m1 = 0, mT = 0;
            for (int k = 0; k < one.count; ++k) m1 = std::max(m1, one.q[k]);
            for (int k = 0; k < st.count; ++k) mT = std::max(mT, st.q[k]);
            if (T < 1.0f) CHECK(mT >= m1 - 1e-12);
            if (T > 1.0f) CHECK(mT <= m1 + 1e-12);
        }
    }
}

TEST(top_k_and_top_p_cut_the_draft_law_by_value_not_by_position) {
    for (int s = 0; s < 300; ++s) {
        const Law L = law_at(9200 + (uint64_t)s, 1.0);
        for (int k : { 1, 2, 3, 5 }) {
            RadMtpCoupleDraw d{};
            d.dscale = 1.0f / 0.7f; d.dtop_k = k;
            RadMtpCoupleState st{};
            CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d, &st));
            CHECK(st.count <= k && st.count >= 1);
            /* the survivors are the k largest scores */
            float kth = -INFINITY;
            std::vector<float> v(L.qval, L.qval + 8);
            std::sort(v.begin(), v.end(), std::greater<float>());
            kth = v[(size_t)k - 1];
            for (int j = 0; j < st.count; ++j) {
                int i = 0;
                while (L.qid[i] != st.ids[j]) ++i;
                CHECK(L.qval[i] >= kth);
            }
            /* the shortlist's order does not matter */
            int32_t id2[8]; float v2[8];
            for (int i = 0; i < 8; ++i) { id2[i] = L.qid[7 - i]; v2[i] = L.qval[7 - i]; }
            RadMtpCoupleState sh{};
            CHECK(rad_mtp_couple_build_law(id2, v2, 1000, d, &sh));
            CHECK_EQ(sh.count, st.count);
            for (int j = 0; j < st.count; ++j) {
                int m = 0;
                while (m < sh.count && sh.ids[m] != st.ids[j]) ++m;
                CHECK(m < sh.count);
                CHECK(std::fabs(sh.q[m] - st.q[j]) < 1e-12);
            }
        }
        for (float p : { 0.3f, 0.6f, 0.9f, 0.95f }) {
            RadMtpCoupleDraw d{};
            d.dscale = 1.0f / 0.7f; d.dtop_p = p;
            RadMtpCoupleState st{}, full{};
            CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d, &st));
            RadMtpCoupleDraw d0{}; d0.dscale = d.dscale;
            CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d0, &full));
            /* the kept set is the smallest top prefix of the scaled law with mass >= p, renormalised */
            std::vector<double> qs;
            for (int k = 0; k < full.count; ++k) qs.push_back(full.q[k]);
            std::sort(qs.begin(), qs.end(), std::greater<double>());
            double cum = 0;
            int need = 0;
            while (need < (int)qs.size()) { cum += qs[(size_t)need++]; if (cum >= (double)p) break; }
            CHECK_EQ(st.count, need);
            double tot = 0;
            for (int k = 0; k < st.count; ++k) tot += st.q[k];
            CHECK(std::fabs(tot - 1.0) < 1e-12);
        }
    }
    /* non-finite scores are build_q's to refuse, with or without a cut */
    const Law L = law_at(9300, 1.0);
    float bad[8];
    std::memcpy(bad, L.qval, sizeof bad);
    bad[3] = NAN;
    RadMtpCoupleDraw d{};
    d.dscale = 2.0f; d.dtop_k = 2; d.dtop_p = 0.5f;
    RadMtpCoupleState st{};
    CHECK(!rad_mtp_couple_build_law(L.qid, bad, 1000, d, &st));
}

/* THE EXACTNESS HOLDS FOR EVERY RECIPE. A chain, a stage-1 tree and a stage-2 tree drafted from a
 * temperature-scaled and truncated Q (the real propose row and the real coupled grower, the real
 * verify) emit the target's law, for a head near and far from the target. The acceptance moves with
 * the recipe -- that is what the flags tune. */
TEST(every_draft_law_recipe_emits_the_target_law) {
    const struct { const char* name; LawRecipe r; } recipes[] = {
        { "T0.5",           { 2.0f, 0, 0 } },
        { "T0.7",           { 1.0f / 0.7f, 0, 0 } },
        { "T1.5",           { 1.0f / 1.5f, 0, 0 } },
        { "T0.7 k3",        { 1.0f / 0.7f, 0, 3 } },
        { "T1.0 p0.8",      { 1.0f, 0.8f, 0 } },
        { "T0.7 k4 p0.9",   { 1.0f / 0.7f, 0.9f, 4 } },
        { "k1 (a point)",   { 0, 0, 1 } },
    };
    const int trials = 200000;
    for (const auto& rc : recipes) {
        g_recipe = rc.r;
        const Result chain = run_kind(CHAIN, trials, 1.0);
        const Result s1    = run_kind(STAGE1, trials, 1.0);
        const Result s2    = run_kind(STAGE2, trials, 1.0);
        const Result far   = run_kind(STAGE1, trials / 2, 3.0);
        std::printf("  %-14s chain z=%.2f acc=%.3f | stage1 z=%.2f acc=%.3f | stage2 z=%.2f acc=%.3f | far z=%.2f\n",
                    rc.name, chain.z, chain.mean_acc, s1.z, s1.mean_acc, s2.z, s2.mean_acc, far.z);
        CHECK(chain.z < 4.5 && chain.first_z < 4.5);
        CHECK(s1.z < 4.5 && s1.first_z < 4.5);
        CHECK(s2.z < 4.5 && s2.first_z < 4.5);
        CHECK(far.z < 4.5 && far.first_z < 4.5);
    }
    /* with pruning too: the slot rule reads the recipe's Q only */
    g_recipe = { 1.0f / 0.7f, 0.9f, 4 };
    const Prune ab{ 0.07, 0.4 };
    double cf;
    const Result p1 = run_pruned(STAGE1, ab, trials, 1.0, &cf);
    const Result p2 = run_pruned(STAGE2, ab, trials, 1.0, &cf);
    CHECK(p1.z < 4.5 && p1.first_z < 4.5);
    CHECK(p2.z < 4.5 && p2.first_z < 4.5);
    g_recipe = LawRecipe{};
}

/* A draft law that is far too sharp (a point at T -> 0) or flat still verifies exactly, and the
 * acceptance of a chain whose head is the target's own sharpened law is higher drafted at the
 * target's temperature than at 1. The toy target is T 1; its head is 0.75 x the target's scores, so
 * a draft scaled by 1 / 0.75 recovers the target's law and accepts more than the legacy T1. */
TEST(drafting_at_the_matching_temperature_accepts_more) {
    const int trials = 200000;
    g_recipe = LawRecipe{};
    const Result base = run_kind(CHAIN, trials, 0.0);
    g_recipe.scale = 1.0f / 0.75f;
    const Result match = run_kind(CHAIN, trials, 0.0);
    g_recipe = LawRecipe{};
    std::printf("  head = 0.75 x target: legacy acc=%.3f, matched acc=%.3f\n", base.mean_acc, match.mean_acc);
    CHECK(base.z < 4.5 && match.z < 4.5);
    CHECK(match.mean_acc > base.mean_acc);
}

/* NEGATIVE CONTROL for the recipe. The drafts are drawn from the recipe's Q but the verify is handed
 * the LEGACY Q (the T1 softmax of the shortlist) in the state, i.e. a build that tuned the draft law
 * without storing it. That is the one way a recipe can break exactness, and the same chi-square
 * statistic must see it. (A chain: no sibling order to disagree with, so the state binds.) */
TEST(verifying_against_a_q_other_than_the_one_drafted_from_is_not_exact) {
    const std::function<void(Tree*, int)> legacy_q = [](Tree* t, int) {
        for (int k = 0; k <= t->n; ++k) {
            RadMtpCoupleState& s = t->st[k];
            if (s.status != 1) continue;
            const Law L = law_at(t->hash[k], 1.0);
            RadMtpCoupleState leg{};
            if (!rad_mtp_couple_build_q(L.qid, L.qval, 1000, &leg)) continue;
            /* keep the drawn token first, as a bound row requires */
            std::memcpy(s.ids, leg.ids, sizeof s.ids);
            std::memcpy(s.q, leg.q, sizeof s.q);
            s.count = leg.count;
            int at = -1;
            for (int i = 0; i < s.count; ++i) if (s.ids[i] == s.x) at = i;
            if (at > 0) { std::swap(s.ids[0], s.ids[at]); std::swap(s.q[0], s.q[at]); }
        }
    };
    const struct { const char* name; LawRecipe r; } recipes[] = {
        { "T0.5", { 2.0f, 0, 0 } }, { "k1", { 0, 0, 1 } }, { "T0.7 p0.8", { 1.0f / 0.7f, 0.8f, 0 } },
    };
    for (const auto& rc : recipes) {
        g_recipe = rc.r;
        const Result ok  = run_kind(CHAIN, 300000, 1.0);
        const Result bad = run_kind(CHAIN, 300000, 1.0, &legacy_q);
        std::printf("  %-10s stored Q z=%.2f | legacy Q in the state (known wrong) z=%.1f\n", rc.name, ok.z, bad.z);
        CHECK(ok.z < 4.5);
        CHECK(bad.z > 8.0);
    }
    g_recipe = LawRecipe{};
}

/* A recipe never reaches a row that is not coupled: a zero recipe next to a hostile one on the same
 * shortlist, a degenerate scale (inf, NaN, negative) falls back or is refused, never a crash or a Q
 * that sums wrong. */
TEST(hostile_recipe_values_give_a_valid_q_or_a_refusal) {
    const Law L = law_at(9400, 1.0);
    const float scales[] = { INFINITY, -INFINITY, NAN, -1.0f, 0.0f, 1e-30f, 1e30f, 3e38f };
    for (float sc : scales)
        for (float tp : { -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, NAN })
            for (int tk : { -5, 0, 1, 8, 100 }) {
                RadMtpCoupleDraw d{};
                d.dscale = sc; d.dtop_p = tp; d.dtop_k = tk;
                RadMtpCoupleState st{};
                const bool ok = rad_mtp_couple_build_law(L.qid, L.qval, 1000, d, &st);
                if (!ok) { CHECK(st.count == 0); continue; }
                double tot = 0;
                for (int k = 0; k < st.count; ++k) { CHECK(st.q[k] > 0 && st.q[k] <= 1.0); tot += st.q[k]; }
                CHECK(std::fabs(tot - 1.0) < 1e-9);
            }
}
/* ====================================================================================================
 * RheoSampling siblings (--draft-siblings rheo, abi/rad_mtp_rrs.h): the top-m tokens of Q plus ONE
 * drawn from the tail law Qt = Q / z outside them, verified by the greedy multi-draft verifier.
 *
 * The law of a node is computed EXACTLY, not sampled: the tail child's draw is an inverse CDF of one
 * uniform and the verify reads two (accept, residual), so each is a step function of its uniform;
 * the test finds every step by bisection on the real code (to 1e-16) and sums the interval lengths
 * with the draw's own weights. What comes out must be P, whatever (P, Q) and however many of the
 * node's slots exist.
 * ================================================================================================== */
namespace {

constexpr int SV = 8;   /* toy vocab bound: tokens 0 .. V-1 with V <= 6 here */

struct SLaw {
    int     V = 0;
    int32_t pid[SV];  float pval[SV];  int pn = 0;   /* the target's candidate plane (its nucleus) */
    double  p[SV] = {};                              /* its exact law over tokens 0 .. V-1 */
    int32_t qid[8];   float qval[8];                 /* the head's shortlist, holes = -1 / -inf */
};

/* A target over `psup` of the V tokens (the others have no candidate, as after top-p) and a head over
 * `qsup` of them, independently chosen: the supports may differ, nest or be disjoint. */
SLaw small_law(uint64_t h, int V, int qsup, int psup, double qspread = 1.0) {
    SLaw L;
    L.V = V;
    int perm[SV], perm2[SV];
    for (int i = 0; i < V; ++i) { perm[i] = i; perm2[i] = i; }
    for (int i = V - 1; i > 0; --i) {
        std::swap(perm[i], perm[(int)(H(h * 31 + (uint64_t)i) % (uint64_t)(i + 1))]);
        std::swap(perm2[i], perm2[(int)(H(h * 37 + 5 + (uint64_t)i) % (uint64_t)(i + 1))]);
    }
    double z = 0;
    for (int j = 0; j < psup; ++j) {
        const float lg = (float)((H(h * 131 + (uint64_t)j * 977 + 1) >> 40) % 4000) / 1000.0f - 2.0f;
        L.pid[L.pn] = perm[j]; L.pval[L.pn++] = lg;
    }
    for (int j = 0; j < L.pn; ++j) z += std::exp((double)L.pval[j]);
    for (int j = 0; j < L.pn; ++j) L.p[L.pid[j]] = std::exp((double)L.pval[j]) / z;
    for (int i = 0; i < 8; ++i) { L.qid[i] = -1; L.qval[i] = -INFINITY; }
    for (int j = 0; j < qsup; ++j) {
        L.qid[j] = perm2[j];
        L.qval[j] = (float)(qspread * (((H(h * 257 + (uint64_t)j * 331 + 7) >> 40) % 4000) / 1000.0 - 2.0));
    }
    return L;
}

/* The step function f(u), u in [0, 1): its pieces (token, length) in order, found by bisection. */
std::vector<std::pair<int, double>> pieces(const std::function<int(double)>& f) {
    std::vector<std::pair<int, double>> out;
    double lo = 0;
    const double last = 1.0 - 0x1p-53;
    while (lo < 1) {
        const int t = f(lo);
        double b = 1.0;
        if (f(last) != t) {
            double a = lo, bb = last;
            for (int it = 0; it < 80; ++it) {
                const double mid = 0.5 * (a + bb);
                if (f(mid) == t) a = mid; else bb = mid;
            }
            b = bb;
        }
        out.emplace_back(t, b - lo);
        if (b >= 1.0) break;
        lo = b;
    }
    return out;
}

struct Branch { double w; RadMtpCoupleState st; double law[SV]; double hit; };

int decide_u(const RadMtpCoupleState& s, const SLaw& L, double ua, double ur) {
    int32_t out = -7;
    CHECK(rad_mtp_rrs_decide_rheo_u(s, L.pid, L.pval, L.pn, 1000, ua, ur, &out));
    return out;
}

/* The node's exact output law, branch by branch of the tail child: weight, state, the law of the
 * emitted token, the probability that it is one of the node's children. nkids is a function of Q's
 * slots only (here a plain number). */
std::vector<Branch> rheo_node(const SLaw& L, int m, int nkids) {
    std::vector<Branch> out;
    RadMtpCoupleState s0{};
    CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &s0));
    s0.draw = make_draw(77, 3); s0.status = 1; s0.round = 1; s0.pos = 5;
    auto drawn = [&](double ut) { RadMtpCoupleState s = s0; rad_mtp_rrs_draw_rheo_u(&s, ut, nkids, m); return s; };
    const bool has_tail = s0.count > m;
    std::vector<std::pair<int, double>> tail_piece;
    if (has_tail) tail_piece = pieces([&](double ut) { return (int)drawn(ut).ids[m]; });
    const int nb = has_tail ? (int)tail_piece.size() : 1;
    double lo = 0;
    for (int bi = 0; bi < nb; ++bi) {
        Branch br{};
        const double len = has_tail ? tail_piece[(size_t)bi].second : 1.0;
        br.w = len;
        br.st = drawn(has_tail ? lo + 0.5 * len : 0.5);
        lo += len;
        const RadMtpCoupleState& s = br.st;
        CHECK(rad_mtp_couple_state_valid(s, 1000));
        const int nt = std::min(m, s.count);
        const bool tail = s.count > nt && s.nkids > nt;
        double A = 0;
        std::vector<std::pair<int, double>> res;
        if (tail) {
            const auto pa = pieces([&](double ua) { return decide_u(s, L, ua, 0.5); });
            for (auto& pc : pa) if (pc.first == s.ids[nt]) A += pc.second;
            res = pieces([&](double ur) { return decide_u(s, L, 1.0 - 0x1p-53, ur); });
            br.law[s.ids[nt]] += A;
        } else {
            res = pieces([&](double ur) { return decide_u(s, L, 0.5, ur); });
        }
        for (auto& pc : res) {
            CHECK(pc.first >= 0 && pc.first < SV);
            br.law[pc.first] += (1.0 - A) * pc.second;
        }
        for (int k = 0; k < s.nkids; ++k) br.hit += br.law[s.ids[k]];
        out.push_back(br);
    }
    return out;
}

/* Thm 8's expected node acceptance for the first `nkids` slots of (P, Q, m): sum_T P + sum_tail
 * min(P, Qt) with the tail slot present, else the P mass of the top-q slots present. */
double thm8_hit(const SLaw& L, int m, int nkids, const RadMtpCoupleState& q0) {
    int ord[8];
    for (int i = 0; i < q0.count; ++i) ord[i] = i;
    std::stable_sort(ord, ord + q0.count, [&](int a, int b) { return q0.q[a] > q0.q[b]; });
    const int nt = std::min(m, q0.count);
    double want = 0, z = 0;
    for (int i = nt; i < q0.count; ++i) z += q0.q[ord[i]];
    if (q0.count > nt && nkids > nt) {
        for (int i = 0; i < nt; ++i) want += L.p[q0.ids[ord[i]]];
        for (int i = nt; i < q0.count; ++i)
            want += std::min(L.p[q0.ids[ord[i]]], q0.q[ord[i]] / (nt == 0 ? 1.0 : z));
    } else {
        for (int i = 0; i < std::min(nkids, nt); ++i) want += L.p[q0.ids[ord[i]]];
    }
    return want;
}

}  /* namespace */

/* (1) The node's law is P, exactly: V = 4 and 5, m = 0..3, every number of present slots (nkids from
 * "only the first top-q token" up to the whole m + 1, the last being the tail child; a smaller nkids
 * is a tail slot that was pruned or cut), P with a nucleus (zeros outside it), Q with a smaller
 * support than P's, P's and Q's supports disjoint. */
TEST(rheo_siblings_one_node_law_is_exactly_the_target) {
    double worst = 0, worst_hit = 0;
    int cases = 0;
    for (int V = 4; V <= 5; ++V)
        for (int m = 0; m <= 3; ++m)
            for (int qsup = 1; qsup <= V; ++qsup)
                for (int psup = 1; psup <= V; ++psup)
                    for (int rep = 0; rep < 3; ++rep) {
                        const SLaw L = small_law((uint64_t)(V * 1000 + m * 100 + qsup * 10 + psup) * 7 + (uint64_t)rep,
                                                 V, qsup, psup, rep == 2 ? 3.0 : 1.0);
                        RadMtpCoupleState q0{};
                        CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &q0));
                        for (int nkids = 1; nkids <= m + 1; ++nkids) {   /* a node with children has at least one */
                            const auto br = rheo_node(L, m, nkids);
                            double law[SV] = {}, wsum = 0, hit = 0;
                            for (const auto& b : br) {
                                wsum += b.w;
                                hit += b.w * b.hit;
                                for (int x = 0; x < SV; ++x) law[x] += b.w * b.law[x];
                            }
                            CHECK(std::fabs(wsum - 1.0) < 1e-12);
                            for (int x = 0; x < SV; ++x) {
                                const double e = x < V ? L.p[x] : 0.0;
                                worst = std::max(worst, std::fabs(law[x] - e));
                                CHECK(std::fabs(law[x] - e) < 1e-12);
                            }
                            const double want = thm8_hit(L, m, nkids, q0);
                            worst_hit = std::max(worst_hit, std::fabs(hit - want));
                            CHECK(std::fabs(hit - want) < 1e-12);
                            ++cases;
                        }
                    }
    std::printf("  rheo node law: %d (P, Q, m, nkids) cases, worst |law - P| = %.2e, worst |hit - Thm 8| = %.2e\n",
                cases, worst, worst_hit);
    CHECK(cases > 600);
}

/* (1) The first two emitted tokens: x1 from the root node, x2 from the node of the child x1 when x1 is
 * a child (its own rheo draw over P(. | x1), Q(. | x1)), else a plain target draw: the joint law is
 * P(x1) P(x2 | x1) to 1e-12, for every branch of the root's tail child. */
TEST(rheo_siblings_two_deep_joint_law_is_exactly_the_target) {
    double worst = 0;
    int cases = 0;
    for (int V = 4; V <= 5; ++V)
        for (int m = 1; m <= 2; ++m)
            for (int cfg = 0; cfg < 12; ++cfg) {
                const int qs0 = 1 + cfg % V, ps0 = 1 + (cfg * 3 + 1) % V;
                const SLaw root = small_law((uint64_t)(900 + V * 50 + m * 7 + cfg), V, qs0, ps0);
                SLaw kid[SV];
                for (int x = 0; x < V; ++x)
                    kid[x] = small_law((uint64_t)(5000 + V * 70 + m * 11 + cfg * 13 + x), V, 1 + (cfg + x) % V,
                                       1 + (cfg + 2 * x + 1) % V);
                const int nk_root = 1 + cfg % (m + 1), nk_kid = 1 + (cfg / 2) % (m + 1);
                const auto rb = rheo_node(root, m, nk_root);
                double joint[SV][SV] = {};
                for (const auto& b : rb) {
                    bool is_child[SV] = {};
                    for (int k = 0; k < b.st.nkids; ++k) is_child[b.st.ids[k]] = true;
                    for (int x1 = 0; x1 < V; ++x1) {
                        if (is_child[x1]) {
                            const auto kb = rheo_node(kid[x1], m, nk_kid);
                            for (const auto& c : kb)
                                for (int x2 = 0; x2 < V; ++x2) joint[x1][x2] += b.w * b.law[x1] * c.w * c.law[x2];
                        } else {
                            for (int x2 = 0; x2 < V; ++x2) joint[x1][x2] += b.w * b.law[x1] * kid[x1].p[x2];
                        }
                    }
                }
                for (int x1 = 0; x1 < V; ++x1)
                    for (int x2 = 0; x2 < V; ++x2) {
                        const double e = root.p[x1] * kid[x1].p[x2];
                        worst = std::max(worst, std::fabs(joint[x1][x2] - e));
                        CHECK(std::fabs(joint[x1][x2] - e) < 1e-12);
                    }
                ++cases;
            }
    std::printf("  rheo two-deep joint: %d cases, worst |joint - P P| = %.2e\n", cases, worst);
}

/* (3) m = 0 is the chain's single-draft maximal coupling: the same token as rad_mtp_couple_decide for
 * the same draft x, the same uniforms, on a grid of both uniforms, for every token of Q's support. */
TEST(rheo_with_m_zero_is_the_single_draft_coupling) {
    long checked = 0, diff = 0;
    for (int V = 3; V <= 6; ++V)
        for (int rep = 0; rep < 40; ++rep) {
            const SLaw L = small_law((uint64_t)(77000 + V * 100 + rep), V, 1 + rep % V, 1 + (rep / 2) % V);
            RadMtpCoupleState chain{};
            CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &chain));
            chain.status = 1; chain.round = 1;
            double tot = 0;
            for (int i = 0; i < chain.count; ++i) tot += chain.q[i];
            for (int k = 0; k < chain.count; ++k) {
                RadMtpCoupleState rheo = chain;
                RadMtpCoupleState ch = chain;
                ch.x = chain.ids[k];
                /* a draw inside token k's interval of the draw's inverse CDF (descending q) */
                int ord[8];
                for (int i = 0; i < chain.count; ++i) ord[i] = i;
                std::stable_sort(ord, ord + chain.count, [&](int a, int b) { return chain.q[a] > chain.q[b]; });
                double lo = 0, wk = 0;
                for (int i = 0; i < chain.count; ++i) {
                    if (ord[i] == k) { wk = chain.q[k]; break; }
                    lo += chain.q[ord[i]];
                }
                rad_mtp_rrs_draw_rheo_u(&rheo, (lo + 0.5 * wk) / tot, 1, 0);
                CHECK_EQ(rheo.ids[0], ch.x);
                CHECK_EQ(rheo.scheme, 1);
                for (int a = 0; a <= 32; ++a)
                    for (int r = 0; r < 32; ++r) {
                        const double ua = a / 33.0, ur = (r + 0.5) / 32.0;
                        int32_t o1 = -1, o2 = -2;
                        const bool ok1 = rad_mtp_couple_decide(ch, L.pid, L.pval, L.pn, 1000, ua, ur, &o1);
                        const bool ok2 = rad_mtp_rrs_decide_rheo_u(rheo, L.pid, L.pval, L.pn, 1000, ua, ur, &o2);
                        CHECK(ok1 && ok2);
                        ++checked;
                        if (o1 != o2) ++diff;
                    }
            }
        }
    std::printf("  rheo m=0 vs chain coupling: %ld decisions, %ld differ\n", checked, diff);
    CHECK_EQ(diff, 0L);
}

/* (4) The draw's invariants: the state stays valid (every q > 0, x = ids[0]), the first m slots are the
 * m largest q in descending order, the tail child has q > 0 and a tail law at most 1, the slot
 * priorities are non-increasing and a function of the multiset of q only. */
TEST(rheo_draw_invariants_and_slot_priorities) {
    for (int s = 0; s < 3000; ++s) {
        const SLaw L = small_law((uint64_t)(31000 + s), 6, 1 + s % 6, 3, 1.5);
        RadMtpCoupleState st{};
        CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st));
        st.draw = make_draw(5, 5);
        st.status = 1; st.round = 1;
        const int m = s % 5;
        RadMtpCoupleState sh = st;
        for (int i = st.count - 1; i > 0; --i) {   /* any permutation of the support */
            const int j = (int)(H((uint64_t)s * 17 + (uint64_t)i) % (uint64_t)(i + 1));
            std::swap(sh.ids[i], sh.ids[j]); std::swap(sh.q[i], sh.q[j]);
        }
        float a[8], b[8];
        rad_mtp_rrs_slot_logq_rheo(st, m, a);
        rad_mtp_rrs_slot_logq_rheo(sh, m, b);
        for (int c = 0; c < 8; ++c) {
            CHECK(a[c] == b[c]);
            if (c > 0) CHECK(a[c] <= a[c - 1]);
        }
        const double ut = ((H((uint64_t)s * 3 + 1) >> 11) + 0.5) * 0x1p-53;
        RadMtpCoupleState d = st;
        rad_mtp_rrs_draw_rheo_u(&d, ut, 8, m);
        CHECK(rad_mtp_couple_state_valid(d, 1000));
        const int nt = std::min(m, d.count);
        CHECK_EQ(d.nkids, nt + (d.count > nt ? 1 : 0));
        CHECK_EQ(d.scheme, m + 1);
        CHECK_EQ(d.x, d.ids[0]);
        for (int i = 1; i < nt; ++i) CHECK(d.q[i - 1] >= d.q[i]);
        for (int i = 0; i < d.count; ++i) CHECK(d.q[i] > 0);   /* a sampled child always has Q > 0 */
        if (d.count > nt) {
            double z = 0, qmax_tail = 0;
            for (int i = nt; i < d.count; ++i) { z += d.q[i]; qmax_tail = std::max(qmax_tail, d.q[i]); }
            CHECK(d.q[nt] / (nt == 0 ? 1.0 : z) <= 1.0 + 1e-15);
            if (nt > 0) CHECK(d.q[nt - 1] >= qmax_tail);
            /* the slot priority of the tail child: log min(q of the last top token, z) */
            const double ref = nt > 0 ? d.q[nt - 1] : 1.0;
            CHECK(std::fabs((double)a[nt] - std::log(std::min(ref, std::min(z, 1.0)))) < 1e-5);
        }
        for (int c = nt + (d.count > nt ? 1 : 0); c < 8; ++c) CHECK(a[c] == -INFINITY);
        /* keep_count: a prefix of the m + 1 slots, never beyond them */
        RadMtpTreeCfg pc{};
        rad_mtp_tree_set_prune(&pc, 0.2, 0.3);
        const int kc = rad_mtp_rrs_slot_keep_count(st, &pc, 8, m + 1);
        CHECK(kc >= 1 && kc <= m + 1 && kc <= st.count);
        CHECK_EQ(rad_mtp_rrs_slot_keep_count(st, nullptr, 8, m + 1), std::min(m + 1, st.count));
    }
}

namespace {
int verify_raw(const RadMtpCoupleState& st, const int32_t* ids, const float* val, int n, uint64_t seed,
               uint64_t pos, int fallback) {
    uint32_t params[24] = {};
    RadSampleParams* p = reinterpret_cast<RadSampleParams*>(params);
    p->seed = seed; p->pos = pos;
    int32_t tok = fallback;
    RadMtpCoupleVerifyCall c{};
    c.idx = ids; c.val = val; c.params = params;
    c.state = reinterpret_cast<const uint32_t*>(&st);
    c.token = &tok;
    c.M = 1; c.n_cand = n; c.vocab = 1000;
    c.ld[0] = n; c.ld[1] = n; c.ld[2] = 24; c.ld[3] = RAD_MTP_COUPLE_STATE_WORDS; c.ld[4] = 1;
    rad_mtp_couple_verify_row(c, 0);
    return tok;
}
}  /* namespace */

/* (1) Monte Carlo through the REAL keyed draw and the real verify row (the plugin's dispatch on
 * state.scheme): 1e7 samples at m = 1 (a V = 6 case, P nucleus of 5, Q over 4 tokens), then m = 2, the
 * pruned-tail nodes (nkids = m) and m = 0: chi-square against P, and the measured hit rate against
 * the exact one. */
TEST(rheo_siblings_monte_carlo_through_the_real_draw_and_verify_row) {
    const SLaw L = small_law(424242, 6, 4, 5, 1.2);
    RadMtpCoupleState q0{};
    CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &q0));
    struct Run { int m, nkids; long trials; };
    const Run runs[] = { { 1, 2, 10000000 }, { 2, 3, 3000000 }, { 1, 1, 2000000 }, { 2, 2, 2000000 }, { 0, 1, 2000000 } };
    for (const Run& rn : runs) {
        std::vector<double> cnt((size_t)6, 0);
        double hits = 0;
        for (long s = 0; s < rn.trials; ++s) {
            RadMtpCoupleState st = q0;
            st.draw = make_draw(0x9e3779b97f4a7c15ull * (uint64_t)(s + 1), (uint64_t)s);
            st.status = 1; st.round = 1; st.pos = 5;
            rad_mtp_rrs_draw_rheo(&st, 0, rn.nkids, rn.m);
            const int y = verify_raw(st, L.pid, L.pval, L.pn, st.draw.seed, 5, -1);
            CHECK(y >= 0 && y < 6);
            ++cnt[(size_t)y];
            for (int k = 0; k < st.nkids; ++k) hits += st.ids[k] == y;
        }
        std::vector<double> e(L.p, L.p + 6);
        const double z = zscore(cnt, e, (double)rn.trials);
        const double want = thm8_hit(L, rn.m, rn.nkids, q0);
        const double hr = hits / (double)rn.trials;
        const double sd = std::sqrt(want * (1 - want) / (double)rn.trials);
        std::printf("  rheo MC m=%d nkids=%d n=%ld: chi z=%.2f hit=%.5f (Thm 8 %.5f, %.1f sd)\n", rn.m, rn.nkids,
                    rn.trials, z, hr, want, sd > 0 ? (hr - want) / sd : 0.0);
        CHECK(z < 4.5);
        CHECK(sd == 0 || std::fabs(hr - want) < 4.5 * sd);
    }
}

/* (2) The whole pipeline (the real propose row, the real coupled grower, slot priorities that include
 * the rheo proxy, pruning by them, the core's prefix rule, the real verify) emits the target's law:
 * stage 1 (trunk chain + leaves), stage 2 and 2b, m = 1 and 2, with and without confidence pruning, a
 * far head. The tail slot is pruned by q-rank where the thresholds say so. The acceptance of the same
 * budget with Gumbel siblings is printed beside it. */
TEST(rheo_siblings_trees_with_and_without_pruning_emit_the_target_law) {
    const int trials = 300000;
    const Prune ab{ 0.07, 0.4 }, hard{ 0.2, 0.17 };
    const struct { const char* name; Kind kind; } kinds[] = { { "stage1", STAGE1 }, { "stage2", STAGE2 }, { "stage2b", STAGE2B } };
    for (const auto& kd : kinds) {
        g_scheme = 0;
        const Result gum = run_kind(kd.kind, trials, 1.0);
        for (int m = 1; m <= 2; ++m)
            for (int pr = 0; pr < 3; ++pr) {
                g_scheme = m + 1;
                g_pr = pr == 1 ? &ab : pr == 2 ? &hard : nullptr;
                g_cut = g_kids = 0;
                const Result r = run_kind(kd.kind, trials, 1.0);
                const double cut = g_kids > 0 ? g_cut / g_kids : 0;
                g_pr = nullptr; g_scheme = 0;
                std::printf("  %-8s rheo m=%d prune=%d: z=%.2f first=%.2f acc=%.3f cut=%.3f | gumbel acc=%.3f\n",
                            kd.name, m, pr, r.z, r.first_z, r.mean_acc, cut, gum.mean_acc);
                CHECK(r.z < 4.5);
                CHECK(r.first_z < 4.5);
            }
    }
    g_scheme = 3; g_pr = nullptr;
    const Result far = run_kind(STAGE2, 150000, 3.0);
    g_scheme = 0;
    CHECK(far.z < 4.5);
    CHECK(far.first_z < 4.5);
}

/* Expected children accepted by one node (hit rate: the emitted token is one of the node's children)
 * with the same number of children k = m + 1, Gumbel-top-k siblings against RheoSampling, on synthetic
 * (P, Q) pairs shaped like the engine's: P the target's top-k 20 / top-p 0.95 law at temperature T
 * over a Zipf-ish logit profile, Q the head's temperature-1 softmax of its noisy top 8. Printed, not
 * asserted: the real number is the engine's `MTP coupled stats` on the model (tools in DESIGN.md). */
TEST(rheo_against_gumbel_siblings_hit_rate_at_equal_children_on_synthetic_laws) {
    const int NODES = 300, DRAWS = 6000;
    for (double T : { 0.7, 1.0 })
        for (double sigma : { 0.6, 1.2 }) {
            double hit_g[4] = {}, hit_r[4] = {};
            for (int n = 0; n < NODES; ++n) {
                const uint64_t h = H((uint64_t)n * 977 + (uint64_t)(T * 10) * 13 + (uint64_t)(sigma * 10));
                const int NV = 40;
                std::vector<double> lg(NV), hl(NV);
                const double sharp = 1.5 + 4.0 * ((H(h + 1) >> 40) % 1000) / 1000.0;
                for (int i = 0; i < NV; ++i) {
                    const double g1 = ((H(h * 3 + (uint64_t)i) >> 40) % 2000) / 1000.0 - 1.0;
                    const double g2 = ((H(h * 5 + (uint64_t)i + 99) >> 40) % 2000) / 1000.0 - 1.0;
                    lg[(size_t)i] = -sharp * std::log((double)(i + 1)) + 1.0 * g1;
                    hl[(size_t)i] = lg[(size_t)i] + sigma * 1.7 * g2;
                }
                /* target law: T, top-k 20, top-p 0.95 */
                std::vector<int> ord(NV);
                for (int i = 0; i < NV; ++i) ord[(size_t)i] = i;
                std::sort(ord.begin(), ord.end(), [&](int a, int b) { return lg[(size_t)a] > lg[(size_t)b]; });
                std::vector<int32_t> pid; std::vector<float> pval;
                double z = 0;
                for (int j = 0; j < 20; ++j) z += std::exp(lg[(size_t)ord[(size_t)j]] / T);
                double cum = 0;
                for (int j = 0; j < 20 && cum < 0.95; ++j) {
                    const double p = std::exp(lg[(size_t)ord[(size_t)j]] / T) / z;
                    pid.push_back(ord[(size_t)j]); pval.push_back((float)(lg[(size_t)ord[(size_t)j]] / T));
                    cum += p;
                }
                /* the head's reranked top 8 */
                std::vector<int> ho(NV);
                for (int i = 0; i < NV; ++i) ho[(size_t)i] = i;
                std::sort(ho.begin(), ho.end(), [&](int a, int b) { return hl[(size_t)a] > hl[(size_t)b]; });
                int32_t qid[8]; float qval[8];
                for (int i = 0; i < 8; ++i) { qid[i] = ho[(size_t)i]; qval[i] = (float)hl[(size_t)ho[(size_t)i]]; }
                RadMtpCoupleState q0{};
                CHECK(rad_mtp_couple_build_q(qid, qval, 1000, &q0));
                for (int k = 1; k <= 3; ++k)
                    for (int scheme = 0; scheme < 2; ++scheme) {
                        double hits = 0;
                        for (int s = 0; s < DRAWS; ++s) {
                            RadMtpCoupleState st = q0;
                            st.draw = make_draw(H(h + (uint64_t)s * 7 + 1), (uint64_t)s);
                            st.status = 1; st.round = 1; st.pos = 5;
                            if (scheme) rad_mtp_rrs_draw_rheo(&st, 0, k, k - 1);
                            else        rad_mtp_rrs_draw(&st, 0, k);
                            const int y = verify_raw(st, pid.data(), pval.data(), (int)pid.size(), st.draw.seed, 5, -1);
                            for (int c = 0; c < st.nkids; ++c) hits += st.ids[c] == y;
                        }
                        (scheme ? hit_r : hit_g)[k] += hits / DRAWS / NODES;
                    }
            }
            std::printf("  synthetic T=%.1f sigma=%.1f: children accepted per node (k children) gumbel %.4f %.4f %.4f | "
                        "rheo(m=k-1) %.4f %.4f %.4f\n", T, sigma, hit_g[1], hit_g[2], hit_g[3], hit_r[1], hit_r[2], hit_r[3]);
        }
}

/* (5) Flag off: scheme 0 is the old draw and the old verify, bit for bit; a rheo state differs only by
 * `scheme` and the order of its children. */
TEST(the_default_scheme_is_the_gumbel_one_bit_for_bit) {
    Tree a, b;
    g_scheme = 0;
    draft_chain(&a, 99, 4, 3, 3, 1.0);
    draft_chain(&b, 99, 4, 3, 3, 1.0);
    for (int k = 0; k < 3; ++k) {
        CHECK_EQ(a.st[k].scheme, 0);
        CHECK(std::memcmp(&a.st[k], &b.st[k], sizeof(RadMtpCoupleState)) == 0);
    }
    CHECK_EQ(a.st[0].nkids, 4);
    draft_grow(&a, 5, 1, 2, 3, 7, 1.0);
    for (int k = 0; k < 7; ++k) CHECK_EQ(a.st[k].scheme, 0);
    g_scheme = 2;
    draft_chain(&b, 99, 4, 3, 3, 1.0);
    g_scheme = 0;
    CHECK_EQ(b.st[0].scheme, 2);
    CHECK_EQ(b.st[0].nkids, 2);
    CHECK_EQ(b.st[1].scheme, 0);   /* the chain's later rounds are single draws */
}

/* ------------------------------------------------------------------ online calibration
 *
 * (2) The controller changes the draw's per-depth draft temperature and calibration codes between
 * steps. The state stores the Q that was drawn from and the verify reads it, so the output law
 * stays P whatever sequence of words the draws carry: chain, stage 1 and stage 2, over words that
 * differ per depth class and per trial (every tau code, every calibration code, the one-hot limit),
 * with the adaptive rule on so the calibration codes act on the tree's shape too. */
std::vector<uint32_t> all_words() {
    std::vector<uint32_t> w;
    uint64_t h = 12345;
    for (int i = 0; i < 64; ++i) {
        uint32_t x = 0;
        for (int d = 0; d < RAD_MTP_DRAW_DEPTHS; ++d) {
            h = rad_mtp_rrs_mix(h + (uint64_t)i * 31 + (uint64_t)d);
            const unsigned tau = (unsigned)(h % 16), cal = (unsigned)((h >> 8) % 16);
            x |= (tau | cal << 4) << (8 * d);
        }
        if (i % 2) x |= RAD_MTP_DRAW_EXPORT_BIT;   /* the verify also exports P: it must not touch the decision */
        w.push_back(x);
    }
    w.push_back(0);   /* a step with no online word at all */
    w.push_back(0x010101u);   /* tau 0.05 (the greedy limit) at every depth */
    return w;
}

TEST(a_draft_law_that_changes_every_step_stays_exact) {
    g_word_cycle = all_words();
    g_recipe = LawRecipe{};
    const int trials = 150000;
    const Result chain = run_kind(CHAIN, trials, 1.0);
    const Result s1    = run_kind(STAGE1, trials, 1.0);
    const Result s2    = run_kind(STAGE2, trials, 1.0);
    const Result far   = run_kind(STAGE1, trials / 2, 3.0);
    std::printf("  changing tau_d / T_d per step: chain z=%.2f acc=%.3f | stage1 z=%.2f | stage2 z=%.2f | far z=%.2f\n",
                chain.z, chain.mean_acc, s1.z, s2.z, far.z);
    CHECK(chain.z < 4.5 && chain.first_z < 4.5);
    CHECK(s1.z < 4.5 && s1.first_z < 4.5);
    CHECK(s2.z < 4.5 && s2.first_z < 4.5);
    CHECK(far.z < 4.5 && far.first_z < 4.5);
    /* with a base recipe underneath (the words override dscale only where a tau code is set) */
    g_recipe = { 1.0f / 0.7f, 0.9f, 4 };
    const Result rc = run_kind(STAGE2, trials, 1.0);
    CHECK(rc.z < 4.5 && rc.first_z < 4.5);
    g_recipe = LawRecipe{};
    g_word_cycle.clear();
}

/* The word picks the Q only: tau codes name the table's temperature at their depth class, the
 * deepest class repeats, code 0 keeps dscale, and a draw with no word is build_law bit for bit. */
TEST(the_online_word_names_the_draft_temperature_per_depth) {
    const Law L = law_at(777, 1.0);
    RadMtpCoupleDraw d0{};
    d0.mode = 1;
    RadMtpCoupleState a{}, b{};
    CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d0, &a));
    for (int depth = 1; depth <= 6; ++depth) {
        CHECK(rad_mtp_couple_build_law_at(L.qid, L.qval, 1000, d0, depth, &b));
        CHECK(std::memcmp(a.q, b.q, sizeof a.q) == 0);   /* no word: the legacy law, at every depth */
    }
    RadMtpCoupleDraw d = d0;
    d.dpad_ = 4u | (8u << 8) | (11u << 16);   /* tau 0.6, 0.9, 1.15 */
    const float taus[3] = { 0.6f, 0.9f, 1.15f };
    for (int depth = 1; depth <= 6; ++depth) {
        const float tau = taus[std::min(depth, 3) - 1];
        RadMtpCoupleDraw e = d0;
        e.dscale = 1.0f / tau;
        CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, e, &a));
        CHECK(rad_mtp_couple_build_law_at(L.qid, L.qval, 1000, d, depth, &b));
        CHECK(a.count == b.count && std::memcmp(a.q, b.q, sizeof a.q) == 0);
    }
    /* code 0 at a depth keeps the draw's own dscale */
    d.dscale = 2.0f;
    d.dpad_ = 8u << 8;
    RadMtpCoupleDraw f = d0;
    f.dscale = 2.0f;
    CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, f, &a));
    CHECK(rad_mtp_couple_build_law_at(L.qid, L.qval, 1000, d, 1, &b));
    CHECK(std::memcmp(a.q, b.q, sizeof a.q) == 0);
    /* the code tables: nearest-in-log lookups invert them */
    for (unsigned c = 1; c < 16; ++c) {
        CHECK_EQ(rad_mtp_tau_code_of(rad_mtp_tau_of_code(c)), c);
        CHECK_EQ(rad_mtp_cal_code_of(rad_mtp_cal_of_code(c), rad_mtp_cal_lift_of_code(c)), c);
    }
}

/* The calibration word reaches the adaptive rule's temperatures: set codes replace the static list
 * at their depth class, the deepest repeats, and an unset word (or adaptive off) leaves it alone. */
TEST(the_calibration_word_overrides_the_static_temperatures) {
    RadMtpAdapt ad{};
    rad_mtp_adapt_parse("on=1;temp=1.3,1.1", &ad);
    RadMtpCoupleDraw d{};
    RadMtpAdapt a1 = ad;
    rad_mtp_calib_apply(&a1, d);   /* no word: untouched */
    CHECK(std::memcmp(&a1, &ad, sizeof ad) == 0);
    d.dpad_ = (3u << 4) | (10u << 20);   /* depth 1: T 1.0, depth 2: static, depth 3: T 2.0 lift 0.2 */
    a1 = ad;
    rad_mtp_calib_apply(&a1, d);
    CHECK(std::fabs(rad_mtp_adapt_temp(&a1, 1) - 1.0f) < 1e-6f);
    CHECK(std::fabs(rad_mtp_adapt_temp(&a1, 2) - 1.1f) < 1e-6f);
    CHECK(std::fabs(rad_mtp_adapt_temp(&a1, 3) - 2.0f) < 1e-6f);
    CHECK(std::fabs(rad_mtp_adapt_temp(&a1, 7) - 2.0f) < 1e-6f);
    CHECK(rad_mtp_adapt_lift(&a1, 1) == 0.0f && rad_mtp_adapt_lift(&a1, 2) == 0.0f);
    CHECK(std::fabs(rad_mtp_adapt_lift(&a1, 3) - 0.2f) < 1e-6f);
    CHECK(std::fabs(rad_mtp_adapt_lift(&a1, 9) - 0.2f) < 1e-6f);
    RadMtpAdapt off{};
    rad_mtp_adapt_parse("temp=1.3", &off);   /* adaptive off: never touched */
    RadMtpAdapt o1 = off;
    rad_mtp_calib_apply(&o1, d);
    CHECK(std::memcmp(&o1, &off, sizeof off) == 0);
}

/* (4) The acceptance label against brute force. A chain row: alpha = sum_x q(x) P(accept | x), the
 * acceptance probability of each drafted token measured by running the real decide over a fine grid
 * of acceptance uniforms. */
double brute_alpha_chain(const RadMtpCoupleState& s, const Law& L) {
    double alpha = 0;
    for (int k = 0; k < s.count; ++k) {
        RadMtpCoupleState t = s;
        t.x = s.ids[k];
        int acc = 0;
        const int G = 4000;
        for (int g = 0; g < G; ++g) {
            const double u = (g + 0.5) / G;
            int32_t out = -1;
            CHECK(rad_mtp_couple_decide(t, L.pid, L.pval, L.pn, 1000, u, 0.5, &out));
            if (out == t.x) ++acc;
        }
        alpha += s.q[k] * (double)acc / G;
    }
    return alpha;
}

void p_on_support(const RadMtpCoupleState& s, const Law& L, double* pq) {
    double z = 0;
    for (int j = 0; j < L.pn; ++j) z += std::exp((double)L.pval[j]);
    for (int k = 0; k < s.count; ++k) {
        pq[k] = 0;
        for (int j = 0; j < L.pn; ++j) if (L.pid[j] == s.ids[k]) pq[k] += std::exp((double)L.pval[j]) / z;
    }
}

TEST(the_alpha_label_equals_the_brute_force_acceptance) {
    for (int t = 0; t < 12; ++t) {
        const Law L = law_at(5000 + (uint64_t)t, 0.6 + 0.2 * t);
        RadMtpCoupleDraw d{};
        d.mode = 1;
        d.dscale = t % 3 == 0 ? 0.0f : 1.0f / (0.6f + 0.15f * t);
        RadMtpCoupleState s{};
        CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d, &s));
        s.status = 1; s.round = 1; s.x = s.ids[0];
        double pq[8], alpha[8];
        p_on_support(s, L, pq);
        CHECK_EQ(rad_calib_labels(s, pq, alpha), 1);
        const double bf = brute_alpha_chain(s, L);
        CHECK(std::fabs(alpha[0] - bf) < 2.5e-3);
        /* and it is exactly sum min(P, Q) */
        double mn = 0;
        for (int k = 0; k < s.count; ++k) mn += std::min(pq[k], s.q[k]);
        CHECK(std::fabs(alpha[0] - mn) < 1e-12);
    }
}

/* A tree row's later labels: child i's label is the acceptance probability of the child in slot i
 * given the rows before it rejected, averaged over the token the draw put there. Estimated by
 * running the real recursive decide over many seeds with the children drawn without replacement
 * (Gumbel order) and counting, per slot, accepted / reached. The label function is evaluated on the
 * same drawn histories, weighted by the probability that the drafted children before the slot were
 * rejected (their own acceptance), and the two must agree. */
TEST(the_sibling_labels_equal_the_recursive_rejection_frequencies) {
    for (int t = 0; t < 4; ++t) {
        const Law L = law_at(6000 + (uint64_t)t, 1.0 + 0.4 * t);
        RadMtpCoupleDraw d{};
        d.mode = 1;
        d.dscale = 1.0f;
        RadMtpCoupleState s{};
        CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d, &s));
        s.status = 1; s.round = 1;
        const int m = 3;
        const int N = 300000;
        double reach[3] = {}, hit[3] = {};
        double e[3] = {}, w[3] = {};   /* label-side: reach-weighted mean label of slot i */
        for (int n = 0; n < N; ++n) {
            RadMtpCoupleState c = s;
            c.draw.seed = 1000 + (uint64_t)n; c.draw.base = 77;
            rad_mtp_rrs_draw(&c, 0, m);
            c.status = 1; c.x = c.ids[0];
            int32_t out = -1;
            if (!rad_mtp_rrs_decide(c, L.pid, L.pval, L.pn, 1000, c.draw.seed, 5, &out)) continue;
            int at = m;
            for (int i = 0; i < m; ++i) if (out == c.ids[i]) { at = i; break; }
            for (int i = 0; i < m && i <= at; ++i) {
                reach[i] += 1;
                if (i == at) hit[i] += 1;
            }
            if (n < 60000) {
                double pc[8], al[8], ar[8];
                p_on_support(c, L, pc);
                rad_calib_labels(c, pc, al, ar);
                double reached = 1.0;
                for (int i = 0; i < m; ++i) {
                    e[i] += reached * al[i];
                    w[i] += reached;
                    reached *= 1.0 - ar[i];   /* the drafted token's own rejection */
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            const double freq = hit[i] / reach[i], lab = e[i] / w[i];
            std::printf("  case %d slot %d: frequency %.4f label %.4f\n", t, i, freq, lab);
            CHECK(std::fabs(freq - lab) < 0.02);
        }
    }
}

/* The export: the verify with the export bit set decides exactly as without it (tokens bit for bit),
 * and rewrites q with P's mass on the state's ids, status 2; a state with no export bit is left
 * alone. */
TEST(the_verify_export_leaves_the_decision_alone_and_reports_p) {
    for (int t = 0; t < 20; ++t) {
        const Law L = law_at(7000 + (uint64_t)t, 1.0);
        for (int kids = 1; kids <= 3; ++kids) {
            RadMtpCoupleDraw d = make_draw(500 + (uint64_t)t, 40);
            RadMtpCoupleState s{};
            CHECK(rad_mtp_couple_build_law(L.qid, L.qval, 1000, d, &s));
            s.draw = d; s.status = 1; s.round = 1; s.pos = 9; s.x = s.ids[0];
            if (kids > 1) rad_mtp_rrs_draw(&s, 0, kids); else s.nkids = 0;
            RadMtpCoupleState plain = s, expo = s;
            expo.draw.dpad_ |= RAD_MTP_DRAW_EXPORT_BIT;
            const int tok_plain = verify(plain, L, d.seed, 9, -7);
            const int tok_expo = verify(expo, L, d.seed, 9, -7);
            CHECK_EQ(tok_plain, tok_expo);
            CHECK(plain.status == 1 && std::memcmp(plain.q, s.q, sizeof s.q) == 0);
            CHECK_EQ(expo.status, RAD_MTP_COUPLE_STATUS_EXPORTED);
            double pq[8];
            p_on_support(s, L, pq);
            for (int k = 0; k < s.count; ++k) CHECK(std::fabs(expo.q[k] - pq[k]) < 1e-12);
        }
    }
}

/* ====================================================================================================
 * Traversal verification, checked EXACTLY and on its own arithmetic (abi/rad_mtp_traversal.h).
 *
 * The walk reads the target through integer masses (rad_mtp_trav_mass: an FP32 exponential in fixed
 * point), so the law it realises for a row is P~ = masses / their sum, a fixed function of the row.
 * The tests below check (1) that P~ is the softmax to FP32 accuracy; (2) by ENUMERATION that the walk
 * emits P~ exactly: every draft tree of a small model is built with its probability (Plackett-Luce
 * prefixes for Gumbel siblings, the tail draw's pieces for rheo), the walk's tests are scripted (test
 * t accepts with its own rate after t - 1 rejections, which the walk reports), and the law of the
 * first two emitted tokens comes out as P~(x1) P~(x2 | x1) to 1e-11; the with-replacement update, the
 * missing (1 - p) term and a rheo node walked as a Gumbel one miss by percents; (3) that the final
 * token's uniform maps onto the residual law (support slots, then the off-support candidates by
 * integer units) on a 700-slot plane; (4) the rheo walk by chi-square through the real drafters.
 * ================================================================================================== */
namespace {

/* The law the walk realises for a candidate plane: its integer masses over the tokens. */
void trav_int_law(const SLaw& L, double* p) {
    for (int x = 0; x < SV; ++x) p[x] = 0;
    if (L.pn <= 0) return;
    float top = -INFINITY;
    for (int j = 0; j < L.pn; ++j) top = std::max(top, L.pval[j]);
    uint64_t m[SV] = {}, sum = 0;
    for (int j = 0; j < L.pn; ++j) m[L.pid[j]] += rad_mtp_trav_mass(L.pval[j], top, RAD_MTP_TRAV_QBITS);
    for (int x = 0; x < SV; ++x) sum += m[x];
    for (int x = 0; x < SV; ++x) p[x] = (double)m[x] / (double)sum;
}

struct ETree {
    int n = 1;
    int par[40] = {};
    int32_t tok[40] = {};
    bool has[40] = {};
    RadMtpCoupleState st[40] = {};
    const SLaw* law[40] = {};
    const RadMtpCoupleState* state(int v) const { return has[v] ? &st[v] : nullptr; }
    bool cand(int v, const int32_t** ids, const float** val, int64_t* nn) const {
        *ids = law[v]->pid; *val = law[v]->pval; *nn = law[v]->pn;
        return true;
    }
    int child(int v, int32_t token) const {
        for (int j = v + 1; j < n; ++j)
            if (par[j] == v && tok[j] == token) return j;
        return -1;
    }
    int64_t vocab() const { return 1000; }
};

struct EDraft { double w = 0; RadMtpCoupleState st{}; int nk = 0; int32_t kid[4] = {}; };

/* Every way a node with the head's law of `L` drafts `want` children, with its probability: the
 * ordered prefixes of sampling without replacement (scheme 0), or the pieces of the rheo tail draw
 * (scheme m + 1). Empty when the head has no law (the node drafts nothing). */
std::vector<EDraft> edrafts(const SLaw& L, int scheme, int want, int round) {
    std::vector<EDraft> out;
    RadMtpCoupleState s0{};
    if (want <= 0 || !rad_mtp_couple_build_q(L.qid, L.qval, 1000, &s0)) return out;
    s0.draw = make_draw(77, 3); s0.status = 1; s0.round = round; s0.pos = 5;
    if (scheme > 0) {
        const int m = scheme - 1;
        auto drawn = [&](double ut) { RadMtpCoupleState s = s0; rad_mtp_rrs_draw_rheo_u(&s, ut, want, m); return s; };
        std::vector<std::pair<int, double>> pc;
        if (s0.count > m) pc = pieces([&](double ut) { return (int)drawn(ut).ids[m]; });
        else pc.emplace_back(0, 1.0);
        double lo = 0;
        for (const auto& x : pc) {
            EDraft d;
            d.w = x.second;
            d.st = drawn(lo + 0.5 * x.second);
            lo += x.second;
            d.nk = d.st.nkids;
            for (int k = 0; k < d.nk; ++k) d.kid[k] = d.st.ids[k];
            out.push_back(d);
        }
        return out;
    }
    const int a = std::min(want, s0.count);
    int idx[4] = {};
    std::function<void(int, double, double)> rec = [&](int depth, double w, double used) {
        if (depth == a) {
            EDraft d;
            d.w = w;
            d.st = s0;
            if (a >= 2) {
                bool taken[8] = {};
                int n = 0;
                for (int i = 0; i < a; ++i) { d.st.ids[n] = s0.ids[idx[i]]; d.st.q[n++] = s0.q[idx[i]]; taken[idx[i]] = true; }
                for (int i = 0; i < s0.count; ++i)
                    if (!taken[i]) { d.st.ids[n] = s0.ids[i]; d.st.q[n++] = s0.q[i]; }
                d.st.x = d.st.ids[0];
                d.st.nkids = a;
            } else {
                d.st.x = s0.ids[idx[0]];   /* the chain's wire: x anywhere in the shortlist's order */
                d.st.nkids = 0;
            }
            d.nk = a;
            for (int i = 0; i < a; ++i) d.kid[i] = s0.ids[idx[i]];
            out.push_back(d);
            return;
        }
        for (int i = 0; i < s0.count; ++i) {
            bool dup = false;
            for (int j = 0; j < depth; ++j) dup = dup || idx[j] == i;
            if (dup) continue;
            idx[depth] = i;
            rec(depth + 1, w * s0.q[i] / (1.0 - used), used + s0.q[i]);
        }
    };
    rec(0, 1.0, 0.0);
    return out;
}

/* The law of the token a walk emits at a path's last node, from the walked tree: the node's
 * residual (support slots, then its off-support candidates by their integer masses), or the plain
 * target when the node carries no law. */
void etree_final_law(const RadTravTree& kt, int c, const SLaw& L, double* law) {
    for (int x = 0; x < SV; ++x) law[x] = 0;
    double tot = 0;
    if (kt.ok[c]) {
        const double woff = kt.moff[c];
        tot = rad_mtp_trav_sum8(kt.m[c]) + woff;
        if (tot > 0) {
            for (int k = 0; k < 8; ++k)
                if (kt.ids[c][k] >= 0) law[kt.ids[c][k]] += kt.m[c][k] / tot;
            if (woff > 0 && kt.off[c] > 0)
                for (int j = 0; j < L.pn; ++j) {
                    bool on = false;
                    for (int k = 0; k < 8; ++k) on = on || kt.ids[c][k] == L.pid[j];
                    if (!on)
                        law[L.pid[j]] += woff / tot *
                            (double)rad_mtp_trav_mass(L.pval[j], kt.top[c], RAD_MTP_TRAV_QBITS) / (double)kt.off[c];
                }
        }
    }
    if (!(tot > 0)) trav_int_law(L, law);
}

/* Adds w * P(the first two emitted tokens = (x1, x2) | this draft tree) to `joint`. */
void etree_joint(const ETree& T, const SLaw* kid, int V, double w, double joint[SV][SV], int ctl,
                 double* leak) {
    double pt[64];
    int nt = 0;
    RadMtpTravOpt o;
    o.ctl = ctl; o.accept_at = 0; o.p_test = pt; o.n_tests = &nt; o.max_tests = 64;
    RadMtpTravOut out;
    rad_mtp_trav_walk(T, 0, 1, 1, &out, o);   /* every test rejects: the rates of tests 1 .. nt */
    double kidlaw[SV][SV];
    for (int x = 0; x < V; ++x) trav_int_law(kid[x], kidlaw[x]);
    double surv = 1;
    for (int t = 1; t <= nt; ++t) {
        const double pa = surv * pt[t - 1];
        surv *= 1.0 - pt[t - 1];
        if (!(pa > 0)) continue;
        RadTravTree kt;
        int ids[RAD_MTP_TRAV_MAX_ROWS];
        RadMtpTravOpt o2;
        o2.ctl = ctl; o2.accept_at = t; o2.u_final = 0.5;
        RadMtpTravOut po;
        CHECK(rad_mtp_trav_walk(T, 0, 1, 1, &po, o2, &kt, ids));
        const int f = po.node[po.n - 1];
        int c = 0;
        while (ids[c] != f) ++c;
        double fl[SV];
        etree_final_law(kt, c, *T.law[f], fl);
        if (po.n == 1) {
            for (int y = 0; y < V; ++y)
                for (int b = 0; b < V; ++b) joint[y][b] += w * pa * fl[y] * kidlaw[y][b];
        } else if (po.n == 2) {
            const int x1 = T.tok[po.node[1]];
            for (int b = 0; b < V; ++b) joint[x1][b] += w * pa * fl[b];
        } else {
            joint[T.tok[po.node[1]]][T.tok[po.node[2]]] += w * pa;
        }
    }
    *leak += w * surv;   /* the root accepts with rate 1: nothing may be left */
}

/* The exact joint law of the first two emitted tokens for a root that drafts `a` children, each
 * drafting `b`; returns the largest deviation from P~(x1) P~(x2 | x1). */
double trav_enum_case(const SLaw& root, const SLaw* kid, int V, int scheme, int a, int b, int ctl, long* trees) {
    double joint[SV][SV] = {}, leak = 0, wsum = 0;
    const std::vector<EDraft> roots = edrafts(root, scheme, a, 1);
    CHECK(!roots.empty());
    for (const EDraft& r0 : roots) {
        std::vector<EDraft> cd[4];
        for (int i = 0; i < r0.nk; ++i) cd[i] = edrafts(kid[r0.kid[i]], scheme, b, 2);
        int sel[4] = {};
        for (;;) {
            ETree T;
            T.has[0] = true; T.st[0] = r0.st; T.law[0] = &root; T.par[0] = -1;
            CHECK(rad_mtp_couple_state_valid(r0.st, 1000));
            double w = r0.w;
            for (int i = 0; i < r0.nk; ++i) {
                const int v = T.n++;
                T.par[v] = 0; T.tok[v] = r0.kid[i]; T.law[v] = &kid[r0.kid[i]];
                T.has[v] = !cd[i].empty();
                if (T.has[v]) { T.st[v] = cd[i][(size_t)sel[i]].st; w *= cd[i][(size_t)sel[i]].w; }
            }
            for (int i = 0; i < r0.nk; ++i) {
                if (cd[i].empty()) continue;
                const EDraft& d = cd[i][(size_t)sel[i]];
                for (int g = 0; g < d.nk; ++g) {
                    const int v = T.n++;
                    T.par[v] = 1 + i; T.tok[v] = d.kid[g]; T.law[v] = &kid[d.kid[g]];
                }
            }
            etree_joint(T, kid, V, w, joint, ctl, &leak);
            wsum += w;
            ++*trees;
            int i = 0;
            for (; i < r0.nk; ++i) {
                if (cd[i].empty()) continue;
                if (++sel[i] < (int)cd[i].size()) break;
                sel[i] = 0;
            }
            if (i >= r0.nk) break;
        }
    }
    CHECK(std::fabs(wsum - 1.0) < 1e-11);
    double p0[SV], worst = ctl ? 0.0 : std::fabs(leak);
    trav_int_law(root, p0);
    for (int x1 = 0; x1 < V; ++x1) {
        double p1[SV];
        trav_int_law(kid[x1], p1);
        for (int x2 = 0; x2 < V; ++x2) worst = std::max(worst, std::fabs(joint[x1][x2] - p0[x1] * p1[x2]));
    }
    return worst;
}

}  /* namespace */

/* (1) The fixed-point exponential against libm's, the integer conversions, and the law they give. */
TEST(the_walks_fixed_point_exponential_tracks_exp) {
    const double unit = (double)(1ull << RAD_MTP_TRAV_QBITS);
    double worst = 0;
    CHECK_EQ(rad_mtp_trav_mass(3.25f, 3.25f, RAD_MTP_TRAV_QBITS), 1ull << RAD_MTP_TRAV_QBITS);
    CHECK_EQ(rad_mtp_trav_mass(-INFINITY, 0.0f, RAD_MTP_TRAV_QBITS), 0ull);
    CHECK_EQ(rad_mtp_trav_mass(-30.0f, 0.0f, RAD_MTP_TRAV_QBITS), 0ull);
    const int N = 3000000;
    for (int i = 0; i <= N; ++i) {
        /* a dense sweep of the whole range, and the same arguments reached from other tops */
        const float x = -28.6f * (float)i / (float)N;
        const float top = i % 3 == 0 ? 0.0f : (float)((int)(H((uint64_t)i) % 6000) - 3000) / 100.0f;
        const float v = top + x;
        const float xr = v - top;   /* the argument the mass sees */
        const uint64_t m = rad_mtp_trav_mass(v, top, RAD_MTP_TRAV_QBITS);
        const double want = xr > -28.5f ? std::exp((double)xr) * unit : 0.0;
        const double err = std::fabs((double)m - want);
        CHECK(err <= want * 0x1p-22 + 1.0);
        if (want > 0x1p32) worst = std::max(worst, err / want);   /* where the floor's one unit is below 2^-32 */
        /* a coarser unit is the floor of the finer one */
        if (i % 16 == 0) CHECK_EQ(rad_mtp_trav_mass(v, top, 31), m >> 9);
    }
    std::printf("  fixed-point exp: worst relative error %.3g (2^-22 = %.3g) over %d arguments\n", worst, 0x1p-22, N);
    CHECK(worst < 0x1p-22);
    CHECK_EQ(rad_mtp_trav_qbits(20), 40);     CHECK_EQ(rad_mtp_trav_qbits(1024), 40);
    CHECK_EQ(rad_mtp_trav_qbits(4096), 40);   CHECK_EQ(rad_mtp_trav_qbits(4097), 39);
    CHECK_EQ(rad_mtp_trav_qbits(8192), 39);   CHECK_EQ(rad_mtp_trav_qbits(1ll << 31), 21);
    for (int i = 0; i < 200000; ++i) {
        const uint64_t x = H((uint64_t)i * 77 + 5) >> (11 + (i % 40));
        CHECK(rad_mtp_trav_u2d(x) == (double)x);
        CHECK_EQ(rad_mtp_trav_d2u(rad_mtp_trav_u2d(x)), x);
        if (x < (1ull << 50)) CHECK_EQ(rad_mtp_trav_d2u(rad_mtp_trav_u2d(x) + 0.75), x);
    }
    /* the law of a plane: the softmax, to FP32 accuracy */
    double wl = 0;
    for (int t = 0; t < 2000; ++t) {
        const Law L = law_at(31000 + (uint64_t)t, 1.0);
        float top = -INFINITY;
        CHECK(rad_mtp_trav_row_top(L.pid, L.pval, L.pn, 1000, &top));
        int32_t sid[8];
        for (int k = 0; k < 8; ++k) sid[k] = L.qid[k];
        RadTravLaw w;
        rad_mtp_trav_row_law(L.pid, L.pval, L.pn, top, sid, 8, &w);
        uint64_t on = 0;
        for (int k = 0; k < 8; ++k) {
            on += w.pq[k];
            wl = std::max(wl, std::fabs((double)w.pq[k] / (double)w.sum - L.p[sid[k]]));
        }
        CHECK_EQ(on + w.off, w.sum);
    }
    std::printf("  integer law against the FP64 softmax: worst |dp| = %.3g\n", wl);
    CHECK(wl < 5e-7);
}

/* (2) The walk emits the law it reads, exactly: Gumbel siblings and rheo (m = 1, 2), roots with one
 * to three children, children with none to two, supports that nest, differ or are disjoint. */
TEST(traversal_two_deep_joint_law_is_exact_by_enumeration) {
    double worst = 0;
    long trees = 0;
    int cases = 0;
    const struct { int scheme, a, b; } shapes[] = {
        { 0, 1, 0 }, { 0, 1, 1 }, { 0, 2, 0 }, { 0, 2, 1 }, { 0, 2, 2 }, { 0, 3, 1 }, { 0, 1, 2 },
        { 0, 3, 2 }, { 0, 2, 3 },
        { 2, 1, 1 }, { 2, 2, 1 }, { 2, 2, 2 }, { 3, 2, 1 }, { 3, 3, 1 }, { 3, 3, 3 }, { 1, 1, 1 }, { 4, 4, 2 },
    };
    for (int V = 4; V <= 6; ++V)
        for (int cfg = 0; cfg < 24; ++cfg) {
            const int qs0 = 1 + (cfg * 2 + 1) % V, ps0 = 1 + (cfg * 3 + 2) % V;
            const SLaw root = small_law((uint64_t)(61000 + V * 50 + cfg), V, qs0, ps0, cfg % 3 == 2 ? 2.5 : 1.0);
            SLaw kid[SV];
            for (int x = 0; x < V; ++x)
                kid[x] = small_law((uint64_t)(67000 + V * 70 + cfg * 13 + x), V, 1 + (cfg + x) % V,
                                   1 + (cfg + 2 * x + 1) % V);
            for (const auto& sh : shapes) {
                const double w = trav_enum_case(root, kid, V, sh.scheme, sh.a, sh.b, 0, &trees);
                worst = std::max(worst, w);
                if (!(w < 1e-11))
                    std::printf("  ENUM FAIL V=%d cfg=%d qs0=%d ps0=%d scheme=%d a=%d b=%d: %g\n", V, cfg, qs0, ps0,
                                sh.scheme, sh.a, sh.b, w);
                CHECK(w < 1e-11);
                ++cases;
            }
        }
    std::printf("  traversal enumeration: %d cases, %ld draft trees, worst |joint - P P| = %.2e\n", cases, trees, worst);
    /* the same statistic sees the mistakes */
    const SLaw root = small_law(61333, 5, 5, 5, 1.0);
    SLaw kid[SV];
    for (int x = 0; x < 5; ++x) kid[x] = small_law((uint64_t)(67333 + x), 5, 4, 5);
    long t2 = 0;
    const double good = trav_enum_case(root, kid, 5, 0, 2, 2, 0, &t2);
    const double c1 = trav_enum_case(root, kid, 5, 0, 2, 2, 1, &t2);
    const double c2 = trav_enum_case(root, kid, 5, 0, 2, 2, 2, &t2);
    const double c3 = trav_enum_case(root, kid, 5, 0, 2, 2, 3, &t2);
    const double rgood = trav_enum_case(root, kid, 5, 2, 2, 2, 0, &t2);
    const double c4 = trav_enum_case(root, kid, 5, 2, 2, 2, 4, &t2);
    std::printf("  controls: exact %.1e | token-level ratios %.3f | no (1 - p) %.3f | no renormalisation %.3f | "
                "rheo exact %.1e, rheo walked as gumbel %.3f\n", good, c1, c2, c3, rgood, c4);
    CHECK(good < 1e-11); CHECK(rgood < 1e-11);
    CHECK(c1 > 1e-3); CHECK(c2 > 1e-3); CHECK(c3 > 1e-3); CHECK(c4 > 1e-3);
}

/* (3) The final token's uniform against the residual law, on a 700-slot plane (holes, a state whose
 * support is partly off the plane): after both children are rejected the root accepts, and the pieces
 * of u -> token have the lengths the walked tree's residual gives each token. */
TEST(the_traversals_final_draw_is_the_residual_law_on_a_wide_plane) {
    constexpr int NC = 700, NV = 5000;
    int mixed = 0;   /* cases where the support slots and the off-support candidates both carry mass */
    for (int rep = 0; rep < 3; ++rep) {
        static int32_t ids[NC];
        static float val[NC];
        int live = 0;
        for (int c = 0; c < NC; ++c) {
            const uint64_t h = H(900000 + (uint64_t)rep * 7919 + (uint64_t)c);
            if (h % 11 == 0) { ids[c] = -1; val[c] = -INFINITY; continue; }
            ids[c] = 1 + c * 7;   /* distinct */
            /* a peaked head and a long tail that still carries a few percent */
            val[c] = (h % 13 == 1) ? -INFINITY
                   : c < 16 ? 6.0f - 0.9f * (float)c - (float)((h >> 20) % 1000) * 0.0005f
                            : 1.5f - 0.012f * (float)c - (float)((h >> 20) % 1000) * 0.002f;
            ++live;
        }
        int32_t qid[8];
        float qv[8];
        for (int k = 0; k < 8; ++k) {
            qid[k] = k < 5 ? 1 + k * 7 : NV - 10 - k;   /* five on the plane's head, three off it */
            qv[k] = 2.0f - (rep == 0 ? 0.35f : rep == 1 ? 0.6f : 0.15f) * (float)k;
        }
        struct WTree {
            RadMtpCoupleState st; int32_t c1, c2;
            const RadMtpCoupleState* state(int v) const { return v == 0 ? &st : nullptr; }
            bool cand(int, const int32_t** i, const float** v, int64_t* n) const { *i = ids; *v = val; *n = NC; return true; }
            int child(int v, int32_t t) const { return v == 0 ? (t == c1 ? 1 : t == c2 ? 2 : -1) : -1; }
            int64_t vocab() const { return NV; }
        } T;
        CHECK(rad_mtp_couple_build_q(qid, qv, NV, &T.st));
        T.st.draw = make_draw(4242 + (uint64_t)rep, 3); T.st.status = 1; T.st.round = 1; T.st.pos = 5;
        rad_mtp_rrs_draw(&T.st, 0, 2);
        T.c1 = T.st.ids[0]; T.c2 = T.st.ids[1];
        RadTravTree kt;
        int map[RAD_MTP_TRAV_MAX_ROWS];
        RadMtpTravOpt o;
        o.accept_at = 3; o.u_final = 0.25;   /* the two leaves reject, the root accepts */
        RadMtpTravOut out;
        CHECK(rad_mtp_trav_walk(T, 0, 1, 1, &out, o, &kt, map));
        CHECK_EQ(out.n, 1);
        CHECK(out.has_y);
        const double woff = kt.moff[0], tot = rad_mtp_trav_sum8(kt.m[0]) + woff;
        std::vector<double> want((size_t)NV, 0.0), got((size_t)NV, 0.0);
        for (int k = 0; k < 8; ++k) want[(size_t)kt.ids[0][k]] += kt.m[0][k] / tot;
        for (int c = 0; c < NC; ++c) {
            if (ids[c] < 0 || val[c] == -INFINITY) continue;
            bool on = false;
            for (int k = 0; k < 8; ++k) on = on || kt.ids[0][k] == ids[c];
            if (!on) want[(size_t)ids[c]] += woff / tot * (double)rad_mtp_trav_mass(val[c], kt.top[0], RAD_MTP_TRAV_QBITS) / (double)kt.off[0];
        }
        const auto pc = pieces([&](double u) {
            RadMtpTravOpt q;
            q.accept_at = 3; q.u_final = u;
            RadMtpTravOut po;
            rad_mtp_trav_walk(T, 0, 1, 1, &po, q);
            return (int)po.y;
        });
        for (const auto& x : pc) { CHECK(x.first >= 0 && x.first < NV); got[(size_t)x.first] += x.second; }
        double worst = 0, off_mass = 0;
        int support = 0;
        for (int y = 0; y < NV; ++y) {
            worst = std::max(worst, std::fabs(got[(size_t)y] - want[(size_t)y]));
            support += want[(size_t)y] > 0;
        }
        off_mass = woff / tot;
        std::printf("  wide final draw %d: %d live slots, %d tokens with mass, off-support share %.3f, %zu pieces, worst |dp| = %.2e\n",
                    rep, live, support, off_mass, pc.size(), worst);
        CHECK(worst < 1e-10);
        CHECK(pc.size() > 300);
        mixed += off_mass > 0.01 && off_mass < 0.99;
    }
    CHECK(mixed >= 2);
}

/* (4) RheoSampling siblings through the walk: the real propose row and grower draft the trees, the
 * walk tries the tail child first and the deterministic ones as point masses. Exact for every kind;
 * the same trees walked as Gumbel ones (what a walk that ignores the scheme does) are not. */
TEST(traversal_verification_of_rheo_siblings_is_exact) {
    const int trials = 200000;
    const Kind kinds[] = { STAGE1, STAGE2, STAGE2B };
    const char* names[] = { "stage1", "stage2", "stage2b" };
    double worst_bad = 0;
    for (int m = 1; m <= 2; ++m)
        for (int k = 0; k < 3; ++k) {
            g_scheme = m + 1;
            const Result r = run_trav(kinds[k], trials, 1.0, RadMtpTravOpt{});
            const double z4 = run_trav4(kinds[k], trials, 1.0);
            const Result row = run_kind(kinds[k], trials, 1.0);
            RadMtpTravOpt bad;
            bad.ctl = 4;
            const Result b = run_trav(kinds[k], trials, 1.0, bad);
            g_scheme = 0;
            std::printf("  trav rheo m=%d %-8s z=%.2f first=%.2f z4=%.2f acc=%.3f (rowwise %.3f) | walked as gumbel: z=%.1f first=%.1f\n",
                        m, names[k], r.z, r.first_z, z4, r.mean_acc, row.mean_acc, b.z, b.first_z);
            CHECK(r.z < 4.5); CHECK(r.first_z < 4.5); CHECK(z4 < 4.5);
            worst_bad = std::max(worst_bad, std::max(b.z, b.first_z));
        }
    CHECK(worst_bad > 20.0);
    /* with pruning (the tail slot cut by q-rank) and a far head */
    const Prune ab{ 0.07, 0.4 };
    g_scheme = 2; g_pr = &ab;
    const Result pr = run_trav(STAGE2, trials, 1.0, RadMtpTravOpt{});
    g_pr = nullptr; g_scheme = 3;
    const Result far = run_trav(STAGE2, 100000, 3.0, RadMtpTravOpt{});
    g_scheme = 0;
    CHECK(pr.z < 4.5); CHECK(pr.first_z < 4.5);
    CHECK(far.z < 4.5); CHECK(far.first_z < 4.5);
}

RAD_TEST_MAIN()
