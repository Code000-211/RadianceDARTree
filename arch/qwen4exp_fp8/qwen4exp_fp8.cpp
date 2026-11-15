/* qwen4exp_fp8 -- Qwen4-Exp (`model_type: qwen4_exp`), the architecture of Qwen3.8-Flash-Next.
 * The checkpoint is dense bf16, quantised ahead of time by a recipe
 * (data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe is the served one): int8 or fp8 trunk linears with
 * matching activations, four-bit routed experts, or bf16 throughout.
 *
 * ============================== RELATION TO QWEN3.5 MOE ==============================
 *
 * In transformers it is `Qwen4ExpTextModel`, a subclass of `Qwen3_5MoeTextModel`, which this tree
 * already serves (arch/qwen35moe_fp8). Most of this file composes the same blocks:
 *
 *   reused                     the Gated DeltaNet (conv width 4, key/value head dim 128), gated
 *                              full attention (`attn_output_gate`: q_proj twice as wide, half of
 *                              it a sigmoid gate on the output), partial rotary (64 of 256 dims),
 *                              the routed feed-forward with a gated shared expert, the MTP
 *                              position.
 *   new                        the gated residual (arch/common/rad_block_hc.h), the n-gram / PLE
 *                              layer (arch/common/rad_block_ple.h), QSA's indexer
 *                              (arch/common/rad_qsa.h), stacked dense expert tensors
 *                              (tools/rad_convert.cpp, slice_stacked), gqa 12, and the delta net's
 *                              sigmoid output gate.
 *
 * Qwen3.8-Flash-Next: 48 layers of 12 x (3 x GatedDeltaNet -> MoE, then 1 x QSA -> MoE), hidden
 * 2560, hc_count 4 so the residual stream is 10240, 512 experts of 640 at top-10 plus a shared 640,
 * 24 query heads and 2 KV heads at head_dim 256, vocab 248320.
 *
 * ============================== THE RESIDUAL STREAM IS FOUR TIMES AS WIDE ==============================
 *
 * The checkpoint has no `input_layernorm`, `post_attention_layernorm` or final `norm`. Instead
 * every block has a gated residual connection: a grouped norm over the 10240-wide stream, a
 * low-rank data-dependent read gate that mixes the four sub-streams down to 2560, and four
 * data-dependent scalars that add the block's output back into all four. There are 96, plus a 97th
 * with no write half that collapses the stream for the lm_head.
 *
 * arch/common/rad_block_hc.h holds the arithmetic and docs/OPS.md's `hc_read` / `hc_write` rows
 * the schema. The blocks do not own their norms: this plugin passes `Src::norm = nullptr`, the fp8
 * blocks' "input already normed and quantised" mode, which forgoes folding the residual add and
 * the all-reduce into the next norm. The all-reduce rides in the gated residual's write instead,
 * where a kernel serves that (HyperConn::Config::reduce, see declare()).
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 * The checkpoint is bf16 throughout (no `quantization_config`), so the quantisation descriptor is
 * empty and the served formats come from a recipe (data/recipes/qwen4exp-*.recipe). The plugin
 * declares each weight at its encoding, and blocks that read several formats choose their ops by
 * it:
 *
 *   block fp8 or int8  self_attn.{q,k,v,o}_proj, linear_attn.{in_proj_qkv, in_proj_z, out_proj},
 *                      the shared expert's three (LinearFP8); the indexer's qk projection
 *                      unless the recipe leaves it bf16, as the served one does
 *   fp8, four-bit      every routed expert's gate_up and down, each layer and projection in its
 *   or bf16            own form (MoeFP8::probe)
 *   E4M3 rows or bf16  each connection's input_mix_weight_{down,up}, whose lowrank of 320 is 2.5
 *                      scale blocks and so is scaled a row group at a time (HyperConn), and the
 *                      MTP head's fc_hidden / fc_embedding
 *   fp8, int8, bf16    the lm_head (whichever logits_gemm kernel reads it)
 *   BF16               every norm gain, A_log, dt_bias, conv1d, in_proj_a / in_proj_b (48 rows is
 *                      not a whole scale block), the router, shared_expert_gate, embed_tokens,
 *                      the PLE projections and inject weights
 *
 * ============================== HOW TENSOR PARALLEL SPLITS AN EXPERT ==============================
 *
 * `moe_intermediate_size` is 640 = 5 x 128. An even tp2 share, 320, is not a whole number of
 * 128-column blocks, on which the fp8 scale block, the 4-bit scale group and the rotation sit. So
 * each rank takes three blocks of the experts of one parity and two of the other's; the router is
 * replicated so both ranks route alike, each rank computes its slice of every routed expert, and
 * the block's existing end-of-block all-reduce sums the partials, with no all-to-all. See the
 * section in declare().
 *
 * ============================== PLE AND QSA ==============================
 *
 * PLE, the n-gram embedding, is 51B of the model's 180B parameters: a 320,001,536-row table of
 * 160-dim embeddings, gathered 16 rows per token by a hash of the last three token ids, read once
 * at layer 1 and injected into the wide stream.
 *
 * QSA's indexer picks the best `indexer_budget / indexer_compress_ratio` = 512 blocks of 4 tokens
 * per query and attends only to those. Below `indexer_budget + indexer_compress_ratio - 1` = 2051
 * tokens of context every complete block is selected, so QSA equals dense causal attention bit for
 * bit; the engine issues the cheaper dense attention there, and it serves as the oracle for QSA.
 * Above it the selection is per query on both prefill and decode (docs/QSA.md,
 * arch/common/rad_qsa.h).
 *
 * The vision tower is declared through arch/common/rad_block_vit.h, only when the container
 * carries one and the deployment asked for media; rad-convert writes only declared weights.
 */
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <arch/rad_arch.h>
#include <rad_prefill_split.h>
#include <rad_mtp_tree.h>
#include <arch/rad_block_attn_gated_fp8.h>
#include <arch/rad_block_dflash2.h>
#include <arch/rad_block_gdn_fp8.h>
#include <arch/rad_block_hc.h>
#include <arch/rad_block_moe_fp8.h>
#include <arch/rad_block_mtp_hc.h>
#include <arch/rad_block_ple.h>
#include <arch/rad_calib_full.h>
#include <arch/rad_qsa.h>
#include <arch/rad_block_vit.h>

namespace qwen4exp_fp8 {

using namespace rad::arch;

/* The delta net's out projection read into the on-die cache beside its decode recurrent update
 * (GdnFP8::op_pf_out): the update runs one workgroup per (sequence, head) and leaves the memory
 * mostly idle, and the projection behind it otherwise streams its weight cold. All of the weight
 * (the read stops at its end), on steps of up to kPrefetchRows rows and kGdnOutPrefetchSeqs
 * sequences. Flash-Next at tp 2 on two R9700s under the draft head at depth 3, ms a step without ->
 * with: one stream 12.78 -> 12.58, two 17.12 -> 16.95, four 23.57 -> 23.52 (level, as eight with
 * the window opened to it). The connection matrices read the same way, beside the recurrent update
 * or beside the writes in front of the reads, cost more than they gave (one stream 12.69 -> 12.88
 * and 13.17): a read here pays ~2 us of launch for what it saves. */
constexpr int64_t kGdnOutPrefetchBytes = (int64_t)8 << 20;
constexpr int64_t kGdnOutPrefetchSeqs  = 4;
/* A full-attention layer's out projection likewise, read behind its q, k and v projections
 * (AttnGatedFP8::op_pf_o): the prologue and the attention between them and the projection are
 * short launches with the memory idle. Twelve layers, so less: five interleaved rounds, ms a step
 * without -> with, one stream 12.51 -> 12.45 (every round), two 16.88 -> 16.85. */
constexpr int64_t kAttnOutPrefetchBytes = (int64_t)8 << 20;

struct Layer {
    int          full = 0;
    HyperConn    hc_mix;     /* in front of the attention or the delta net */
    HyperConn    hc_ffn;     /* in front of the routed feed-forward */
    AttnGatedFP8 attn;
    QsaIndexer   qsa;        /* full-attention layers only; inert when the checkpoint has none */
    GdnFP8       gdn;
    MoeFP8       mlp;
    /* The control vector's projection, applied to the wide stream right after this layer's
     * feed-forward write -- where llama.cpp's `build_cvec(res_hc, il)` sits in the `qwen4exp`
     * graph. Inert (no op) when the deployment has no vector or this layer is outside its range
     * (arch/common/rad_block_hc.h). */
    HcProject    proj;
    /* The layer's calibration inputs (rad_calib_full.h, site_i8), -1 where this run records none:
     * the mixer's input (the delta net's in projection, or q, k and v) and its output projection's. */
    int          cs_in = -1, cs_o = -1;
};

struct Model {
    Names          nm{""};
    Geom           g{};
    GdnFP8::Config gcfg{};
    MoeFP8::Config moecfg{};
    /* The trunk's linears are int8 (LinearFP8::i8), as read from the first delta-net layer's input
     * projection before the activation buffers are declared. */
    bool           trunk_i8 = false;
    /* The trunk's int8 linears are stored rotated (i8a8h), read from the same weight: every
     * reader of the block input then takes the rotated codes the connections write. */
    bool           trunk_rot = false;
    /* --calib-dir over the int8 trunk: the Hessian of each mixer linear's input, from the codes its
     * GEMM reads (rad_calib_full.h). */
    CalibFull      calib;
    HyperConn::Config hccfg{};
    QsaIndexer::Config qsacfg{};
    QsaIndexer::Wire   qsaw{};
    rad_kvgroup        kv_qsa_bkey = 0;   /* created by the first QSA layer's declare() */

    int64_t n_rot = 0;
    /* A draft tree (RadBuildCtx::draft_tree with a head round count): see declare(). */
    bool    tree_on = false;
    /* Stage 2 (RadBuildCtx::draft_tree_width > 1): the head grows the tree itself. The frontier
     * width W, the head's rounds R and the node budget N the selector grows to; the head's own
     * tree-attention operands (the selector writes them for each frontier row) and its state. */
    int64_t tree_w = 1, tree_r = 0, tree_n = 0;
    /* Stage 2's buffers: the proposal (tokens) and its parents plane at one pitch, the grower's
     * state and the next pass's tree-attention operands (mtp_tree_grow), which cross passes. */
    rad_buf b_mtp_ptree = 0, b_mtp_ptok = 0, b_mtp_gstate = 0;
    rad_buf b_mtp_tsel = 0, b_mtp_tnsel = 0, b_mtp_tvmask = 0, b_mtp_trb = 0;
    /* The head's per-node hidden-state store and the routing into it (MtpHcBlock::Wire). */
    rad_buf b_mtp_hstore = 0, b_mtp_tsrc = 0, b_mtp_tdst = 0;
    int64_t tree_pitch = 0;
    static constexpr int kMaxRounds = 16;
    rad_op  op_grow[kMaxRounds] = {};

    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;
    /* PLE lives on exactly one layer; -1 when the checkpoint has none or this build does not
     * declare that layer. */
    PleLayer    ple;
    int64_t     ple_layer = -1;

    /* The wide stream and the narrow block input. `b_h` is the residual stream, hc * n_embd per
     * token; `a_x` is what a connection's read produces and where its block writes the delta. One
     * of each for the whole model: a read is immediately followed by its block and its write, so
     * nothing in `a_x` or `b_inj` outlives the write that consumes it. Declaration order determines
     * the planner's liveness (see the note at declare()), so this holds by this file's order. */
    rad_buf b_h = 0;
    ActFP8  a_x{};
    rad_buf b_inj = 0;

    /* the attention block's own activations, shared across the 12 attention layers */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0, b_rope_cs = 0, b_gate = 0;
    ActFP8  a_attn{};

    /* the delta net's */
    ActFP8  a_in{}, a_go{};
    rad_buf b_ab = 0, b_gq = 0, b_gk = 0, b_gv = 0, b_gdec = 0, b_beta = 0, b_kkt = 0;

    /* the routed feed-forward's */
    rad_buf b_rlogits = 0, b_eids = 0, b_ew = 0, b_sorted = 0, b_eoff = 0, b_ecnt = 0;
    rad_buf b_egu = 0, b_edn = 0, b_emoe = 0, b_sgu = 0, b_sout = 0, b_sgate = 0;
    ActFP8  a_eff{}, a_sff{};
    /* The rotated codes of the block input, for the routed arm only: `q` and `s` are used and `x`
     * stays zero, since the bf16 source is `a_x`, unrotated. Declared only when the container asks
     * for a rotation; zero handles tell the MoE block the pair is absent. */
    ActFP8  a_hr{};
    /* The routed arm's activations are int8 (MoeFP8::a8): int8 connection codes under experts
     * with an int8 row. */
    bool    arm8 = false;

    rad_buf b_hout = 0, b_logits = 0;

    /* ---- the multi-token-prediction head (arch/common/rad_block_mtp_hc.h). Declared only when
     * the deployment speculates, since its ~2.5 GiB of weights would otherwise be loaded and never
     * issued. rad-convert declares at a non-zero max_spec so the container carries everything the
     * model can serve. */
    bool        have_mtp = false;
    MtpHcBlock  mtp{};
    rad_buf     b_mtp_hin = 0, b_mtp_hw = 0, b_mtp_e = 0, b_mtp_hout = 0;
    rad_buf     b_mtp_dq = 0, b_mtp_ds = 0, b_mtp_dsum = 0;
    rad_buf     b_mtp_dlog = 0, b_mtp_didx = 0, b_mtp_dval = 0;
    /* --draft-coupling: [max_seqs] draws staged before the serial passes and [max_spec,
     * max_seqs] states read back after them. PERSIST: the engine writes them from the host and the
     * head writes the states on the card, across a step boundary. */
    rad_buf     b_mtp_couple_draws = 0, b_mtp_couple_state = 0;
    rad_buf     b_mtp_dpair = 0, b_mtp_dgath = 0;

    /* ---- the DFlash2 drafter (arch/common/rad_block_dflash2.h), declared in place of the MTP head
     * when the deployment chose it (drafter_mode) and the container carries one (`draft.`
     * metadata, merged by `rad-convert --draft-model` or appended in place). It conditions on the
     * trunk's completed layers 5, 15, 25, 35, 45, each the mean of the four hyper-connection
     * streams (the vendor's capture patch), copied into one column slice of `b_dfsrc` as the
     * layer finishes. */
    bool          have_df = false;
    Dflash2       df{};
    Dflash2::Wire dfw{};
    rad_kvgroup   kv_draft = 0;
    /* A DFlash2 tree (RadBuildCtx::draft_tree): the drafter grows `max_spec` nodes over its
     * trained block's lattice (dflash_tree) and the trunk verifies them through the same tree
     * attention a serial head's tree uses. */
    bool          df_tree = false;
    enum { MAX_TAPS = 8 };
    int           n_taps = 0;
    int           tap_layer[MAX_TAPS] = {};
    rad_buf       b_dfsrc = 0;                 /* [max_tok, n_taps * n_embd] */
    rad_op        op_tap = 0;

    HcEnter   enter;
    HyperConn mixer;            /* the 97th connection: no write half */
    /* Whether this rank has said that it pipelines a prefill step (step_pipelined), once. */
    bool      pipe_noted = false;
    /* Whether the deployment spans (RadBuildCtx::max_mixed_tok below a max_tok wider than a MoE
     * pass): a step wider than a pass is then one sequence, the connections read it a pass at a
     * time and the indexer scores it in windows. `pipe_on` adds ranks whose blocks issue their own
     * all-reduce at such a step (declare): it is pipelined (step_pipelined) and its all-reduces
     * are declared at `overlap` 1. Without a span neither is set and none of these paths is
     * declared or taken. A sizing declare runs at a max_tok no wider than a pass and takes both
     * from the real declare, so it declares the same ops. */
    bool      span_on = false;
    bool      pipe_on = false;
    /* {0, pass, T} for a step wider than a pass (pipe_bounds): the windows of the indexer's score
     * plane and the delta net's pieces. A transient (declare), uploaded by every step that reads
     * it -- the first two from the pinned host copy `pipe_cu_h`, made once. */
    rad_buf   b_pipe_cu = 0;
    int32_t*  pipe_cu_h = nullptr;
    bool      pipe_cu_tried = false;

    std::vector<Layer> layers;

    rad_weight w_tok = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_rope_cs = 0, op_gather = 0, op_logits = 0;

    /* ---- media (arch/common/rad_block_vit.h): the vision tower and the scatter of its rows over
     * the token embeddings, declared only when the container carries a tower and the deployment
     * asked for media (RadBuildCtx::max_enc_patches). */
    bool        media = false;
    VisionTower vit{};
    MediaRows   mrows{};

};

static Model g_model[MAX_SLOTS];

/* The n-gram table's row count, by prime search.
 *
 * Each of the `heads` hash heads has its own vocabulary: the 1st, 2nd, ... nth prime above
 * `ngram_vocab_size_base - 1`; the table is their sum, padded to `divisor`
 * (`Qwen4ExpTextNGramEmbedding.__init__`). The per-head sizes are in the checkpoint
 * (`ngram_heads_vocab_sizes`) and the served hash reads them there; only the total is derived,
 * because a weight's shape must be declared before any tensor is opened. rad-convert checks it:
 * a multi-source concat's declared leading extent must equal its sources'.
 *
 * Cost: 16 primes near 2e7 by trial division, once at declare. */
static bool ngram_is_prime(int64_t v) {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    for (int64_t d = 3; d * d <= v; d += 2)
        if (v % d == 0) return false;
    return true;
}

static int64_t ngram_table_rows(int64_t base, int64_t heads, int64_t divisor) {
    if (base <= 1 || heads <= 0) return 0;
    int64_t total = 0, p = base - 1;
    for (int64_t i = 0; i < heads; ++i) {
        do { ++p; } while (!ngram_is_prime(p));
        total += p;
    }
    return divisor > 1 ? ((total + divisor - 1) / divisor) * divisor : total;
}

/* A rank's place in the shared expert's uneven block split, where the first places take one
 * block more. MoeFP8::declare hands a plain expert's extra 64-wide tiles to the first ranks by
 * index; the shared expert's extra blocks start at the rank after the last of those, so the two
 * remainders land on different ranks when there are enough. (Routed experts leave no remainder:
 * the parity split at two ranks and the four-class split at four are even.) At four ranks and 640
 * columns (ten tiles, five blocks) ranks 0 and 1 take a tile more and rank 2 the block; at two the
 * place is the rank. */
static int64_t shared_place(int64_t n_ff_all, int world, int rank) {
    const int64_t first = (n_ff_all / 64) % world;
    return (rank - first + world) % world;
}

/* Columns [lo, hi) of the slice at `place` when `nb` fp8 blocks split over `world` places, the
 * first nb % world places taking one block more. */
static void block_slice(int64_t nb, int world, int64_t place, int64_t* lo, int64_t* hi) {
    const int64_t base = nb / world, extra = nb % world;
    const int64_t b0 = place * base + std::min<int64_t>(place, extra);
    *lo = b0 * RAD_FP8_BLOCK;
    *hi = (b0 + base + (place < extra ? 1 : 0)) * RAD_FP8_BLOCK;
}

/* The serial draft tree's default node budget at the operating point of probe(): the head's 3
 * rounds' greedy chain plus the engine's default 3 first-round leaves (`--draft-tree-leaves`,
 * leaves = min(L, N - R)), 7 verify rows with the anchor. */
static constexpr int kTreeNodes = 6;

/* The KV page of the trunk's full-attention layers and of the head's: tokens a block. A draft tree
 * reads its in-flight rows through a 64-key region based at a page boundary (the grower's
 * `block_size`), and the QSA block keys that region's tree attention scores are `ratio` keys wide, so
 * a tree is declared only where the two agree. */
static constexpr int kKvBlock = 4;

/* Stage 2's buffers (RadDrafterDecl::tree_width > 1), declared with the head's candidate planes. The
 * proposal and its parents share a pitch of at least the node budget; the grower's state and the
 * tree-attention operands it leaves for the next pass live across passes (each pass is a step of
 * its own), so they persist. */
static int declare_head_tree_buffers(RadBuilder* b, Names& nm, Model& m) {
    const Geom& g = m.g;
    const int64_t W = m.tree_w, N = m.tree_n, R = m.tree_r;
    m.tree_pitch = std::max<int64_t>(N, kDraftTopR);
    const int64_t sw = rad_mtp_tree_state_words((int)N, (int)W, (int)R);
    /* Persistent as well: a round's frontier is written at the end of one pass and read by the
     * core before the next, whose first op reads the head's carry, which a transient proposal
     * could share bytes with. */
    m.b_mtp_ptok   = decl_b(b, nm.f("mtp_ptok"),  RAD_I32, {g.max_seqs, m.tree_pitch},
                            RAD_BUF_PERSIST);
    m.b_mtp_ptree  = decl_b(b, nm.f("mtp_ptree"), RAD_I32, {g.max_seqs, m.tree_pitch},
                            RAD_BUF_PERSIST);
    m.b_mtp_gstate = decl_b(b, nm.f("mtp_gstate"), RAD_I32, {g.max_seqs, sw}, RAD_BUF_PERSIST);
    m.b_mtp_tsel   = decl_b(b, nm.f("mtp_tsel"),  RAD_I32, {g.max_seqs * W, m.qsacfg.topk + 1},
                            RAD_BUF_PERSIST);
    m.b_mtp_tnsel  = decl_b(b, nm.f("mtp_tnsel"), RAD_I32, {g.max_seqs * W}, RAD_BUF_PERSIST);
    m.b_mtp_tvmask = decl_b(b, nm.f("mtp_tvmask"), RAD_I64, {g.max_seqs * W}, RAD_BUF_PERSIST);
    m.b_mtp_trb    = decl_b(b, nm.f("mtp_trb"),   RAD_I32, {g.max_seqs}, RAD_BUF_PERSIST);
    /* Every node a head pass embeds keeps its output, because the best-first pool may hang a node
     * off any earlier pass's node, not only the last's: Q = 1 + (R - 1) * W rows a sequence (the
     * root and the frontiers of rounds 1 .. R - 1), row `s * Q + node id`. */
    const int64_t Q = 1 + (R - 1) * W;
    m.b_mtp_hstore = decl_b(b, nm.f("mtp_hstore"), g.act_dtype, {g.max_seqs * Q, m.hccfg.hc * g.n_embd},
                            RAD_BUF_PERSIST);
    m.b_mtp_tsrc   = decl_b(b, nm.f("mtp_tsrc"), RAD_I32, {g.max_seqs * W}, RAD_BUF_PERSIST);
    m.b_mtp_tdst   = decl_b(b, nm.f("mtp_tdst"), RAD_I32, {g.max_seqs * W}, RAD_BUF_PERSIST);
    if (!m.b_mtp_ptok || !m.b_mtp_ptree || !m.b_mtp_gstate || !m.b_mtp_tsel || !m.b_mtp_tnsel ||
        !m.b_mtp_tvmask || !m.b_mtp_trb || !m.b_mtp_hstore || !m.b_mtp_tsrc || !m.b_mtp_tdst)
        return RAD_E_INVAL;
    return RAD_OK;
}


/* ============================== THE DRAFTER CHOICE ==============================
 *
 * A container may carry both the model's MTP head and a DFlash2 drafter (`draft.` metadata:
 * `rad-convert --in-place --draft-model`). A deployment serves one of them, chosen by the
 * environment variable RADIANCE_DRAFTER, read by the probe (the default depth) and by declare:
 *
 *     mtp      (default) the serial MTP head, the DARTree serial trees
 *     dflash2  the block drafter; a chain of 3 or 7 drafts (the block is a power of two), or with
 *              --draft-tree on a tree of --num-speculative-tokens nodes over its trained block
 *     both     rad-convert only: declares both so a container written in place keeps and carries
 *              every weight of either
 *
 * The probe cannot see the command line, hence the environment: a first-class flag would be a
 * RadBuildCtx field and a config key in the core. */
enum class Drafter { Mtp, Dflash2, Both };

static bool has_dflash2(const RadModelMeta* meta) {
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    return da && std::strstr(da, "DFlash2");
}

static Drafter drafter_mode(const RadModelMeta* meta) {
    const char* e = std::getenv("RADIANCE_DRAFTER");
    if (!e || !*e || !std::strcmp(e, "mtp")) return Drafter::Mtp;
    if (!std::strcmp(e, "both")) return has_dflash2(meta) ? Drafter::Both : Drafter::Mtp;
    if (!std::strcmp(e, "dflash2")) return has_dflash2(meta) ? Drafter::Dflash2 : Drafter::Mtp;
    return Drafter::Mtp;
}

/* The DFlash2 drafter: its geometry, KV group, buffers and block (as qwen35_fp8's, which this
 * follows). `base_layer` is the first absolute layer of its own KV group. */
static int declare_dflash2(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                           Model& m, int base_layer, bool primary) {
    Names& nm = m.nm;
    Geom   dg{};
    RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "DFlash2"));

    Dflash2::Config dc{};
    dc.n_taps   = m.n_taps;
    dc.block    = ctx->max_spec + 1;
    dc.taps     = rad_meta_geti(meta, "draft.dflash_config.conv_kernel_size", 2);
    dc.group    = rad_meta_geti(meta, "draft.dflash_config.conv_group_size", 16);
    dc.sel_rank = rad_meta_geti(meta, "draft.dflash_config.selector_rank", 0);
    dc.sel_topk = rad_meta_geti(meta, "draft.dflash_config.selector_top_k", 0);
    dc.causal   = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.dflash_config.swa_window_size",
                                  rad_meta_geti(meta, "draft.sliding_window", 0));
    if (dc.sel_rank <= 0 || dc.sel_topk <= 0) {
        fprintf(stderr, "radiance: qwen4exp: the DFlash2 drafter states no selector geometry "
                        "(rank %lld, top_k %lld)\n", (long long)dc.sel_rank, (long long)dc.sel_topk);
        return RAD_E_INVAL;
    }
    const long long trained = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    /* A tree is `max_spec` nodes grown over the lattice of the block the drafter was trained at
     * (gemma4's way): the block is the trained one and --num-speculative-tokens counts nodes. */
    if (m.df_tree && trained > 1) {
        dc.block      = trained;
        dc.tree_nodes = ctx->max_spec;
    }
    if (dc.block < 2 || (dc.block & (dc.block - 1))) {
        fprintf(stderr, "radiance: qwen4exp: DFlash2's convolution masks the block position with "
                        "block_size - 1, so the query block has to be a power of two. This serve "
                        "asks for %d draft tokens, which makes it %lld; use 3 or 7 (a chain), or "
                        "--draft-tree on for a tree of any size up to 15 nodes.\n",
                ctx->max_spec, (long long)dc.block);
        return RAD_E_UNSUPPORTED;
    }
    if (!m.df_tree && trained > 0 && trained != dc.block)
        rad_note(b, "DFlash2 was trained at block_size %lld (%lld draft tokens) and runs at "
                    "%lld; it accepts less away from its trained block length",
                 trained, trained - 1, (long long)dc.block);
    const double escale = rad_meta_getf(meta, "draft.dflash_config.input_embedding_scale", 1.0);
    const double omul   = rad_meta_getf(meta, "draft.dflash_config.output_multiplier", 1.0);
    const double scap   = rad_meta_getf(meta, "draft.dflash_config.final_logit_softcapping", 0.0);
    if (escale != 1.0 || omul != 1.0 || scap != 0.0) {
        fprintf(stderr, "radiance: qwen4exp: this DFlash2 checkpoint scales its input embedding by "
                        "%g and its draft logits by %g, and soft-caps them at %g; none of the three "
                        "is implemented\n", escale, omul, scap);
        return RAD_E_UNSUPPORTED;
    }

    /* The window bounds the cache: the drafter's 2048 tokens, not the context length. */
    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, &m.kv_draft));

    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dflash2::Wire& dw = m.dfw;
    dw.src     = m.b_dfsrc;
    dw.c       = decl_b(b, nm.f("df_c"),    dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.ck      = decl_b(b, nm.f("df_ck"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.cv      = decl_b(b, nm.f("df_cv"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.x       = decl_b(b, nm.f("df_x"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h       = decl_b(b, nm.f("df_h"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h2      = decl_b(b, nm.f("df_h2"),   dg.act_dtype, {qr, dg.n_embd});
    dw.coef    = decl_b(b, nm.f("df_coef"), dg.act_dtype, {qr, dc.coef_dim(dg.n_embd)});
    dw.q       = decl_b(b, nm.f("df_q"),    dg.act_dtype, {qr, qd});
    dw.k       = decl_b(b, nm.f("df_k"),    dg.act_dtype, {qr, kvd});
    dw.v       = decl_b(b, nm.f("df_v"),    dg.act_dtype, {qr, kvd});
    dw.attn    = decl_b(b, nm.f("df_attn"), dg.act_dtype, {qr, qd});
    dw.gate_up = decl_b(b, nm.f("df_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn     = decl_b(b, nm.f("df_ffn"),  dg.act_dtype, {qr, dg.n_ff});
    /* The activation pairs the linears read: the IQR-V4 drafter's six linears and its fc are
     * rotated (trellis or int8, group 128), so every pair is int8 under the Hadamard
     * (had_quant_act_i8g); `probe_i8` tells it from the fc and makes its name map. */
    RAD_ARCH_TRY(Dflash2::probe_i8(b, nm, "draft", &dc.act_i8, &dc.act_g8));
    dc.fc_mapped = true;
    const uint32_t dqt = dc.act_i8 ? RAD_I8 : RAD_F8E4M3;
    const int64_t  dqg = dc.act_i8 ? dc.act_g8 : (int64_t)RAD_FP8_BLOCK;
    if (dc.act_i8 && ((dc.n_taps * dg.n_embd) % dqg || (dg.n_embd % dqg) || (qd % dqg) ||
                      (dg.n_ff % dqg))) {
        fprintf(stderr, "radiance: qwen4exp: a rotated drafter takes widths in whole %lld-column "
                        "scale groups; this one is n_embd %lld, q %lld, n_ff %lld\n", (long long)dqg,
                (long long)dg.n_embd, (long long)qd, (long long)dg.n_ff);
        return RAD_E_INVAL;
    }
    auto dqs = [&](int64_t n, int64_t gr) { return (n + gr - 1) / gr; };
    dw.srcq    = decl_b(b, nm.f("df_srcq"), dqt, {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32, {dg.max_tok, dqs(dc.n_taps * dg.n_embd, dqg)});
    dw.cq      = decl_b(b, nm.f("df_cq"),   dqt,     {dg.max_tok, dg.n_embd});
    dw.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32, {dg.max_tok, dqs(dg.n_embd, dqg)});
    dw.h2q     = decl_b(b, nm.f("df_h2q"),  dqt,     {qr, dg.n_embd});
    dw.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32, {qr, dqs(dg.n_embd, dqg)});
    dw.atq     = decl_b(b, nm.f("df_atq"),  dqt,     {qr, qd});
    dw.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32, {qr, dqs(qd, dqg)});
    dw.ffq     = decl_b(b, nm.f("df_ffq"),  dqt,     {qr, dg.n_ff});
    dw.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32, {qr, dqs(dg.n_ff, dqg)});
    dw.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    dw.dq      = decl_b(b, nm.f("df_dq"),   RAD_I8,  {sr, dg.n_embd});
    dw.ds      = decl_b(b, nm.f("df_ds"),   RAD_F32, {sr, 1});
    dw.dsum    = decl_b(b, nm.f("df_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    /* Candidates rescored against the trunk's int8 head (as the MTP head's are). */
    {
        const uint32_t hd = weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16);
        if ((hd == RAD_F8E4M3 || hd == RAD_I8) && (Dflash2::head_sharded(dg) || dg.world <= 1)) {
            m.df.w_lm  = m.w_lm_head;
            m.df.top_w = Dflash2::kRerankR;
        }
    }
    const int64_t tw = m.df.w_lm ? m.df.top_w : dc.sel_topk;
    dw.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, tw});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, tw});
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, tw, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * tw, 2});
    }
    const int64_t nodes = dc.tree_nodes > 0 ? dc.tree_nodes : dc.steps();
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, nodes});
    if (m.df_tree) {
        dc.tree      = true;
        dc.tree_temp = 1.25;
        dw.parents   = decl_b(b, nm.f("dflash_parents"), RAD_I32, {dg.max_seqs, nodes});
    }
    if (ctx->draft_coupling && !m.df_tree) {
        if (dc.steps() > RAD_MTP_COUPLE_MAX_ROUNDS || dc.sel_topk > RAD_DFLASH_COUPLE_MAX_K ||
            dc.sel_rank % RAD_DFLASH_EDGE_PARTS || dc.sel_rank > RAD_DFLASH_COUPLE_MAX_R) {
            rad_note(b, "--draft-coupling on: DFlash2's coupled walk takes at most %d drafts, %d "
                        "candidates and a selector rank that is a multiple of %d up to %d",
                     RAD_MTP_COUPLE_MAX_ROUNDS, RAD_DFLASH_COUPLE_MAX_K, RAD_DFLASH_EDGE_PARTS,
                     RAD_DFLASH_COUPLE_MAX_R);
            return RAD_E_UNSUPPORTED;
        }
        dc.q_temp = 0.6;
        dw.couple_draws = decl_b(b, nm.f("mtp_couple_draws"), RAD_U32,
                                 {dg.max_seqs, RAD_MTP_COUPLE_DRAW_WORDS}, RAD_BUF_PERSIST);
        dw.couple_state = decl_b(b, nm.f("mtp_couple_state"), RAD_U32,
                                 {dc.steps(), dg.max_seqs, RAD_MTP_COUPLE_STATE_WORDS},
                                 RAD_BUF_PERSIST);
        if (!dw.couple_draws || !dw.couple_state) return RAD_E_INVAL;
    } else if (ctx->draft_coupling) {
        fprintf(stderr, "radiance: qwen4exp: --draft-coupling does not combine with a DFlash2 "
                        "tree\n");
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), dw));
    m.have_df = true;
    if (m.df_tree && !m.df.cfg.tree) {
        fprintf(stderr, "radiance: qwen4exp: a DFlash2 tree of %lld nodes needs dflash_tree, which "
                        "no library serves here\n", (long long)nodes);
        return RAD_E_UNSUPPORTED;
    }

    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = nodes;
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: qwen4exp: this DFlash2 checkpoint declares no "
                        "dflash_config.mask_token_id\n");
        return RAD_E_FORMAT;
    }
    /* Not the program's drafter when the MTP head is (a convert declares both, only to carry
     * every weight): one step has one draft head. */
    if (!primary) return RAD_OK;
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    if (m.df_tree) {
        RAD_ARCH_TRY(rad_drafter_tree(b, dw.parents));
        rad_note(b, "DFlash2 drafter: a tree of %lld nodes a step (softmax temperature %.3g) over "
                    "the lattice of its %lld-row block, verified in %lld rows", (long long)nodes,
                 dc.tree_temp, (long long)dc.block, (long long)nodes + 1);
    }
    rad_note(b, "DFlash2 drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x %lld, "
                "ffn %lld, %s attention with a %lld-token window, %d trunk tap(s) (the mean of the "
                "%lld hyper-connection streams); one forward proposes %lld tokens through the "
                "candidate selector, its head %s",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head_all, (long long)dg.n_head_kv_all, (long long)dg.head_dim,
             (long long)dg.n_ff_all, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)m.hccfg.hc, (long long)dc.steps(),
             m.df.head_i8 ? "8-bit" : "2-bit");
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;

    /* A sizing declare writes scratch: the step reads g_model, filled by the real declare at the
     * real max_tok (RadBuildCtx::shape_probe). */
    static thread_local Model probe_model;
    Model& m = ctx->shape_probe ? probe_model : g_model[rad_build_slot(ctx)];
    /* The reset drops the pinned window copy (pipe_bounds); free it first -- a re-declare of the
     * same rank (every rebuild, every test case) otherwise overwrites the pointer and leaks the
     * pin. The copy is read only by a step's memcpys, and no step runs across a declare. */
    if (m.pipe_cu_h) rad_dev_free(m.pipe_cu_h, RAD_MEM_HOST_PINNED);
    m = Model{};
    m.nm = Names(ctx->scope ? ctx->scope : "");
    Names& nm = m.nm;

    /* arch_splits_ff: the plugin splits the routed experts' width (640 = five fp8 blocks) itself,
     * by whole blocks and parity, at any world up to the block count (including three, which 640
     * does not divide), so geom_from must not refuse it. attn_world_ok: this plugin's attention
     * blocks (rad_block_attn_gated_fp8.h) serve a smaller attention world when the two KV heads do
     * not divide the world. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16, /*arch_splits_ff=*/true,
                           /*attn_world_ok=*/true));
    Geom& g = m.g;

    /* DARTree: the engine asks for a tree (RadBuildCtx::draft_tree) where the probe advertised one
     * (probe, tree_nodes), with `max_spec` the node budget N, `draft_rounds` the head's rounds R
     * and the leaf count and frontier width stage 1 and stage 2 grow by. Everything that depends on
     * the deepest speculation (`g.max_spec`: the tail's ring, the window groups the core sizes,
     * the logits rows, the work list) is already N, since the Geom carries it. A tree needs the
     * head, so it is off at max_spec 0 or with no head round. */
    const Drafter dmode = drafter_mode(meta);
    const bool    dmode_df = dmode == Drafter::Dflash2 && ctx->max_spec > 0;
    /* A DFlash2 tree verifies as a serial head's does, through the trunk's tree attention, with
     * no head rounds (a block drafter proposes the whole lattice at once). */
    m.df_tree = dmode_df && ctx->draft_tree != 0;
    m.tree_on = m.df_tree ||
                (!dmode_df && ctx->draft_tree != 0 && ctx->draft_rounds > 0 && ctx->max_spec > 0 &&
                 ctx->draft_rounds <= ctx->max_spec);
    if (m.tree_on) {
        /* The region's visibility mask is 64 keys: (ratio - 1) committed keys before the anchor
         * and every row of the verify block (docs/QSA.md). The ratio is not known yet; the exact
         * bound max_spec + ratio <= 64 is checked once the QSA geometry is read, this is the
         * grower's own node limit. */
        if (ctx->max_spec > 60) {
            fprintf(stderr, "radiance: qwen4exp: a draft tree of %d nodes does not fit the QSA "
                            "region mask (60 at most)\n", ctx->max_spec);
            return RAD_E_UNSUPPORTED;
        }
        /* The delta net's tree verify walks at most 16 rows a sequence (8 on the fold, which the
         * tree steps do not issue). */
        if (ctx->max_spec + 1 > 16) {
            fprintf(stderr, "radiance: qwen4exp: a draft tree of %d nodes exceeds the delta net's "
                            "16 rows a verify\n", ctx->max_spec);
            return RAD_E_UNSUPPORTED;
        }
        if (ctx->draft_rounds > Model::kMaxRounds) {
            fprintf(stderr, "radiance: qwen4exp: a draft tree of %d head rounds (%d at most)\n",
                    ctx->draft_rounds, Model::kMaxRounds);
            return RAD_E_UNSUPPORTED;
        }
        m.tree_r = ctx->draft_rounds;
        m.tree_n = ctx->max_spec;
        m.tree_w = std::max(ctx->draft_tree_width, 1);
        /* Stage 2 grows W frontier rows a pass out of at most N nodes: a wider frontier is a
         * misconfiguration, refused rather than clamped (the core stages the width it declared). */
        if (m.tree_w > m.tree_n) {
            fprintf(stderr, "radiance: qwen4exp: --draft-tree-width %lld exceeds the %lld nodes of "
                            "the draft tree\n", (long long)m.tree_w, (long long)m.tree_n);
            return RAD_E_UNSUPPORTED;
        }
        /* --draft-coupling with a tree: recursive rejection sampling over the siblings
         * (abi/rad_mtp_rrs.h). Stage 1 draws its leaves in round 1's propose op, whose state
         * plane is one row a round; stage 2's grower writes one state row a node. */
        if (ctx->draft_coupling) {
            if (m.tree_w <= 1 && ctx->draft_rounds > RAD_MTP_COUPLE_MAX_ROUNDS) {
                fprintf(stderr, "radiance: qwen4exp: --draft-coupling with a draft tree of %d head "
                                "rounds (%d at most)\n", ctx->draft_rounds,
                        RAD_MTP_COUPLE_MAX_ROUNDS);
                return RAD_E_UNSUPPORTED;
            }
            if (m.tree_w > 1 && ctx->max_spec > RAD_MTP_COUPLE_MAX_ROWS) {
                fprintf(stderr, "radiance: qwen4exp: --draft-coupling with a stage-2 draft tree of "
                                "%d nodes (%d at most)\n", ctx->max_spec, RAD_MTP_COUPLE_MAX_ROWS);
                return RAD_E_UNSUPPORTED;
            }
            if (m.tree_w > RAD_MTP_RRS_MAX_WIDTH) {
                fprintf(stderr, "radiance: qwen4exp: --draft-coupling with a draft tree frontier of "
                                "%lld rows (%d at most)\n", (long long)m.tree_w,
                        RAD_MTP_RRS_MAX_WIDTH);
                return RAD_E_UNSUPPORTED;
            }
        }
        /* A tree step writes the KV of rows the verify may drop and compacts them in place
         * (tree_compact); a streamed group's host mirror and ring have no such step. */
        if (ctx->kv_stream_ok && ctx->kv_resident_blocks > 0) {
            fprintf(stderr, "radiance: qwen4exp: a draft tree does not combine with a streamed KV "
                            "cache (--kv-resident)\n");
            return RAD_E_UNSUPPORTED;
        }
    }

    /* Qwen4-Exp's norms are all `x_hat * (1 + w)` (`Qwen3NextRMSNorm.forward` is
     * `_norm(x.float()) * (1.0 + weight.float())`, weight initialised to zeros); the delta net's
     * gated output norm is plain `weight * x` (initialised to ones). Taken from the reference. */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;

    /* ---- the gated residual's geometry. */
    m.hccfg.hc      = meta_int2(meta, "qwen4exp.hc_count",   "hc_count",   0);
    m.hccfg.lowrank = meta_int2(meta, "qwen4exp.hc_lowrank", "hc_lowrank", 0);
    if (m.hccfg.hc <= 1 || m.hccfg.lowrank <= 0) {
        fprintf(stderr, "radiance: qwen4exp: the container does not describe the gated residual "
                        "(hc_count %lld, hc_lowrank %lld). Every norm in this architecture is one "
                        "of these connections, so there is nothing to fall back to.\n",
                (long long)m.hccfg.hc, (long long)m.hccfg.lowrank);
        return RAD_E_INVAL;
    }
    /* Every block's all-reduce rides in the following write where a kernel serves that
     * (HyperConn::Config::reduce): the block's partial sum is the write's input, so the pair is one
     * launch. Each block is told below whether its connection's fused op resolved. */
    m.hccfg.reduce = 1;
    const int64_t HN = m.hccfg.hc * g.n_embd;

    /* ---- partial rotary: 0.25 of head_dim 256 is 64; the other 192 dims pass through. */
    {
        const double pr = rad_meta_getf(meta, "text_config.rope_parameters.partial_rotary_factor",
                          rad_meta_getf(meta, "rope_parameters.partial_rotary_factor",
                          rad_meta_getf(meta, "partial_rotary_factor", 1.0)));
        m.n_rot = (int64_t)((double)g.head_dim * pr);
        if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot % 2)) {
            fprintf(stderr, "radiance: qwen4exp: partial_rotary_factor %.6g of head_dim %lld is "
                            "%lld rotary dims, which is not an even count in (0, head_dim]\n",
                    pr, (long long)g.head_dim, (long long)m.n_rot);
            return RAD_E_INVAL;
        }
    }
    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));
    /* There is deliberately no serve-time layer slice. A model too large for the device needs a
     * converter flag that writes a named partial container, not a switch that silently serves less
     * than the model. */

    /* ---- the layer schedule. */
    const int64_t interval = meta_int2(meta, "qwen4exp.full_attention_interval",
                                       "full_attention_interval", 0);
    if (interval <= 1) {
        fprintf(stderr, "radiance: qwen4exp: full_attention_interval is %lld; this is a hybrid "
                        "architecture and the interval decides which layers are linear\n",
                (long long)interval);
        return RAD_E_INVAL;
    }
    /* The config file says "full_attention" for the QSA layers (HuggingFace's post_init rewrites
     * them to "qwen_sparse_attention" in memory), so both spellings mean a full layer. */
    RAD_ARCH_TRY(check_layer_schedule("qwen4exp", meta_text_gets(meta, "layer_types", nullptr),
                                      g.n_layer, interval,
                                      {"full_attention", "qwen_sparse_attention"}));

    /* ---- the delta net. */
    {
        GdnFP8::Config& gc = m.gcfg;
        const int64_t nk = meta_int2(meta, "qwen4exp.ssm.key_head_count",
                                     "linear_num_key_heads", 0);
        const int64_t nv = meta_int2(meta, "qwen4exp.ssm.value_head_count",
                                     "linear_num_value_heads", 0);
        gc.head_k     = meta_int2(meta, "qwen4exp.ssm.key_head_dim",   "linear_key_head_dim",   0);
        gc.head_v     = meta_int2(meta, "qwen4exp.ssm.value_head_dim", "linear_value_head_dim", 0);
        gc.conv_width = meta_int2(meta, "qwen4exp.ssm.conv_kernel",    "linear_conv_kernel_dim", 0);
        gc.chunk      = 64;
        /* `output_gate_type` is a checkpoint property: Qwen3.5/3.6 leave it unset and mean silu,
         * Qwen4-Exp states "sigmoid". libr4d's gated norm has both. */
        const char* ogt = rad_meta_gets(meta, "text_config.output_gate_type",
                          rad_meta_gets(meta, "output_gate_type", nullptr));
        if (!ogt || !*ogt) ogt = rad_meta_gets(meta, "text_config.hidden_act",
                                rad_meta_gets(meta, "hidden_act", "silu"));
        gc.act = ogt;
        if (nk <= 0 || nv <= 0 || gc.head_k <= 0 || gc.head_v <= 0 || gc.conv_width <= 0) {
            fprintf(stderr, "radiance: qwen4exp: the container does not describe the linear "
                            "layers (key heads %lld, value heads %lld, key width %lld, value "
                            "width %lld, conv width %lld)\n",
                    (long long)nk, (long long)nv, (long long)gc.head_k, (long long)gc.head_v,
                    (long long)gc.conv_width);
            return RAD_E_INVAL;
        }
        /* Heads split evenly when they can and unevenly otherwise: 16 K heads over three ranks
         * are 6, 5 and 5, each rank's V heads the same multiple of its K heads (48 / 16 = 3 ->
         * 18, 15, 15), declared as spans of the container's rows (rad_block_gdn_fp8.h,
         * Config::k_first). Nothing is padded; the kernels take any head count. A V count that is
         * not a multiple of the K count is refused. */
        gc.n_head_k_all = nk;
        gc.n_head_v_all = nv;
        if (g.world > 1 && (nk % g.world != 0 || nv % g.world != 0)) {
            if (nv % nk != 0) {
                fprintf(stderr, "radiance: qwen4exp: %lld value heads over %lld key heads do not "
                                "group, so an uneven split over %d ranks is not defined\n",
                        (long long)nv, (long long)nk, g.world);
                return RAD_E_INVAL;
            }
            const int64_t ratio = nv / nk, base = nk / g.world, rem = nk % g.world;
            /* Extra heads go first to ranks without attention (rad_arch.h remainder_place);
             * with none, this is the rank's own place. */
            const int64_t pl = remainder_place(g, g.rank, 0);
            gc.n_head_k = base + (pl < rem ? 1 : 0);
            gc.k_first  = pl * base + (pl < rem ? pl : rem);
            gc.n_head_v = ratio * gc.n_head_k;
            gc.v_first  = ratio * gc.k_first;
            rad_note(b, "rank %d takes delta-net K heads [%lld, %lld) of %lld and V heads "
                        "[%lld, %lld) of %lld", g.rank,
                    (long long)gc.k_first, (long long)(gc.k_first + gc.n_head_k), (long long)nk,
                    (long long)gc.v_first, (long long)(gc.v_first + gc.n_head_v), (long long)nv);
        } else {
            int s;
            if ((s = divide_or_fail(nk, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0) return s;
            if ((s = divide_or_fail(nv, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0) return s;
        }
    }

    /* ---- the routed feed-forward.
     *
     * n_ff is `moe_intermediate_size`, divided by world_size when the world divides it and whole
     * otherwise (geom_from's `arch_splits_ff`). At more than one rank the split below sets the
     * rank's width, by whole fp8 blocks and parity (see the file header). */
    {
        MoeFP8::Config& mc = m.moecfg;
        mc.n_expert    = meta_int2(meta, "qwen4exp.expert_count",      "num_experts", 0);
        mc.top_k       = meta_int2(meta, "qwen4exp.expert_used_count", "num_experts_per_tok", 0);
        mc.n_ff_exp    = g.n_ff;
        mc.n_ff_shared = meta_int2(meta, "qwen4exp.expert_shared_feed_forward_length",
                                   "shared_expert_intermediate_size", 0);
        /* The shared expert is split by whole fp8 scale blocks: evenly when possible, otherwise
         * as evenly as whole blocks allow, by the rank's place (shared_place). Its 640 columns are
         * five blocks: three on rank 0 and two on rank 1; at four ranks two on rank 2 and one on
         * each other. See MoeFP8::Config::shared_lo. */
        if (mc.n_ff_shared > 0 && g.world > 1 && (mc.n_ff_shared / g.world) % RAD_FP8_BLOCK == 0) {
            int s;
            if ((s = divide_or_fail(mc.n_ff_shared, g.world,
                                    "shared_expert_intermediate_size", &mc.n_ff_shared)) < 0)
                return s;
        } else if (mc.n_ff_shared > 0 && g.world > 1) {
            if (mc.n_ff_shared % RAD_FP8_BLOCK) {
                fprintf(stderr, "radiance: the shared expert is %lld columns, which is not a whole "
                                "number of %d-column fp8 scale blocks, so it cannot be split "
                                "across %d ranks.\n", (long long)mc.n_ff_shared, RAD_FP8_BLOCK,
                        g.world);
                return RAD_E_INVAL;
            }
            const int64_t nb = mc.n_ff_shared / RAD_FP8_BLOCK;
            block_slice(nb, g.world, shared_place(g.n_ff_all, g.world, g.rank),
                        &mc.shared_lo, &mc.shared_hi);
            mc.n_ff_shared = mc.shared_hi - mc.shared_lo;
            rad_note(b, "the shared expert does not split evenly into %d-column fp8 scale blocks: "
                        "rank %d computes columns [%lld, %lld) of it, %lld of %lld blocks, and the "
                        "all-reduce sums the partials", RAD_FP8_BLOCK, g.rank,
                     (long long)mc.shared_lo, (long long)mc.shared_hi,
                     (long long)(mc.n_ff_shared / RAD_FP8_BLOCK), (long long)nb);
        }
        mc.norm_topk   = 1;
        /* GPTQ calibration run, enabled by the same setting that names the output directory (a
         * calibration with nowhere to write would be a 1.34 GB allocation and two launches per
         * layer for nothing). A separate run from serving; see MoeFP8::Config::calib. */
        mc.calib       = calib_dir() != nullptr ? 1 : 0;
        /* Tokens per pass. 2048 is the linear-state checkpoint interval the scheduler clamps the
         * prefill chunk to, so the shipped configuration is one pass. It also sizes four buffers at
         * `rows * top_k`, so a smaller max_tok lowers it. */
        mc.rows        = g.max_tok < 2048 ? g.max_tok : 2048;
        if (mc.n_expert <= 0 || mc.top_k <= 0 || mc.top_k > mc.n_expert) {
            fprintf(stderr, "radiance: qwen4exp: the container describes %lld expert(s) with "
                            "top_k %lld\n", (long long)mc.n_expert, (long long)mc.top_k);
            return RAD_E_INVAL;
        }
        /* ---- the experts' stored format (block fp8, four-bit, four-bit rotated, or plain bf16),
         * read from one expert's encoding (MoeFP8::probe) before anything is sized by it: the
         * split between ranks, the rotated copy of the block input, the connection in front of
         * every routed block, and for bf16 every linear in the model. The encoding comes from the
         * container, checkpoint or recipe being applied, so a container is always served in the
         * format it holds.
         *
         * The probed expert is the first one this rank owns under expert parallelism: the answer
         * decides whether that split is used, and the expert is declared on this rank either way
         * (whole, or a slice under the parity split), so the block skips exactly the map the probe
         * made. */
        mc.probed_expert = (g.world > 1 && mc.n_expert % g.world == 0)
                               ? (int64_t)g.rank * (mc.n_expert / g.world) : 0;
        {
            MoeFP8::Src ps{};
            RAD_ARCH_TRY(moe_expert_sources(b, nm, "format",
                                            "model.language_model.layers.0.mlp.experts", &ps));
            RAD_ARCH_TRY(MoeFP8::probe(b, nm, 0, ps, &mc));
        }
        /* Plain experts mean a container converted without a quantising recipe, so every linear
         * is plain too: the blocks issue bf16 GEMMs over bf16 activations and nothing quantises
         * (Geom::w_bf16). Each linear refuses a quantised encoding then. The converse is allowed:
         * a quantised container may keep a linear plain (the QSA indexer's projection), which
         * LinearFP8 reads as bf16. */
        g.w_bf16 = mc.expert_bf16 != 0;
        /* The connections in front of the blocks then emit no codes, since nothing reads them. */
        m.hccfg.quant = g.w_bf16 ? 0 : 1;
        /* ============================== EVERY RANK HOLDS A SLICE OF EVERY EXPERT ==============================
         *
         * An expert is 640 columns, five 128-column blocks (the fp8 scale block, the 4-bit scale
         * group and the rotation width), so two ranks cannot halve one.
         *
         * The split is by parity: each parity's five blocks are cut contiguously over the ranks,
         * and the rank taking the extra block alternates with the expert's parity
         * (MoeFP8::Config::ff_lo). At two ranks, rank r's slice of expert e is the first three
         * blocks when e % 2 == r and the last two otherwise. The router is replicated, every rank
         * computes its slice of every routed expert, and the block's all-reduce sums the partials.
         *
         * At four ranks parity alone would give ranks 0 and 1 three blocks of every expert pair
         * and ranks 2 and 3 two, leaving 2 and 3 waiting in the all-reduce. So the extra block
         * rotates over four places by expert number mod 4 (four slice tables, `moe_gemm_q_mod4`),
         * and every rank computes five blocks of every four experts. A layer keeping two experts
         * plain bf16 has 510 quantised ones, and its first two classes are one expert longer.
         * Other remainders stay apart: the plain experts' extra tiles go to ranks 0 and 1
         * (MoeFP8::declare), the shared expert's extra block to rank 2 (shared_place).
         *
         * Not expert parallelism (rank r owning experts [r*E/W, (r+1)*E/W) whole): bytes per rank
         * are equal on average, but routing is random, and a verify step's forty routed slots leave
         * one rank about five experts ahead, costing about a millisecond per step at one sequence
         * in the all-reduce wait. With parity the ranks differ by a fifth of an expert per unit of
         * even/odd imbalance. Whole experts would also break the recipe's pairing of protected
         * experts, which keeps each rank's quantised-expert count even for the two parity tables.
         *
         * A calibration run does split whole experts per rank: the down projection's Hessian spans
         * an expert's whole K, and a slice would lack the cross terms. Whole experts give each rank
         * the full Hessian of every expert it holds, which the packer sums (rad_arch.h,
         * calib_dir). */
        mc.n_expert_all = mc.n_expert;
        /* bf16 experts are also split whole per rank: a slice of a bf16 expert is two row ranges
         * and a column range, which the file tier cannot read as one run, and a bf16 model is
         * served to calibrate and compare, not for decode speed. */
        if (g.world > 1 && (mc.calib || mc.expert_bf16)) {
            int s;
            int64_t per = 0;
            if ((s = divide_or_fail(mc.n_expert_all, g.world, "expert_count", &per)) < 0) return s;
            mc.n_ff_exp    = g.n_ff_all;
            mc.n_expert    = per;
            mc.expert_base = (int64_t)g.rank * per;
            rad_note(b, "%s: rank %d owns experts [%lld, %lld) of %lld whole.",
                     mc.calib ? "calibration (each expert's Hessian is over its whole width)"
                              : "bf16 experts", g.rank,
                     (long long)mc.expert_base, (long long)(mc.expert_base + mc.n_expert),
                     (long long)mc.n_expert_all);
        } else if (g.world > 1) {
            if (g.n_ff_all % RAD_FP8_BLOCK) {
                fprintf(stderr, "radiance: qwen4exp: an expert is %lld columns, which is not a whole "
                                "number of %d-column blocks and cannot be split across ranks.\n",
                        (long long)g.n_ff_all, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            const int64_t nb = g.n_ff_all / RAD_FP8_BLOCK;
            if (nb % g.world == 0) {
                mc.n_ff_exp = g.n_ff_all / g.world;
            } else if (g.world <= nb) {
                /* The uneven split is two slices per rank: each parity's blocks are cut
                 * contiguously across the ranks and the rank taking the odd block alternates with
                 * the parity, so slices are whole blocks and every block has exactly one owner.
                 * Widths are ceil(nb/W) and floor(nb/W). Every rank must get at least one block of
                 * each parity, since an empty slice still joins the all-reduce, so the world may
                 * not exceed nb. */
                for (int q = 0; q < 2; ++q) {
                    /* This rank's place in expert parity q's split: first for its own parity.
                     * Experts do not go remainder-first (rad_arch.h remainder_place): an expert
                     * block is a whole GEMM over every routed token, and a rank taking the extra
                     * block of every class becomes the slowest rank (at three ranks on
                     * Flash-Next, a 50/25/25 split cost 25% per step single-stream). Only the
                     * delta net's remainder, a head's worth of work, goes to the rank without
                     * attention. */
                    block_slice(nb, g.world, (g.rank - q + g.world) % g.world, &mc.ff_lo[q],
                                &mc.ff_hi[q]);
                }
                mc.n_ff_exp = std::max(mc.ff_hi[0] - mc.ff_lo[0], mc.ff_hi[1] - mc.ff_lo[1]);
                /* At four ranks, four classes by expert number mod 4 (MoeFP8::Config::ff_lo4),
                 * used by every layer instead: the same cut with the rank's place shifted by the
                 * class, so the extra block rotates over all four ranks. Every block still has one
                 * owner. */
                if (g.world == 4) {
                    for (int q = 0; q < 4; ++q) {
                        block_slice(nb, 4, (g.rank - q + 4) % 4, &mc.ff_lo4[q], &mc.ff_hi4[q]);
                        mc.n_ff_exp = std::max(mc.n_ff_exp, mc.ff_hi4[q] - mc.ff_lo4[q]);
                    }
                    rad_note(b, "routed experts at four classes: rank %d computes columns [%lld, "
                                "%lld) [%lld, %lld) [%lld, %lld) [%lld, %lld) of the experts "
                                "numbered 0, 1, 2 and 3 mod 4, %lld blocks per four experts like "
                                "every rank",
                             g.rank, (long long)mc.ff_lo4[0], (long long)mc.ff_hi4[0],
                             (long long)mc.ff_lo4[1], (long long)mc.ff_hi4[1],
                             (long long)mc.ff_lo4[2], (long long)mc.ff_hi4[2],
                             (long long)mc.ff_lo4[3], (long long)mc.ff_hi4[3], (long long)nb);
                } else {
                    rad_note(b, "routed experts: rank %d computes columns [%lld, %lld) of each "
                                "even expert and [%lld, %lld) of each odd one, of %lld; %lld "
                                "blocks of %d do not split evenly over %lld ranks, so the extra "
                                "block alternates with the expert's parity",
                             g.rank, (long long)mc.ff_lo[0], (long long)mc.ff_hi[0],
                             (long long)mc.ff_lo[1], (long long)mc.ff_hi[1],
                             (long long)g.n_ff_all, (long long)nb, RAD_FP8_BLOCK,
                             (long long)g.world);
                }
            } else {
                fprintf(stderr, "radiance: qwen4exp: an expert is %lld blocks of %d, which %d ranks "
                                "cannot each take a whole block of in both parities. Lower the rank "
                                "count, or serve whole experts (a bf16 container or a calibration "
                                "run does).\n",
                        (long long)nb, RAD_FP8_BLOCK, g.world);
                return RAD_E_INVAL;
            }
        }
    }

    /* ---- the n-gram embedding's layer and the indexer's geometry, each noted where it changes
     * what this build computes. */
    {
        const char* ple = rad_meta_gets(meta, "text_config.ple_layer_ids",
                          rad_meta_gets(meta, "ple_layer_ids", nullptr));
        if (ple && *ple) {
            /* One-indexed: `ple_layer_ids: [2]` is layer_idx 1 (the reference uses
             * `config.ple_layer_ids.index(layer_idx + 1)` and validates [1, num_hidden_layers]). */
            m.ple_layer = (int64_t)std::strtoll(ple, nullptr, 10) - 1;
            if (m.ple_layer < 0 || m.ple_layer >= g.n_layer) {
                /* Outside the declared layers: the n-gram embedding is not in this build, and the
                 * note says so rather than dropping it silently. */
                rad_note(b, "ple_layer_ids = %s is outside the %lld layers declared, so the n-gram "
                            "embedding is not in this build.", ple, (long long)g.n_layer);
                m.ple_layer = -1;
            }
        }
    }
    {
        const int64_t budget = meta_int2(meta, "qwen4exp.indexer_budget", "indexer_budget", 0);
        const int64_t ratio  = meta_int2(meta, "qwen4exp.indexer_compress_ratio",
                                         "indexer_compress_ratio", 0);
        m.qsacfg.budget   = budget;
        m.qsacfg.ratio    = ratio;
        m.qsacfg.heads    = meta_int2(meta, "qwen4exp.indexer_n_heads", "indexer_n_heads", 0);
        m.qsacfg.head_dim = meta_int2(meta, "qwen4exp.indexer_head_dim", "indexer_head_dim", 0);
        m.qsacfg.rot        = m.n_rot;
        m.qsacfg.topk       = (ratio > 0) ? budget / ratio : 0;
        m.qsacfg.max_blocks = (ratio > 0) ? (g.max_ctx + ratio - 1) / ratio : 0;
        m.qsacfg.max_work   = (ratio > 0) ? (g.max_tok + ratio - 1) / ratio + g.max_seqs : 0;
        /* A tree deployment's heal rebuilds, a sequence at most, the block keys the previous
         * step's committed tokens completed: ceil(max_spec / ratio) beyond the work a plain step
         * has (docs/QSA.md). */
        if (m.tree_on && ratio > 0)
            m.qsacfg.max_work += g.max_seqs * ((g.max_spec + ratio - 1) / ratio);
        /* The trunk's layers follow spec_parent and spec_path; the head's own layer, copied from
         * this Config, switches it off again (see the head's declaration). The region tables are
         * RW wide: (deepest path + ratio) / ratio + 1 blocks, 8 at least. */
        m.qsacfg.tree       = m.tree_on && ratio > 0;
        m.qsacfg.region     = std::max<int64_t>(8, ratio > 0 ? (g.max_spec + ratio) / ratio + 1 : 0);
        /* The tail's ring (see QsaIndexer::Config::ring), sized for `max_spec`, the deepest
         * speculation any step can carry. */
        m.qsacfg.ring       = ratio + g.max_spec;
        if (budget > 0 && ratio > 0) {
            /* The exactness bound: with `pos + 1 <= budget + ratio - 1` tokens visible there are
             * at most budget/ratio complete blocks, the indexer's top-k keeps all of them, and the
             * tail is always attended, so QSA equals dense causal attention bit for bit. */
            const int64_t exact_to = budget + ratio - 1;
            /* The context bound is not capped: prefill selects per query exactly as decode does
             * (arch/common/rad_qsa.h, docs/QSA.md). There is deliberately no force-dense switch,
             * since above the bound dense attention is not the trained model. */
            if (g.max_ctx > exact_to)
                rad_note(b, "QSA: sparse past %lld tokens of context, %lld blocks of %lld per "
                            "query at prefill and decode; at or below that the engine issues the "
                            "bit-identical dense attention",
                         (long long)exact_to, (long long)(budget / ratio), (long long)ratio);
            else
                rad_note(b, "QSA: dense attention is exact to %lld tokens of context and "
                            "--max-model-len is %lld, so the indexer selects every visible block",
                         (long long)exact_to, (long long)g.max_ctx);
        }
    }

    rad_note(b, "gated residual: %lld streams of %lld (%lld wide), low-rank gate %lld; %lld "
                "connections plus the mixer",
             (long long)m.hccfg.hc, (long long)g.n_embd, (long long)HN,
             (long long)m.hccfg.lowrank, (long long)(2 * g.n_layer));

    /* ---- the KV groups: the same three as qwen35_fp8. */
    {
        RadKVGroupDecl fd{};
        fd.kind      = RAD_KV_FULL;
        fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
        fd.n_head_kv = g.n_head_kv;
        fd.head_dim  = g.head_dim;
        /* The MTP head's layer shares this group. */
        m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
        g.kv_dtype = rad_dtype_name(fd.dtype);

        RadKVGroupDecl sd{};
        sd.kind         = RAD_KV_LINEAR;
        sd.dtype        = RAD_F32;   /* f16 trades draft acceptance for a sliver of step time */
        sd.n_head_kv    = m.gcfg.n_head_v;
        sd.state_dim[0] = m.gcfg.head_v;
        sd.state_dim[1] = m.gcfg.head_k;
        m.kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);

        RadKVGroupDecl cd{};
        cd.kind       = RAD_KV_CONV;
        cd.dtype      = RAD_BF16;
        cd.n_head_kv  = 2 * m.gcfg.n_head_k + m.gcfg.n_head_v;
        cd.head_dim   = m.gcfg.head_k;
        cd.conv_width = m.gcfg.conv_width;
        m.kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);
        if (!m.kv_full || !m.kv_state || !m.kv_conv) return RAD_E_INVAL;

        /* ---- the QSA indexer's tail, which is state, not a buffer.
         *
         * It carries an unfinished block's raw index keys across steps, so it is indexed by
         * sequence and layer, which a step-local buffer cannot express (arch/common/rad_qsa.h,
         * Wire). As a linear group the engine keeps a slot stable for the sequence's life, zeroes
         * it on allocation, and saves and restores it with the linear state on prefix-cache
         * checkpoints, so a resume onto a half-built block reads defined data.
         *
         * `n_head_kv` 1 and `state_dim` {ring, head_dim}: `ring` raw keys, a few KiB per layer per
         * sequence. */
        if (m.qsacfg.heads > 0 && m.qsacfg.head_dim > 0 && m.qsacfg.ring > 0) {
            RadKVGroupDecl td{};
            td.kind         = RAD_KV_LINEAR;
            td.dtype        = g.act_dtype;
            td.n_head_kv    = 1;
            td.state_dim[0] = m.qsacfg.ring;
            td.state_dim[1] = m.qsacfg.head_dim;
            m.qsacfg.kv_tail = rad_decl_kv_group(b, nm.f("kv_qsa_tail"), &td);
            if (!m.qsacfg.kv_tail) return RAD_E_INVAL;
        }
    }

    /* ---- the trunk's stored format, read from the first delta-net layer's input projection as
     * the experts' is from one expert. The activation buffers below carry an int8 code pair only
     * when the trunk's linears are int8 (LinearFP8::i8), so it must be known first. The probe maps
     * the weight through the layer's own LinearFP8, which then does not map it again. */
    m.layers.resize((size_t)g.n_layer);
    {
        const char* pre0 = "model.language_model.layers";
        for (int64_t l = 0; l < g.n_layer; ++l) {
            if (is_full_attention(l, interval)) continue;
            RAD_ARCH_TRY(m.layers[(size_t)l].gdn.in.probe(
                b, nm, g, nm.f("blk.%lld.ssm_inz", (long long)l),
                { nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight", pre0, (long long)l),
                  nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight", pre0, (long long)l) }));
            m.trunk_i8 = m.layers[(size_t)l].gdn.in.probed_i8;
            /* Stored rotated (i8a8h: int8 codes or a narrower form on their grid, /fwht128): its
             * readers take had_quant_act_i8g's codes, one scale a 128 of K as the connections
             * write them. */
            RadEncoding enc{};
            m.trunk_rot = m.trunk_i8 &&
                          weight_enc(b, nm.f("blk.%lld.ssm_inz.weight", (long long)l), &enc) &&
                          (linear_is_w8h(enc) || linear_is_wnh(enc));
            if (m.trunk_rot && linear_w8h_group(enc) != RAD_FP8_BLOCK) {
                fprintf(stderr, "radiance: qwen4exp: the trunk's rotated int8 linears keep a scale "
                                "a %lld of K; the connections write the block input's codes at one "
                                "a %d\n", (long long)linear_w8h_group(enc), RAD_FP8_BLOCK);
                return RAD_E_UNSUPPORTED;
            }
            break;
        }
    }

    /* ---- buffers. */
    const int64_t q_dim    = g.q_dim();
    const int64_t kv_dim   = g.kv_dim();
    const int64_t v_dim    = m.gcfg.v_dim();
    const int64_t conv_dim = m.gcfg.conv_dim();
    const int64_t T        = g.max_tok;
    const int64_t ER       = m.moecfg.rows * m.moecfg.top_k;

    m.b_h   = decl_b(b, nm.f("h_wide"), g.act_dtype, {T, HN});
    m.a_x.x = decl_b(b, nm.f("x"),      g.act_dtype, {T, g.n_embd});
    if (ctx->shape_probe) {
        m.span_on = g_model[rad_build_slot(ctx)].span_on;
        m.pipe_on = g_model[rad_build_slot(ctx)].pipe_on;
    } else {
        m.span_on = ctx->max_mixed_tok > 0 && ctx->max_mixed_tok < g.max_tok &&
                    g.max_tok > m.moecfg.rows;
        /* Only where the blocks issue their own all-reduce at a step wider than a pass: at two
         * ranks the gated residual's fused write takes every step's (ar_exact_write_rows), so such
         * a step runs serially. */
        m.pipe_on = m.span_on && g.world > 1 && ar_exact_write_rows(g) < g.max_tok && !dmode_df;
    }
    /* The pipelined step's all-reduces run on the second lane, in place over `x`, so it is the one
     * buffer that lane touches; it is live over the whole program anyway. */
    if (m.pipe_on) RAD_ARCH_TRY(rad_buf_concurrent(b, m.a_x.x));
    /* A transient although only its last entry changes per step: a persistent buffer sits in the
     * fixed region under every transient, and one more alignment unit there moves every
     * activation by 256 bytes, which on Flash-Next at four ranks slows the MoE gather by 73 % and
     * a prefill by 3 %. Every step that reads it rewrites it (pipe_bounds). */
    if (m.span_on)
        m.b_pipe_cu = decl_b(b, nm.f("pipe_cu"), RAD_I32, {4});
    /* Under a span every connection reads a step wider than a pass a pass at a time
     * (HyperConn::read_pieces, and the pipelined step's own reads), so the reads are declared no
     * wider than a pass and their scratch is sized for that band. */
    if (m.span_on) m.hccfg.read_rows = m.moecfg.rows;
    RAD_ARCH_TRY(m.a_x.declare_qs(b, nm, g, "x", g.n_embd, 0, m.trunk_i8));
    /* The connections write int8 codes themselves when every reader of their codes is an int8
     * linear: the trunk's are, and the routed experts then read the rotated copy or the bf16
     * plane. When experts read the plain E4M3 codes, the connections keep those and the int8
     * linears quantise for themselves. */
    /* A read's rotated copy takes its plain codes' dtype, so beside int8 codes the routed arm is
     * int8 too: the experts read had_quant_act_i8g's codes through the grouped GEMM's int8 rows
     * (MoeFP8::a8), which the rotated four-bit plane, the codebook at E4M3 scales a 64 or 32 of K
     * and rotated int8 experts have. The codebook at a bf16 scale a 128 and the five-bit one do
     * not: under them the connections keep E4M3 codes, as for unrotated experts. */
    const bool exp8 = m.moecfg.expert_rot && !m.moecfg.expert_w5 &&
                      (!m.moecfg.expert_nl || m.moecfg.expert_g64);
    m.hccfg.codes_i8 = (m.trunk_i8 && (exp8 || m.moecfg.expert_bf16)) ? 1 : 0;
    m.a_x.q8_fed     = m.hccfg.codes_i8 != 0;
    m.arm8           = m.hccfg.codes_i8 != 0 && exp8;
    /* A rotated trunk and the routed experts read one pair: the connections write the block
     * input's rotated codes as its own (HyperConn::xrot), and the routed arm takes those
     * (MoeFP8's gate_up input) with no second copy. */
    if (m.trunk_rot) {
        if (!m.hccfg.codes_i8) {
            fprintf(stderr, "radiance: qwen4exp: a rotated int8 trunk shares the block input's "
                            "rotated codes with the routed experts, which then are rotated and "
                            "have an int8 row (four-bit, the codebook at a 64 or 32 of K, int8), "
                            "or plain bf16\n");
            return RAD_E_UNSUPPORTED;
        }
        m.a_x.q8_rot = true;
    }
    m.b_inj = decl_b(b, nm.f("hc_inj"), g.act_dtype, {T, m.hccfg.hc});

    m.b_qg     = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {T, g.n_head, 2 * g.head_dim});
    m.b_k      = decl_b(b, nm.f("attn_k"),   g.act_dtype, {T, kv_dim});
    m.b_v      = decl_b(b, nm.f("attn_v"),   g.act_dtype, {T, kv_dim});
    m.b_q      = decl_b(b, nm.f("attn_q"),   g.act_dtype, {T, q_dim});
    m.a_attn.x = decl_b(b, nm.f("attn_out"), g.act_dtype, {T, q_dim});
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim, 0, m.trunk_i8));
    /* The gate epilogue in front of the int8 out projection writes its int8 codes itself. */
    m.a_attn.q8_fed = m.trunk_i8;
    m.a_attn.q8_rot = m.trunk_rot;
    /* The bf16 gate's own plane (AttnGatedFP8::Wire::gate); the fp8 path fuses it away. */
    if (g.w_bf16) m.b_gate = decl_b(b, nm.f("attn_gate"), g.act_dtype, {T, q_dim});
    /* The QSA indexer's planes, shared by every full-attention layer like `attn_out`. Declared
     * only when the checkpoint carries an indexer; otherwise the layer's declare() is a no-op. */
    if (m.qsacfg.heads > 0 && m.qsacfg.head_dim > 0) {
        m.qsaw.qk = decl_b(b, nm.f("qsa_qk"), g.act_dtype,
                           { g.max_tok, (m.qsacfg.heads + 1) * m.qsacfg.head_dim });
        m.qsaw.q  = decl_b(b, nm.f("qsa_q"),  g.act_dtype,
                           { g.max_tok, m.qsacfg.heads * m.qsacfg.head_dim });
        const int64_t hd = m.qsacfg.head_dim, rt = m.qsacfg.ratio, wk = m.qsacfg.max_work;
        m.qsaw.stage = decl_b(b, nm.f("qsa_stage"), g.act_dtype, { wk * rt, hd });
        m.qsaw.page  = decl_b(b, nm.f("qsa_page"),  RAD_I32, { wk });
        /* Three components a block under M-RoPE (arch/common/rad_qsa.h). */
        m.qsaw.bpos  = g.rope_mc ? decl_b(b, nm.f("qsa_bpos"), RAD_I32, { wk, 3 })
                                 : decl_b(b, nm.f("qsa_bpos"), RAD_I32, { wk });
        /* Padded to int_affine's 16 entries a launch when a tree is verified: the region bases
         * rb = ratio * nc are made from it (QsaIndexer::step). */
        const int64_t nc_rows = m.qsacfg.tree ? ((g.max_seqs + 15) / 16) * 16 : g.max_seqs;
        m.qsaw.nc    = decl_b(b, nm.f("qsa_nc"),    RAD_I32, { nc_rows });
        if (m.qsacfg.tree) {
            /* A tree verify's scratch and outputs, indexed by absolute batch row, so max_tok tall
             * (a mixed step's prefill rows are inert and still index). */
            const int64_t RW = m.qsacfg.region;
            m.qsaw.skey   = decl_b(b, nm.f("qsa_skey"),   g.act_dtype, { g.max_tok, hd });
            m.qsaw.sstage = decl_b(b, nm.f("qsa_sstage"), g.act_dtype, { g.max_tok * rt, hd });
            m.qsaw.spage  = decl_b(b, nm.f("qsa_spage"),  RAD_I32, { g.max_tok });
            m.qsaw.sbpos  = g.rope_mc ? decl_b(b, nm.f("qsa_sbpos"), RAD_I32, { g.max_tok, 3 })
                                      : decl_b(b, nm.f("qsa_sbpos"), RAD_I32, { g.max_tok });
            m.qsaw.rnc    = decl_b(b, nm.f("qsa_rnc"),    RAD_I32, { g.max_tok });
            m.qsaw.rcu    = decl_b(b, nm.f("qsa_rcu"),    RAD_I32, { g.max_tok + 1 });
            m.qsaw.rtab   = decl_b(b, nm.f("qsa_rtab"),   RAD_I32, { g.max_tok, RW });
            m.qsaw.rscore = decl_b(b, nm.f("qsa_rscore"), RAD_F32, { g.max_tok, RW });
            m.qsaw.vmask  = decl_b(b, nm.f("qsa_vmask"),  RAD_I64, { g.max_tok });
            m.qsaw.rb     = decl_b(b, nm.f("qsa_rb"),     RAD_I32, { nc_rows });
            if (!m.qsaw.skey || !m.qsaw.sstage || !m.qsaw.spage || !m.qsaw.sbpos || !m.qsaw.rnc ||
                !m.qsaw.rcu || !m.qsaw.rtab || !m.qsaw.rscore || !m.qsaw.vmask || !m.qsaw.rb)
                return RAD_E_INVAL;
        }
        /* A pass tall under a span whose shared steps fit in a pass (RadBuildCtx::
         * max_mixed_tok): a wider step is one sequence and scores and selects in windows of a
         * pass (QsaIndexer::step, with the bounds in b_pipe_cu). The plane is the largest buffer
         * the span would otherwise grow, max_ctx / ratio floats a query row (512 MiB more at 4096
         * rows and 256K of context). */
        const int64_t qsa_rows = (m.span_on && ctx->max_mixed_tok <= m.moecfg.rows)
                                     ? m.moecfg.rows : g.max_tok;
        m.qsacfg.plane_rows = qsa_rows < g.max_tok ? qsa_rows : 0;
        /* One row per query, not per sequence (arch/common/rad_qsa.h). The score plane is the
         * only buffer whose size is the product of two deployment bounds, so an oversized one is
         * refused here, naming the two flags, rather than failing later as an allocation. */
        const int64_t sc_elems = qsa_rows * m.qsacfg.max_blocks;
        if (sc_elems > (int64_t)512 * 1024 * 1024) {
            fprintf(stderr,
                "radiance: qwen4exp: the QSA score plane would be %lld MiB -- %lld rows "
                "(--max-num-batched-tokens %lld) by --max-model-len %lld / compress ratio %lld. "
                "Lower either.\n",
                (long long)(sc_elems * 4 / (1024 * 1024)), (long long)qsa_rows,
                (long long)g.max_tok, (long long)g.max_ctx, (long long)m.qsacfg.ratio);
            return RAD_E_UNSUPPORTED;
        }
        m.qsaw.score = decl_b(b, nm.f("qsa_score"), RAD_F32,
                              { qsa_rows, m.qsacfg.max_blocks });
        m.qsaw.sel   = decl_b(b, nm.f("qsa_sel"),   RAD_I32,
                              { g.max_tok, m.qsacfg.topk + 1 });
        m.qsaw.nsel  = decl_b(b, nm.f("qsa_nsel"),  RAD_I32, { g.max_tok });
        m.qsaw.sequ  = decl_b(b, nm.f("qsa_sequ"),  RAD_I32, { g.max_tok });
        if (!m.qsaw.qk || !m.qsaw.q || !m.qsaw.stage || !m.qsaw.page ||
            !m.qsaw.bpos || !m.qsaw.nc || !m.qsaw.score || !m.qsaw.sel || !m.qsaw.nsel ||
            !m.qsaw.sequ)
            return RAD_E_INVAL;
    }
    /* A draft tree verifies through the QSA tree attention (`attn_paged_tree_sel`) of every full
     * layer: it needs the indexer's tree scratch and region mask, and a mask wide enough for the
     * anchor, the (ratio - 1) committed keys of its block and N rows. The probe advertised a tree
     * for the head's presence alone, so this is where a container without the indexer is refused. */
    if (m.tree_on) {
        const int64_t ratio = m.qsacfg.ratio;
        if (ratio <= 0 || !m.qsacfg.tree || !m.qsaw.qk || !m.qsaw.vmask || !m.qsaw.rb) {
            fprintf(stderr, "radiance: qwen4exp: a draft tree needs the QSA indexer's tree "
                            "attention, which this container has no indexer for\n");
            return RAD_E_UNSUPPORTED;
        }
        if (ratio != kKvBlock) {
            fprintf(stderr, "radiance: qwen4exp: a draft tree needs the indexer's compress ratio "
                            "(%lld) to equal the KV page (%d tokens)\n", (long long)ratio,
                    kKvBlock);
            return RAD_E_UNSUPPORTED;
        }
        if (ctx->max_spec + 1 + (ratio - 1) > 64) {
            fprintf(stderr, "radiance: qwen4exp: a draft tree of %d nodes does not fit the QSA "
                            "region mask (%lld at most at ratio %lld)\n", ctx->max_spec,
                    (long long)(64 - ratio), (long long)ratio);
            return RAD_E_UNSUPPORTED;
        }
    }

    m.a_in.x = decl_b(b, nm.f("gdn_in"), g.act_dtype, {T, conv_dim + v_dim});
    RAD_ARCH_TRY(m.a_in.declare_qs(b, nm, g, "gdn_in", conv_dim + v_dim));
    m.b_ab   = decl_b(b, nm.f("gdn_ab"),   g.act_dtype, {T, m.gcfg.ab_dim()});
    m.b_gq   = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {T, m.gcfg.n_head_k, m.gcfg.head_k});
    m.b_gk   = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {T, m.gcfg.n_head_k, m.gcfg.head_k});
    m.b_gv   = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {T, m.gcfg.n_head_v, m.gcfg.head_v});
    /* Both f32, not the activation dtype; `gdn_conv_prep` requires it and otherwise fails the
     * first step with "dtype this kernel does not serve". The log decay is a cumulative sum whose
     * differences the chunked scan exponentiates, and bf16's eight mantissa bits are not enough for
     * it or for beta. */
    m.b_gdec = decl_b(b, nm.f("gdn_gdec"), RAD_F32,  {T, m.gcfg.n_head_v});
    m.b_beta = decl_b(b, nm.f("gdn_beta"), RAD_F32,  {T, m.gcfg.n_head_v});
    m.b_kkt  = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16, {T, m.gcfg.n_head_v, m.gcfg.chunk});
    m.a_go.x = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {T, m.gcfg.n_head_v, m.gcfg.head_v});
    RAD_ARCH_TRY(m.a_go.declare_qs(b, nm, g, "gdn_o", v_dim, 0, m.trunk_i8));
    /* The delta net's output epilogue also writes its int8 codes; its prefill quantiser is the
     * int8 one. */
    m.a_go.q8_fed = m.trunk_i8;
    m.a_go.q8_rot = m.trunk_rot;

    /* The model's expert count, not this rank's: the router is replicated and scores every
     * expert. */
    m.b_rlogits = decl_b(b, nm.f("moe_logits"), g.act_dtype, {T, m.moecfg.n_expert_all});
    m.b_eids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,     {T, m.moecfg.top_k});
    m.b_ew      = decl_b(b, nm.f("moe_ew"),     g.act_dtype, {T, m.moecfg.top_k});
    m.b_eoff    = decl_b(b, nm.f("moe_eoff"),   RAD_I32,     {m.moecfg.n_expert + 1});
    /* PERSIST, not transient. `moe_scatter` writes the per-expert histogram and no declared op
     * reads it (rad_route_report is a handoff, not an op), so as a transient liveness would let a
     * later buffer alias it, and a wide prefill chunk would corrupt the counts (even negative).
     *
     * The model's routing table (rad_route_table): a row for each trunk layer and one for the
     * draft head, at the model's expert count, which is the core's pitch whatever share of the
     * experts this rank computes. Each router writes its layer's row, so a report copies nothing
     * and a layer's counts last until its next pass. */
    m.b_ecnt    = decl_b(b, nm.f("moe_ecnt"),   RAD_I32,
                         {g.n_layer + 1, m.moecfg.n_expert_all}, RAD_BUF_PERSIST);
    RAD_ARCH_TRY(rad_route_table(b, m.b_ecnt));
    m.b_sorted  = decl_b(b, nm.f("moe_sorted"), RAD_I32,     {ER});
    m.b_egu     = decl_b(b, nm.f("moe_egu"),    g.act_dtype, {ER, 2 * m.moecfg.n_ff_exp});
    m.a_eff.x   = decl_b(b, nm.f("moe_eff"),    g.act_dtype, {ER, m.moecfg.n_ff_exp});
    /* ER rows, not max_tok: an expert plane has one row per (token, slot). declare_qs's default
     * would make it `top_k` times too short and the gated quantiser would overrun it (see
     * ActFP8::declare_qs). */
    if (m.arm8) {
        /* The int8 routed arm's two pairs, rotated at a scale a 128 and written by their
         * producers: the gated product's by gated_had_quant_i8g, the input's by the read. */
        RAD_ARCH_TRY(m.a_eff.declare_qs(b, nm, g, "moe_eff", m.moecfg.n_ff_exp, ER, true, false,
                                        RAD_FP8_BLOCK));
        m.a_eff.q8_fed = m.a_eff.q8_rot = true;
        if (!m.trunk_rot) {
            RAD_ARCH_TRY(m.a_hr.declare_qs(b, nm, g, "moe_hr", g.n_embd, 0, true, false,
                                           RAD_FP8_BLOCK));
            m.a_hr.q8_fed = m.a_hr.q8_rot = true;
        }
    } else {
        RAD_ARCH_TRY(m.a_eff.declare_qs(b, nm, g, "moe_eff", m.moecfg.n_ff_exp, ER));
        if (m.moecfg.expert_rot)
            RAD_ARCH_TRY(m.a_hr.declare_qs(b, nm, g, "moe_hr", g.n_embd));
    }
    m.b_edn     = decl_b(b, nm.f("moe_edn"),    g.act_dtype, {ER, g.n_embd});
    m.b_emoe    = decl_b(b, nm.f("moe_out"),    g.act_dtype, {T, g.n_embd});
    if (m.moecfg.n_ff_shared > 0) {
        m.b_sgu   = decl_b(b, nm.f("moe_sgu"),   g.act_dtype, {T, 2 * m.moecfg.n_ff_shared});
        m.a_sff.x = decl_b(b, nm.f("moe_sff"),   g.act_dtype, {T, m.moecfg.n_ff_shared});
        RAD_ARCH_TRY(m.a_sff.declare_qs(b, nm, g, "moe_sff", m.moecfg.n_ff_shared, 0,
                                         m.trunk_i8));
        /* The shared expert's gated quantiser in front of its int8 down projection writes its codes
     * too. */
        m.a_sff.q8_fed = m.trunk_i8;
        m.a_sff.q8_rot = m.trunk_rot;
        m.b_sout  = decl_b(b, nm.f("moe_sout"),  g.act_dtype, {T, g.n_embd});
        m.b_sgate = decl_b(b, nm.f("moe_sgate"), g.act_dtype, {T, 1});
    }

    m.b_hout   = decl_b(b, nm.f("hout"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
    /* The DFlash2 taps, only when the container carries the drafter and the deployment chose it:
     * the trunk's completed layers, each the mean of the hc streams (`mhc_contract`: one launch,
     * written into the tap's column slice at the buffer's row pitch), 25 KiB a token at five
     * taps. */
    if (g.max_spec > 0 && (dmode == Drafter::Dflash2 || dmode == Drafter::Both)) {
        m.n_taps = parse_layer_ids(rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr),
                                   g.n_layer, m.tap_layer, Model::MAX_TAPS);
        if (m.n_taps <= 0) {
            fprintf(stderr, "radiance: qwen4exp: the DFlash2 drafter names no target layer this "
                            "%lld-layer trunk has (draft.dflash_config.target_layer_ids)\n",
                    (long long)g.n_layer);
            return RAD_E_FORMAT;
        }
        m.b_dfsrc = decl_b(b, nm.f("dflash_src"), g.act_dtype, {T, (int64_t)m.n_taps * g.n_embd});
        m.op_tap = rw(b, RAD_OP(b, "mhc_contract",
                          RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                     RAD_INT("hc", m.hccfg.hc), RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {m.b_h}, {m.b_dfsrc});
        if (!m.b_dfsrc || !m.op_tap) return RAD_E_INVAL;
    }
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges, both row-sharded as in qwen35_fp8. This checkpoint ships
     * `lm_head.weight` (tie_word_embeddings is false); the alternative source covers a smaller
     * family member that ties them.
     *
     * The head is declared over all its planes at the codes' dtype (bf16 as shipped, block fp8
     * where a recipe makes it so); the selector then picks a `logits_gemm` kernel whose layout
     * reads that encoding. The name maps come first, since they are how the head is queried. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    if (!m.w_tok || !m.w_lm_head) return RAD_E_INVAL;
    /* The rank's rows, stated explicitly: a vocabulary the world does not divide (248320 at three
     * ranks) is split by rad_arch.h's rule (ceil to a 128-row block, last rank short), not the
     * core's even share. Where it divides, the span equals the even share. */
    if (g.world > 1) {
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_tok,     g.vocab_off, g.vocab_off + g.n_vocab));
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_lm_head, g.vocab_off, g.vocab_off + g.n_vocab));
    }

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", g.n_embd),
                                  RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                  RAD_INT("vocab_offset", g.vocab_off)),
                       RAD_WEIGHTS(m.w_tok)),
                   {}, {m.a_x.x});
    /* The encoder's rows overwrite the lookup before its reduction (see MediaRows). */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, T, g.n_embd, m.a_x.x));
    /* Always exact: this sums disjoint rows, a token's embedding coming from exactly one rank. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                              RAD_PARAMS(RAD_INT("world_size", g.world),
                                         RAD_RANGE("numel", g.n_embd, T * g.n_embd),
                                         RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                              RAD_NOWEIGHTS),
                          {m.a_x.x}, {m.a_x.x});

    /* The stream's initial value: hc identical copies of the embedding. */
    RAD_ARCH_TRY(m.enter.declare(b, g, m.hccfg.hc, m.a_x.x, m.b_h));

    /* The control vector's per-token switch, when this deployment found one (HcProject). All zeros
     * is the stock model, so a request that does not ask for the vector is unaffected. */
    const rad_buf proj_on = HcProject::declare_mask(b, nm.f("projection_on"), g, ctx->cvec);

    /* ---- the layers.
     *
     * Declaration order is the buffer planner's liveness: a transient lives over [first
     * definition, last use] in op indices, and transients with disjoint ranges may share bytes
     * (core/build/rad_bufplan.cpp). The buffers above are shared by every layer, so the order
     * below must match step()'s issue order (read, block, write, ...). Declaring a connection out
     * of order would alias buffers live in the same step, silently corrupting the output.
     */
    m.op_rope_cs = rw(b, decl_rotary(b, g, "rope_table",
                         RAD_PARAMS(RAD_RANGE("M", 1, qk_fuse_rows(g)),
                                    RAD_INT("rot", m.n_rot), RAD_F64("theta", g.theta),
                                    RAD_F64("scale", g.rope_scale), RAD_STR("dtype", "f32")),
                         RAD_NOWEIGHTS),
                     {}, {m.b_rope_cs = decl_b(b, nm.f("rope_cs"), RAD_F32,
                                               {rope_table_rows(g), m.n_rot})});

    AttnGatedFP8::Wire aw{};
    aw.x = m.b_h; aw.h = m.a_x; aw.qg = m.b_qg; aw.k = m.b_k; aw.v = m.b_v;
    aw.q = m.b_q; aw.attn = m.a_attn; aw.gate = m.b_gate;
    aw.cos_sin = m.op_rope_cs ? m.b_rope_cs : 0;
    /* The indexer reads the same rope table: its rotation uses `m.n_rot`, `g.theta` and
     * `g.rope_scale`, as the table does, which saves the trunk layers and the head's passes a norm
     * and rope each. Set here because the table is declared after `m.qsaw`'s buffers;
     * `tw.qsa = m.qsaw` below passes it on to the MTP head. */
    m.qsaw.cos_sin = aw.cos_sin;

    GdnFP8::Wire gw{};
    gw.x = m.b_h; gw.h = m.a_x; gw.in = m.a_in; gw.ab = m.b_ab;
    gw.q = m.b_gq; gw.k = m.b_gk; gw.v = m.b_gv;
    gw.gdec = m.b_gdec; gw.beta = m.b_beta; gw.kkt = m.b_kkt; gw.o = m.a_go;

    MoeFP8::Wire mw{};
    mw.x = m.b_h; mw.h = m.a_x; mw.hr = m.a_hr;
    mw.logits = m.b_rlogits; mw.ids = m.b_eids; mw.ew = m.b_ew;
    mw.sorted = m.b_sorted;  mw.eoff = m.b_eoff; mw.ecnt = m.b_ecnt;
    mw.ecnt_pitch = m.moecfg.n_expert_all;
    mw.egu = m.b_egu; mw.eff = m.a_eff; mw.edn = m.b_edn; mw.moe = m.b_emoe;
    mw.sgu = m.b_sgu; mw.sff = m.a_sff; mw.sout = m.b_sout; mw.sgate = m.b_sgate;

    HyperConn::Wire hw{};
    hw.h = m.b_h; hw.x = m.a_x; hw.inj = m.b_inj;

    m.layers.resize((size_t)g.n_layer);
    const char* pre = "model.language_model.layers";
    /* The pipelined step's all-reduces run beside the other piece's work, so each block declares
     * its collective a second time at `overlap` 1, for the rows that leave the CUs to that work
     * (the lane rows of libr4d/r4d_rows.cpp). */
    const int pipe_overlap = m.pipe_on ? 1 : 0;
    /* GPTQ calibration of the trunk (rad_calib_full.h). A bf16 trunk is recorded at any rank count
     * from the bf16 inputs, a column-split one gathered to its whole width; an int8 trunk at one
     * rank from the codes the connections and the mixers write themselves, so each tap reads what
     * the GEMM after it reads. */
    if (ctx->calib_dir && *ctx->calib_dir && m.trunk_i8 &&
        (g.world > 1 || !m.a_x.q8_fed || !m.a_attn.q8_fed || !m.a_go.q8_fed)) {
        fprintf(stderr, "radiance: qwen4exp: --calib-dir records an int8 trunk under int8-routed "
                        "experts at --tp 1 only\n");
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(m.calib.begin(b, nm, g, ctx, g.n_layer, std::max(q_dim, v_dim) * g.world,
                               std::max<int64_t>(g.n_embd, std::max(q_dim, v_dim) * g.world)));
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;
        lay.attn.ar_overlap = lay.gdn.ar_overlap = lay.mlp.ar_overlap = pipe_overlap;

        /* PLE comes first, before the attention connection reads the stream: the reference adds
         * it into `hidden_states` at the top of the decoder layer, so the following gated residual
         * normalises a stream that already includes the n-gram features. It is declared here, not
         * hoisted, so its buffers' liveness starts where they are used. */
        if (l == m.ple_layer) {
            PleLayer::Config pc{};
            pc.hc         = m.hccfg.hc;
            pc.embed_dim  = meta_int2(meta, "qwen4exp.ple_embed_dim", "ple_embed_dim", 0);
            pc.ngram      = meta_int2(meta, "qwen4exp.ngram_size", "ngram_size", 0);
            const int64_t per_ngram =
                meta_int2(meta, "qwen4exp.heads_per_ngram", "heads_per_ngram", 0);
            pc.heads      = (pc.ngram - 1) * per_ngram;
            pc.eos        = meta_int2(meta, "qwen4exp.eos_token_id", "eos_token_id", -1);
            pc.shards     = meta_int2(meta, "qwen4exp.split_ngram_parts", "split_ngram_parts", 0);
            pc.conv_width = meta_int2(meta, "qwen4exp.ple_conv_kernel_size",
                                      "ple_conv_kernel_size", 0);
            /* The dilation is the n-gram size (`conv_dilation = config.ngram_size` in the
             * reference), not a separate key and not 1. */
            pc.dilation   = pc.ngram;
            pc.rows       = ngram_table_rows(
                meta_int2(meta, "qwen4exp.ngram_vocab_size_base", "ngram_vocab_size_base", 0),
                pc.heads,
                meta_int2(meta, "qwen4exp.make_ngram_vocab_size_divisible_by",
                          "make_ngram_vocab_size_divisible_by", 1));

            PleLayer::Wire pw{};
            pw.h = m.b_h;

            PleLayer::Src ps{};
            ps.shard_prefix = nm.ckpt("%s.%lld.ple.ple_embedding.ngram_embedding.shard_",
                                      pre, (long long)l);
            ps.shard_suffix = ".weight";
            ps.mult         = nm.ckpt("%s.%lld.ple.ple_embedding.layer_multipliers",
                                      pre, (long long)l);
            ps.vocab_sizes  = nm.ckpt("%s.%lld.ple.ple_embedding.ngram_heads_vocab_sizes",
                                      pre, (long long)l);
            ps.offsets      = nm.ckpt("%s.%lld.ple.ple_embedding.ngram_heads_offsets",
                                      pre, (long long)l);
            ps.key_proj     = nm.ckpt("%s.%lld.ple.key_proj.weight",   pre, (long long)l);
            ps.value_proj   = nm.ckpt("%s.%lld.ple.value_proj.weight", pre, (long long)l);
            ps.norm_key     = nm.ckpt("%s.%lld.ple.norm_key.weight",   pre, (long long)l);
            ps.norm_query   = nm.ckpt("%s.%lld.ple.norm_query.weight", pre, (long long)l);
            ps.norm_conv    = nm.ckpt("%s.%lld.ple.norm_conv.weight",  pre, (long long)l);
            ps.conv1d       = nm.ckpt("%s.%lld.ple.conv1d.weight",     pre, (long long)l);
            m.ple.tree = m.tree_on;     /* the windows' compaction ops, for a tree deployment */
            RAD_ARCH_TRY(m.ple.declare(b, nm, g, pc, (int)l, pw, ps));
        }

        HyperConn::Src mixs{};
        mixs.norm     = nm.ckpt("%s.%lld.attn_hyper_connection.hc_norm.weight", pre, (long long)l);
        mixs.mix_down = nm.ckpt("%s.%lld.attn_hyper_connection.input_mix_weight_down.weight",
                                pre, (long long)l);
        mixs.mix_up   = nm.ckpt("%s.%lld.attn_hyper_connection.input_mix_weight_up.weight",
                                pre, (long long)l);
        mixs.inject_w = nm.ckpt("%s.%lld.attn_hyper_connection.block_inject_weight.weight",
                                pre, (long long)l);
        RAD_ARCH_TRY(lay.hc_mix.declare(b, nm, g, nm.f("blk.%lld.attn_hc", (long long)l),
                                        m.hccfg, hw, mixs));

        /* `Src::norm = nullptr` is the blocks' "input already normed and quantised" mode, and
         * `add_out = false` stops them adding into the stream; the connection's write half does
         * that with the four gains the read produced. `fold` and `ar_in` are off because both fuse
         * into a norm these blocks do not have. */
        if (lay.full) {
            AttnGatedFP8::Src as{};
            as.qg     = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
            as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
            as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
            as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight", pre, (long long)l);
            as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight", pre, (long long)l);
            as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l);
            /* The KV page is the QSA compress block, so the indexer's selection is a paged block
             * table rather than a new attention kernel (docs/QSA.md); the attention rows must
             * declare the same 4 or nothing resolves. Page size only affects the block-table walk:
             * pages of 4 and 16 cost the same at long context and give identical text. */
            lay.attn.kv_block = kKvBlock;
            /* The selection must be set before the attention is declared: `attn_paged` declares
             * `sel` and `seqused` as reads, and a read declared after the op does not count
             * (rad_block_attn_gated_fp8.h). The indexer itself is declared after the attention
             * block, where the sparse attention reads its ops.
             *
             * Below the indexer's budget the selection is every visible block, so sparse and dense
             * paths must give identical text; any difference is a defect. */
            if (m.qsaw.qk) {
                lay.attn.qsa_sel      = m.qsaw.sel;
                lay.attn.qsa_nsel     = m.qsaw.nsel;
                lay.attn.qsa_sequ     = m.qsaw.sequ;
                lay.attn.qsa_topk     = m.qsacfg.topk;
                /* The tree form of the attention reads the selection's region mask and bases. */
                lay.attn.qsa_vmask    = m.qsaw.vmask;
                lay.attn.qsa_rb       = m.qsaw.rb;
                /* The exactness bound, below which prefill issues the cheaper dense attention.
                 * Decode ignores it and always takes the sparse path, keeping that path exercised
                 * on every step. */
                lay.attn.qsa_exact_to = m.qsacfg.budget + m.qsacfg.ratio - 1;
            }
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
            /* An attention-zero rank keeps no attention cache: its block is a fill of zeros
             * (rad_block_attn_gated_fp8.h), and the indexer reads only the shared block table. The
             * call is idempotent. */
            if (g.attn_zero && m.kv_full) RAD_ARCH_TRY(rad_kv_group_zero(b, m.kv_full));
            lay.attn.ar_out_take = lay.hc_mix.ar_take();
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                          false, false, kArNone, lay.hc_mix.op_ar_write != 0));
            /* The indexer, after the attention block so its ops sit where the sparse attention
             * reads them. It reads the same normed fp8 input as the projections
             * (arch/common/rad_qsa.h, docs/QSA.md). */
            if (m.qsaw.qk) {
                QsaIndexer::Config qc = m.qsacfg;
                qc.mode    = lay.attn.mode;
                qc.qk_proj = nm.ckpt("%s.%lld.self_attn.indexer.index_qk_proj.weight",
                                     pre, (long long)l);
                qc.q_norm  = nm.ckpt("%s.%lld.self_attn.indexer.q_layernorm.weight",
                                     pre, (long long)l);
                qc.k_norm  = nm.ckpt("%s.%lld.self_attn.indexer.k_layernorm.weight",
                                     pre, (long long)l);
                qc.kv_attn  = m.kv_full;
                qc.emit_sel = lay.attn.qsa_sel != 0;
                RAD_ARCH_TRY(lay.qsa.declare(b, nm, g, (int)l, qc, m.qsaw, aw.h,
                                             &m.kv_qsa_bkey));
            }
        } else {
            GdnFP8::Src gs{};
            gs.in_qkv   = nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight", pre, (long long)l);
            gs.in_a     = nm.ckpt("%s.%lld.linear_attn.in_proj_a.weight",   pre, (long long)l);
            gs.in_b     = nm.ckpt("%s.%lld.linear_attn.in_proj_b.weight",   pre, (long long)l);
            gs.in_z     = nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight",   pre, (long long)l);
            gs.conv1d   = nm.ckpt("%s.%lld.linear_attn.conv1d.weight",      pre, (long long)l);
            gs.a_log    = nm.ckpt("%s.%lld.linear_attn.A_log",              pre, (long long)l);
            gs.dt_bias  = nm.ckpt("%s.%lld.linear_attn.dt_bias",            pre, (long long)l);
            gs.out_norm = nm.ckpt("%s.%lld.linear_attn.norm.weight",        pre, (long long)l);
            gs.out      = nm.ckpt("%s.%lld.linear_attn.out_proj.weight",    pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_conv));
            lay.gdn.ar_out_take = lay.hc_mix.ar_take();
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, m.gcfg, (int)l, m.kv_state, m.kv_conv, gw, gs,
                                         false, false, kArNone, lay.hc_mix.op_ar_write != 0));
        }

        HyperConn::Src ffns{};
        ffns.norm     = nm.ckpt("%s.%lld.mlp_hyper_connection.hc_norm.weight", pre, (long long)l);
        ffns.mix_down = nm.ckpt("%s.%lld.mlp_hyper_connection.input_mix_weight_down.weight",
                                pre, (long long)l);
        ffns.mix_up   = nm.ckpt("%s.%lld.mlp_hyper_connection.input_mix_weight_up.weight",
                                pre, (long long)l);
        ffns.inject_w = nm.ckpt("%s.%lld.mlp_hyper_connection.block_inject_weight.weight",
                                pre, (long long)l);
        /* The rotated expert plane's input codes come out of this read, beside the plain ones
         * the shared arm takes, so the routed arm declares no quantiser of its own. */
        HyperConn::Config fcfg = m.hccfg;
        HyperConn::Wire   fw   = hw;
        if (m.moecfg.expert_rot && !m.trunk_rot) {
            fcfg.rotate = 1; fw.rq = m.a_hr.cq(); fw.rs = m.a_hr.cs();
        }
        RAD_ARCH_TRY(lay.hc_ffn.declare(b, nm, g, nm.f("blk.%lld.ffn_hc", (long long)l),
                                        fcfg, fw, ffns));

        MoeFP8::Src ms{};
        ms.cfg              = m.moecfg;
        ms.router           = nm.ckpt("%s.%lld.mlp.gate.weight", pre, (long long)l);
        RAD_ARCH_TRY(moe_expert_sources(b, nm, "layer",
                                        nm.ckpt("%s.%lld.mlp.experts", pre, (long long)l), &ms));
        if (m.moecfg.n_ff_shared > 0) {
            ms.shared      = nm.ckpt("%s.%lld.mlp.shared_expert", pre, (long long)l);
            ms.shared_gate = nm.ckpt("%s.%lld.mlp.shared_expert_gate.weight", pre, (long long)l);
        }
        lay.mlp.ar_out_take = lay.hc_ffn.ar_take();
        lay.mlp.hr_in = fcfg.rotate != 0;
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, false, false, kArNone,
                                     lay.hc_ffn.op_ar_write != 0));
        /* The write behind the block takes its gather where it can, declared after the block so
         * the buffers it reads are the block's outputs in declaration order. */
        {
            HyperConn::GatherSrc gs{};
            gs.ye     = lay.mlp.w.edn;
            gs.ew     = lay.mlp.w.ew;
            gs.sorted = lay.mlp.w.sorted;
            if (lay.mlp.gfold) { gs.sh = lay.mlp.w.sout; gs.sg = lay.mlp.w.sgate; }
            gs.top_k  = lay.mlp.c.top_k;
            gs.rows   = lay.mlp.c.rows;
            RAD_ARCH_TRY(lay.hc_ffn.declare_gather(b, gs));
            lay.mlp.gather_out_rows  = lay.hc_ffn.gather_rows;
            lay.mlp.gather_out_rows6 = lay.hc_ffn.gather_rows6;
        }

        /* The control vector's projection for this layer, if the vector steers it. */
        RAD_ARCH_TRY(lay.proj.declare(b, nm.f("blk.%lld.proj", (long long) l), g, l, ctx->cvec,
                                      proj_on, m.b_h, m.hccfg.hc));

        if (m.calib.records(l)) {
            /* Named by the container's weights, which is how a quantiser asks for a Hessian. */
            const ActFP8& ao = lay.full ? m.a_attn : m.a_go;
            const int64_t ko = lay.full ? q_dim : v_dim;      /* this rank's columns of the output */
            const std::string L = "blk." + std::to_string(l);
            const std::string in_stem = L + (lay.full ? ".attn_qkv.input" : ".ssm_inz.input");
            const std::string in_lin  = lay.full ? L + ".attn_qg," + L + ".attn_k," + L + ".attn_v"
                                                 : L + ".ssm_inz";
            const std::string o_stem  = L + (lay.full ? ".attn_output.input" : ".ssm_out.input");
            const std::string o_lin   = L + (lay.full ? ".attn_output" : ".ssm_out");
            lay.cs_in = (int)m.calib.sites.size();
            if (m.trunk_i8) {
                const uint32_t dom = m.trunk_rot ? (uint32_t)RAD_FP8_BLOCK : 0u;
                RAD_ARCH_TRY(m.calib.site_i8(b, g, l, in_stem, in_lin, m.a_x.q8, m.a_x.s8, g.n_embd, dom));
                lay.cs_o = (int)m.calib.sites.size();
                RAD_ARCH_TRY(m.calib.site_i8(b, g, l, o_stem, o_lin, ao.q8, ao.s8, ko, dom));
            } else {
                RAD_ARCH_TRY(m.calib.site(b, g, l, in_stem, in_lin, m.a_x.x, g.n_embd, g.n_embd));
                lay.cs_o = (int)m.calib.sites.size();
                RAD_ARCH_TRY(m.calib.site(b, g, l, o_stem, o_lin, ao.x, ko, ko * g.world));
            }
        }
    }

    /* Each routed layer predicts the next one's experts at its router input (MoeFP8::xp_next). */
    for (size_t l = 0; l + 1 < m.layers.size(); ++l) m.layers[l].mlp.xp_next = &m.layers[l + 1].mlp;

    /* ---- the 97th connection: no write half, no fp8 copy. It produces the bf16 hidden state the
     * lm_head reads; its `hc_norm` takes the place of a final norm (the checkpoint has no
     * `model.language_model.norm.weight`). */
    {
        HyperConn::Config mc = m.hccfg;
        mc.inject = 0;
        mc.quant  = 0;
        HyperConn::Wire mixw{};
        mixw.h = m.b_h; mixw.x.x = m.a_x.x;       /* q/s deliberately absent */
        HyperConn::Src mixs{};
        mixs.norm     = nm.ckpt("model.language_model.hyper_connection_mixer.hc_norm.weight");
        mixs.mix_down = nm.ckpt("model.language_model.hyper_connection_mixer."
                                "input_mix_weight_down.weight");
        mixs.mix_up   = nm.ckpt("model.language_model.hyper_connection_mixer."
                                "input_mix_weight_up.weight");
        RAD_ARCH_TRY(m.mixer.declare(b, nm, g, nm.f("output_hc"), mc, mixw, mixs));
    }

    /* ---- the delta nets' out projections, read beside their decode recurrent updates
     * (kGdnOutPrefetchBytes), and the full-attention layers' beside the attention
     * (kAttnOutPrefetchBytes). At more than one rank; --weight-prefetch off declares none. */
    if (g.world > 1 && !ctx->weight_prefetch_off && rad_op_declared(b, "prefetch")) {
        for (Layer& lay : m.layers) {
            if (lay.full) {
                lay.attn.op_pf_o = decl_prefetch(b, lay.attn.o.w, kAttnOutPrefetchBytes);
                continue;
            }
            lay.gdn.op_pf_out   = decl_prefetch(b, lay.gdn.out.w, kGdnOutPrefetchBytes);
            lay.gdn.pf_out_rows = kPrefetchRows;
            lay.gdn.pf_out_seqs = kGdnOutPrefetchSeqs;
        }
    }

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                   RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {m.a_x.x}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(m.w_lm_head)),
                    {m.b_hout}, {m.b_logits});
    if (!m.op_embed || !m.op_gather || !m.op_logits) return RAD_E_INVAL;

    /* ---- the multi-token-prediction head (arch/common/rad_block_mtp_hc.h).
     *
     * Declared after op_logits; do not move it. Declaration order is the planner's liveness, and a
     * drafter declared before the trunk's tail ops would share a trunk buffer with one of the
     * head's, giving correct output with zero acceptance.
     *
     * Declared only when the checkpoint carries a head and the deployment speculates; at
     * max_spec 0 the weights stay off the card (the container still carries them, since
     * rad-convert declares at a non-zero max_spec). */
    /* Local debug switch: RAD_DEBUG_NO_MTP=1 leaves the head undeclared, so --spec-ngram with
     * --debug-draft-oracle can test the verify path on this container. */
    const int64_t nextn = std::getenv("RAD_DEBUG_NO_MTP") ? 0
                        : meta_int2(meta, "qwen4exp.mtp_num_hidden_layers",
                                    "mtp_num_hidden_layers", 0);
    if (nextn > 0 && g.max_spec > 0 && !dmode_df) {
        const char* mpre = "mtp";
        /* Sized at max_tok, not max_logit_rows: the history pass that fills the head's own K and V
         * has one row per token over a whole prefill chunk. */
        m.b_mtp_hin  = decl_b(b, nm.f("mtp_hin"),  g.act_dtype, {T, HN});
        m.b_mtp_hw   = decl_b(b, nm.f("mtp_hw"),   g.act_dtype, {T, HN});
        m.b_mtp_e    = decl_b(b, nm.f("mtp_e"),    g.act_dtype, {T, g.n_embd});
        m.b_mtp_hout = decl_b(b, nm.f("mtp_hout"), g.act_dtype, {g.max_logit_rows, g.n_embd});
        /* The 2-bit draft head, when the model holds one: asked once here, since the answer
         * decides the head's buffers and the block reads it back from them. */
        if (mtp_draft_head_q2(b, MtpHcBlock::head_name(nm), nm.ckpt("lm_head.weight"))) {
            const int64_t dmpad = ((g.max_logit_rows + 15) / 16) * 16;
            m.b_mtp_dq   = decl_b(b, nm.f("mtp_dq"),   RAD_I8,  {g.max_logit_rows, g.n_embd});
            m.b_mtp_ds   = decl_b(b, nm.f("mtp_ds"),   RAD_F32, {g.max_logit_rows, 1});
            m.b_mtp_dsum = decl_b(b, nm.f("mtp_dsum"), RAD_F32, {g.n_embd / 128, dmpad});
            /* This rank's rows, not the model's: the 2-bit head is vocab-parallel and the merge
             * reassembles it (rad_arch.h). */
            m.b_mtp_dlog = decl_b(b, nm.f("mtp_dlog"), g.act_dtype,
                                  {g.max_logit_rows, vocab_head_rows(g)});
            if (vocab_head_sharded(g)) {
                m.b_mtp_dpair = decl_b(b, nm.f("mtp_dpair"), RAD_F32,
                                       {g.max_logit_rows, kDraftTopR, 2});
                m.b_mtp_dgath = decl_b(b, nm.f("mtp_dgath"), RAD_F32,
                                       {g.max_logit_rows, (int64_t)g.world * kDraftTopR, 2});
            }
            /* The proposal: a draft round's token comes from row_topk, never the sampler.
             * kDraftTopR columns, of which a draft reads column 0; the rest are the coarse head's
             * other candidates, which `logit_rerank` scores first, and the width also gives the
             * vocab-parallel all-gather a 16-byte row (rad_block_mtp_hc.h, kDraftTopR). */
            m.b_mtp_didx = decl_b(b, nm.f("mtp_didx"), RAD_I32, {g.max_logit_rows, kDraftTopR});
            m.b_mtp_dval = decl_b(b, nm.f("mtp_dval"), RAD_F32, {g.max_logit_rows, kDraftTopR});
            /* Stage 2 of a draft tree: the head grows the tree from these candidates. */
            if (m.tree_on && m.tree_w > 1) {
                if (!m.qsaw.qk) {
                    rad_note(b, "a stage-2 draft tree reads the QSA geometry of the head's "
                                "attention, which this container lacks");
                    return RAD_E_UNSUPPORTED;
                }
                RAD_ARCH_TRY(declare_head_tree_buffers(b, nm, m));
            }

            /* --draft-coupling (RadBuildCtx::draft_coupling). Q is the law of the 2-bit candidate
             * head's reranked top-8; a plain bf16 head leaves one candidate and nothing to couple
             * over. The engine checks the drafter shape (a linear serial chain). */
            if (ctx->draft_coupling) {
                /* One state row per head round (a chain, a stage-1 tree), or per node a stage-2
                 * tree's grower may give children to. */
                const int64_t crows = !m.tree_on ? (int64_t)g.max_spec
                                    : m.tree_w > 1 ? (int64_t)g.max_spec : m.tree_r;
                const int64_t cmax = m.tree_on && m.tree_w > 1 ? RAD_MTP_COUPLE_MAX_ROWS
                                                               : RAD_MTP_COUPLE_MAX_ROUNDS;
                if (!m.b_mtp_didx || crows < 1 || crows > cmax) {
                    rad_note(b, "--draft-coupling on requires the 2-bit candidate head's reranked "
                                "top8 and 1..%d coupled rounds (nodes of a stage-2 tree)",
                             (int)cmax);
                    return RAD_E_UNSUPPORTED;
                }
                m.b_mtp_couple_draws = decl_b(b, nm.f("mtp_couple_draws"), RAD_U32,
                                              {g.max_seqs, RAD_MTP_COUPLE_DRAW_WORDS},
                                              RAD_BUF_PERSIST);
                m.b_mtp_couple_state = decl_b(b, nm.f("mtp_couple_state"), RAD_U32,
                                              {crows, g.max_seqs,
                                               RAD_MTP_COUPLE_STATE_WORDS},
                                              RAD_BUF_PERSIST);
                if (!m.b_mtp_couple_draws || !m.b_mtp_couple_state) return RAD_E_INVAL;
                rad_note(b, "MTP coupled: the head samples x~Q from its reranked top8 each round "
                            "(FP64), and the sampler couples the target's own draw to it; %lld "
                            "round(s)%s, fixed-seed tokens are NOT preserved",
                         (long long)(m.tree_on ? m.tree_r : g.max_spec),
                         m.tree_on ? ", siblings drawn without replacement and verified by "
                                     "recursive rejection sampling" : "");
            }
        }

        MtpHcBlock::Wire tw{};
        tw.src  = m.b_h;        tw.hin = m.b_mtp_hin;  tw.hw = m.b_mtp_hw;
        tw.e    = m.b_mtp_e;    tw.inj = m.b_inj;      tw.hout = m.b_mtp_hout;
        tw.logits = m.b_logits; tw.x = m.a_x;
        tw.dq   = m.b_mtp_dq;   tw.ds = m.b_mtp_ds;    tw.dsum = m.b_mtp_dsum;
        tw.dlog = m.b_mtp_dlog; tw.didx = m.b_mtp_didx; tw.dval = m.b_mtp_dval;
        tw.dpair = m.b_mtp_dpair; tw.dgath = m.b_mtp_dgath;
        tw.hstore = m.b_mtp_hstore; tw.tsrc = m.b_mtp_tsrc; tw.tdst = m.b_mtp_tdst;
        tw.couple_draws = m.b_mtp_couple_draws; tw.couple_state = m.b_mtp_couple_state;
        /* The head reuses the trunk's working buffers, all dead by the time a draft round runs (a
         * second set would be 150 MiB). Only the wide stream and the carry are its own, since
         * they must survive the round that produced them. */
        tw.attn = aw; tw.attn.x = m.b_mtp_hw;
        tw.mlp  = mw; tw.mlp.x  = m.b_mtp_hw;
        /* The QSA wire is within-step scratch and a draft round is its own step, so the head
         * shares all of it. The QSA tail is per-layer state (`kv_qsa_tail`), not in the wire, and
         * declare() binds the head's layer to it like any other. */
        tw.qsa  = m.qsaw;

        MtpHcBlock::Src ts{};
        ts.fc_hidden          = nm.ckpt("%s.fc_hidden.weight", mpre);
        ts.fc_embedding       = nm.ckpt("%s.fc_embedding.weight", mpre);
        ts.pre_norm_hidden    = nm.ckpt("%s.pre_fc_norm_hidden.weight", mpre);
        ts.pre_norm_embedding = nm.ckpt("%s.pre_fc_norm_embedding.weight", mpre);

        ts.hc_attn.norm     = nm.ckpt("%s.layers.0.attn_hyper_connection.hc_norm.weight", mpre);
        ts.hc_attn.mix_down = nm.ckpt("%s.layers.0.attn_hyper_connection."
                                      "input_mix_weight_down.weight", mpre);
        ts.hc_attn.mix_up   = nm.ckpt("%s.layers.0.attn_hyper_connection."
                                      "input_mix_weight_up.weight", mpre);
        ts.hc_attn.inject_w = nm.ckpt("%s.layers.0.attn_hyper_connection."
                                      "block_inject_weight.weight", mpre);
        ts.hc_mlp.norm      = nm.ckpt("%s.layers.0.mlp_hyper_connection.hc_norm.weight", mpre);
        ts.hc_mlp.mix_down  = nm.ckpt("%s.layers.0.mlp_hyper_connection."
                                      "input_mix_weight_down.weight", mpre);
        ts.hc_mlp.mix_up    = nm.ckpt("%s.layers.0.mlp_hyper_connection."
                                      "input_mix_weight_up.weight", mpre);
        ts.hc_mlp.inject_w  = nm.ckpt("%s.layers.0.mlp_hyper_connection."
                                      "block_inject_weight.weight", mpre);
        /* The head's own mixer has no `block_inject_weight`, like the trunk's 97th connection:
         * nothing follows it to write back into. */
        ts.hc_mix.norm      = nm.ckpt("%s.hyper_connection_mixer.hc_norm.weight", mpre);
        ts.hc_mix.mix_down  = nm.ckpt("%s.hyper_connection_mixer.input_mix_weight_down.weight",
                                      mpre);
        ts.hc_mix.mix_up    = nm.ckpt("%s.hyper_connection_mixer.input_mix_weight_up.weight", mpre);

        ts.attn.qg     = nm.ckpt("%s.layers.0.self_attn.q_proj.weight", mpre);
        ts.attn.k      = nm.ckpt("%s.layers.0.self_attn.k_proj.weight", mpre);
        ts.attn.v      = nm.ckpt("%s.layers.0.self_attn.v_proj.weight", mpre);
        ts.attn.q_norm = nm.ckpt("%s.layers.0.self_attn.q_norm.weight", mpre);
        ts.attn.k_norm = nm.ckpt("%s.layers.0.self_attn.k_norm.weight", mpre);
        ts.attn.o      = nm.ckpt("%s.layers.0.self_attn.o_proj.weight", mpre);

        ts.mlp.cfg             = m.moecfg;
        ts.mlp.router          = nm.ckpt("%s.layers.0.mlp.gate.weight", mpre);
        RAD_ARCH_TRY(moe_expert_sources(b, nm, "mtp", nm.ckpt("%s.layers.0.mlp.experts", mpre),
                                        &ts.mlp));
        if (m.moecfg.n_ff_shared > 0) {
            ts.mlp.shared      = nm.ckpt("%s.layers.0.mlp.shared_expert", mpre);
            ts.mlp.shared_gate = nm.ckpt("%s.layers.0.mlp.shared_expert_gate.weight", mpre);
        }

        /* Set before declare(), as for the trunk: `attn_paged` declares `sel` and `seqused` as
         * reads, which must precede the op. */
        m.mtp.attn.kv_block = kKvBlock;
        if (m.b_mtp_tsel) {
            /* Stage 2: the rounds' rows attend over the committed pages and their own ancestors,
             * the operands mtp_tree_grow wrote for them. Divergence from the chain head: rounds
             * >= 2 read the last `topk + 1` committed pages (a recency window), not QSA's top-k
             * choice, and the indexer is not run for them. At or below the exactness bound
             * (budget + ratio - 1 tokens of context) the window holds every page and the
             * stage-2 head equals the chain head's attention; above it the drafts differ, which
             * costs acceptance only, since the trunk's verify is exact. */
            m.mtp.attn.tree_sel   = m.b_mtp_tsel;
            m.mtp.attn.tree_nsel  = m.b_mtp_tnsel;
            m.mtp.attn.qsa_vmask  = m.b_mtp_tvmask;
            m.mtp.attn.qsa_rb     = m.b_mtp_trb;
            m.mtp.attn.tree_any   = true;
            m.mtp.tree_rounds     = true;
            m.mtp.tree_w          = (int)m.tree_w;
        }
        if (m.qsaw.qk) {
            m.mtp.attn.qsa_sel      = m.qsaw.sel;
            m.mtp.attn.qsa_nsel     = m.qsaw.nsel;
            m.mtp.attn.qsa_sequ     = m.qsaw.sequ;
            m.mtp.attn.qsa_topk     = m.qsacfg.topk;
            m.mtp.attn.qsa_exact_to = m.qsacfg.budget + m.qsacfg.ratio - 1;
            ts.qsa         = m.qsacfg;
            /* The head's tail is stored in committed order by its history pass: its layer keeps
             * the classic issue sequence (rad_qsa.h, Config::tree). */
            ts.qsa.tree    = false;
            ts.qsa.mode    = m.mtp.attn.mode;
            ts.qsa.qk_proj = nm.ckpt("%s.layers.0.self_attn.indexer.index_qk_proj.weight", mpre);
            ts.qsa.q_norm  = nm.ckpt("%s.layers.0.self_attn.indexer.q_layernorm.weight", mpre);
            ts.qsa.k_norm  = nm.ckpt("%s.layers.0.self_attn.indexer.k_layernorm.weight", mpre);
            ts.qsa.kv_attn  = m.kv_full;
            ts.qsa.emit_sel = m.mtp.attn.qsa_sel != 0;
        }

        /* Stage 1 with coupling: round 1's propose op draws the leaves beside the trunk token. */
        if (m.tree_on && m.tree_w <= 1 && ctx->draft_coupling) {
            m.mtp.couple_leaves = std::min<int>(std::max(ctx->draft_tree_leaves, 0),
                                                (int)kDraftTopR - 1);
            /* The leaves' confidence pruning decided by q-rank inside the propose op, never by
             * the drawn token's own probability (DESIGN-coupling.md 3.5). */
            m.mtp.couple_prune_abs = std::max<double>(ctx->draft_tree_prune_abs, 0.0);
            m.mtp.couple_prune_rel = std::max<double>(ctx->draft_tree_prune_rel, 0.0);
        }
        m.mtp.media = m.media;
        RAD_ARCH_TRY(m.mtp.declare(b, nm, g, (int)g.n_layer, m.kv_full, &m.kv_qsa_bkey, m.n_rot,
                                   m.w_tok, m.w_lm_head, m.hccfg,
                                   m.tree_on ? ctx->draft_rounds : (int)g.max_spec, tw, ts));
        /* A head pass refills the rope table ahead of the head's ops (step), while the carry
         * `mtp_hin` still holds the rows the previous pass left. The refill is the trunk's
         * `rope_table` op, so the table's planned life would end at its last declared reader,
         * which is in the trunk when no head op reads the table (no fused prologue), and the
         * planner could then put the carry in the table's bytes. Declared as written by the
         * head's first op, the table is live where the carry is. */
        if (m.op_rope_cs && rad_op_writes(b, m.mtp.op_gather, &m.b_rope_cs, 1) < 0)
            return RAD_E_INVAL;
        m.have_mtp = true;

        /* Serial drafter: `max_spec` dependent passes, each embedding the last one's proposal.
         * The 2-bit head writes its own argmax into `mtp_didx`; without it `mtp_didx` is 0 and the
         * bf16 head goes through the ordinary sampler chain. */
        RadDrafterDecl dd{};
        dd.name           = "mtp";
        dd.kind           = RAD_DRAFT_SERIAL;
        dd.depth          = ctx->max_spec;
        dd.proposal       = m.b_mtp_didx;
        dd.proposal_pitch = kDraftTopR;   /* the pick is column 0; see kDraftTopR */
        dd.mask_token     = -1;
        if (m.tree_on) {
            /* A serial tree: R rounds (the head's depth), N = max_spec nodes a verify. Stage 1 (W
             * = 1) is the greedy chain 1..R plus the first round's runners-up, columns 1..L of the
             * proposal row `didx` (pitch 8, so L is at most 7); the core takes them before round 2
             * overwrites. */
            if (!m.b_mtp_didx) {
                rad_note(b, "a draft tree needs the 2-bit candidate head's reranked top8; this "
                            "container's head has none");
                return RAD_E_UNSUPPORTED;
            }
            dd.depth       = ctx->draft_rounds;
            dd.tree_nodes  = ctx->max_spec;
            dd.tree_leaves = std::min<int>(std::max(ctx->draft_tree_leaves, 0),
                                           (int)kDraftTopR - 1);
            dd.tree_width  = (int)m.tree_w;   /* checked against N above, never clamped here */
            /* Stage 1's confidence pruning of the first round's leaves is the core's (advance.h),
             * which reads the candidates' logits beside their ids. With --draft-coupling the
             * leaves are drawn tokens in draft order, not the shortlist's columns: the propose
             * op prunes them by q-rank itself and the core's capture takes what it left. */
            if (dd.tree_width == 1 && !ctx->draft_coupling &&
                (ctx->draft_tree_prune_abs > 0.0f || ctx->draft_tree_prune_rel > 0.0f))
                dd.proposal_score = m.b_mtp_dval;
            m.tree_r = dd.depth;
            m.tree_n = dd.tree_nodes;
            if (dd.tree_width > 1) {
                /* The head grows the tree: its proposal plane is the tokens (and frontier) it
                 * writes, `mtp_ptree` the parents at its pitch, one grower op a round (the ABI
                 * freezes `round` as a parameter). */
                dd.proposal       = m.b_mtp_ptok;
                dd.proposal_pitch = m.tree_pitch;
                /* Confidence pruning (--draft-tree-prune / -rel): the thresholds are op parameters
                 * only when on, so the unpruned graph is the one it always was. */
                const double pabs = std::max<double>(ctx->draft_tree_prune_abs, 0.0);
                const double prel = std::max<double>(ctx->draft_tree_prune_rel, 0.0);
                const bool   prune = pabs > 0.0 || prel > 0.0;
                for (int r = 1; r <= (int)m.tree_r; ++r) {
                    /* With --draft-coupling the grower also draws each frontier row's children
                     * without replacement and writes the node states (abi/rad_mtp_tree_rrs.h);
                     * its pruning then reads slot priorities only (DESIGN-coupling.md 3.5). */
                    if (ctx->draft_coupling) {
                        const rad_op grow = prune
                            ? RAD_OP(b, "mtp_tree_grow",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("round", r),
                                           RAD_INT("rounds", m.tree_r), RAD_INT("nodes", m.tree_n),
                                           RAD_INT("width", m.tree_w),
                                           RAD_INT("block_size", kKvBlock),
                                           RAD_INT("sel_pitch", m.qsacfg.topk + 1),
                                           RAD_F64("prune_abs", pabs), RAD_F64("prune_rel", prel),
                                           RAD_INT("n_vocab", g.n_vocab_all)),
                                RAD_NOWEIGHTS)
                            : RAD_OP(b, "mtp_tree_grow",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("round", r),
                                           RAD_INT("rounds", m.tree_r), RAD_INT("nodes", m.tree_n),
                                           RAD_INT("width", m.tree_w),
                                           RAD_INT("block_size", kKvBlock),
                                           RAD_INT("sel_pitch", m.qsacfg.topk + 1),
                                           RAD_INT("n_vocab", g.n_vocab_all)),
                                RAD_NOWEIGHTS);
                        m.op_grow[r - 1] = rw(b, grow,
                            { m.b_mtp_didx, m.b_mtp_dval, m.b_mtp_gstate, m.b_mtp_couple_draws },
                            { m.b_mtp_ptok, m.b_mtp_ptree, m.b_mtp_gstate, m.b_mtp_tsel,
                              m.b_mtp_tnsel, m.b_mtp_tvmask, m.b_mtp_trb, m.b_mtp_tsrc,
                              m.b_mtp_tdst, m.b_mtp_couple_state });
                    } else {
                    const rad_op grow = prune
                        ? RAD_OP(b, "mtp_tree_grow",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("round", r),
                                       RAD_INT("rounds", m.tree_r), RAD_INT("nodes", m.tree_n),
                                       RAD_INT("width", m.tree_w), RAD_INT("block_size", kKvBlock),
                                       RAD_INT("sel_pitch", m.qsacfg.topk + 1),
                                       RAD_F64("prune_abs", pabs), RAD_F64("prune_rel", prel)),
                            RAD_NOWEIGHTS)
                        : RAD_OP(b, "mtp_tree_grow",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("round", r),
                                       RAD_INT("rounds", m.tree_r), RAD_INT("nodes", m.tree_n),
                                       RAD_INT("width", m.tree_w), RAD_INT("block_size", kKvBlock),
                                       RAD_INT("sel_pitch", m.qsacfg.topk + 1)),
                            RAD_NOWEIGHTS);
                    m.op_grow[r - 1] = rw(b, grow,
                        { m.b_mtp_didx, m.b_mtp_dval, m.b_mtp_gstate },
                        { m.b_mtp_ptok, m.b_mtp_ptree, m.b_mtp_gstate, m.b_mtp_tsel,
                          m.b_mtp_tnsel, m.b_mtp_tvmask, m.b_mtp_trb, m.b_mtp_tsrc,
                          m.b_mtp_tdst });
                    }
                    if (!m.op_grow[r - 1]) {
                        rad_note(b, "a stage-2 draft tree needs the mtp_tree_grow op, which no "
                                    "loaded library serves");
                        return RAD_E_UNSUPPORTED;
                    }
                }
            }
            RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
            RAD_ARCH_TRY(rad_drafter_tree(b, dd.tree_width > 1 ? m.b_mtp_ptree : 0));
            if (dd.tree_width > 1)
                rad_note(b, "MTP head: serial draft tree, %d round(s), %d node(s) a verify, stage 2 "
                            "(the head grows the tree, frontier width %d)",
                         dd.depth, dd.tree_nodes, dd.tree_width);
            else
                rad_note(b, "MTP head: serial draft tree, %d round(s), %d node(s) a verify, stage 1 "
                            "(the greedy chain plus %d first-round leaves)",
                         dd.depth, dd.tree_nodes, dd.tree_leaves);
        } else {
            RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
            rad_note(b, "MTP head: one gated-attention layer with 512 routed experts over the wide "
                        "stream, %d serial draft round(s)", ctx->max_spec);
        }
    } else if (nextn > 0 && !dmode_df) {
        rad_note(b, "the checkpoint's `mtp.` head is not loaded: this deployment does not "
                    "speculate");
    } else if (nextn > 0) {
        rad_note(b, "the checkpoint's `mtp.` head is not loaded: this deployment drafts with its "
                    "DFlash2 drafter (RADIANCE_DRAFTER)");
    }

    /* ---- the DFlash2 drafter, after the head (a convert declares both, so the container carries
     * every weight of either; the head's layer is n_layer, the drafter's group follows it). */
    if (g.max_spec > 0 && (dmode == Drafter::Dflash2 || dmode == Drafter::Both))
        RAD_ARCH_TRY(declare_dflash2(b, meta, ctx, m,
                                     (int)g.n_layer + (dmode == Drafter::Both ? 1 : 0),
                                     dmode == Drafter::Dflash2));

    /* The vision tower, last: an encoder pass issues nothing else, so its activations share the
     * arena with every buffer above. Its weight groups start past every text layer and the MTP
     * head. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    return RAD_OK;
}

/* ============================== A prefill step wider than one MoE pass ==============================
 *
 * Every layer of a prefill chunk makes two all-reduces, the mixer's and the MoE's, each in a hard
 * chain: mixer -> all-reduce -> write -> read -> MoE -> all-reduce -> write -> read. At four ranks
 * a chunk's 97 tiered-int8 two-shots are about a quarter of its time, with nothing beside them on
 * one stream.
 *
 * Under --prefill-span the scheduler hands a prefill alone in its step several checkpoint
 * intervals, each piece the interval-sized chunk a span of 1 makes. Everything a piece can run by
 * itself runs a piece at a time, and every all-reduce goes out on the second lane, beside the
 * other piece's work on the first:
 *
 *   the mixer       the delta net runs whole a piece at a time (projection, convolution, solve and
 *                   scan over the piece's bounds, the state handed on through the cache as between
 *                   two chunks), and the attention runs its gated sparse path a piece at a time;
 *                   each hands its first piece's all-reduce off behind that piece's out projection
 *                   (ArPieces), so it runs beside the second piece's mixer;
 *   the MoE         a pass a piece: the mixer's all-reduce of piece 1 runs beside piece 0's write,
 *                   read and pass, the MoE's of piece 0 beside piece 1's;
 *   the last one    the MoE's all-reduce of piece 1 runs beside piece 0's write and the next
 *                   layer's read.
 *
 * The indexer's selection and the embedding run over the whole step.
 *
 * A piece computes the bytes the narrower chunk does: every all-reduce message has a piece's rows,
 * so the tiered-int8 shards and their owners are the ones a span of 1 sends; every connection read
 * is a piece, so its down product splits K the same way (HyperConn::read_pieces); every MoE pass
 * and delta-net piece is a chunk's, and attention rows are their own. The first interval of a
 * prompt is never spanned (Scheduler::plan_chunk), so the dense attention below QSA's exactness
 * bound is taken where it was.
 *
 * The second lane's requirements (Ctx::lane1_) hold: every collective of the step is on it,
 * including the embedding's; none carries an arrival counter; and the one buffer it touches is out
 * of the sharing pool (declare). The collectives there are declared at `overlap` 1, whose rows
 * leave the CUs to the work beside them. */
static bool pipelined(const Model& m, int64_t T) {
    const int64_t R = m.moecfg.rows;
    if (!m.pipe_on || R <= 0 || T <= R) return false;
    if (m.moecfg.calib || m.calib.on || dbg_resid_on() || !m.op_embed_ar) return false;
    for (const Layer& l : m.layers) {
        if (!(l.full ? l.attn.issues_ar(T) : l.gdn.issues_ar(T))) return false;
        if (!l.mlp.issues_ar(T, R)) return false;
    }
    return true;
}

/* The mixer's reduction of a piece the block has finished (ArPieces), on the second lane behind
 * the out projection that wrote it. */
struct MixAr {
    RadCtx*      c = nullptr;
    const Layer* l = nullptr;
};
static void mix_ar_piece(void* u, int64_t r0, int64_t n) {
    const MixAr& s = *(const MixAr*)u;
    rad_lane(s.c, 1);
    rad_lane_join(s.c, 0, 1);
    if (s.l->full) s.l->attn.issue_ar(s.c, r0, n);
    else           s.l->gdn.issue_ar(s.c, r0, n);
    rad_lane(s.c, 0);
}

/* The bounds a step wider than a pass is cut by, on the card: {0, pass, T}. The first two are the
 * window of the indexer's score plane (QsaIndexer::step, Config::plane_rows) and the delta net's
 * first piece (ArPieces::cu), copied from the pinned host copy; T is copied on the card from the
 * step's cu_seqlens. Both are issued on the step's stream ahead of every layer by each step of one
 * sequence wider than a pass, since the buffer is a transient (declare). Null where no step can be
 * wider than a pass (no b_pipe_cu) or the pinned copy could not be allocated. */
static const int32_t* pipe_bounds(RadCtx* c, const RadBatch* batch, Model& m) {
    if (!m.b_pipe_cu) return nullptr;
    if (!m.pipe_cu_tried) {
        m.pipe_cu_tried = true;
        m.pipe_cu_h = (int32_t*)rad_dev_alloc(2 * sizeof(int32_t), RAD_MEM_HOST_PINNED);
        if (m.pipe_cu_h) {
            m.pipe_cu_h[0] = 0;
            m.pipe_cu_h[1] = (int32_t)m.moecfg.rows;
        } else if (m.g.rank == 0) {
            fprintf(stderr, "radiance: qwen4exp: no pinned host memory for the window bounds; a "
                            "step wider than a pass cannot run\n");
        }
    }
    if (!m.pipe_cu_h) return nullptr;
    int32_t* d = (int32_t*)rad_buf_ptr(c, m.b_pipe_cu);
    if (batch->n_tok > m.moecfg.rows && batch->n_seq == 1) {
        const RadStream s = rad_stream(c);
        if (rad_memcpy_async(d, m.pipe_cu_h, 2 * sizeof(int32_t), s) != RAD_OK ||
            rad_memcpy_async(d + 2, batch->cu_seqlens + 1, sizeof(int32_t), s) != RAD_OK)
            return nullptr;
    }
    return d;
}

static void step_pipelined(RadCtx* c, const RadBatch* batch, Model& m, const int32_t* pcu) {
    const Geom& g = m.g;
    const int64_t T = batch->n_tok, R = m.moecfg.rows;
    const int64_t K = (T + R - 1) / R;
    auto r0 = [&](int64_t p) { return p * R; };
    auto rn = [&](int64_t p) { return T - p * R < R ? T - p * R : R; };
    /* A collective on the second lane, ordered behind lane 0 where it reads what lane 0 just
     * wrote (a join to a lane not yet built is a no-op). */
    auto on1 = [&](bool behind0, auto&& issue) {
        rad_lane(c, 1);
        if (behind0) rad_lane_join(c, 0, 1);
        issue();
        rad_lane(c, 0);
    };
    auto wait1 = [&] { rad_lane_join(c, 1, 0); };
    if (!m.pipe_noted) {
        m.pipe_noted = true;
        if (g.rank == 0)
            fprintf(stderr, "radiance: qwen4exp: prefill steps wider than a MoE pass are pipelined: "
                            "%lld rows in %lld pieces of %lld, all-reduces on the second lane\n",
                    (long long)T, (long long)K, (long long)R);
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    on1(true, [&] { RAD_ISSUE_N(c, m.op_embed_ar, T * g.n_embd, brows(m.a_x.x, T), RAD_NONE); });
    wait1();
    if (m.ple_layer >= 0) m.ple.ids(c, batch);
    m.enter.step(c, T);
    if (m.op_rope_cs && T <= qk_fuse_rows(g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));

    /* The delta net runs a piece at a time over the bounds (ArPieces::cu) in a step of one
     * prefill sequence in two pieces. */
    const int32_t* gcu = pcu && K == 2 && batch->n_seq == 1 && batch->phase == RAD_PHASE_PREFILL
                       ? pcu : nullptr;

    const int64_t n = (int64_t)m.layers.size();
    if (n > 0) {
        if (m.ple_layer == 0) m.ple.step(c, batch);
        for (int64_t p = 0; p < K; ++p) m.layers[0].hc_mix.read(c, T, r0(p), rn(p));
    }
    for (int64_t li = 0; li < n; ++li) {
        const Layer& l = m.layers[(size_t)li];
        /* The mixer finishes its output a piece at a time where its last stages are per row and
         * hands the first pieces' reductions to the second lane as it goes (ArPieces), so the
         * first piece's all-reduce runs beside the next piece's attention or out projection. */
        MixAr ms{c, &l};
        ArPieces pc{};
        pc.rows = R; pc.done = mix_ar_piece; pc.u = &ms; pc.cu = gcu;
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch, pcu); l.attn.step(c, batch, false, &pc); }
        else        l.gdn.step(c, batch, false, &pc);
        auto ar_mix = [&](int64_t p) {
            if (l.full) l.attn.issue_ar(c, r0(p), rn(p));
            else        l.gdn.issue_ar(c, r0(p), rn(p));
        };
        /* A block that handed nothing leaves the first piece's all-reduce to be issued here. */
        if (pc.handed == 0) on1(true, [&] { ar_mix(0); });
        wait1();
        int64_t issued = pc.handed > 0 ? pc.handed : 1;
        bool behind = pc.handed == 0;     /* lane 1 already behind the whole mixer */
        for (int64_t p = 0; p < K; ++p) {
            if (p + 1 < K && p + 1 >= issued) {
                on1(!behind, [&] { ar_mix(p + 1); });
                behind = true;
                issued = p + 2;
            }
            l.hc_mix.write(c, T, r0(p), rn(p));
            l.hc_ffn.read(c, T, r0(p), rn(p));
            l.mlp.pass(c, T, r0(p), rn(p), false);
            /* Lane 0 waits here for the next piece's mixer all-reduce and the last piece's MoE
             * one, both issued before this pass; lane 1 then takes this piece's behind its
             * gather. */
            wait1();
            on1(true, [&] { l.mlp.issue_ar(c, r0(p), rn(p)); });
        }
        l.mlp.route_report(c);
        /* The writes, and the next layer's reads beside the last piece's all-reduce. PLE adds into
         * every row of the stream before the next layer reads it, so there the reads wait. */
        const Layer* nx = li + 1 < n ? &m.layers[(size_t)li + 1] : nullptr;
        const bool ple_next = nx && li + 1 == m.ple_layer;
        for (int64_t p = 0; p < K; ++p) {
            if (p == K - 1) wait1();
            l.hc_ffn.write(c, T, r0(p), rn(p));
            /* The projection of this piece, before the next layer reads it. */
            l.proj.step(c, batch, r0(p), rn(p));
            if (nx && !ple_next) nx->hc_mix.read(c, T, r0(p), rn(p));
        }
        if (ple_next) {
            m.ple.step(c, batch);
            for (int64_t p = 0; p < K; ++p) nx->hc_mix.read(c, T, r0(p), rn(p));
        }
    }
}

/* The trunk one stream at a time over the whole step: every step a span of 1 makes, and every
 * other where step_pipelined's conditions do not hold. */
static void step_serial(RadCtx* c, const RadBatch* batch, Model& m, const int32_t* pcu) {
    const Geom& g = m.g;
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    /* Unconditional, unlike qwen35_fp8's: the first block has no norm to fold it into, only a
     * gated-residual read. */
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, T * g.n_embd, brows(m.a_x.x, T), RAD_NONE);
    /* The PLE's hash, ahead of every layer: its host gather waits for it, and the layers issued
     * between the two keep the device busy while the host gathers (rad_block_ple.h). */
    if (m.ple_layer >= 0) m.ple.ids(c, batch);
    m.enter.step(c, T);

    if (m.op_rope_cs && T <= qk_fuse_rows(g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));

    for (int64_t li = 0; li < (int64_t)m.layers.size(); ++li) {
        const Layer& l = m.layers[(size_t)li];
        /* Before the attention connection reads the stream, which is where the reference puts it. */
        if (li == m.ple_layer) m.ple.step(c, batch);
        l.hc_mix.read_pieces(c, T);
        /* The calibration taps read the codes the mixer's GEMMs read: its input before it runs,
         * its output projection's input before the feed-forward's connection reuses the pair. */
        if (l.cs_in >= 0) m.calib.tap(c, (size_t)l.cs_in, T);
        /* The indexer runs first, since the attention reads its selection. The block's input is
         * already normed and quantised by the connection (`ext_in`), so `w.h` is final. */
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch, pcu); l.attn.step(c, batch); }
        else        l.gdn.step(c, batch);
        if (l.cs_o >= 0) m.calib.tap(c, (size_t)l.cs_o, T);
        l.hc_mix.write(c, T, 0, T);
        dbg_resid(c, (int)li, "mix", g.n_embd, m.b_h, m.a_x.x);
        l.hc_ffn.read_pieces(c, T);
        l.mlp.step(c, batch);
        l.hc_ffn.write(c, T, 0, T);
        /* The request's control vector, on the stream the write just finished. */
        l.proj.step(c, batch, 0, T);
        dbg_resid(c, (int)li, "ffn", g.n_embd, m.b_h, m.a_x.x);
        /* The DFlash2 tap, after the layer's feed-forward write and control vector: the mean of
         * the hc streams of the completed layer, into this tap's column slice. */
        for (int t = 0; t < m.n_taps; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, m.op_tap, brows(m.b_h, T),
                      bcol(m.b_dfsrc, (int64_t)t * g.n_embd, g.n_embd, T));
        }
    }
    m.calib.after_step(c, T);
}

/* Stage 2: after a head pass (the history pass is round 1, then rounds 2..R of W rows a sequence)
 * the grower turns the pass's top-8 candidates into the next frontier and, after the last round,
 * the final tree, in the proposal and parents planes (docs/OPS.md "Serial draft tree growth"). */
static void head_tree_grow(RadCtx* c, const RadBatch* batch, const Model& m) {
    const int r = batch->draft_pass;
    if (r < 0 && batch->n_draft_out <= 0) return;      /* a history pass that drafts nothing */
    const int round = r < 0 ? 1 : r;
    if (round < 1 || round > (int)m.tree_r || !m.op_grow[round - 1]) {
        rad_step_fail(c, "the draft tree's head was handed a round it has no grower for");
        return;
    }
    const RadKVGroupBatch* kvb = kv_batch(batch, m.kv_full);
    if (!kvb || !kvb->block_table) {
        rad_step_fail(c, "the draft tree's grower needs the head's block table");
        return;
    }
    const int64_t W = m.tree_w;
    /* Round 1 is the history pass's draft rows, one a drafting sequence; a later round is W rows
     * a sequence of a pass that carries drafting sequences only. */
    const int64_t S    = r < 0 ? batch->n_draft_out : batch->n_seq;
    const int64_t rows = r < 0 ? batch->n_draft_out : batch->n_tok;
    if (r > 1 && rows != S * W) {
        rad_step_fail(c, "the draft tree's round pass is not W rows a sequence");
        return;
    }
    RAD_ISSUE_N(c, m.op_grow[round - 1], S,
                brows(m.b_mtp_didx, rows), brows(m.b_mtp_dval, rows),
                brows(m.b_mtp_gstate, S),
                r < 0 ? praw(batch->positions, RAD_I32, batch->n_tok) : RAD_NONE,
                r < 0 ? praw(batch->draft_out_ids, RAD_I32, S) : RAD_NONE,
                praw2(kvb->block_table, RAD_I32, S, kvb->block_table_pitch),
                brows(m.b_mtp_ptok, S), brows(m.b_mtp_ptree, S),
                brows(m.b_mtp_tsel, S * W), brows(m.b_mtp_tnsel, S * W),
                brows(m.b_mtp_tvmask, S * W), brows(m.b_mtp_trb, S),
                brows(m.b_mtp_tsrc, S * W), brows(m.b_mtp_tdst, S * W),
                m.b_mtp_couple_draws ? praw(batch->seq_ids, RAD_I32, batch->n_seq) : RAD_NONE,
                m.b_mtp_couple_draws ? brows(m.b_mtp_couple_draws, S) : RAD_NONE,
                m.b_mtp_couple_draws ? RAD_B(m.b_mtp_couple_state) : RAD_NONE);
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_slot(c)];
    const Geom& g = m.g;
    const int64_t T = batch->n_tok;

    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    /* An MTP pass runs only the head. The trunk already left its wide stream in `h`; the head
     * turns the rows it wants into a draft token or one more position of its own attention
     * history, by the sign of `draft_pass` (rad_block_mtp_hc.h). Zero is a trunk step. */
    if (batch->draft_pass != 0) {
        /* DFlash2: positive is the query pass proposing the whole block, negative the context
         * pass turning the tokens the trunk just committed into the drafter's own K and V. */
        if (m.have_df) {
            /* No refill of the trunk's cos/sin table: the drafter rotates with its own ops, and the
             * table's planned life ends in the trunk (the head's refill is declared to extend it,
             * the drafter's is not), so a write here could land in the drafter's buffers. */
            if (batch->draft_pass > 0) m.df.step(c, batch);
            else                       m.df.context(c, batch);
            return;
        }
        if (!m.have_mtp) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `mtp.` head, or declare ran at max_spec 0");
            return;
        }
        /* Fill the head's own cos/sin rows. A draft round runs past the committed end (round k
         * writes K and V at position head_i + k - 1), at rows the preceding trunk step never
         * wrote. Without this, later rounds would rotate by stale rows: only acceptance drops, and
         * because the indexer reads the same table, the head's QSA selection and so the accepted
         * text shift at tie level at long context. A row depends only on its position, so
         * re-filling is idempotent. */
        if (m.op_rope_cs && T <= qk_fuse_rows(g) && !rope_mixed(batch))
            RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));
        /* The n-gram rows of the pass's newest tokens, for the verify step that runs them:
         * hashed ahead of the head, fetched once the whole pass is issued (PleLayer::hint_ids). */
        if (m.ple_layer >= 0 && !(m.tree_w > 1 && batch->spec_parent)) m.ple.hint_ids(c, batch);
        m.mtp.qsa_win_cu = pipe_bounds(c, batch, m);
        m.mtp.step(c, batch);
        if (m.ple_layer >= 0 && !(m.tree_w > 1 && batch->spec_parent)) m.ple.hint_fetch(c, batch);
        if (m.tree_w > 1) head_tree_grow(c, batch, m);
        return;
    }

    const int32_t* pcu = pipe_bounds(c, batch, m);
    if (pipelined(m, T)) step_pipelined(c, batch, m, pcu);
    else                 step_serial(c, batch, m, pcu);

    m.mixer.read_pieces(c, T);

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_x.x, T), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

/* The default draft depth, used by `--num-speculative-tokens auto` and by rad-convert.
 *
 * 3 is an operating point, not a model property (`mtp_num_hidden_layers` only says the one-layer
 * head exists), so `draft_depth_fixed` stays 0. Throughput peaks at 3 on this model: per-position
 * acceptance stays high at every depth, but each extra verify row costs real time because the
 * weights do not fit device memory (more experts streamed, more n-gram rows faulted from the
 * mapped table). */
uint64_t pass_key(RadCtx* c, const RadBatch* batch, const int32_t* used_max_h) {
    if (!c || !batch) return 0;
    if (batch->enc) return 1;
    /* A decode pass, a speculative verify or an MTP draft included, issues every launch at the
     * step's keyed context bound (RadArgs::kv_used_max_h), never the live maximum, so the core's
     * key holds its geometry. */
    if (batch->phase == RAD_PHASE_DECODE) return 1;
    /* Mixed batches split the query rows into decode and prefill subsets before attention, which
     * read the live maximum. The whole-batch host summary cannot key those subset geometries
     * exactly, nor a drafter's or a verify's rows inside a prefill pass. */
    if (batch->phase == RAD_PHASE_MIXED || batch->draft_pass != 0 || batch->n_spec > 0) return 0;

    Model& m = g_model[rad_slot(c)];
    bool has_full_attention = false;
    for (const Layer& layer : m.layers) {
        if (!layer.full) continue;
        if (layer.attn.kv_block != 4) return 0;
        has_full_attention = true;
    }
    if (!has_full_attention) return 1;
    if (m.g.head_dim != 256 || (m.g.gqa() != 6 && m.g.gqa() != 12) || !m.kv_full ||
        !used_max_h || !batch->kv || batch->n_kv_groups <= 0)
        return 0;

    int group = -1;
    for (int i = 0; i < batch->n_kv_groups; ++i)
        if (batch->kv[i].group == m.kv_full) { group = i; break; }
    if (group < 0) return 0;

    /* Qwen always passes cu_seqlens, so fill_paged takes q_len = min(band, total_q). That equals
     * n_tok only for one sequence; a ragged or multi-sequence launch's band is not the host group
     * max, and mixed decode-prefix / prefill-suffix launches use the suffix's own KV lengths. */
    if (batch->n_seq != 1 || batch->n_tok <= 0 || m.g.max_ctx <= 0) return 0;
    const int q_len = (int)batch->n_tok;
    const int total_q = (int)batch->n_tok;
    const int num_seqs = 1;
    const int kv_heads = (int)m.g.n_head_kv;
    const int used = used_max_h[group];
    if (used <= 0) return 0;  // Missing host summary is not the same as an empty KV cache.
    const int wgps = rad_ctx_wgp_count(c);
    if (wgps <= 0) return 0;
    constexpr int tile = 48, nwarps = 24;
    const int bq = nwarps * 16 / m.g.gqa();
    const int splits = rad_prefill_splits_for_live(bq, tile, q_len, total_q, num_seqs,
                                                    kv_heads, m.g.max_ctx, used, wgps);
    return rad_prefill_grid_key(q_len, bq, splits, kv_heads, num_seqs, nwarps * 32);
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    /* The gated attention block declares the streamed store and the resolve
     * (rad_block_attn_gated_fp8.h). */
    out->kv_stream_ok = 1;
    out->draft_depth = 0;
    if (drafter_mode(meta) == Drafter::Dflash2) {
        /* The block drafter: a chain of 3 (a block of 4, the vendor's tested block) by default;
         * 7 (its trained block) and trees (--draft-tree on, --num-speculative-tokens N nodes up to
         * 15, the delta net's 16 verify rows) are asked for. `tree_pays` stays 0 until measured. */
        out->draft_depth    = 3;
        out->tree_linear_ok = 1;
        out->tree_pays      = 0;
    } else if (meta_int2(meta, "qwen4exp.mtp_num_hidden_layers", "mtp_num_hidden_layers", 0) > 0) {
        out->draft_depth = 3;
        /* The serial draft tree (DARTree), advertised with the head: N = 6 nodes at the default
         * operating point (the greedy chain of the head's 3 rounds plus 3 depth-1 siblings, 7 verify
         * rows with the anchor, which the delta net's tree walk and the QSA region mask both take; at
         * most 16 rows and, at the indexer's ratio, 64 - ratio nodes). A container with no indexer,
         * or a streamed KV cache, is refused at declare. `tree_linear_ok`: the delta net ops, the PLE
         * windows and the QSA tail and block keys all follow RadBatch::spec_parent and spec_path
         * (rad_block_gdn_fp8.h, rad_block_ple.h, rad_qsa.h).
         * `tree_pays` stays 0 until a measurement says the tree beats the chain on the real model:
         * `--draft-tree on` forces it, `auto` keeps the chain. */
        out->tree_nodes     = kTreeNodes;
        out->tree_nodes_max = 15;   /* the delta net's tree walk takes 16 rows a verify, anchor included */
        out->tree_linear_ok = 1;
        out->tree_pays      = 0;
    }
    out->shape_probe_ok = 1;      /* declare() writes scratch under shape_probe */
    /* The gated residual applies a control vector after every block's write (HcProject). */
    out->control_vector_ok = 1;
    /* A pass is about 930 launches, nearly all dependent, and the kernels of its decode heights
     * have chained forms: on two R9700 a full pass under the draft head went from 13.9 to 12.5 ms
     * and the step without the head from 9.98 to 9.17 ms at one sequence. */
    out->chain_pays = RAD_CHAIN_ALWAYS;
    return RAD_OK;
}

}  /* namespace qwen4exp_fp8 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(qwen4exp_fp8)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(qwen4exp_fp8, "qwen4exp", "", "0.1.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next): hybrid Gated DeltaNet / gated attention, a gated "
                "residual four streams wide and 512 routed experts, from a dense bf16 checkpoint "
                "quantised ahead of time by a recipe")
extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* batch,
                                      const int32_t* used_max_h) {
    return qwen4exp_fp8::pass_key(c, batch, used_max_h);
}
#endif
