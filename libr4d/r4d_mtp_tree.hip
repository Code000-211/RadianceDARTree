// A serial draft head's best-first tree growth (op mtp_tree_grow), one thread per sequence.
// docs/OPS.md ("Serial draft tree growth") is the contract and abi/rad_mtp_tree.h the algorithm,
// which libref runs on the host unchanged (ref_mtp_tree_grow), so the two agree by construction up
// to the float expf / logf of the candidates' log-softmax.
//
// A pass of the head leaves, per frontier row, its top-8 candidates; this launch pools them against
// their parents' cumulative log probability, creates the next frontier (or, after the last round,
// the rest of the node budget), and writes the tokens, parents and the next pass's tree-attention
// operands. Everything is a few hundred words a sequence in global memory, so the launch is one
// thread a sequence and latency, not bandwidth, is the cost (about the dozen small launches of a
// draft round's sampler chain).
#include <hip/hip_runtime.h>
#include "r4d_gdn_args.h"
#include "rad_mtp_tree.h"

struct MtpTreeArgs {
  const int* didx;  const float* dval;
  long row_stride;                // elements between candidate rows (didx and dval alike)
  int* state;       long state_stride;
  const int* head_pos;  const int* head_row;   // round 1 only, else null
  const int* table; long table_stride;
  int* tok;  long tok_stride;
  int* par;  long par_stride;
  int* sel;  int* nsel;  long long* vmask;  int* rb;
  int M;
  RadMtpTreeCfg cfg;
  long per;                       // candidate rows a sequence: 1 in round 1, else width
};

__global__ __launch_bounds__(64) void r4d_mtp_tree_grow_kernel(MtpTreeArgs a) {
  const int m = blockIdx.x * 64 + threadIdx.x;
  if (m >= a.M) return;
  int head = -1;
  if (a.cfg.round == 1 && a.head_pos && a.head_row) head = a.head_pos[a.head_row[m]];
  const bool att = a.cfg.sel_pitch > 0 && a.sel && a.nsel && a.vmask && a.rb && a.table;
  rad_mtp_tree_grow_seq(&a.cfg, a.state + (long)m * a.state_stride, a.didx, a.dval,
                        (long)m * a.per, a.row_stride, 1, head,
                        att ? a.table + (long)m * a.table_stride : nullptr,
                        a.tok + (long)m * a.tok_stride, a.par + (long)m * a.par_stride,
                        att ? a.sel + (long)m * a.cfg.width * a.cfg.sel_pitch : nullptr,
                        att ? a.nsel + (long)m * a.cfg.width : nullptr,
                        att ? a.vmask + (long)m * a.cfg.width : nullptr,
                        att ? a.rb + m : nullptr);
}

extern "C" int r4d_rad_mtp_tree_grow(const RadArgs* a, RadStream s) {
  const RadTensor *didx = r4d_opt(a, 0), *dval = r4d_opt(a, 1), *st = r4d_opt(a, 2);
  const RadTensor *hpos = r4d_opt(a, 3), *hrow = r4d_opt(a, 4), *tab = r4d_opt(a, 5);
  const RadTensor *tok = r4d_opt(a, 6), *par = r4d_opt(a, 7);
  const RadTensor *sel = r4d_opt(a, 8), *nsel = r4d_opt(a, 9), *vmask = r4d_opt(a, 10),
                  *rbt = r4d_opt(a, 11);
  if (!didx || !dval || !st || !tok || !par) return R4D_NO(RAD_E_INVAL);
  if (didx->dtype != RAD_I32 || dval->dtype != RAD_F32 || st->dtype != RAD_I32 ||
      tok->dtype != RAD_I32 || par->dtype != RAD_I32)
    return R4D_NO(RAD_E_DTYPE);
  MtpTreeArgs k{};
  k.cfg.round  = (int)r4d_geti_or(a, "round", 0);
  k.cfg.rounds = (int)r4d_geti_or(a, "rounds", 0);
  k.cfg.nodes  = (int)r4d_geti_or(a, "nodes", 0);
  k.cfg.width  = (int)r4d_geti_or(a, "width", 0);
  k.cfg.block  = (int)r4d_geti_or(a, "block_size", 0);
  k.cfg.sel_pitch = (int)r4d_geti_or(a, "sel_pitch", 0);
  if (st->rank != 2 || tok->rank != 2 || par->rank != 2 || didx->rank != 2 || dval->rank != 2)
    return R4D_NO(RAD_E_SHAPE);
  const long long M = st->shape[0];
  k.cfg.pitch = (int)tok->shape[1];
  if (M <= 0 || k.cfg.round < 1 || k.cfg.rounds < 1 || k.cfg.nodes < 1 || k.cfg.width < 1 ||
      k.cfg.round > k.cfg.rounds || k.cfg.nodes > RAD_MTP_TREE_MAX_NODES ||
      k.cfg.pitch < k.cfg.nodes)
    return R4D_NO(RAD_E_SHAPE);
  if (st->shape[1] < rad_mtp_tree_state_words(k.cfg.nodes, k.cfg.width, k.cfg.rounds) ||
      tok->shape[0] < M || par->shape[0] < M || par->shape[1] < k.cfg.nodes)
    return R4D_NO(RAD_E_SHAPE);
  k.per = k.cfg.round == 1 ? 1 : k.cfg.width;
  if (didx->shape[1] != RAD_MTP_TREE_K || dval->shape[1] != RAD_MTP_TREE_K ||
      didx->shape[0] < M * k.per || dval->shape[0] < M * k.per)
    return R4D_NO(RAD_E_SHAPE);
  if (didx->stride[1] != 1 || dval->stride[1] != 1 || st->stride[1] != 1 || tok->stride[1] != 1 ||
      par->stride[1] != 1 || didx->stride[0] != dval->stride[0])
    return R4D_NO(RAD_E_STRIDE);
  const bool att = k.cfg.sel_pitch > 0;
  if (att) {
    if (!sel || !nsel || !vmask || !rbt || !tab || k.cfg.block <= 0 || sel->rank != 2 ||
        sel->shape[1] < k.cfg.sel_pitch || sel->shape[0] < M * k.cfg.width ||
        sel->stride[0] != k.cfg.sel_pitch || sel->dtype != RAD_I32 || nsel->dtype != RAD_I32 ||
        vmask->dtype != RAD_I64 || rbt->dtype != RAD_I32 || tab->dtype != RAD_I32 ||
        r4d_numel(nsel) < M * k.cfg.width || r4d_numel(vmask) < M * k.cfg.width ||
        r4d_numel(rbt) < M || tab->rank != 2)
      return R4D_NO(RAD_E_SHAPE);
  }
  if (k.cfg.round == 1 && hpos && hrow && (hpos->dtype != RAD_I32 || hrow->dtype != RAD_I32))
    return R4D_NO(RAD_E_DTYPE);
  k.didx = (const int*)didx->data;   k.dval = (const float*)dval->data;
  k.row_stride = (long)didx->stride[0];
  k.state = (int*)st->data;          k.state_stride = (long)st->stride[0];
  k.head_pos = (k.cfg.round == 1 && hpos && hrow) ? (const int*)hpos->data : nullptr;
  k.head_row = (k.cfg.round == 1 && hpos && hrow) ? (const int*)hrow->data : nullptr;
  k.table = att ? (const int*)tab->data : nullptr;
  k.table_stride = att ? (long)tab->stride[0] : 0;
  k.tok = (int*)tok->data;  k.tok_stride = (long)tok->stride[0];
  k.par = (int*)par->data;  k.par_stride = (long)par->stride[0];
  k.sel = att ? (int*)sel->data : nullptr;
  k.nsel = att ? (int*)nsel->data : nullptr;
  k.vmask = att ? (long long*)vmask->data : nullptr;
  k.rb = att ? (int*)rbt->data : nullptr;
  k.M = (int)M;
  hipLaunchKernelGGL(r4d_mtp_tree_grow_kernel, dim3((unsigned)((M + 63) / 64)), dim3(64), 0,
                     (hipStream_t)s, k);
  return R4D_RAW((int)hipGetLastError());
}
