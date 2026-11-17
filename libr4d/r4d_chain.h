// r4d_chain.h -- a chained launch's wait and completion (RadKernelChainFn, abi/rad_abi.h).
//
// Between two dependent kernels on one queue the packet processor waits for the first to finish
// before it launches the second (the AQL barrier bit): on 4x R9700, ~1.3 ms of a 14 ms DFlash2 step
// on the 31B. A chained launch drops the bit. The engine writes a record into the
// kernel's argument slot, and the kernel waits for its queue's completion counter to reach the
// blocks issued before it, then adds its own blocks as they finish. A kernel's chained form
// (`CHAIN` true) brackets its body with r4d_chain_wait / r4d_chain_done; the plain form never reads
// the record, so a launch from outside the engine (r4d_selftest, rad-kbench) is untouched.
//
// Every block must reach r4d_chain_done exactly once, so a kernel whose blocks return early keeps
// its body in a __device__ function called from the chained wrapper.
#pragma once
#include <hip/hip_runtime.h>
#include "rad_abi.h"

/* A kernel template's chain parameter, for a kernel that brackets its own body: `r4dchain`
 * instantiates the chained form and puts the name the backend knows a chained kernel by into the
 * instantiation's symbol, R4dUnchained the plain form. */
struct r4dchain     { static constexpr bool on = true; };
struct R4dUnchained { static constexpr bool on = false; };

#if !defined(__HIP__)
/* The host SIMT emulator (tests/r4d_emu) compiles some of these sources as C++: no engine record
 * ever reaches a launch there, so a chained form is its plain form. */
__device__ __forceinline__ void r4d_chain_wait() {}
__device__ __forceinline__ void r4d_chain_done() {}
#else

struct R4dChainRec {
    unsigned long long  magic, target;
    unsigned long long* ctr;
    unsigned long long  check;
};

// The record, or null where it is not the engine's. Read from the argument segment (scalar loads)
// at both ends of the kernel, so nothing is held across the body: a register-heavy body (the
// split-KV decode attention) would otherwise pay for the pointer.
__device__ __forceinline__ const R4dChainRec* r4d_chain_rec() {
    const R4dChainRec* r = (const R4dChainRec*)((const char*)__builtin_amdgcn_kernarg_segment_ptr() +
                                                RAD_CHAIN_KARG_OFF);
    const unsigned long long magic = r->magic, target = r->target, check = r->check;
    const unsigned long long* ctr = r->ctr;
    if (magic != RAD_CHAIN_MAGIC ||
        check != (magic ^ target ^ (unsigned long long)(uintptr_t)ctr ^ RAD_CHAIN_CHECK))
        return nullptr;
    return r;
}

// Before the first read of an operand: one lane waits for the counter to reach the target.
// Every resident block polls the same word, so a wide kernel launched behind a long one (a GEMM
// behind a junction's all-reduce) would put hundreds of loads into one L2 channel every few dozen
// clocks for as long as the wait lasts, slowing whatever else that channel serves. The first
// R4D_CHAIN_FAST polls are ~64 clocks apart, the rest R4D_CHAIN_SLOW x 64: a long wait ends at most
// that much late and puts 1/R4D_CHAIN_SLOW of the load on the channel.
//
// A grid of a few blocks puts little on the channel however often it polls, and a decode step is
// mostly such grids behind one another (a reduction's four blocks, a quantiser's, a router's
// seventeen), each ending its wait up to a slow period late: so up to R4D_CHAIN_TINY blocks keep
// the fast period for the whole wait and up to R4D_CHAIN_SMALL poll every R4D_CHAIN_MID x 64 clocks,
// which is at most the load a 256-block grid puts there at the slow period. Qwen3.8-Flash-Next at
// tp 2, chain 3 with coupling, ~670 dependent launches a verify pass: -0.32% a step; every grid at
// 8 or 4 instead of 16 gave -0.16% and -0.25% for two and four times the polls of the wide ones.
#ifndef R4D_CHAIN_FAST
#define R4D_CHAIN_FAST 8
#endif
#ifndef R4D_CHAIN_SLOW
#define R4D_CHAIN_SLOW 16
#endif
#ifndef R4D_CHAIN_TINY
#define R4D_CHAIN_TINY 16
#endif
#ifndef R4D_CHAIN_SMALL
#define R4D_CHAIN_SMALL 64
#endif
#ifndef R4D_CHAIN_MID
#define R4D_CHAIN_MID 4
#endif
__device__ __forceinline__ void r4d_chain_wait() {
    const R4dChainRec* r = r4d_chain_rec();
    if (!r) return;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const unsigned long long target = r->target;
        unsigned long long* ctr = r->ctr;
        const unsigned nblk = gridDim.x * gridDim.y * gridDim.z;
        int n = 0;
        while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
            if (n < R4D_CHAIN_FAST) {
                __builtin_amdgcn_s_sleep(1);
                ++n;
            } else if (nblk <= R4D_CHAIN_TINY) {
                __builtin_amdgcn_s_sleep(1);
            } else if (nblk <= R4D_CHAIN_SMALL) {
                __builtin_amdgcn_s_sleep(R4D_CHAIN_MID);
            } else {
                __builtin_amdgcn_s_sleep(R4D_CHAIN_SLOW);
            }
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

// After the block's last write: release, then one add a block.
__device__ __forceinline__ void r4d_chain_done() {
    const R4dChainRec* r = r4d_chain_rec();
    if (!r) return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif  /* __HIP__ */
