#!/usr/bin/env python3
# stampseq.py -- where a recorded pass's time goes, record by record, from RADIANCE_AQL_STAMP's CSV.
#
#   RADIANCE_AQL_STAMP=1 RADIANCE_AQL_STAMP_OUT=/tmp/st.csv build/radiance ...   (drive a load)
#   scripts/stampseq.py /tmp/st.csv [--min 500] [--max N] [--seq] [--top 40]
#
# stampgaps.py gives a pass's wall, idle share and the kernels by summed dispatch time. With
# chained launches that sum is mostly waiting: a kernel is on the card long before the one it
# depends on ends, so its start-to-end time says little about what it costs the pass. This script
# attributes the pass's wall instead: the dispatches of a pass in order of their end, each charged
# the time by which it moved the latest end forward ("inc"). The incs of a pass add up to its wall.
#
# Per kernel (name and workgroup count): how many a pass, the median over passes of its summed inc,
# its share of the wall, inc and start-to-end time a dispatch. With --seq, per record of the tape:
# p10 / p50 / p90 of its inc, its start-to-end time, and its lead (how long before the latest end so
# far the card began launching it: a lead near zero is a launch the chain waited for).
#
# Reading it: the card reports a dispatch's end only once every dispatch before it on the queue has
# ended, and no two ends closer than ~2.7 us. So a kernel that ran beside the one before it (an
# unordered launch, a read that adds its blocks at once) still shows an inc of ~3 us, taken from the
# next long kernel's; segments between two long kernels are exact, the split inside a run of short
# ones is not. Measured on Qwen3.8-Flash-Next at tp 2 by adding launches to a pass: a dependent
# launch of a few blocks costs the step ~1.6 us, an unordered one ~0.5 us (the card launches about
# one packet every 1.7 us, and a wide kernel in front holds the next launch until its own blocks
# are all placed).
#
# Passes of fewer records than --min (draft rounds) or more than --max are left out; a pass with
# fewer lines than the longest of its kind, or a line without times, is dropped. Standard library
# only.
import csv
import re
import statistics as st
import sys
from collections import defaultdict


def name(k):
    k = k.replace("_ZN12_GLOBAL__N_1", "")
    k = re.sub(r"^_Z\d+", "", k)
    k = re.sub(r"^\d+", "", k)
    k = re.sub(r"^r4dchain_", "", k)
    k = re.sub(r"^r4d_", "", k)
    m = re.match(r"([a-z0-9_]+?)(?:_kernel)?(I.*|E.*)?$", k)
    return m.group(1) if m else k[:40]


def main():
    args = sys.argv[1:]
    opt = {"--min": 500, "--max": 10 ** 9, "--top": 40}
    seq = "--seq" in args
    if seq:
        args.remove("--seq")
    for o in list(opt):
        if o in args:
            i = args.index(o)
            opt[o] = int(args[i + 1])
            del args[i:i + 2]
    if not args:
        sys.exit("usage: stampseq.py CSV [--min RECORDS] [--max RECORDS] [--seq] [--top K]")
    passes = defaultdict(list)
    with open(args[0], newline="") as f:
        for r in csv.DictReader(f):
            try:
                rec = int(r["records"])
                if rec < opt["--min"] or rec > opt["--max"] or int(r["pass"]) == 0:
                    continue
                grid, wg = int(r.get("grid") or 0), int(r.get("wg") or 0)
                passes[int(r["pass"])].append((int(r["seq"]), int(r["start_ns"]), int(r["end_ns"]),
                                               r["kernel"], grid // wg if wg else 0,
                                               r["ordered"], r["chained"]))
            except (ValueError, KeyError):
                pass
    if not passes:
        sys.exit("no pass of that size in " + args[0])
    full = max(len(v) for v in passes.values())
    ps = [v for v in passes.values() if len(v) == full]
    walls, tot, cnt, dur = [], defaultdict(list), {}, defaultdict(list)
    sinc, sdur, slead, smeta = defaultdict(list), defaultdict(list), defaultdict(list), {}
    for v in ps:
        t0 = min(x[1] for x in v)
        prev = t0
        acc, d, c = defaultdict(float), defaultdict(float), defaultdict(int)
        for x in sorted(v, key=lambda x: x[2]):
            inc = max(0, x[2] - prev)
            key = "%s wgs=%d" % (name(x[3]), x[4])
            acc[key] += inc
            d[key] += x[2] - x[1]
            c[key] += 1
            sinc[x[0]].append(inc)
            sdur[x[0]].append(x[2] - x[1])
            slead[x[0]].append(prev - x[1])
            smeta[x[0]] = (name(x[3]), x[4], x[5], x[6])
            prev = max(prev, x[2])
        for k in acc:
            tot[k].append(acc[k])
            dur[k].append(d[k])
            cnt[k] = c[k]
        walls.append(max(x[2] for x in v) - t0)
    walls.sort()
    w50 = st.median(walls)
    print("%d passes of %d dispatches: wall p10 %.1f  p50 %.1f  p90 %.1f  mean %.1f us" % (
        len(ps), full, walls[len(walls) // 10] / 1e3, w50 / 1e3, walls[len(walls) * 9 // 10] / 1e3,
        st.mean(walls) / 1e3))
    print("%-44s %5s %9s %7s %9s %9s" % ("kernel", "n", "inc us", "share", "inc/disp", "dur/disp"))
    for m, k in sorted(((st.median(v), k) for k, v in tot.items()), reverse=True)[:opt["--top"]]:
        print("%-44s %5d %9.1f %6.1f%% %9.2f %9.2f" % (k, cnt[k], m / 1e3, 100 * m / w50,
                                                      m / 1e3 / cnt[k],
                                                      st.median(dur[k]) / 1e3 / cnt[k]))
    if seq:
        print()
        print("%4s %-30s %5s o c %7s %7s %7s %7s %7s %9s" % ("rec", "kernel", "wgs", "inc10", "inc50",
                                                            "inc90", "dur", "lead", "cum us"))
        cum = 0.0
        for s in sorted(smeta):
            a = sorted(sinc[s])
            n = len(a)
            cum += st.median(a)
            print("%4d %-30s %5d %s %s %7.1f %7.1f %7.1f %7.1f %7.1f %9.1f" % (
                s, smeta[s][0][:30], smeta[s][1], smeta[s][2], smeta[s][3], a[n // 10] / 1e3,
                st.median(a) / 1e3, a[n * 9 // 10] / 1e3, st.median(sdur[s]) / 1e3,
                st.median(slead[s]) / 1e3, cum / 1e3))


if __name__ == "__main__":
    main()
