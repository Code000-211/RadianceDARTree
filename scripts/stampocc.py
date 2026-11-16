#!/usr/bin/env python3
"""stampocc.py stamps.csv : per kind of pass, time-weighted CU fill from dispatch grid/workgroup size.
CU fill of a dispatch = min(1, workgroups/64) (a workgroup occupies at least one CU; 64 CUs per R9700).
This is a LOWER bound of busy CUs for kernels with < 64 workgroups and an upper bound of idleness."""
import csv,sys,collections,statistics as st
CU=64
by=collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(sys.argv[1])):
    try:
        s,e=int(r['start_ns']),int(r['end_ns']); g=int(r['grid']); w=int(r['wg'])
    except: continue
    if e<=s or w==0: continue
    by[int(r['records'])][int(r['pass'])].append((s,e,max(1,g//w),r['kernel'],int(r['queue'])))
def runs(iv):  # union length
    iv=sorted(iv); tot=0; ca=cb=None
    for a,b in iv:
        if cb is None or a>cb:
            if cb is not None: tot+=cb-ca
            ca,cb=a,b
        elif b>cb: cb=b
    if cb is not None: tot+=cb-ca
    return tot
for rec,ps in sorted(by.items(),key=lambda x:-len(x[1])):
    if len(ps)<5: continue
    stats=[]; small=collections.Counter(); tcount=collections.Counter()
    for p,v in ps.items():
        wall=max(e for s,e,*_ in v)-min(s for s,e,*_ in v)
        # CU-time: integrate min(CU, sum of wgs of concurrently running dispatches) over time
        ev=[]
        for s,e,n,k,q in v: ev.append((s,n)); ev.append((e,-n))
        ev.sort(); cur=0; last=ev[0][0]; cut=0.0; busy=0
        for t,d in ev:
            if cur>0: cut+=(t-last)*min(1.0,cur/CU); busy+=t-last
            cur+=d; last=t
        stats.append((wall,busy,cut))
        for s,e,n,k,q in v:
            tcount[k[:60]]+=(e-s)
            if n<CU: small[k[:60]]+=(e-s)
    W=st.median(x[0] for x in stats); B=st.median(x[1] for x in stats); C=st.median(x[2] for x in stats)
    print(f"pass of {rec} records: {len(ps)} passes, wall {W/1e3:.0f}us busy {B/1e3:.0f}us CU-fill(lower bound) {100*C/W:.0f}% of wall, {100*C/B:.0f}% of busy; idle-CU time bound {100*(1-C/W):.0f}%")
    tot=sum(tcount.values())
    print("   share of dispatch time in kernels with <64 workgroups: %.0f%%"%(100*sum(small.values())/tot))
