#!/usr/bin/env python3
"""stamppasses.py stamps.csv : timeline of consecutive passes (N=1 stamps): per kind of pass wall, gap to next."""
import csv,sys,statistics as st
from collections import defaultdict
rows=defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    try: rows[int(r['pass'])].append((int(r['start_ns']),int(r['end_ns']),int(r['records']),r['kernel'],int(r['queue'])))
    except: pass
ps=sorted(rows)
info={}
for p in ps:
    v=rows[p]
    s=min(a for a,b,*_ in v); e=max(b for a,b,*_ in v)
    if s<=0 or e<=0: continue
    info[p]=(s,e,v[0][2],len(v))
kinds=defaultdict(list); gaps=defaultdict(list)
prev=None
for p in sorted(info):
    s,e,rec,n=info[p]
    kinds[rec].append((e-s)/1e3)
    if prev is not None and prev[0]==p-1:
        gaps[(prev[2],rec)].append((s-prev[1])/1e3)
    prev=(p,e,rec)
print("kind(records)  n  wall_us(p50)")
for k,v in sorted(kinds.items(),key=lambda x:-len(x[1])):
    if len(v)>=5: print(k,len(v),round(st.median(v),1))
print("gap_us between consecutive pass kinds (prev->next): n p50 p90")
for k,v in sorted(gaps.items(),key=lambda x:-len(x[1])):
    if len(v)>=5:
        v=sorted(v); print(k,len(v),round(st.median(v),1),round(v[int(.9*len(v))],1))
