#!/usr/bin/env python3
"""gapan.py stamps.csv [show]: breakdown of all card-0 dispatches (tape passes + live) between the end of the verify pass and the start of the next verify."""
import csv,sys,statistics as st,re
from collections import defaultdict
R=[]
for r in csv.DictReader(open(sys.argv[1])):
    try: R.append((int(r['start_ns']),int(r['end_ns']),int(r['pass']),int(r['records']),r['kernel'],int(r['grid']),int(r['wg'])))
    except: pass
R.sort()
def nm(k):
    m=re.match(r'_Z\d+(\w+?)(?:I|E|v|P|L)',k)
    k=k.replace('_ZN12_GLOBAL__N_','')
    return re.sub(r'^\d+','',k)[:48]
# verify passes: records>500
vi=[i for i,x in enumerate(R) if x[2]>0 and x[3]>500]
# group verify dispatches by pass id
vp=defaultdict(list)
for i in vi: vp[R[i][2]].append(i)
passes=sorted(vp)
seqs=[]
for a,b in zip(passes,passes[1:]):
    ea=max(R[i][1] for i in vp[a]); sb=min(R[i][0] for i in vp[b])
    seqs.append((a,ea,sb,max(vp[a]),min(vp[b])))
# per step: items with start in [ea, sb)
agg=defaultdict(list); tot=[]; idle=[]
ex=None
for a,ea,sb,ia,ib in seqs[5:-5]:
    items=[x for x in R if x[0]>=ea-1 and x[0]<sb and not (x[2]>0 and x[3]>500)]
    if len(items)<20: continue
    tot.append((sb-ea)/1e3)
    # segment timeline: gaps
    cur=ea; busy=0
    for x in items:
        agg[(nm(x[4]),x[2]>0 and x[3] or 0)].append((x[1]-x[0])/1e3)
    ex=ex or (items,ea)
print("steps",len(tot),"verify-end -> next verify-start p50 %.0f us"%st.median(tot))
if len(sys.argv)>2:
    items,ea=ex
    for x in items: print("%7.1f %6.1f pass=%d rec=%d wg=%d %s"%((x[0]-ea)/1e3,(x[1]-x[0])/1e3,x[2],x[3],x[5] and x[5] and x[5], nm(x[4])))
