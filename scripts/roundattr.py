import csv,sys,statistics as st,re
from collections import defaultdict
P=defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    try: P[int(r['pass'])].append((int(r['seq']),int(r['start_ns']),int(r['end_ns']),r['kernel'],int(r['grid']),int(r['wg']),int(r['records'])))
    except: pass
def nm(k):
    k=k.replace('_ZN12_GLOBAL__N_','')
    k=re.sub(r'^_Z\d+','',k); k=re.sub(r'^\d+','',k)
    return k[:38]
inc=defaultdict(list); wall=defaultdict(list)
for p,v in P.items():
    if v[0][6] not in (41,42): continue   # round 2/3 (1 row)
    v.sort(key=lambda x:x[0])
    t0=min(x[1] for x in v); 
    prev=t0
    # incremental by end time order
    order=sorted(v,key=lambda x:x[2])
    for x in order:
        e=x[2]
        d=max(0,e-prev); prev=max(prev,e)
        inc[(x[0],nm(x[3]),x[4]//max(x[5],1))].append(d/1e3)
    wall[0].append((max(x[2] for x in v)-t0)/1e3)
print("round wall p50 %.1f us (n=%d)"%(st.median(wall[0]),len(wall[0])))
rows=sorted(inc.items())
tot=0
for (s,n,g),v in rows:
    m=st.median(v); tot+=m
    print("%2d %-40s wgs=%-6d incr_p50=%6.1f"%(s,n,g,m))
print("sum incr",tot)
