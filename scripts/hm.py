import csv,sys,statistics as st
from collections import defaultdict
R=[]
for r in csv.DictReader(open('stamps2.csv')):
    try: R.append((int(r['start_ns']),int(r['end_ns']),int(r['pass']),int(r['records']),r['kernel']))
    except: pass
R.sort()
M=[]
for l in open('marks2.csv'):
    t,rk,tag=l.strip().split(',',2); M.append((int(t),int(rk),tag))
M.sort()
vp=defaultdict(list)
for x in R:
    if x[2]>0 and x[3]>500: vp[x[2]].append(x)
ps=sorted(vp)
def tg(k):
    for t in ['sample_temp','sample_topk','ar_gather_2rank','merge_topk','narrow_kernel','sample_pick','couple_verify','step_advance','aql_copy_link','aql_copy_ranges']:
        if t in k: return t
    return None
acc=defaultdict(list)
import bisect
mt=[m[0] for m in M]
for a,b in zip(ps,ps[1:]):
    ea=max(x[1] for x in vp[a]); sb=min(x[0] for x in vp[b])
    lo=bisect.bisect_left(mt,ea-30000000); hi=bisect.bisect_left(mt,sb+1000)
    # marks in window: take the ones after ea-? ; tag ordinal
    cnt=defaultdict(int)
    for m in M[lo:hi]:
        k="%s[%d]#%d"%(m[2],m[1],cnt[(m[2],m[1])]); cnt[(m[2],m[1])]+=1
        acc['H '+k].append((m[0]-ea)/1e3)
    pre=[x for x in R if ea-2000<=x[0]<sb and x[2]==0 and not (x[2]>0 and x[3]>500)]
    c2=defaultdict(int)
    for x in pre:
        t=tg(x[4])
        if not t: continue
        k="G %s #%d"%(t,c2[t]); c2[t]+=1
        acc[k+" start"].append((x[0]-ea)/1e3); acc[k+" end"].append((x[1]-ea)/1e3)
    fh=min([x[0] for x in R if ea<=x[0]<sb and x[2]>0 and x[3]<500] or [sb])
    acc['G first hist start'].append((fh-ea)/1e3)
out=[(st.median(v),k,len(v)) for k,v in acc.items() if len(v)>20]
for m,k,n in sorted(out): print("%8.1f  %s (n=%d)"%(m,k,n))
