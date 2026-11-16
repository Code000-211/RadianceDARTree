import csv,sys,statistics as st,re
from collections import defaultdict
R=[]
for r in csv.DictReader(open(sys.argv[1])):
    try: R.append((int(r['start_ns']),int(r['end_ns']),int(r['pass']),int(r['records']),r['kernel'],int(r['grid']),int(r['wg'])))
    except: pass
R.sort()
vp=defaultdict(list)
for x in R:
    if x[2]>0 and x[3]>500: vp[x[2]].append(x)
ps=sorted(vp)
def tag(k):
    for t in ['sample_temp','sample_topk','ar_gather_2rank','merge_topk','narrow_kernel','sample_pick','couple_verify','step_advance','aql_copy_link','aql_copy_ranges','aql_copy2d','wtab_patch']:
        if t in k: return t
    return k[:30]
acc=defaultdict(list); tot=[]
for a,b in zip(ps,ps[1:]):
    ea=max(x[1] for x in vp[a]); sb=min(x[0] for x in vp[b])
    # last verify dispatch start
    items=[x for x in R if ea-2000<=x[0]<sb and not(x[2]>0 and x[3]>500)]
    # draft passes: records<500, pass>0
    seq=[]; 
    first_hist=min([x[0] for x in items if x[2]>0] or [sb])
    pre=[x for x in items if x[2]==0 and x[0]<first_hist]
    if not pre: continue
    tot.append((first_hist-ea)/1e3)
    acc['gap_total(verifyend->first hist dispatch)'].append((first_hist-ea)/1e3)
    prev=ea
    for x in pre:
        acc['start '+tag(x[4])+' #'+str(sum(1 for y in pre[:pre.index(x)] if tag(y[4])==tag(x[4])))].append((x[0]-ea)/1e3)
        acc['dur '+tag(x[4])+' #'+str(sum(1 for y in pre[:pre.index(x)] if tag(y[4])==tag(x[4])))].append((x[1]-x[0])/1e3)
    acc['sum_busy_pre'].append(sum(x[1]-x[0] for x in pre)/1e3)
    acc['pre_end'].append((max(x[1] for x in pre)-ea)/1e3)
    # verify tape tail: last 3 dispatch end relative to ea
for k,v in acc.items():
    if len(v)>20: print("%-50s n=%d p50=%.1f p10=%.1f p90=%.1f"%(k,len(v),st.median(v),sorted(v)[len(v)//10],sorted(v)[len(v)*9//10]))
