#!/usr/bin/env python3
"""Recover recorded results without executing inference. Standard library only."""
import argparse,csv,hashlib,statistics,collections,shutil
from pathlib import Path

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows):
    if not rows: return
    p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def read(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.reader(f))
def recover(root,out):
    summary=[]; details=[]; hw=[]
    for p in sorted((root/'Accuracy_Ensemble').rglob('accuracy_*.csv')):
        rel=p.relative_to(root); parts=rel.parts
        if parts[1] not in ('VGG8_CIFAR10_All_bits','VGG8_CIFAR100','ResNet20t_CIFAR10_ver2') or 'Trash' in parts: continue
        scaling='VGG8_CIFAR10_Increased_M' in parts
        expected=3 if scaling else 2
        if parts[expected:expected+2]!=('Inference_pytorch','Result') or len(parts) not in (expected+3,expected+4) or (len(parts)==expected+4 and parts[-2].lower()!='saved'): continue
        if scaling and p.parent.name.lower()=='saved': cohort='vgg8_cifar10_paper_candidate'
        elif scaling: cohort='vgg8_cifar10_scaling'
        elif parts[1]=='VGG8_CIFAR10_All_bits': cohort='vgg8_cifar10_later_unmatched'
        elif parts[1]=='VGG8_CIFAR100': cohort='vgg8_cifar100_unmatched'
        else: cohort='resnet20_cifar10_partial' 
        rows=read(p); meta={r[0].lstrip('# ').strip():r[1] for r in rows if len(r)>1 and r[0].startswith('#')}
        idx=next((i for i,r in enumerate(rows) if r and r[0]=='type'),None)
        if idx is None: raise ValueError(f'No detail header: {p}')
        base=dict(cohort=cohort,source_file=str(rel),sha256=digest(p),weight_bits=int(meta['--wl_weight']),activation_bits=int(meta['--wl_activate']),sigma=float(meta['--vari']),requested_replicas=int(meta['--num_models(N)']))
        groups=collections.defaultdict(list); seen=set()
        for r in rows[idx+1:]:
            if not r: continue
            d=dict(zip(rows[idx],r)); typ=d['type']; t=int(d['ensemble_size']); key=(typ,t,d['id'])
            if key in seen: raise ValueError(f'Duplicate {key}: {p}')
            seen.add(key); a=float(d['accuracy']); c=int(d['correct']); n=int(d['total'])
            if n<=0 or not 0<=c<=n or abs(a-c/n)>1e-7: raise ValueError(f'Invalid counts: {p} {key}')
            groups[t].append((a,c,n)); details.append({**base,**d})
        for t,vals in sorted(groups.items()):
            a=[v[0] for v in vals]; summary.append(dict(**base,ensemble_size=t,records=len(a),mean_accuracy_percent=100*statistics.mean(a),std_population_pp=100*statistics.pstdev(a),weighted_accuracy_percent=100*sum(v[1] for v in vals)/sum(v[2] for v in vals),sample_evaluations=sum(v[2] for v in vals),record_status='recorded',individual_count_matches=(len(a)==base['requested_replicas'] if t==1 else '')))
        shutil.copyfile(p, (dest:=out/'accuracy'/'raw'/rel).parent.mkdir(parents=True,exist_ok=True) or dest)
    write(out/'accuracy'/'single_per_replica.csv',[r for r in details if r['type']=='individual']); write(out/'accuracy'/'ensemble_per_instance.csv',[r for r in details if r['type']=='ensemble']); write(out/'accuracy'/'summary.csv',summary)
    for mode in ('single','ensemble'):
        for tech in ('SRAM','RRAM','FeFET'):
            folder=root/f'HW_{mode}'/f'HW_{mode}_{tech}'/'Inference_pytorch'/'summary'
            for p in sorted(folder.glob('*summary*.csv')):
                d={r[0]:r[1] for r in read(p) if len(r)>1}; rel=p.relative_to(root)
                if not d.get('PIPE_ReadLatency_(us)'): continue
                bits=int(p.name.split('SynBit')[1].split('_')[0]); dyn=float(d['PIPE_ReadDynamicEnergy_(uJ)']); leak=float(d['PIPE_LeakageEnergy_(uJ)'])
                selected=(p.name.startswith('summary_') and ((mode=='ensemble' and '20260118' in p.name) or (mode=='single' and ('20260119' in p.name if tech=='SRAM' else '20260117' in p.name))))
                hw.append(dict(technology=tech,ensemble_size=1 if mode=='single' else 2,weight_bits=bits,source_file=str(rel),sha256=digest(p),paper_candidate=selected,area_mm2=float(d['ChipArea_(mm^2)']),latency_us=float(d['PIPE_ReadLatency_(us)']),dynamic_energy_uJ=dyn,leakage_energy_uJ=leak,total_energy_uJ=dyn+leak))
                dest=out/'hardware'/'raw'/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
    write(out/'hardware'/'all_candidates.csv',hw); write(out/'hardware'/'paper_candidates.csv',[r for r in hw if r['paper_candidate']]); print(f'{len(details)} detail records; {len(summary)} run/size summaries; {len(hw)} hardware files')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); recover(a.source,a.output)
