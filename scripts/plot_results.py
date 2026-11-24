import csv,collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]; out=root/'figures';out.mkdir(exist_ok=True)
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
for c in read(root/'results/cohorts.csv'):
 d=root/'results'/c['directory']; rows=read(d/'single_summary.csv')+read(d/'ensemble_summary.csv'); groups=collections.defaultdict(list)
 for r in rows:groups[(r['weight_bits'],r['ensemble_size'])].append((float(r['sigma']),float(r['mean_accuracy_percent'])))
 fig,ax=plt.subplots(figsize=(8,5))
 for (b,t), vals in sorted(groups.items()):
  vals.sort(); x,y=zip(*vals); dup=len(set(x))!=len(x);ax.plot(x,y,marker='o',linestyle='None' if dup else '-',label=f'{b} bit, T={t}')
 ax.set(xlabel='Conductance variation',ylabel='Mean accuracy (%)',title=c['cohort']);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/(c['cohort']+'.png'),dpi=180);plt.close(fig)
rows=read(root/'results/hardware/paper_candidates.csv')
for k,label in [('area_mm2','Area (mm²)'),('latency_us','Latency (µs)'),('total_energy_uJ','Energy (µJ)')]:
 fig,ax=plt.subplots();ax.bar([f"{r['technology']}\n{r['weight_bits']}b T={r['ensemble_size']}" for r in rows],[float(r[k]) for r in rows]);ax.set_ylabel(label);fig.tight_layout();fig.savefig(out/(k+'.png'),dpi=180);plt.close(fig)
