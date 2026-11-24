from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {'NO_ORDINAL_SINGLE_CELL_SUPERVISION': 'No ordinal loss',
            'NO_CORAL': 'No CORAL', 'NO_MMD': 'No MMD',
            'NO_ENDPOINT_GENE_WEIGHTING': 'No gene weighting'}
COHORTS = ['GSE79362', 'GSE94438', 'GSE107993_GSE107994_combined']
ENDPOINTS = ['E_6M', 'E_12M', 'E_COX']

def main():
    d = pd.read_csv(ROOT / 'data/result3_seeds.tsv', sep='\t')
    assert not d.duplicated(['variant', 'seed', 'cohort', 'endpoint']).any()
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 7.3), sharex=True, layout='constrained')
    for i, cohort in enumerate(COHORTS):
        for j, endpoint in enumerate(ENDPOINTS):
            ax = axes[i, j]
            for y, variant in enumerate(VARIANTS):
                z = d[(d.cohort == cohort) & (d.endpoint == endpoint) & (d.variant == variant)].delta.dropna()
                if len(z):
                    ax.scatter(z, np.full(len(z), y), s=13, color='#8295a5', alpha=.7)
                    ax.plot(z.mean(), y, 'o', color='#b84941', markersize=4)
            if not len(d[(d.cohort == cohort) & (d.endpoint == endpoint)].delta.dropna()):
                ax.text(.5, .5, 'Not estimable', transform=ax.transAxes, ha='center', va='center', fontsize=8)
            ax.axvline(0, color='0.5', linestyle='--', linewidth=.7)
            ax.set_yticks(range(4), list(VARIANTS.values()) if j == 0 else [])
            ax.invert_yaxis(); ax.spines[['top', 'right']].set_visible(False)
            if i == 0: ax.set_title(['180 days', '365 days', 'Cox'][j])
            if j == 0: ax.set_ylabel(cohort.replace('_combined', '').replace('_', ' / '))
            if i == 2: ax.set_xlabel('Change in performance')
    fig.legend(handles=[Line2D([], [], marker='o', color='#8295a5', linestyle='', label='Individual seed'),
                        Line2D([], [], marker='o', color='#b84941', linestyle='', label='Seed mean')],
               loc='outside upper center', ncol=2, frameon=False)
    out = ROOT / 'figures'; out.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'): fig.savefig(out / f'result3_ablation.{ext}', dpi=300)
    plt.close(fig)
    print('Rendered seed-level changes; empty panel denotes non-estimable endpoint')

if __name__ == '__main__': main()
