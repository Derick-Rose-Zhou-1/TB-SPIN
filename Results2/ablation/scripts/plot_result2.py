from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'NO_ORDINAL': 'No ordinal loss', 'NO_CROSSCOV': 'No cross-covariance',
          'NO_IDENTITY': 'No identity supervision', 'NO_KL': 'No KL',
          'MEAN_ONLY': 'Mean pooling only',
          'NO_PARTICIPANT_BALANCING': 'No participant balancing'}

def main():
    d = pd.read_csv(ROOT / 'data/result2_summary.tsv', sep='\t')
    d = d[(d.task == 'T2') & (d.metric == 'AUROC')].set_index('variant').loc[list(LABELS)]
    assert len(d) == 6 and d.index.is_unique
    fig, ax = plt.subplots(figsize=(6.6, 3.4), layout='constrained')
    y = list(range(len(d)))
    ax.hlines(y, d.ci_low, d.ci_high, color='#516b82', linewidth=1.2)
    ax.scatter(d.delta_mean, y, color='#b84941', zorder=3, s=24)
    ax.axvline(0, color='0.5', linestyle='--', linewidth=.8)
    ax.set(yticks=y, yticklabels=[LABELS[x] for x in d.index],
           xlabel='Change in AUROC (ablation minus full)', title='Clinical-state model: T2')
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    out = ROOT / 'figures'; out.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'result2_ablation.{ext}', dpi=300)
    plt.close(fig)
    print('Rendered 6 ablations from frozen summary')

if __name__ == '__main__':
    main()
