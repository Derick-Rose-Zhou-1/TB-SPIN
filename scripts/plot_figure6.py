from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['TB-SPIN', 'SIDISH', 'scAB', 'scPAS', 'DEGAS']
COLORS = ['#b9473e', '#66869b', '#8da786', '#bc9b68', '#8b80a5']

def validate(data):
    columns = ['cohort', 'endpoint', 'method', 'estimate', 'ci_low', 'ci_high']
    if not set(columns).issubset(data.columns) or data[columns].isna().any().any():
        raise ValueError('Missing benchmark columns or values')
    if data.duplicated(['cohort', 'endpoint', 'method']).any():
        raise ValueError('Duplicate benchmark rows')
    if not (data.ci_low.ge(0) & data.ci_low.le(data.estimate) &
            data.estimate.le(data.ci_high) & data.ci_high.le(1)).all():
        raise ValueError('Invalid AUC or confidence interval')
    if not set(data.method).issubset(METHODS):
        raise ValueError('Unexpected method')

def main():
    data = pd.read_csv(ROOT / 'data' / 'figure6_benchmarks.csv')
    validate(data)
    expected = {(c, e) for c in ['GSE79362', 'GSE94438'] for e in ['H1', 'H2', 'H3']}
    if set(zip(data.cohort, data.endpoint)) != expected or len(data) != 24:
        raise ValueError('Expected 24 rows across two cohorts and three endpoints')
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True, layout='constrained')
    for ax, cohort in zip(axes, ['GSE79362', 'GSE94438']):
        subset = data[data.cohort == cohort]
        for group, endpoint in enumerate(['H1', 'H2', 'H3']):
            rows = subset[subset.endpoint == endpoint].set_index('method')
            present = [m for m in METHODS if m in rows.index]
            for j, method in enumerate(present):
                row = rows.loc[method]
                x = group + (j - (len(present) - 1) / 2) * .19
                ax.bar(x, row.estimate, width=.18, color=COLORS[METHODS.index(method)])
                ax.errorbar(x, row.estimate, yerr=[[row.estimate-row.ci_low],
                            [row.ci_high-row.estimate]], fmt='none', color='black',
                            capsize=2, linewidth=.7)
        ax.axhline(.5, color='0.5', linestyle='--', linewidth=.7)
        ax.set(xticks=range(3), xticklabels=['Complete follow-up', '180 days', '365 days'],
               title=cohort, ylim=(0, 1.05), xlabel='Endpoint')
        ax.tick_params(axis='x', labelrotation=15)
    axes[0].set_ylabel('AUC (95% CI)')
    fig.legend(handles=[Patch(color=c, label=m) for m, c in zip(METHODS, COLORS)],
               loc='outside upper center', ncol=5, frameon=False)
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'figure6_from_supplement.{ext}', dpi=300)
    plt.close(fig)
    print(f'Plotted {len(data)} benchmark rows from Supplementary Table 9')

if __name__ == '__main__':
    main()
