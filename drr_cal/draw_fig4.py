#!/usr/bin/env python3
"""Fig.4-style plot: our reproduced DRR vs paper's published DRR."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = 'results/dataset_results.csv'   # 改成你的实际输出路径

# 论文 Fig.4 读图值 (±0.03 精度), 论文名 -> (paper_DRR)
paper = {
  'FFM-250-50-0.50-SAT-1': 0.97, 'Scrum1k': 0.94, 'nasa93dem': 0.84,
  'xomo_flight': 0.74, 'xomo_osp': 0.75, 'xomo_ground': 0.70,
  'xomo_osp2': 0.63, 'SS-M': 0.81, 'SS-U': 0.57, 'Wine_quality': 0.60,
  'pom3d': 0.44, 'pom3a': 0.44, 'SS-D': 0.67, 'SS-B': 0.67,
  'SS-T': 0.32, 'rs-6d-c3_obj2': 0.33,
  'Health-Commits0000': 0.60, 'Health-ClosedIssues0000': 0.63,
}  # Health 两行对应论文 Health-Easy/Hard, 映射本身存疑, 图上会标注

# ---- 按论文 Fig.4 从左到右的顺序 (仅 SE/蓝点; 原图 x 轴无量化含义) ----
fig4_order = [
    ('FFM-250-50-0.50-SAT-1',   'FFM-250'),
    ('Scrum1k',                 'SCRUM'),
    ('nasa93dem',               'nasa93dem'),
    ('xomo_flight',             'Xomo Flight'),
    ('xomo_ground',             'Xomo Ground'),
    ('xomo_osp',                'Xomo OSP'),
    ('xomo_osp2',               'Xomo OSP2'),
    ('SS-U',                    'SS-U'),
    ('SS-M',                    'SS-M'),
    ('Wine_quality',            'Wine Quality'),
    ('pom3d',                   'pom3d'),
    ('pom3a',                   'Pom3a'),
    ('Health-ClosedIssues0000', 'Health-Hard (?)'),
    ('Health-Commits0000',      'Health-Easy (?)'),
    ('SS-D',                    'SS-D'),
    ('SS-B',                    'SS-B'),
    ('rs-6d-c3_obj2',           'rs-6d-c3-obj2'),
    ('SS-T',                    'SS-T'),
]
order_map  = {k: i for i, (k, _) in enumerate(fig4_order)}
label_map  = dict(fig4_order)

df = pd.read_csv(RESULTS)
df['paper'] = df['Dataset'].map(paper)
df['pos']   = df['Dataset'].map(order_map)
df = df.dropna(subset=['pos']).sort_values('pos').reset_index(drop=True)
df['label'] = df['Dataset'].map(label_map)

fig, ax = plt.subplots(figsize=(11, 5.5))
x = range(len(df))
ax.scatter(x, df['paper'], marker='o', s=55, color='#1f77b4',
           label='Paper Fig.4 (read off, ±0.03)')
ax.scatter(x, df['DRR'], marker='x', s=55, color='#d62728',
           label='Reproduced (code-as-released, l1)')
for i, r in df.iterrows():                       # 连线显示落差
    ax.plot([i, i], [r['paper'], r['DRR']], color='gray', lw=.7, alpha=.6)

# 标出 fallback 产物 (I == R-5 或 I == 0.5R)
fb = df[(df['I'] == df['R'] - 5) | (df['I'] == df['R'] // 2)]
# ax.scatter(fb.index, fb['DRR'], marker='s', s=130, facecolors='none',
#            edgecolors='red', lw=1.2, label='I is hardcoded fallback')

ax.axhline(1/3, color='k', ls='--', lw=1, label='DRR = 1/3 threshold')
ax.set_xticks(list(x))
ax.set_xticklabels(df['label'], rotation=70, ha='right', fontsize=8)
ax.set_ylabel('DRR = 1 - I/R'); ax.set_ylim(-0.02, 1.02)
ax.set_title('Reproduction of Lustosa & Menzies Fig.4 with released DRR calculator')
ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')
fig.tight_layout()
fig.savefig('fig4_reproduction.png', dpi=150)
print('saved fig4_reproduction.png')