#!/usr/bin/env python3
"""
make_fig_hin_sample.py -- data-driven visualisation of a sampled sub-graph of
the dynamic HIN used in DynHIN / R-HGT (Figure 2 candidate).

The graph is built with EXACTLY the rules of src/code-rhgt-v2.py:
  * per-architecture PCAP/SAR/STRACE parquet files are aggregated per Hash
    (AVG of numeric columns, FIRST of MalwareFamily/Arch) and inner-joined;
  * label = (MalwareFamily != 'benign'); duplicates on (Hash, arch) dropped;
  * syscall columns  = 'Call_*' (or containing 'syscall');
    protocol columns = the 14 fixed names used by the pipeline;
    file features    = PCAP + SAR columns (everything else), z-scored with
                       statistics of the TRAINING partition only;
  * edges: file -executes-> syscall  if value > 0,
           file -uses_proto-> protocol if value > 0,
           file -runs_on-> arch,
           file -similar-> file  = kNN (k = 3, Euclidean) on standardised
                                   features, computed over the WHOLE training
                                   partition (training graph only).
  * the split is the same hash-disjoint StratifiedGroupKFold selection
    (group_safe_stratified_split) for the chosen seed.

Sampling for display (the full graph has ~13.6k file nodes):
  * pick `--per-cell` seed files for every (architecture x label) cell of the
    training partition, then add their k = 3 nearest training neighbours, so
    the drawn 'similar' edges are real edges of the training graph;
  * all 4 arch nodes and all protocol nodes touched by the sample are drawn;
    only the `--top-syscalls` most frequent syscalls of the sample are drawn
    (state this in the caption).

Usage:
  python make_fig_hin_sample.py --data-dir data/cic-ynu-iotmal --out paper/figures/fig_hin_sample
Requirements: pandas, numpy, duckdb, pyarrow, scikit-learn, networkx, matplotlib
"""
import argparse, os, gc, json
import numpy as np, pandas as pd

# ------------------------------------------------------------------ loading
def read_parquet_with_duckdb(file_path):
    import duckdb
    con = duckdb.connect(database=':memory:')
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}', union_by_name=True) LIMIT 0").df()
    numeric = ['DOUBLE','FLOAT','REAL','DECIMAL','TINYINT','SMALLINT','INTEGER','BIGINT','HUGEINT',
               'UTINYINT','USMALLINT','UINTEGER','UBIGINT']
    num_cols, cat_cols = [], []
    for _, r in desc.iterrows():
        c, t = r['column_name'], str(r['column_type']).upper()
        if c == 'Hash': continue
        if any(x in t for x in numeric): num_cols.append(c)
        elif c in ['MalwareFamily', 'Arch', 'arch', 'label']: cat_cols.append(c)
    parts = [f'AVG(TRY_CAST("{c}" AS FLOAT)) as "{c}"' for c in num_cols] + [f'FIRST("{c}") as "{c}"' for c in cat_cols]
    df = con.execute(f'SELECT "Hash", {", ".join(parts)} FROM read_parquet(\'{file_path}\', union_by_name=True) GROUP BY "Hash"').df()
    for c in df.select_dtypes(include=['float64']).columns: df[c] = df[c].astype(np.float32)
    con.close(); return df

def load_and_merge(base, cache=None):
    if cache and os.path.exists(cache): return pd.read_parquet(cache)
    dfs = []
    for arch in ['arm', 'mips', 'mipsel', 'x86']:
        d = os.path.join(base, arch)
        p, s, t = (read_parquet_with_duckdb(os.path.join(d, f'{m}.parquet')) for m in ('pcap', 'sar', 'strace'))
        cs = [c for c in s.columns if c not in p.columns or c == 'Hash']
        ct = [c for c in t.columns if (c not in p.columns and c not in s.columns) or c == 'Hash']
        m = p.merge(s[cs], on='Hash', how='inner').merge(t[ct], on='Hash', how='inner')
        m['arch'] = arch; dfs.append(m); del p, s, t; gc.collect()
    df = pd.concat(dfs, ignore_index=True)
    df['label'] = (df['MalwareFamily'].astype(str).str.lower() != 'benign').astype(int)
    df = df.drop_duplicates(subset=['Hash', 'arch']).reset_index(drop=True)
    if cache: df.to_parquet(cache, index=False)
    return df

def split_columns(df):
    cols = df.columns.tolist()
    sys_c = [c for c in cols if c.startswith('Call_') or 'syscall' in c.lower()]
    proto = [c for c in ['TCP','UDP','ICMP','HTTP','HTTPS','DNS','Telnet','SSH','IRC','DHCP','ARP','IGMP','IPv','LLC'] if c in cols]
    excl = set(['label','arch','Arch','Hash','MalwareFamily','stratify_col'] + sys_c + proto)
    pcap = [c for c in cols if c in ['Header_Length','Protocol Type','Time_To_Live','Rate','Tot sum','Min','Max','AVG','Std',
                                     'Tot size','IAT','Number','Variance'] or c.endswith('_flag_number') or c.endswith('_count')]
    sar = [c for c in cols if c not in excl and c not in pcap]
    return pcap + sar, sys_c, proto, len(pcap), len(sar)

def clean(df, cols):
    v = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    return np.clip(np.nan_to_num(v, nan=0.0, posinf=1e9, neginf=-1e9), -1e9, 1e9).astype(np.float32)

def group_safe_split(df, seed, test_size=0.2):
    from sklearn.model_selection import StratifiedGroupKFold
    strat = df['label'].astype(str) + '_' + df['arch'].astype(str)
    groups = df['Hash'].astype(str).values
    n = min(max(2, int(round(1 / test_size))), df['Hash'].nunique())
    glob = strat.value_counts(normalize=True); best, bs = None, np.inf
    for tr, te in StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(df, strat.values, groups=groups):
        ts = strat.iloc[te].value_counts(normalize=True); idx = glob.index.union(ts.index)
        sc = (glob.reindex(idx, fill_value=0) - ts.reindex(idx, fill_value=0)).abs().sum() + 2 * abs(len(te) / len(df) - test_size)
        if sc < bs: bs, best = sc, (tr, te)
    return df.iloc[best[0]].reset_index(drop=True), df.iloc[best[1]].reset_index(drop=True)

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/cic-ynu-iotmal')
    ap.add_argument('--out', default='fig_hin_sample')
    ap.add_argument('--cache', default=None, help='optional merged-parquet cache path')
    ap.add_argument('--seed', type=int, default=42, help='split seed (one of the 10 paper seeds)')
    ap.add_argument('--per-cell', type=int, default=2, help='seed files per (arch x label) cell')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--top-syscalls', type=int, default=14)
    ap.add_argument('--sample-seed', type=int, default=7)
    a = ap.parse_args()

    df = load_and_merge(a.data_dir, a.cache)
    df['stratify_col'] = df['label'].astype(str) + '_' + df['arch'].astype(str)
    keep = df['stratify_col'].value_counts()[lambda x: x > 1].index
    df = df[df['stratify_col'].isin(keep)].reset_index(drop=True)
    num_cols, sys_c, proto, n_pcap, n_sar = split_columns(df)
    print(f'samples={len(df)}  benign={int((df.label==0).sum())}  malware={int(df.label.sum())}')
    print(f'file features={len(num_cols)} (PCAP {n_pcap} + SAR {n_sar}); syscalls={len(sys_c)}; protocols={len(proto)}')

    tr, te = group_safe_split(df, a.seed)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit(clean(tr, num_cols)).transform(clean(tr, num_cols)).astype(np.float32)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=a.k + 1, algorithm='brute', metric='euclidean').fit(X)
    SYS = clean(tr, sys_c) > 0; PRO = clean(tr, proto) > 0

    rng = np.random.default_rng(a.sample_seed)
    seeds = []
    for (arch, lab), g in tr.groupby(['arch', 'label']):
        seeds += rng.choice(g.index.values, size=min(a.per_cell, len(g)), replace=False).tolist()
    _, nb = nn.kneighbors(X[seeds])
    knn_edges = set()
    for s, row in zip(seeds, nb):
        for j in row[1:a.k + 1]:
            if j != s: knn_edges.add((int(s), int(j)))
    files = sorted(set(seeds) | {j for _, j in knn_edges})
    # kNN edges among ALL drawn files (true training-graph edges)
    _, nb_all = nn.kneighbors(X[files]); fs = set(files)
    for s, row in zip(files, nb_all):
        for j in row[1:a.k + 1]:
            if int(j) in fs and int(j) != s: knn_edges.add((int(s), int(j)))
    knn_edges = {tuple(sorted(e)) for e in knn_edges}

    freq = SYS[files].sum(0)
    top_sys = [sys_c[i] for i in np.argsort(-freq)[:a.top_syscalls] if freq[i] > 0]
    used_proto = [proto[i] for i in range(len(proto)) if PRO[files, i].any()]
    stats = dict(seed=a.seed, n_train=len(tr), n_test=len(te), drawn_files=len(files), seeds=len(seeds),
                 knn_edges=len(knn_edges), syscalls_shown=len(top_sys), syscalls_total=len(sys_c),
                 protocols_shown=used_proto, file_dim=len(num_cols),
                 benign_drawn=int((tr.loc[files, 'label'] == 0).sum()), malware_drawn=int(tr.loc[files, 'label'].sum()),
                 mean_syscalls_per_file_train=round(float(SYS.sum(1).mean()), 1),
                 mean_protocols_per_file_train=round(float(PRO.sum(1).mean()), 1),
                 executes_edges_drawn=None, uses_proto_edges_drawn=None)
    # homophily of the TRAINING kNN graph (all training files, not only the sample)
    _, nb_tr = nn.kneighbors(X)
    src = np.repeat(np.arange(len(tr)), a.k); dst = nb_tr[:, 1:a.k + 1].reshape(-1)
    ar_ = tr['arch'].values; lb_ = tr['label'].values
    stats['knn_same_arch_rate_train'] = round(float((ar_[src] == ar_[dst]).mean()), 4)
    stats['knn_same_label_rate_train'] = round(float((lb_[src] == lb_[dst]).mean()), 4)

    # ------------------------------------------------------------- graph
    import networkx as nx
    G = nx.Graph()
    lab = lambda c: c.replace('Call_', '').lstrip('_')
    for f in files: G.add_node(('f', f), t='file', y=int(tr.loc[f, 'label']), arch=tr.loc[f, 'arch'])
    for ar in ['arm', 'mips', 'mipsel', 'x86']: G.add_node(('a', ar), t='arch')
    for p in used_proto: G.add_node(('p', p), t='proto')
    for s in top_sys: G.add_node(('s', s), t='sys')
    sidx = {c: i for i, c in enumerate(sys_c)}; pidx = {c: i for i, c in enumerate(proto)}
    for f in files:
        G.add_edge(('f', f), ('a', tr.loc[f, 'arch']), r='runs_on')
        for s in top_sys:
            if SYS[f, sidx[s]]: G.add_edge(('f', f), ('s', s), r='executes')
        for p in used_proto:
            if PRO[f, pidx[p]]: G.add_edge(('f', f), ('p', p), r='uses_proto')
    for u, v in knn_edges: G.add_edge(('f', u), ('f', v), r='similar')
    stats['executes_edges_drawn'] = sum(1 for *_, d in G.edges(data=True) if d['r'] == 'executes')
    stats['uses_proto_edges_drawn'] = sum(1 for *_, d in G.edges(data=True) if d['r'] == 'uses_proto')
    print(json.dumps(stats, indent=1))

    # layout: (1) files are placed around their architecture hub using only the
    # runs_on + similar edges (keeps kNN neighbourhoods readable); (2) syscall and
    # protocol anchors are placed on outer arcs because every file is linked to
    # most of them, which would otherwise collapse the layout.
    hubs = {('a', 'arm'): (-0.55, 0.42), ('a', 'x86'): (0.55, 0.42), ('a', 'mips'): (-0.55, -0.42), ('a', 'mipsel'): (0.55, -0.42)}
    H = nx.Graph(); H.add_nodes_from(hubs)
    for u, v, d in G.edges(data=True):
        if d['r'] in ('runs_on', 'similar'): H.add_edge(u, v, weight=1.0 if d['r'] == 'runs_on' else 2.5)
    pos = nx.spring_layout(H, pos=dict(hubs), fixed=list(hubs), k=0.13, iterations=600, seed=a.sample_seed, weight='weight')
    for i, s_ in enumerate(top_sys):
        th = np.deg2rad(-62 + 124 * i / max(len(top_sys) - 1, 1)); pos[('s', s_)] = np.array([1.05 + 0.32 * np.cos(th), 1.0 * np.sin(th)])
    for i, p in enumerate(used_proto):
        th = np.deg2rad(-50 + 100 * i / max(len(used_proto) - 1, 1)); pos[('p', p)] = np.array([-1.05 - 0.30 * np.cos(th), 0.85 * np.sin(th)])

    # ------------------------------------------------------------- draw
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    C = dict(file='#2F6DB5', fileb='#A9C8EC', sys='#E07B22', arch='#3D9A57', proto='#8656B8', ink='#262B33')
    fig, ax = plt.subplots(figsize=(6.48, 4.3)); ax.set_axis_off()
    E = lambda r: [(u, v) for u, v, d in G.edges(data=True) if d['r'] == r]
    nx.draw_networkx_edges(G, pos, E('executes'), edge_color=C['sys'], alpha=0.10, width=0.4, ax=ax)
    nx.draw_networkx_edges(G, pos, E('uses_proto'), edge_color=C['proto'], alpha=0.12, width=0.4, ax=ax)
    nx.draw_networkx_edges(G, pos, E('runs_on'), edge_color=C['arch'], alpha=0.75, width=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, E('similar'), edge_color=C['file'], alpha=1.0, width=1.0, style=(0, (2.5, 1.4)), ax=ax)
    N = lambda t, **k: [n for n, d in G.nodes(data=True) if d['t'] == t and all(d.get(a) == b for a, b in k.items())]
    nx.draw_networkx_nodes(G, pos, N('file', y=0), node_color=C['fileb'], edgecolors=C['file'], linewidths=0.8, node_size=55, ax=ax)
    nx.draw_networkx_nodes(G, pos, N('file', y=1), node_color=C['file'], edgecolors='#1B4579', linewidths=0.8, node_size=55, ax=ax)
    nx.draw_networkx_nodes(G, pos, N('sys'), node_shape='s', node_color='#FBE3CF', edgecolors=C['sys'], linewidths=0.9, node_size=70, ax=ax)
    nx.draw_networkx_nodes(G, pos, N('proto'), node_shape='h', node_color='#E8DDF4', edgecolors=C['proto'], linewidths=0.9, node_size=150, ax=ax)
    nx.draw_networkx_nodes(G, pos, N('arch'), node_shape='D', node_color=C['arch'], edgecolors='#24693A', linewidths=1.0, node_size=900, ax=ax)
    for n in N('arch'):
        x, y = pos[n]; ax.text(x, y, {'arm':'ARM','mips':'MIPS','mipsel':'MIPSEL','x86':'x86'}[n[1]],
                               ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    for n in N('sys'):
        x, y = pos[n]; ax.text(x + 0.05, y, lab(n[1]), ha='left', va='center', fontsize=6.5, color='#8A4A12')
    for n in N('proto'):
        x, y = pos[n]; ax.text(x - 0.07, y, n[1], ha='right', va='center', fontsize=6.5, color=C['proto'])
    leg = [Line2D([], [], marker='o', ls='', mfc=C['fileb'], mec=C['file'], ms=5, label='file (benign)'),
           Line2D([], [], marker='o', ls='', mfc=C['file'], mec='#1B4579', ms=5, label='file (malware)'),
           Line2D([], [], marker='s', ls='', mfc='#FBE3CF', mec=C['sys'], ms=5, label='syscall'),
           Line2D([], [], marker='D', ls='', mfc=C['arch'], mec='#24693A', ms=5, label='arch'),
           Line2D([], [], marker='h', ls='', mfc='#E8DDF4', mec=C['proto'], ms=6, label='protocol'),
           Line2D([], [], color=C['sys'], alpha=.6, lw=1, label='executes'),
           Line2D([], [], color=C['arch'], lw=1, label='runs_on'),
           Line2D([], [], color=C['proto'], lw=1, label='uses_proto'),
           Line2D([], [], color=C['file'], lw=1.1, ls=(0, (3, 1.6)), label=f'similar ($k$NN, $k$={a.k})')]
    ax.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, -0.005), ncol=5, frameon=False, fontsize=6.8,
              handletextpad=0.4, columnspacing=1.1)
    fig.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out + '.pdf'); fig.savefig(a.out + '.png', dpi=300)
    json.dump(stats, open(a.out + '_stats.json', 'w'), indent=1)
    print('saved', a.out + '.pdf/.png/_stats.json')

if __name__ == '__main__':
    main()
