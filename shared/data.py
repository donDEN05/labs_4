"""Complete Journey → leakage-controlled customer/category/week panel."""
import hashlib
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from shared.metrics import save

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'shared/data'


def panel(rows, households, categories, window=4):
    start, end = int(rows.week.min()), int(rows.week.max())
    agg = rows.groupby(['household', 'category', 'week'])[['quantity', 'sales_value', 'retail_disc']].sum()
    records = []
    weeks = np.arange(start + window, end + 1)
    a, b = int(len(weeks) * .6), int(len(weeks) * .8)
    train_end, val_end = int(weeks[a - 1]), int(weeks[b - 1])
    for h in households:
        for c in categories:
            hist = agg.reindex(pd.MultiIndex.from_product([[h], [c], np.arange(start, end + 1)], names=agg.index.names), fill_value=0).to_numpy(float)
            for w in weeks:
                past = hist[w - start - window:w - start]
                split = 'train' if w <= train_end else 'validation' if w <= val_end and w - window > train_end else 'test' if w - window > val_end else 'purged'
                if split == 'purged':
                    continue
                # ponytail: graph and text encode the same 4-week history; add independent modalities only with timestamped data.
                nums = np.concatenate([past[:, 0], past[:, 1], past[:, 2]])
                text = f'Category: {c}. Last {window} weeks, oldest first. Units: ' + ','.join(f'{x:g}' for x in past[:, 0]) + '. Spend: ' + ','.join(f'{x:.2f}' for x in past[:, 1]) + '. Retail discount: ' + ','.join(f'{x:.2f}' for x in past[:, 2]) + '.'
                records.append(dict(id=f'{h}_{c}_{w}', household=h, category=c, week=int(w), history_start=int(w - window), history_end=int(w - 1), target_week=int(w), split=split, text=text, numbers=nums.tolist(), label=int(hist[w - start, 0] > 0)))
    return pd.DataFrame(records)


def prepare(root=DATA, config=None):
    import pyreadr
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = config or dict(households=200, categories=8, window=4, transfer_households=20)
    api = 'https://api.github.com/repos/bradleyboehmke/completejourney/commits/master'
    commit_file = root / 'source_revision.json'
    if not commit_file.exists():
        with urllib.request.urlopen(api, timeout=60) as r:
            save(commit_file, {'commit': json.load(r)['sha']})
    revision = json.loads(commit_file.read_text(encoding='utf-8'))['commit']
    manifest = dict(source='https://bradleyboehmke.github.io/completejourney/', revision=revision, license='CC0 per package documentation', config=config, files=[])
    objects = {}
    for name in ['transactions.rds', 'products.rda', 'campaigns.rda', 'coupon_redemptions.rda']:
        path = root / name
        url = f'https://raw.githubusercontent.com/bradleyboehmke/completejourney/{revision}/data/{name}'
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        manifest['files'].append(dict(name=name, url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size, accessed='2026-10-22'))
        result = pyreadr.read_r(str(path))
        objects[name] = next(iter(result.values()))
    tx, products = objects['transactions.rds'], objects['products.rda']
    save(root / 'schemas.json', {k:{'rows':len(v), 'columns':{c:str(t) for c,t in v.dtypes.items()}} for k,v in objects.items()})
    original = len(tx)
    missing = tx.isna().sum().to_dict()
    duplicates = int(tx.duplicated().sum())
    tx = tx.drop_duplicates().copy()
    tx['household'] = tx.household_id.astype(str)
    products['product_id'] = products.product_id.astype(str)
    tx['product_id'] = tx.product_id.astype(str)
    tx = tx.merge(products[['product_id', 'product_category']], on='product_id', how='left', validate='many_to_one').rename(columns={'product_category':'category'})
    bad = tx['category'].isna() | (tx.quantity <= 0) | (tx.sales_value < 0) | (tx.retail_disc < 0)
    tx.loc[bad].to_csv(root / 'excluded.csv', index=False)
    tx = tx.loc[~bad].copy()
    # Use source week only; undocumented civil timestamps/time zones are never invented.
    tx['week'] = tx.week.astype(int)
    cutoff = int(tx.week.min() + .6 * (tx.week.max() - tx.week.min()))
    train = tx[(tx.week <= cutoff) & ~tx.category.isin(['COUPON/MISC ITEMS', 'FUEL'])]
    cats = train.groupby('category').quantity.sum().nlargest(config['categories']).index.tolist()
    hs = sorted(train.household.unique(), key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    main = hs[:config['households']]
    transfer = hs[config['households']:config['households'] + config['transfer_households']]
    selected = tx[tx.household.isin(main + transfer) & tx.category.isin(cats)]
    selected.to_csv(root / 'transactions_selected.csv.gz', index=False)
    frame = panel(selected, main + transfer, cats, config['window'])
    frame['cohort'] = np.where(frame.household.isin(transfer), 'transfer', 'main')
    # Training statistics must exclude held-out households even in early weeks.
    frame = frame[~((frame.cohort == 'transfer') & (frame.split != 'test'))].copy()
    frame.to_json(root / 'panel.jsonl', orient='records', lines=True, force_ascii=False)
    stats = dict(raw_rows=original, duplicates=duplicates, missing={k:int(v) for k,v in missing.items()}, excluded=int(bad.sum()), category_exclusions=['COUPON/MISC ITEMS: accounting entries, not goods', 'FUEL: outside chosen e-commerce goods scope'], weeks=[int(tx.week.min()),int(tx.week.max())], categories=cats, selected_households=len(main), transfer_households=len(transfer), panel_rows=len(frame), by_split=frame.groupby(['cohort','split','label']).size().reset_index(name='n').to_dict('records'), sample_hash=hashlib.sha256((root / 'panel.jsonl').read_bytes()).hexdigest(), timezone='Source week index; no exchange calendar or invented timezone', discount_sign='Nonnegative in this processed Complete Journey version')
    save(root / 'audit.json', stats)
    save(root / 'manifest.json', manifest)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return frame


def load():
    return pd.read_json(DATA / 'panel.jsonl', lines=True)


if __name__ == '__main__':
    prepare()
