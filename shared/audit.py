"""Additional descriptive statistics and exact/near profile leakage audit."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from shared.data import load,DATA
from shared.metrics import save

frame=load()
assert not frame.id.duplicated().any()
for earlier,later in [('train','validation'),('validation','test')]:
    assert frame[frame.split==earlier].target_week.max()<frame[frame.split==later].history_start.min()
numbers=pd.DataFrame(np.stack(frame.numbers),columns=[f'{kind}_lag{lag}' for kind in ['units','spend','discount'] for lag in [4,3,2,1]])
numbers.describe(percentiles=[.05,.5,.95]).to_csv(DATA/'feature_describe.csv')
frame.assign(text_length=frame.text.str.len()).groupby(['split','label']).text_length.describe().to_csv(DATA/'text_describe.csv')
groups=frame.groupby('text').agg(n=('id','size'),parts=('split','nunique'),households=('household','nunique'))
tr=frame[frame.split=='train'];te=frame[(frame.split=='test')&(frame.cohort=='main')].sample(500,random_state=11)
vector=TfidfVectorizer(max_features=128,ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b',dtype=np.float32)
train=vector.fit_transform(tr.text);test=vector.transform(te.text)
similarity=(test@train.T).toarray()
idx=similarity.argmax(1);score=similarity[np.arange(len(te)),idx]
near=pd.DataFrame(dict(test_id=te.id.to_numpy(),train_id=tr.iloc[idx].id.to_numpy(),cosine=score,same_household=te.household.to_numpy()==tr.iloc[idx].household.to_numpy()))
near.to_csv(DATA/'near_duplicates.csv',index=False)
save(DATA/'profile_audit.json',dict(unique_texts=len(groups),repeated_text_groups=int((groups.n>1).sum()),exact_profiles_crossing_parts=int((groups.parts>1).sum()),near_test_sample=len(te),near_threshold=.97,near_matches=int((score>=.97).sum()),near_same_household=int(((score>=.97)&near.same_household).sum()),interpretation='Repeated zero/similar histories can belong to independent households. No future event appears in earlier features; overlap alone is not proof of target leakage.',limitation='Near-duplicate check covers a fixed 500-object test sample, not every possible pair.'))
print('PASS audit: unique object ids, purged temporal boundaries; descriptive statistics and 500 nearest-profile checks saved')
