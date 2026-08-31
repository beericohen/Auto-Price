"""Export the trained XGBoost model and preprocessing metadata for GitHub Pages."""
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'site'/'assets'; OUT.mkdir(parents=True,exist_ok=True)
model=joblib.load(ROOT/'Models'/'XGBoost_fine.pkl')
df=pd.read_csv(ROOT/'data'/'preprocessing.csv')
booster=model.get_booster(); features=list(booster.feature_names)

def categories(prefix):
    return sorted(c[len(prefix)+1:] for c in df.columns if c.startswith(prefix+'_'))

def related(prefix, parent_prefix, parent_value):
    parent_col=f'{parent_prefix}_{parent_value}'
    rows=df[df[parent_col]==1]
    return sorted(c[len(prefix)+1:] for c in df.columns if c.startswith(prefix+'_') and rows[c].sum()>0)

cats={p:categories(p) for p in ['manufacturer','model','submodel','fuel','transmission','drive_type']}
meta={'feature_index':{n:i for i,n in enumerate(features)},'categories':cats,
      'manufacturer_models':{m:related('model','manufacturer',m) for m in cats['manufacturer']},
      'model_submodels':{m:related('submodel','model',m) for m in cats['model']}}
fs=joblib.load(ROOT/'data'/'scaler.pkl'); ps=joblib.load(ROOT/'data'/'minmax_scaler.pkl')
meta['scaler']={'mean':fs.mean_.tolist(),'scale':fs.scale_.tolist()}
meta['price_scaler']={'min':float(ps.min_[0]),'scale':float(ps.scale_[0])}
trees=[json.loads(t) for t in booster.get_dump(dump_format='json')]
model_json={'feature_names':features,'feature_index':{n:i for i,n in enumerate(features)},'base_score':float(booster.base_score),'trees':trees}
(OUT/'model.json').write_text(json.dumps(model_json,separators=(',',':')))
(OUT/'metadata.json').write_text(json.dumps(meta,separators=(',',':')))
print(f'Exported {len(trees)} trees and {len(features)} features.')
