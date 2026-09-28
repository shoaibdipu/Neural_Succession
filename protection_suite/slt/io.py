from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml

def load_yaml(path):
    with open(path) as f: return yaml.safe_load(f)

def append_csv(row,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame([row]); df.to_csv(p,mode='a',header=not p.exists(),index=False)
