"""Regenerate the single example: python -m scripts.generate_examples."""
from pathlib import Path
from edi.example_data import generate,FILENAME
if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'data'/FILENAME
    path.parent.mkdir(exist_ok=True)
    df=generate();df.to_csv(path,index=False)
    print(f'{path.name}: {len(df):,} individual rows')
