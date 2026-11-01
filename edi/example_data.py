"""One coherent synthetic user-level export; no outcome-dependent seed selection."""
import numpy as np
import pandas as pd
from edi.engine import demo

FILENAME='experiment_users.csv'


def generate():
    # Original fixed-seed A case has p≈0.098: no conventional primary significance.
    df=demo('A',seed=42).rename(columns={'os':'platform','crashes_7d':'crash_guardrail'})
    rng=np.random.default_rng(108);n=len(df)
    df['unit_id']=[f'user_{i:06d}' for i in range(1,n+1)]
    assigned=pd.Timestamp('2026-09-01',tz='UTC')+pd.to_timedelta(rng.integers(0,21*86400,n),unit='s')
    df['assigned_at']=assigned.strftime('%Y-%m-%dT%H:%M:%SZ')
    df['window_end_at']=(assigned+pd.Timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
    df['observed_through']='2026-10-01T00:00:00Z'
    df['country']=rng.choice(['ES','PT','FR','DE'],n,p=[.4,.2,.25,.15])
    df['customer_type']=rng.choice(['new','returning'],n,p=[.35,.65])
    df['sessions_pre_7d']=rng.poisson(3,n)
    df['sessions_7d']=rng.poisson(4,n)+df.converted_7d
    df['revenue_7d']=np.round(df.converted_7d*rng.lognormal(3.4,1.4,n),2)
    # First-purchase timing profile: disjoint equal-duration windows, same users.
    allocation=rng.random(n)
    early_share=np.where(df.variant=='treatment',.65,.45)
    df['converted_early']=((df.converted_7d==1)&(allocation<early_share)).astype(int)
    df['converted_late']=((df.converted_7d==1)&(allocation>=early_share)&(allocation<.9)).astype(int)
    return df[['unit_id','variant','converted_7d','crash_guardrail','assigned_at',
               'window_end_at','observed_through','platform','country','customer_type','sessions_pre_7d',
               'sessions_7d','revenue_7d','converted_early','converted_late']]


DICTIONARY=pd.DataFrame([
 ('unit_id','Required','Unique randomized unit. unir_id is accepted as a spelling alias.'),
 ('variant','Required','Assigned arm. Map your two labels to control/treatment in the app.'),
 ('converted_7d','Example primary','Purchase within 7 days (0/1). Any numeric primary column can be selected instead.'),
 ('crash_guardrail','Optional safety metric','At least one crash (0/1). Every column ending _guardrail is detected in Results; specify its direction and tolerated harm.'),
 ('assigned_at','Optional UTC timestamp','Assignment time in UTC. Weekly charts compare users assigned in each Monday–Sunday calendar week; this is not a daily conversion-event log.'),
 ('window_end_at / observed_through','Optional UTC timestamps','End of outcome window and export cutoff; enable maturity checks.'),
 ('platform / country / customer_type','Optional pre-treatment splits','Segment overview, balance diagnostics and target-population scenarios.'),
 ('sessions_pre_7d','Optional pre-treatment numeric covariate','Balance diagnostic; not silently used for CUPED or segmentation.'),
 ('sessions_7d','Optional count / denominator','Session count. Can pair with revenue_7d as a ratio of sums.'),
 ('revenue_7d','Optional numeric metric / numerator','Heavy-tailed spend per assigned user. Enables numeric inference and p99 winsorized sensitivity when selected.'),
 ('converted_early / converted_late','Optional equal-window outcomes','First purchase in days 1–3 and 4–6, respectively. Measured on all the same assigned users; enables a paired time contrast.')
],columns=['Column','Role','Definition'])
