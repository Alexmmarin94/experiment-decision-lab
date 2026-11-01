"""Small presentation helpers shared by the guided tabs."""
import json
import numpy as np
import pandas as pd
import plotly.express as px


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def dumps(value):return json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)


def forest(rows,label='level',binary=False):
    data=pd.DataFrame(rows).copy();scale=100 if binary else 1
    data['Effect']=data.effect*scale
    data['Upper error']=(data.ci_high-data.effect)*scale
    data['Lower error']=(data.effect-data.ci_low)*scale
    fig=px.scatter(data,x='Effect',y=label,error_x='Upper error',error_x_minus='Lower error',
                   color='segment' if 'segment' in data else None,hover_data=['n_control','n_treatment'],
                   labels={'Effect':'Treatment − control (pp)' if binary else 'Treatment − control (metric units)'})
    fig.add_vline(x=0,line_dash='dash',line_color='#64748B')
    fig.update_layout(height=max(300,45*len(data)+100),yaxis={'autorange':'reversed'})
    return fig


def effects_frame(rows,binary=False):
    scale=100 if binary else 1
    return pd.DataFrame([{'Group':r.get('level',r['metric']),'Control n':r['n_control'],'Treatment n':r['n_treatment'],
        'Control':r['control']*scale,'Treatment':r['treatment']*scale,'Effect':r['effect']*scale,
        'CI lower':r['ci_low']*scale,'CI upper':r['ci_high']*scale,'Raw p':r['p_value'],
        **({'Holm p':r['p_holm'],'BH p':r['p_bh']} if 'p_bh' in r else {})} for r in rows])
