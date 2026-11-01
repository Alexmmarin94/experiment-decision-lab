"""Readable visual explanations; calculations remain in the evidence engines."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def posterior_chart(relative_benefit,goal):
    values=np.asarray(relative_benefit)*100;n=len(values)
    low=float(values.min());high=float(values.max())
    width=max((high-low)/60,1e-9)
    fig=go.Figure()
    for label,mask,color in [('Could make things worse',values<0,'#E76F51'),
                              ('Helps, but below your goal',(values>=0)&(values<goal),'#94A3B8'),
                              ('Meets your improvement goal',values>=goal,'#0D9488')]:
        fig.add_trace(go.Histogram(x=values[mask],y=np.full(int(mask.sum()),100/n),histfunc='sum',
            xbins={'start':low-width/2,'end':high+width,'size':width},name=label,marker_color=color,
            hovertemplate='Relative improvement: %{x:.1f}%<br>Share of modeled possibilities: %{y:.1f}%<extra></extra>'))
    fig.add_vline(x=0,line_dash='dot',annotation_text='No improvement')
    fig.add_vline(x=goal,line_dash='dash',annotation_text='Your goal')
    fig.update_layout(barmode='stack',xaxis_title='Possible relative improvement (%)',yaxis_title='Share of modeled possibilities (%)',legend_title_text='Under the selected Bayesian model')
    return fig


def probability_chart(probabilities):
    labels=['Helps at all','Meets your improvement goal','Causes meaningful harm']
    keys=['probability_benefit','probability_useful_benefit','probability_material_harm']
    values=[probabilities[k]*100 for k in keys]
    fig=go.Figure(go.Bar(x=values,y=labels,orientation='h',text=[f'{v:.1f}%' for v in values],textposition='auto',marker_color=['#0D9488','#0F766E','#E76F51']))
    fig.update_layout(xaxis={'range':[0,100],'title':'Probability under this model (%)'},yaxis={'autorange':'reversed'},height=270)
    return fig


def pooling_chart(groups,reference,direction):
    fig=go.Figure();scale=100/max(abs(reference),1e-12)
    for i,row in enumerate(groups):
        old=direction*row['unpooled_effect']*scale;new=direction*row['pooled_effect']*scale
        lo,hi=sorted([direction*row['credible_low']*scale,direction*row['credible_high']*scale])
        fig.add_trace(go.Scatter(x=[old,new],y=[row['group']]*2,mode='lines',line={'color':'#CBD5E1','width':4},showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[lo,hi],y=[row['group']]*2,mode='lines',line={'color':'#0D9488','width':2},name='95% range after sharing information',showlegend=i==0))
        fig.add_trace(go.Scatter(x=[old],y=[row['group']],mode='markers',marker={'color':'#64748B','symbol':'circle-open','size':11},name='Estimate from this group alone',showlegend=i==0))
        fig.add_trace(go.Scatter(x=[new],y=[row['group']],mode='markers',marker={'color':'#0D9488','size':11},name='Estimate after sharing information',showlegend=i==0))
    fig.add_vline(x=0,line_dash='dash')
    fig.update_layout(xaxis_title='Improvement (% of the overall control reference)',yaxis={'autorange':'reversed'},height=max(320,len(groups)*65+130))
    return fig


def sensitivity_view(rows,business,opportunity,baseline):
    labels={'Contribution margin':'Value per extra outcome','Eligible population':'Total eligible population','Reach':'Share actually receiving the change',
            'Effect retention':'How long the improvement lasts','Implementation cost':'One-off launch cost','Maintenance per month':'Ongoing maintenance cost'}
    def inputs(row):
        lo,hi=row['low_input'],row['high_input'];name=row['assumption']
        if name=='Reach':return f'{lo:.0%} affected',f'{hi:.0%} affected'
        if name=='Effect retention':return f'{lo**(business.months-1):.0%} of initial benefit left at month {business.months}',f'{hi**(business.months-1):.0%} of initial benefit left at month {business.months}'
        if name=='Eligible population':return f'{lo:,.0f} eligible / month',f'{hi:,.0f} eligible / month'
        return f'€{lo:,.2f}',f'€{hi:,.2f}'
    fig=go.Figure();table=[]
    for row in rows:
        name=labels[row['assumption']];lo_input,hi_input=inputs(row)
        lo_value=row['low_expected_value']-opportunity;hi_value=row['high_expected_value']-opportunity
        color='#0D9488' if row['assumption']=='Reach' else '#64748B'
        fig.add_trace(go.Scatter(x=[lo_value,hi_value],y=[name]*2,mode='lines+markers',line={'width':7 if row['assumption']=='Reach' else 4,'color':color},marker={'size':11},showlegend=False,
            customdata=[lo_input,hi_input],hovertemplate='%{y}<br>Assumption: %{customdata}<br>Expected net value: €%{x:,.0f}<extra></extra>'))
        table.append({'Assumption':name,'Lower setting':lo_input,'Expected value at lower setting':f'€{lo_value:,.0f}',
                      'Higher setting':hi_input,'Expected value at higher setting':f'€{hi_value:,.0f}','Total swing (€)':row['swing']})
    fig.add_vline(x=0,line_dash='dash',line_color='#E76F51',annotation_text='Break-even')
    fig.add_vline(x=baseline,line_dash='dot',line_color='#0F172A',annotation_text='Current estimate',annotation_position='bottom right')
    fig.update_layout(xaxis_title='Expected net value (€): left of break-even loses money',yaxis={'autorange':'reversed'},height=470)
    return fig,pd.DataFrame(table)


def reach_threshold(mean_benefit,business,opportunity):
    benefit_at_full_reach=business.monthly_users*mean_benefit*business.margin*sum(business.persistence**m for m in range(business.months))
    if benefit_at_full_reach<=0:return None
    return (business.implementation+business.maintenance*business.months+opportunity)/benefit_at_full_reach


def observed_effect_chart(rows,kind,label='level',metric_name='Outcome'):
    """Plot original-scale effect intervals; never mislabel them as relative intervals."""
    data=pd.DataFrame(rows).copy();scale=100 if kind=='binary' else 1
    data['Change']=data.effect*scale
    data['Upper']=(data.ci_high-data.effect)*scale;data['Lower']=(data.effect-data.ci_low)*scale
    data['Control display']=data.control*scale;data['Treatment display']=data.treatment*scale
    unit='percentage points' if kind=='binary' else f'{metric_name} units'
    rate_unit='%' if kind=='binary' else f' {metric_name} units'
    fig=go.Figure()
    for attribute,part in data.groupby('segment',sort=False) if 'segment' in data else [('Estimated change',data)]:
        fig.add_trace(go.Scatter(x=part.Change,y=part[label],mode='markers',name='Estimate ◆ with 95% confidence interval',marker={'size':13,'color':'#0F172A','symbol':'diamond','line':{'color':'white','width':1}},showlegend=len(fig.data)==0,
            error_x={'type':'data','symmetric':False,'array':part.Upper,'arrayminus':part.Lower,'color':'#2563EB','thickness':3,'width':8},
            customdata=np.column_stack([part['Control display'],part['Treatment display'],part.n_control,part.n_treatment,part.ci_low*scale,part.ci_high*scale]),
            hovertemplate=f'Estimated change: %{{x:+.3f}} {unit}<br>95% confidence interval: %{{customdata[4]:+.3f}} to %{{customdata[5]:+.3f}} {unit}<br>Control: %{{customdata[0]:.2f}}{rate_unit}<br>Treatment: %{{customdata[1]:.2f}}{rate_unit}<br>Users: %{{customdata[2]:,.0f}} control / %{{customdata[3]:,.0f}} treatment<extra></extra>'))
    fig.add_vline(x=0,line_dash='dash',line_color='#E76F51',annotation_text='No change (zero)',line_width=2)
    fig.update_layout(title='Estimated change and 95% confidence interval',xaxis_title=f'Treatment minus control ({unit})',yaxis={'categoryorder':'array','categoryarray':list(data[label]),'autorange':'reversed','automargin':True},
        height=max(300,45*len(data)+100),margin={'l':15,'r':20,'t':45,'b':45},legend={'orientation':'h','y':-0.25})
    return fig


def weekly_effect_chart(rows,kind,metric_name='Outcome'):
    ordered=sorted(rows,key=lambda r:r['week_start'])
    figure=observed_effect_chart(ordered,kind,label='week_label',metric_name=metric_name)
    for trace in figure.data:
        trace.customdata=[[float(r['control'])*(100 if kind=='binary' else 1),float(r['treatment'])*(100 if kind=='binary' else 1),r['n_control'],r['n_treatment'],r['ci_low']*(100 if kind=='binary' else 1),r['ci_high']*(100 if kind=='binary' else 1),r['observed_assignment_start'],r['observed_assignment_end']] for r in ordered]
        trace.hovertemplate=trace.hovertemplate.replace('<extra>', '<br>Observed assignment dates: %{customdata[6]} to %{customdata[7]}<extra>')
    figure.update_layout(title='How did the overall effect vary by assignment week?',yaxis_title='Assignment week · earliest at top',height=max(340,len(rows)*65+130))
    return figure


def scenario_change(previous,scenario,result,workspace):
    """Describe a business-state change without rerunning or changing any inference."""
    if not previous or previous['workspace']!=workspace or previous['scenario']==scenario:return None
    old=previous['scenario'];labels={'monthly_users':'Eligible population','reach':'Affected share','margin':'Contribution per extra outcome','implementation':'Launch cost','maintenance':'Monthly maintenance',
        'months':'Comparison months','persistence':'Benefit durability','pilot_cost':'Gradual-launch cost','iteration_cost':'Iteration budget','opportunity_cost':'Displaced-project cost','include_gradual':'Consider gradual launch'}
    changes=[]
    for group in ['business','options','preferences','uncertainty']:
        for field,new in scenario[group].items():
            before=old[group].get(field)
            if before!=new:changes.append({'Assumption':labels.get(field,field.replace('_',' ').capitalize()),'Previous':str(before),'Current':str(new)})
    if not changes:return None
    old_global=next(r for r in previous['result']['actions'] if r['action']=='Global launch')
    new_global=next(r for r in result['actions'] if r['action']=='Global launch')
    return {'changes':changes,'previous_value':old_global['expected_net_value'],'current_value':new_global['expected_net_value'],
            'previous_loss_risk':old_global['loss_probability'],'current_loss_risk':new_global['loss_probability'],
            'previous_preference':previous['result']['assessment']['preferred_action'],'current_preference':result['assessment']['preferred_action']}


def financial_outcome_chart(values,months):
    values=np.asarray(values);width=max(float(values.max()-values.min())/70,1e-6)
    fig=go.Figure()
    for label,mask,color in [('Modeled loss',values<0,'#E76F51'),('Modeled gain',values>=0,'#0D9488')]:
        fig.add_trace(go.Histogram(x=values[mask],y=np.full(int(mask.sum()),100/len(values)),histfunc='sum',
            xbins={'start':float(values.min())-width/2,'end':float(values.max())+width,'size':width},name=label,marker_color=color,
            hovertemplate='Net value: €%{x:,.0f}<br>Share of modeled scenarios: %{y:.1f}%<extra></extra>'))
    fig.add_vline(x=0,line_dash='dash',annotation_text='Break-even: no net gain or loss')
    fig.update_layout(title=f'Possible global-launch value over {months} months',barmode='stack',xaxis_title='Incremental net value after costs (€)',yaxis_title='Share of simulated scenarios (%)',legend={'orientation':'h','y':-0.25})
    return fig
