"""Scoped, deterministic interpretations of already calculated evidence."""


def segment_readout(evidence):
    rows=evidence.get('segments',[])
    if not rows:return 'No usable baseline segment analysis was available. A targeted strategy is not assessed.'
    direction=1 if (evidence.get('posterior') or {}).get('directional_probabilities',{}).get('beneficial_direction','increase')=='increase' else -1
    positive=[r for r in rows if r.get('p_holm',1)<.05 and direction*r['effect']>0]
    negative=[r for r in rows if r.get('p_holm',1)<.05 and direction*r['effect']<0]
    name=lambda r:f'{r["segment"]}={r["level"]}'
    parts=[]
    if positive:parts.append('Exploratory improvements survive the subgroup correction in '+', '.join(name(r) for r in positive[:3])+'. These are candidates for strategically relevant follow-up, not validated targeted-launch recommendations.')
    if negative:parts.append('Exploratory deterioration survives correction in '+', '.join(name(r) for r in negative[:3])+'. Investigate product fit, instrumentation and safety before exposing those users further.')
    if not positive and not negative:parts.append('No subgroup passes the Holm multiple-comparison check across all explored groups. Positive points (and some individual 95% intervals above zero) can still appear: those intervals assess one group at a time, while Holm asks for stronger evidence when selecting findings from many groups. The effect estimates have not changed; their evidence is insufficient for that broader claim. This is not evidence of no effect.')
    different=[r['attribute'] for r in evidence.get('heterogeneity_tests',[]) if r.get('p_holm',1)<.05]
    if different:parts.append('The separate across-group comparison suggests differences for '+', '.join(different[:2])+', still exploratory.')
    else:parts.append('The separate corrected comparison does not establish different responses between groups.')
    parts.append('Uncertain groups are not proven nonresponders. Segment value, user pain and extra implementation cost need business context; no segmented-policy payoff is modeled here.')
    return ' '.join(parts)
