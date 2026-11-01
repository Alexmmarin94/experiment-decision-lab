"""Plain-language diagnostic meanings; computed statuses and facts remain authoritative."""


def check_title(name):
    return {'Sample ratio mismatch':'Do the assigned arm sizes match the plan?',
        'Missing unit IDs':'Does every user have an ID?','Repeated units':'Does each user appear only once?',
        'Exact repeated records':'Are any records repeated exactly?','Assignment labels':'Are the two assigned groups identified?',
        'Independent sample size':'Are there enough users in both groups?',
        'Pre-treatment covariate balance':'Were measured baseline attributes comparable?',
        'Window maturity':'Did every user have a complete outcome window?',
        'Exposure diagnostics':'Could we check delivery of the changed experience?',
        'Design declaration':'Has the owner confirmed the supported design?',
        'Supported design':'Does this analysis match the declared design?'}.get(name,name)


def check_explanation(check):
    name=check['check'];status=check['status']
    if status=='pass':
        if name.endswith('observed finite numbers'):return 'All values needed for this calculation are present and numeric.'
        return {'Pre-treatment covariate balance':'No large measured baseline imbalance was flagged; this does not prove correct randomization.',
            'Sample ratio mismatch':'The arm sizes do not show a flagged mismatch with the original allocation plan.',
            'Window maturity':'The declared outcome windows are complete at the export cutoff.',
            'Missing unit IDs':'Every row has a usable user ID.','Repeated units':'No user appears more than once or in both arms.',
            'Exact repeated records':'No identical duplicate rows were found.','Assignment labels':'Both expected arms are present.',
            'Independent sample size':'Each arm has enough assigned users for the supported procedures.'}.get(name,'No issue was flagged by this specific check.')
    if name=='Exposure diagnostics':return 'The supplied fields cannot establish comparable opportunities to see the change. Confirm delivery/logging with the owner while retaining all assigned users. Missing optional exposure information is not itself evidence of a logging defect.'
    if 'Exposure coverage' in name:return 'Some exposure records are missing, or coverage differs between arms/platforms. Confirm how exposure was recorded and whether delivery differed. Retain all assigned users rather than selecting only those who were exposed.'
    if 'tails' in name:return 'A small set of unusually large or skewed values could strongly influence the average. Check whether they are real high-value users or logging errors; compare the original and capped-result sensitivity.'
    if name=='Window maturity':
        return 'The declared cutoff precedes at least one window end, so some outcomes are incomplete. Obtain a complete export; do not quietly filter users to make the check pass.' if status=='block' else 'The required window/cutoff fields were not supplied, so completeness could not be verified. Ask the owner to confirm mature observation; missing evidence is not a detected incomplete window.'
    if name=='Pre-treatment covariate balance':
        return 'No usable baseline attributes were selected or supplied, so this comparison was not performed. That is not evidence of imbalance; these covariates are optional for the core analysis.' if status=='unavailable' else 'Some measured baseline differences deserve investigation. Check assignment/export and baseline missingness; do not search for covariates that make the effect significant.'
    if name=='Sample ratio mismatch':return 'The assignment split is inconsistent with the plan. Investigate assignment, exclusions and export filters before interpreting effects.'
    if status=='block':return 'This finding prevents a supported analysis. Resolve it in the export or design; AI cannot bypass the issue.'
    if status=='unavailable':return 'The supplied columns cannot support this check. Confirm it with the experiment owner; this is missing evidence, not a detected defect.'
    return 'Investigate this finding with the metric or experiment owner before making an irreversible decision.'


BRIEF_DISCLAIMER=('**Disclaimer · context beyond this model**\n\n'
    'This model cannot capture every secondary outcome or interaction among measured variables, '
    'an iterative release’s learning value, strategic bets, or priorities such as acquisition or margin over immediate profit. '
    'These can broaden the decision. Discuss them in **05 · Explore with AI**, grounded in the calculated results; '
    'untested interactions and future benefits remain assumptions, and data/safety constraints still apply.')
