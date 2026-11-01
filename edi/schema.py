"""CSV parsing and explicit column mapping, separate from analytical validity checks."""
from io import BytesIO,StringIO
import csv
import hashlib
import numpy as np
import pandas as pd

METADATA={'unit_id','variant','assigned_at','exposed_at','window_end_at','observed_through'}


def parse_csv(contents):
    """Preserve ID spelling until the user identifies which columns carry units/arms."""
    try:
        header=next(csv.reader(StringIO(contents.decode('utf-8-sig'))))
        if len(set(header))!=len(header):raise ValueError('Column names must be unique; rename the repeated headers before uploading.')
        frame=pd.read_csv(BytesIO(contents),dtype=str,encoding='utf-8-sig')
    except (pd.errors.ParserError,pd.errors.EmptyDataError,UnicodeError,StopIteration) as error:
        raise ValueError('Could not read the file. Export a UTF-8 CSV with comma separators and a header row; check that every row has the same number of fields.') from error
    if not len(frame.columns):raise ValueError('The CSV has no columns. Add a header and assigned-user records.')
    if len(frame.columns)==1 and any(mark in str(frame.columns[0]) for mark in [';','\t']):
        raise ValueError('The file appears to use semicolons or tabs. Export it with comma separators so the columns can be read separately.')
    if not len(frame):raise ValueError('The file has column headers but no user rows. Export the assigned users and their measured outcomes.')
    if frame.columns.duplicated().any():raise ValueError('Column names must be unique. Give each outcome and identifier its own name.')
    return frame


def load_upload(upload):
    if upload is None:return None,None
    data=upload.getvalue()
    return parse_csv(data),hashlib.sha256(data).hexdigest()


def suggested_column(columns,role):
    names=list(columns)
    candidates=['unit_id','unir_id','user_id','customer_id','participant_id'] if role=='unit' else ['variant','assignment','arm','experiment_group','group']
    return next((c for c in candidates if c in names),None)


def normalize_schema(frame,unit_column,assignment_column,control_label=None):
    """Return a new normalized frame; never repair IDs, remove rows or fill outcomes."""
    if unit_column not in frame or assignment_column not in frame:raise ValueError('Choose an existing user-ID column and an existing assignment column.')
    if unit_column==assignment_column:raise ValueError('User ID and assignment must be different columns. IDs identify people; assignment identifies their experimental arm.')
    renames={unit_column:'unit_id',assignment_column:'variant'}
    for source,target in renames.items():
        if source!=target and target in frame and target not in renames:
            raise ValueError(f'The file already has a {target} column. Choose that column, or rename the conflicting field in the export before mapping it.')
    data=frame.rename(columns=renames).copy()
    if data.columns.duplicated().any():raise ValueError('These mappings create duplicate column names. Use distinct identifier, assignment and outcome columns.')
    data['unit_id']=data.unit_id.map(lambda value:str(value) if pd.notna(value) else None)
    arms=list(data.variant.dropna().unique())
    mapping={}
    if control_label is not None:
        if control_label not in arms or len(arms)!=2:raise ValueError('The selected control label must belong to a file with exactly two assigned arms.')
        treatment=next(a for a in arms if a!=control_label)
        mapping={str(control_label):'control',str(treatment):'treatment'}
        data['variant']=data.variant.map({control_label:'control',treatment:'treatment'})
    elif set(arms)=={'control','treatment'}:mapping={'control':'control','treatment':'treatment'}
    for name in data:
        if name in METADATA or name.endswith('_id') or name in {'platform','country','os','customer_type','device','segment'}:continue
        converted=pd.to_numeric(data[name],errors='coerce')
        observed=data[name].notna()
        if observed.any() and converted[observed].notna().all():data[name]=converted
    return data,{'unit_column':str(unit_column),'assignment_column':str(assignment_column),'arm_mapping':mapping}


def outcome_candidates(frame):
    return [c for c in frame if c not in METADATA and not c.endswith(('_guardrail','_id')) and pd.to_numeric(frame[c],errors='coerce').notna().any()]


def setup_issues(frame,metric=None,denominator=None):
    errors=[]
    for column in ['unit_id','variant']:
        if column not in frame:errors.append(f'Missing {column}: map the corresponding column or add it to the export.')
    if errors:return errors
    if frame.unit_id.isna().any() or frame.unit_id.astype(str).str.strip().eq('').any():errors.append('Some user IDs are missing. Export a real unique identifier for every assigned user; do not replace it with generated row numbers.')
    arms=set(frame.variant.dropna())
    if arms!={'control','treatment'} or frame.variant.isna().any():errors.append('Assignments must contain exactly two arms with no missing labels. Choose the control label and retain both arms.')
    if metric is not None:
        for name in [metric]+([denominator] if denominator else []):
            if name not in frame:errors.append(f'The selected outcome field {name} is missing.')
            else:
                bad=~np.isfinite(pd.to_numeric(frame[name],errors='coerce'))
                if bad.any():errors.append(f'{name}: {int(bad.sum()):,} values are missing, nonnumeric or infinite. Correct the measurement/export; replace with zero only when zero was actually observed.')
    return errors


def minimal_template():
    return 'unit_id,variant,converted_7d\nuser_001,control,0\nuser_002,treatment,1\nuser_003,control,1\nuser_004,treatment,0\n'
