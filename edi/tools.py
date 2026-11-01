"""A tiny, allowlisted tool surface. Proposals and previews never mutate a scenario."""
import json
from edi.scenarios import proposed_scenario,fingerprint,run
from edi.ui import clean

PATCH_SCHEMA={'type':'object','properties':{key:{'type':'number'} for key in [
    'monthly_users','reach','margin','implementation','maintenance','annual_maintenance_eur','months','persistence','margin_uncertainty',
    'pilot_fraction','pilot_cost','iteration_cost','opportunity_cost']},'additionalProperties':False}
PATCH_SCHEMA['properties']['global_reversible']={'type':'boolean'}
TOOL_DEFINITIONS=[{'type':'function','function':{'name':'get_decision_summary','description':'Read the authoritative current decision and action numbers.','parameters':{'type':'object','properties':{},'additionalProperties':False}}},
 {'type':'function','function':{'name':'get_decision_drivers','description':'Read computed binding constraints, feasible alternatives, payoff trade-offs, bounded counterfactuals and their numerical limits. Does not change any active state.','parameters':{'type':'object','properties':{},'additionalProperties':False}}},
 {'type':'function','function':{'name':'inspect_quality_check','description':'Read one computed check, without changing its status.','parameters':{'type':'object','properties':{'check':{'type':'string'}},'required':['check'],'additionalProperties':False}}},
 *[{'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':{'changes':PATCH_SCHEMA,'reason':{'type':'string'}},'required':['changes','reason'],'additionalProperties':False}}}
   for name,description in [('compare_scenario','Calculate a read-only counterfactual with permitted business inputs. Does not change active values.'),
                            ('propose_scenario_update','Preview a permitted business change and prepare an explicit user confirmation. No active changes until Confirm and recalculate is clicked. Annual maintenance is converted to monthly by the engine; do not estimate an unpriced two-week project delay yourself.')]]]


class ToolSession:
    def __init__(self,evidence,effects,read_only=False):
        self.evidence=evidence;self.effects=effects;self.proposal=None;self.events=[];self.read_only=read_only
        self._drivers=evidence.get('decision_drivers')
    @property
    def definitions(self):
        return [d for d in TOOL_DEFINITIONS if not self.read_only or d['function']['name']!='propose_scenario_update']
    def execute(self,name,args):
        try:
            if not isinstance(args,dict):raise ValueError('Arguments must be an object.')
            if name=='get_decision_summary':
                if args:raise ValueError('No arguments expected.')
                result={k:self.evidence.get(k) for k in ['assessment','actions','primary','posterior','evsi']}
            elif name=='get_decision_drivers':
                if args:raise ValueError('No arguments expected.')
                if self._drivers is None:
                    from edi.decision_drivers import build_decision_drivers
                    self._drivers=build_decision_drivers(self.evidence,self.effects)
                result=self._drivers
            elif name=='inspect_quality_check':
                if set(args)!={'check'}:raise ValueError('Supply only check.')
                result=[c for c in self.evidence['quality']['checks'] if c['check']==args['check']]
                if not result:raise ValueError('Unknown check name.')
            elif name in ('compare_scenario','propose_scenario_update'):
                if self.read_only and name=='propose_scenario_update':raise ValueError('The strategic review is read-only. Request a business update in the follow-up conversation.')
                if set(args)!={'changes','reason'} or not isinstance(args['reason'],str):raise ValueError('Supply changes and a text reason.')
                if self.effects is None:raise ValueError('Scenario recalculation is blocked by data validity.')
                if 'include_gradual' in args['changes']:raise ValueError('Choose which alternatives to consider manually in Model decisions.')
                new,changes=proposed_scenario(self.evidence['scenario'],args['changes'],args['reason'])
                preview,_,_=run(self.effects,new,self.evidence['quality'],self.evidence['guardrails'])
                result={'active_scenario_unchanged':True,'normalized_changes':changes,'preview':preview,'reason':args['reason']}
                if name=='propose_scenario_update':
                    self.proposal={'base_fingerprint':fingerprint(self.evidence['scenario']),'evidence_hash':fingerprint(self.evidence),
                                   'changes':changes,'reason':args['reason'],'scenario':new,'preview':preview}
                    result['requires_explicit_confirmation']=True
            else:raise ValueError('Unknown tool; only the listed functions are permitted.')
            result=clean(result)
        except (ValueError,TypeError,KeyError) as error:result={'error':str(error),'active_scenario_unchanged':True}
        self.events.append({'tool':name,'arguments':args,'result':result})
        return result


def confirm_proposal(current,proposal,current_evidence):
    if proposal['base_fingerprint']!=fingerprint(current) or proposal['evidence_hash']!=fingerprint(current_evidence):
        raise ValueError('The evidence or scenario changed after this proposal. Request a fresh preview.')
    # Revalidate the allowlist at confirmation; do not trust an arbitrary proposed scenario object.
    updated,changes=proposed_scenario(current,proposal['changes'],proposal['reason'])
    return updated,{'type':'confirmed_business_update','before':fingerprint(current),'after':fingerprint(updated),
                    'changes':changes,'reason':proposal['reason'],'protected_evidence_unchanged':True,
                    'before_scenario':current,'after_scenario':updated,
                    'before_assessment':current_evidence.get('assessment'),'before_actions':current_evidence.get('actions')}
