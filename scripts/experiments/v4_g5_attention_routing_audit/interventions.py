"""Use verified full-AR capture/replay, with Value-only routing hooks."""
from contextlib import contextmanager
from scripts.experiments.v4_g5_cross_attention_audit.interventions import (
    intervention as original_capture,replay_step as original_replay,one_frame,SELECTED)
from .routing import MODES,routing_blocks


def metadata(mode):
    if mode not in MODES:raise ValueError('Unknown routing mode')
    return dict(name=mode,value_route_only=True,original_scores_execute=True,
                motion12_mass4_use='alpha_learned_current_query',
                null_preserved=mode!='UniformRaw',
                hand_mass_preserved=mode in ('Normal',*MODES[3:]),
                null_value_zero=True)


@contextmanager
def intervention(model,mode='Normal',only_step=None):
    with original_capture(model,'Normal') as rows:
        with routing_blocks(model,mode,only_step) as routes:
            try:
                yield rows
            finally:
                for k,row in enumerate(rows):
                    row['routing_calls']=[r for r in routes if r['step']==k]
                    row['capture_is_normal']=mode=='Normal'
                    row['routing_capture_mode']=mode


def replay_step(model,inputs,normal_trace,captures,step,mode='Normal'):
    with routing_blocks(model,mode) as routes:
        result=original_replay(model,inputs,normal_trace,captures,step,'Normal')
    result['routing_calls']=routes
    return result
