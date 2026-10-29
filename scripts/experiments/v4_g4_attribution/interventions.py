"""Output-only interventions; all original controller/attention paths stay intact."""
from contextlib import contextmanager
import torch
from scripts.training.model import masked
from scripts.training.gs_decoder import interpolate_anchor_displacements

MODES=('Normal','NoGlobal','NoLeft','NoRight')


def effective_hands(hands,mode):
    if mode not in MODES:raise ValueError(mode)
    if mode=='Normal':return hands
    value=hands.clone()
    if mode=='NoGlobal':value.zero_()
    elif mode=='NoLeft':value[:,0]=0
    else:value[:,1]=0
    return value


@contextmanager
def intervention(model,mode,only_step=None):
    if mode not in MODES:raise ValueError(mode)
    def hook(module,args,output):
        step=int(args[1]);delta,trace=output
        if mode=='Normal' or (only_step is not None and step!=only_step):return output
        hands=effective_hands(trace['hand_contribution'],mode)
        return hands.sum(1),dict(trace,hand_contribution=hands,
                                raw_hand_contribution=trace['hand_contribution'],raw_global_delta=delta)
    handle=model.global_motion.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def direct_positions(inputs,normal_trace,step,mode):
    """Same observed Normal history, same base_d, replace ONLY this step's shift."""
    s=normal_trace['steps'][step];field=normal_trace['field'];keep=inputs['gs_mask']
    hands=effective_hands(s['hand_contribution'],mode)
    d=masked(s['base_cumulative_d']+hands.sum(1)[:,None],field.anchor_mask)
    return masked(inputs['gs_xyz']+interpolate_anchor_displacements(d,field,keep),keep)


def model_digest(model):
    import hashlib
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
