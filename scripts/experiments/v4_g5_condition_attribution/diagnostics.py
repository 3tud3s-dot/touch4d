"""K/V and Query diagnostics; dimensionless feature RMS is not world mm."""
import torch
from scripts.experiments.v4_g5_cross_attention_audit.diagnostics import feature_rms,anchor_regions
from scripts.experiments.v4_g5_attention_routing_audit.diagnostics import entropy


def cross_rows(inputs,field,calls,normal_calls,scope,mode,history,horizon):
    rows=[];regions=anchor_regions(inputs,field)
    for call,normal in zip(calls,normal_calls):
        alpha=call['alpha_used'][0];reference=normal['alpha_used'][0]
        for region,ids in regions.items():
            if not len(ids):continue
            for head in range(4):
                valid=call['inputs']['tool_valid'][0]
                k=(call['k']-normal['k'])[0].reshape(60,4,32)[valid,head]
                v=(call['v']-normal['v'])[0].reshape(60,4,32)[valid,head]
                row=dict(**scope,mode=mode,history=history,horizon=horizon,layer=call['layer']+1,head=head,region=region,
                    valid_controllers=int(valid.sum()),key_change_feature_rms=feature_rms(k),value_change_feature_rms=feature_rms(v),
                    null_mass=float(alpha[ids,head,-1].mean()),left_mass=float(alpha[ids,head,:30].sum(-1).mean()),
                    right_mass=float(alpha[ids,head,30:60].sum(-1).mean()),
                    entropy_nats=float(entropy(alpha[ids,head]).mean()),
                    attention_total_variation=float(.5*(alpha-reference)[ids,head].abs().sum(-1).mean()),
                    query_change_feature_rms=feature_rms((call['query_after']-normal['query_after'])[0,ids]),
                    aggregate_change_feature_rms=feature_rms((call['aggregate']-normal['aggregate'])[0,ids].reshape(len(ids),4,32)[:,head]),
                    projected_change_feature_rms_joint_heads=feature_rms((call['projected_update']-normal['projected_update'])[0,ids]))
                rows.append(row)
    return rows


def assert_preserved_inputs(raw,normal,mode):
    """Only compare same-history current step, never assert unchanged AR Query."""
    a,b=raw['condition_content'],normal['condition_content']
    for key in ('cp','valid','hand_tokens'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    if mode in ('DirectHandTokenOff','LocalController6Off','SignedHand12Off'):
        torch.testing.assert_close(a['tools'],b['tools'],rtol=0,atol=0)
    # Same x current and initial endpoints must survive all feature interventions.
    ga=raw['direct_intervention']['global_kwargs_original'];gb=normal['direct_intervention']['global_kwargs_original']
    for key in ('cp','cp_valid','previous_cp','previous_valid'):
        if ga[key] is not None:torch.testing.assert_close(ga[key],gb[key],rtol=0,atol=0)
