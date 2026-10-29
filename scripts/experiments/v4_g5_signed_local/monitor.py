"""Passive trace observer: no new parameters, detached summaries, unchanged loss."""
import torch
from torch import nn
from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions


class EpochMonitor:
    def __init__(self,scale,propagate):self.factor=scale*1000;self.propagate=propagate;self.rows={}

    @torch.no_grad()
    def add(self,x,trace,phase):
        if len(x['gs_xyz'])!=1:raise ValueError('Formal diagnostic observer expects batch1')
        field=trace['field'];valid=field.anchor_mask[0];count=valid.sum().clamp_min(1)
        far=torch.isin(field.anchor_indices[0],fixed_regions(x)['far']) & valid
        values=[];names=None
        for step,s in enumerate(trace['steps']):
            row={};hands=s['hand_valid'][0]
            d=s['direct_seed_per_hand'][0];p=s.get('propagated_seed_per_hand',s['direct_seed_per_hand'])[0]
            for j,hand in enumerate(('left','right')):
                row[hand+'_valid']=hands[j].float()
                for label,vector in (('direct',d[:,j]),('propagated',p[:,j]),('mix_change',p[:,j]-d[:,j])):
                    vv=vector[valid].detach()*self.factor
                    for axis,k in zip('xyz',range(3)):row[f'{hand}_{label}_{axis}_mm']=vv[:,k].mean()
                    row[f'{hand}_{label}_rms_mm']=vv.square().sum(-1).mean().sqrt()
                for axis,k in zip('xyz',range(3)):row[f'{hand}_g4_{axis}_mm']=s['hand_contribution'][0,j,k].detach()*self.factor
                beta=s['controller_cumulative_weights'][0,:,j].detach();observed=far & (beta.sum(-1)>0)
                w=beta.max(-1).values
                row[hand+'_far_beta_rows']=observed.sum().float()
                row[hand+'_far_weight_max']=torch.where(observed,w,0.).max()
                entropy=-(beta*beta.clamp_min(torch.finfo(beta.dtype).tiny).log()).sum(-1)
                row[hand+'_far_entropy']=torch.where(observed,entropy,0.).sum()/observed.sum().clamp_min(1)
                geometry=s['controller_geometry'][0,:,j].detach().abs().amax(-1)
                row[hand+'_far_geometry_abs_max']=torch.where(observed,geometry,0.).max()
                if self.propagate:
                    attention=s['anchor_attention_weights'][0,valid,j].detach()
                    row[hand+'_a2a_self']=attention[:,0].mean()
                    row[hand+'_a2a_entropy']=-(attention*attention.clamp_min(torch.finfo(attention.dtype).tiny).log()).sum(-1).mean()
            for axis,k in zip('xyz',range(3)):
                row['g4_global_'+axis+'_mm']=s['g4_global_delta'][0,k].detach()*self.factor
                row['local_gs_mean_'+axis+'_mm']=s['local_gs_mean'][0,k].detach()*self.factor
            row['local_gs_demeaned_rms_mm']=s['local_gs_demeaned_rms'][0].detach()*self.factor
            names=list(row);values.append(torch.stack([v.float() for v in row.values()]))
        packed=torch.stack(values).cpu()
        if not torch.isfinite(packed).all():raise FloatingPointError('Nonfinite G5 training trace')
        for h,vector in enumerate(packed.tolist(),1):
            row=dict(zip(names,vector));target=self.rows.setdefault((phase,h),dict(samples=0,sums={},maxima={},counts={}))
            target['samples']+=1
            for key,value in row.items():
                hand='left' if key.startswith('left_') else 'right' if key.startswith('right_') else None
                if hand and not row[hand+'_valid'] and not key.endswith('_valid'):continue
                if key.endswith('_max'):target['maxima'][key]=max(target['maxima'].get(key,0.),value)
                else:
                    target['sums'][key]=target['sums'].get(key,0.)+value
                    target['counts'][key]=target['counts'].get(key,0)+1

    def summary(self):
        result={}
        for (phase,h),row in sorted(self.rows.items()):
            result.setdefault(phase,{})[str(h)]=dict(samples=row['samples'],
                mean_batch_diagnostics={k:v/row['counts'][k] for k,v in row['sums'].items()},
                observed_maxima=row['maxima'],valid_observation_counts=row['counts'])
        return dict(phases=result,propagation_applicable=self.propagate,
                    note='Detached diagnostic batch means/maxima, never used by Loss or Best selection. Missing hands excluded. Far uses original initial GS quartiles.')


class ObservedPredictor(nn.Module):
    """Enable the existing trace path; return precisely the original predictions."""
    def __init__(self,base,monitor):super().__init__();self.base=base;self.monitor=monitor
    @property
    def config(self):return self.base.config
    @property
    def global_motion(self):return self.base.global_motion
    def forward(self,x):
        prediction,trace=self.base(x,return_trace=True)
        self.monitor.add(x,trace,'train' if self.training else 'val')
        return prediction
