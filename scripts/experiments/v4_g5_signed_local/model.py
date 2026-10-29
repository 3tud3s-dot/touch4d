"""One G5 rollout shared by Direct and Propagate; original G4 is untouched."""
import torch
from scripts.training.model import initial_knn,masked
from scripts.training.v4_model import fixed_field
from scripts.training.gs_decoder import pool_anchor_features,interpolate_anchor_displacements
from scripts.experiments.v4_g4_global_motion.model import G4GlobalMotionPredictor
from .motion import AnchorSignedDirect
from .propagation import build_anchor_graph,SignedAnchorPropagation


class _G5Predictor(G4GlobalMotionPredictor):
    def __init__(self,*,propagate,horizon=15,prediction_interval=1):
        super().__init__(horizon=horizon,prediction_interval=prediction_interval)
        # Append only after G4; Propagate appends AFTER identical Direct init.
        self.direct_motion=AnchorSignedDirect(width=128)
        self.use_propagation=propagate
        mode='propagate' if propagate else 'direct'
        self.config=dict(self.config,architecture=f'v4_g5_signed_local_{mode}_v1',
            local_motion='per_anchor_per_hand_18d_v1',local_zero_mean=False,
            local_mapping_width=128,local_bandwidth_multiplier=1.,
            anchor_propagation=propagate,anchor_graph_other_neighbors=8 if propagate else 0,
            anchor_graph_self=propagate,propagation_layers=1 if propagate else 0,
            propagation_lambda=.5 if propagate else 0.,checkpoint_kind=f'g5_{mode}_v1')
        if propagate:self.propagation=SignedAnchorPropagation(width=128)

    def forward(self,x,*,return_trace=False):
        h=self.config['horizon'];frame_ids=x.get('condition_frame_ids')
        if (frame_ids is None or frame_ids.dtype!=torch.int64 or frame_ids.shape!=(len(x['gs_xyz']),h)
                or not torch.all(frame_ids[:,1:]-frame_ids[:,:-1]==self.config['prediction_interval'])):
            raise ValueError('G5 condition_frame_ids must match configured H/interval')
        if any(x[key].shape[1]!=h for key in ('action_state','controller_points','tactile')):
            raise ValueError('G5 requires exactly H sampled conditions')
        keep=x['gs_mask']
        if keep.dtype!=torch.bool or not keep.any(1).all():raise ValueError('Each sample requires valid GS')
        initial,attr=masked(x['gs_xyz'],keep),masked(x['gs_attr'],keep)
        if not torch.isfinite(initial).all() or not torch.isfinite(attr).all():raise ValueError('Nonfinite initial GS')
        field=fixed_field(initial,keep);neighbors=initial_knn(initial,keep,16)
        batches=torch.arange(len(initial),device=initial.device)[:,None]
        initial_anchors=masked(initial[batches,field.anchor_indices],field.anchor_mask)
        if self.use_propagation:graph_indices,graph_valid=build_anchor_graph(initial_anchors,field.anchor_mask)
        current=initial;previous_anchor=initial_anchors;previous_d=torch.zeros_like(initial_anchors)
        previous_cp=previous_valid=None;states=[];steps=[]
        for step in range(h):
            features=masked(self.gs_encoder(torch.cat((current,attr),-1)),keep)
            features=self.local(features,current,keep,neighbors)
            query=masked(pool_anchor_features(features,field)+self.pred_time.weight[step],field.anchor_mask)
            anchors=masked(current[batches,field.anchor_indices],field.anchor_mask)
            tools,valid,cp,condition_trace=self.conditions(x,step)
            for block in self.cross:
                query,alpha=block(query,tools,anchors,cp,field.bandwidth,field.anchor_mask,valid)
            adjacent=torch.zeros_like(valid) if step==0 else valid&previous_valid
            cp_delta=torch.zeros_like(cp) if step==0 else masked(cp-previous_cp,adjacent)
            weights=alpha[...,:60]
            tool_motion=torch.einsum('bahj,bjd->bahd',weights,cp_delta).flatten(-2)
            history=torch.zeros_like(anchors) if step==0 else anchors-previous_anchor
            head_mass=(weights*adjacent[:,None,None]).sum(-1)
            anchor_history_valid=field.anchor_mask[...,None].to(query.dtype)*(step>0)
            motion_input=torch.cat((query,tool_motion,history,previous_d,head_mass,anchor_history_valid),-1)
            base_d=masked(self.motion(motion_input),field.anchor_mask)
            global_delta,global_trace=self.global_motion(x,step,query=query,anchor_valid=field.anchor_mask,
                hand_tokens=condition_trace['tokens62'][:,:2],cp=cp,cp_valid=valid,
                previous_cp=previous_cp,previous_valid=previous_valid)
            seed,direct_trace=self.direct_motion(x,step,query=query,anchor_xyz=anchors,
                anchor_valid=field.anchor_mask,hand_tokens=condition_trace['tokens62'][:,:2],
                hand_basis12=global_trace['hand_basis'],hand_valid=global_trace['hand_valid'],cp=cp,cp_valid=valid,
                previous_cp=previous_cp,previous_valid=previous_valid,anchor_history=history,previous_d=previous_d,
                global_hand_contribution=global_trace['hand_contribution'],sigma=field.bandwidth)
            if self.use_propagation:
                propagated,anchor_weights=self.propagation(query,condition_trace['tokens62'][:,:2],
                    direct_trace['anchor_hand_basis18'],seed,initial_anchors,anchors,field.anchor_mask,
                    global_trace['hand_valid'],graph_indices,graph_valid)
                local=masked(propagated.sum(2),field.anchor_mask)
            else:local=masked(seed.sum(2),field.anchor_mask)
            # Exact original G4 addition first; local0 is an identity, never centered.
            g4_d=masked(base_d+global_delta[:,None],field.anchor_mask)
            d=masked(g4_d+local,field.anchor_mask)
            next_xyz=masked(initial+interpolate_anchor_displacements(d,field,keep),keep)
            if not torch.isfinite(next_xyz).all():raise FloatingPointError('Nonfinite G5 prediction')
            states.append(next_xyz)
            if return_trace:
                local_gs=interpolate_anchor_displacements(local,field,keep)
                count=keep.sum(1,keepdim=True).clamp_min(1).to(local_gs.dtype)
                mean=masked(local_gs,keep).sum(1)/count
                residual=masked(local_gs-mean[:,None],keep)
                rms=torch.linalg.vector_norm(residual,dim=(1,2))/count.squeeze(-1).sqrt()
                trace=dict(alpha=alpha,anchor_xyz=anchors,anchor_history=history,tool_motion=tool_motion,
                    adjacent_valid=adjacent,head_history_mass=head_mass,base_cumulative_d=base_d,
                    global_delta=global_delta,cumulative_d=d,previous_d=previous_d,state=next_xyz,
                    g4_global_delta=global_delta,hand_basis12=global_trace['hand_basis'],
                    direct_seed_per_hand=seed,local_correction=local,
                    final_cumulative_anchor_displacement=d,local_gs_mean=mean,local_gs_demeaned_rms=rms,
                    **global_trace,**condition_trace,**direct_trace)
                if self.use_propagation:
                    trace.update(propagated_seed_per_hand=propagated,anchor_attention_weights=anchor_weights,
                                 anchor_neighbor_indices=graph_indices,anchor_neighbor_mask=graph_valid)
                steps.append(trace)
            previous_anchor,previous_d=anchors,d;previous_cp,previous_valid=cp,valid
            current=next_xyz  # Full H-step graph, no detach or target state replacement.
        prediction={'xyz':torch.stack(states,1)}
        for key,begin,end in (('log_scales',0,3),('quaternions',3,7),('opacity_logits',7,8),('sh_dc',8,11)):
            prediction[key]=attr[:,None,:,begin:end].expand(-1,h,-1,-1)
        trace={'steps':steps,'field':field}
        if self.use_propagation:trace.update(anchor_neighbor_indices=graph_indices,anchor_neighbor_mask=graph_valid)
        return (prediction,trace) if return_trace else prediction


class G5DirectPredictor(_G5Predictor):
    def __init__(self,horizon=15,prediction_interval=1):
        super().__init__(propagate=False,horizon=horizon,prediction_interval=prediction_interval)


class G5PropagatePredictor(_G5Predictor):
    def __init__(self,horizon=15,prediction_interval=1):
        super().__init__(propagate=True,horizon=horizon,prediction_interval=prediction_interval)
