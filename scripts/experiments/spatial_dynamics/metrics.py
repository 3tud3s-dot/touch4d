"""World-millimetre, independent-set diagnostics; never GT row correspondence."""
import torch
from scripts.training.losses import nearest_distances
from scripts.training.model import initial_knn
from scripts.experiments.fixed_window.fixed_fit import prediction_metrics


class CaptureRenderer:
    def __init__(self,renderer):self.renderer=renderer
    def __call__(self,*args):
        self.pred=self.renderer(*args);return self.pred
    def render_targets(self,*args):
        self.gt=self.renderer.render_targets(*args);return self.gt


@torch.no_grad()
def enriched_metrics(pred,batch,processor,renderer,trace=None):
    capture=CaptureRenderer(renderer)
    result=prediction_metrics(pred,batch,processor,capture,512)
    x=batch['inputs'];keep=x['gs_mask'][0];initial=x['gs_xyz'][0,keep]
    units=processor.state_dict()['scale']*1000
    neighbors,edge_valid=initial_knn(initial[None],torch.ones(1,len(initial),device=initial.device,dtype=torch.bool),16)
    neighbors=neighbors[0];edge_valid=edge_valid[0] & (neighbors!=torch.arange(len(initial),device=initial.device)[:,None])
    d0=(initial[:,None]-initial[neighbors]).norm(dim=-1)
    rows=[]
    previous_ids=None
    for k in range(pred['xyz'].shape[1]):
        p=pred['xyz'][0,k,keep]
        t=batch['targets']['target_xyz'][k][0,batch['targets']['target_mask'][k][0]]
        centered_p=p-p.mean(0);centered_t=t-t.mean(0)
        pc=nearest_distances(centered_p,centered_t);tc=nearest_distances(centered_t,centered_p)
        pcd=nearest_distances(p,t);tcd=nearest_distances(t,p)
        tools=x['controller_points'][0,k].reshape(60,3)
        valid=x['controller_mask'][0,k].reshape(60) & x['operation_hands'][0].repeat_interleave(30)
        if valid.any():
            pd=torch.cdist(p,tools[valid]).amin(-1);td=torch.cdist(t,tools[valid]).amin(-1)
        else:pd=p.new_full((len(p),),float('inf'));td=t.new_full((len(t),),float('inf'))
        region={}
        for name,pm,tm in [('near',pd<=.1,td<=.1),('far',pd>.1,td>.1)]:
            region[name+'_cd_mm']=float((pcd[pm].mean()+tcd[tm].mean())*.5*units) if pm.any() and tm.any() else None
            region[name+'_counts']=[int(pm.sum()),int(tm.sum())]
        dist=(p[:,None]-p[neighbors]).norm(dim=-1)
        delta=p-initial
        span0=(initial.amax(0)-initial.amin(0)).clamp_min(1e-8)
        area_pred=(capture.pred['alpha'][0,k]>.5).sum((-1,-2))
        area_gt=(capture.gt['alpha'][0,k]>.5).sum((-1,-2))
        row=dict(centered_cd_mm=float((pc.mean()+tc.mean())*.5*units),
            centroid_error_mm=float((p.mean(0)-t.mean(0)).norm()*units),
            predicted_centroid_motion_mm=((p.mean(0)-initial.mean(0))*units).tolist(),
            gt_centroid_motion_mm=((t.mean(0)-initial.mean(0))*units).tolist(),
            displacement_rms_mm=float(delta.square().sum(-1).mean().sqrt()*units),
            bbox_span_ratio=((p.amax(0)-p.amin(0))/span0).tolist(),
            neighbor_distance_change_mm=float((dist-d0)[edge_valid].abs().mean()*units) if edge_valid.any() else 0.,
            rendered_area_pixels=area_pred.tolist(),gt_rendered_area_pixels=area_gt.tolist(),**region)
        if trace:
            ids=trace[k]['anchor_ids'][0,trace[k]['anchor_mask'][0]]
            row['anchor_switch_fraction']=0. if previous_ids is None else 1-float(torch.isin(ids,previous_ids).float().mean())
            previous_ids=ids
            full=pred['xyz'][0,k];anchor_pos=full[ids]
            row['anchor_coverage_mean_mm']=float(torch.cdist(p,anchor_pos).amin(-1).mean()*units)
            fixed_keep=trace[k]['fixed_mask'][0]
            anchor_pos=full[trace[k]['fixed_ids'][0,fixed_keep]]
            if len(ids)>1:
                dd=torch.cdist(anchor_pos,anchor_pos);dd.fill_diagonal_(float('inf'));nearest=dd.argmin(-1)
                ad=trace[k]['displacement'][0,fixed_keep]
                row['anchor_field_gradient']=float(((ad-ad[nearest]).norm(dim=-1)/dd.amin(-1).clamp_min(1e-6)).mean())
        rows.append(row)
    result['motion']=rows
    return result
