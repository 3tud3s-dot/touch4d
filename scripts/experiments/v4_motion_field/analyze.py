"""CPU-only audit of saved arrays; never constructs a model or an optimizer."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from scripts.training.gs_decoder import build_initial_anchor_field
from scripts.training.model import initial_knn
from .metrics import EPS_M, vector_comparison, neighbor_metrics, rms


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')


def csv_write(p,rows):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)


def analyze(source,output):
    if (output/'provenance/analysis.json').exists():raise FileExistsError('Do not overwrite completed audit')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    paths=sorted(list((source/'arrays').glob('*'))+[source/'results/geometry_metrics.csv']+list((source/'provenance').glob('*.json')))
    original_manifest=json.loads((source/'provenance/file_manifest.json').read_text())
    expected={x['path']:x['sha256'] for x in original_manifest}
    before=[dict(path=p.relative_to(source).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    for row in before:
        if row['path'] in expected and row['sha256']!=expected[row['path']]:raise ValueError('Input changed: '+row['path'])
    for d in ('results','figures','provenance','logs','fields'):(output/d).mkdir(parents=True,exist_ok=True)
    proc=json.loads((source/'provenance/checkpoint.json').read_text())['processor']
    scale=proc['scale'];center=torch.tensor(proc['center'],dtype=torch.float32)
    rows=[];region_rows=[];structure_rows=[];gates=[];selection=[]
    for ep,start in ((7,105),(3,135)):
        window=f'ep{ep}_start{start}';reference=None
        for h in (5,10,15):
            key=f'{window}_h{h:02}'
            with np.load(source/'arrays'/f'{key}.npz') as zz:z={k:zz[k] for k in zz.files}
            x=z['initial'];n=z['normal'].astype(np.float64)-x.astype(np.float64)
            if not np.array_equal(x,z['copy']):raise ValueError('Copy differs from initial')
            if reference is not None:
                for k in ('initial','near','far'):
                    if not np.array_equal(z[k],reference[k]):raise ValueError('Initial/regions drift across H')
            reference=z
            normalized=(torch.from_numpy(x)-center)/scale;keep=torch.ones((1,len(x)),dtype=torch.bool)
            field=build_initial_anchor_field(normalized[None],keep,128,16,1.)
            weights=field.weights[0].numpy().astype(np.float64);ai=field.anchor_indices[0].numpy()
            if not np.allclose(weights.sum(1),1,atol=2e-7):raise ValueError('RBF normalization')
            ids,valid=initial_knn(normalized[None],keep,16)
            ids=ids[0].numpy();valid=valid[0].numpy()&(ids!=np.arange(len(x))[:,None])
            # Same normalized 1e-8 exclusion as original Oracle structural audit.
            valid &= np.linalg.norm(normalized.numpy()[:,None]-normalized.numpy()[ids],axis=-1)>1e-8
            groups={'all':np.arange(len(x)),'near':z['near'],'far':z['far']}
            groups['middle']=np.setdiff1d(groups['all'],np.concatenate([groups['near'],groups['far']]))
            assert not np.intersect1d(groups['near'],groups['far']).size
            assert all(np.all((v>=0)&(v<len(x))) for v in groups.values())
            nd=torch.load(source/'arrays'/f'{key}_cd_normal.pt',map_location='cpu',weights_only=True)['initial_d'][0].numpy().astype(np.float64)*scale
            fields=dict(initial=x,normal=n,anchor_initial=x[ai],normal_anchor_control=nd,
                        weights=weights,anchor_indices=ai,near=z['near'],far=z['far'],knn_ids=ids,knn_valid=valid)
            errors=[dict(method='normal',max_mapping_error_mm=float(np.linalg.norm(weights@nd-n,axis=1).max())*1000)]
            structure_rows.append(dict(window=window,horizon=h,method='normal',snapshot='fixed',**neighbor_metrics(x,n,ids,valid)))
            for loss in ('cd','joint'):
                for init in ('zero','normal'):
                    state=torch.load(source/'arrays'/f'{key}_{loss}_{init}.pt',map_location='cpu',weights_only=True)
                    initial=state['initial_d'][0].numpy()*scale
                    if init=='normal' and not np.allclose(initial,nd,atol=1e-8,rtol=1e-6):raise ValueError('Normal D mismatch')
                    if init=='zero' and np.any(initial):raise ValueError('Nonzero zero initialization')
                    for snapshot in ('best','last'):
                        name=f'{loss}_{init}_{snapshot}';o=z[name].astype(np.float64)-x.astype(np.float64)
                        od=state[snapshot+'_d'][0].numpy().astype(np.float64)*scale
                        fields[name]=o;fields[name+'_anchor_control']=od
                        err=float(np.linalg.norm(weights@od-o,axis=1).max())*1000
                        errors.append(dict(method=name,max_mapping_error_mm=err))
                        base=dict(window=window,split='train' if ep==7 else 'val',horizon=h,oracle=loss,initialization=init,snapshot=snapshot,
                                  selected_step=int(state['best_step']) if snapshot=='best' else 300)
                        for level,a,b in [('gs',n,o),('anchor_control',nd,od),('fps_actual',n[ai],o[ai])]:
                            metrics,point=vector_comparison(a,b)
                            rows.append(dict(**base,level=level,**metrics))
                            if level=='gs':
                                fields[name+'_angle_deg']=point['angle_deg'];fields[name+'_difference_mm']=point['difference_mm']
                                top=np.argsort(point['difference_mm'])[-max(1,len(x)//10):]
                                selection.append(dict(**base,top_error_decile_count=len(top),
                                    top_error_near_fraction=float(np.isin(top,z['near']).mean()),top_error_far_fraction=float(np.isin(top,z['far']).mean()),
                                    centroid_world_m=x[top].mean(0).tolist(),min_world_m=x[top].min(0).tolist(),max_world_m=x[top].max(0).tolist()))
                        for region,ix in groups.items():
                            metrics,_=vector_comparison(n[ix],o[ix]);region_rows.append(dict(**base,region=region,count=len(ix),**metrics))
                        structure_rows.append(dict(**base,method=name,**neighbor_metrics(x,o,ids,valid)))
            # Predetermined float32 world-export/reconstruction tolerance, 0.1 micrometre.
            if max(e['max_mapping_error_mm'] for e in errors)>1e-4:
                raise ValueError(f'{key}: cannot reproduce original mapping; stop and export original tensors')
            gates.append(dict(window=key,points=len(x),anchors=len(ai),sigma_world_mm=float(field.bandwidth[0])*scale*1000,
                              max_weight_sum_error=float(abs(weights.sum(1)-1).max()),mapping=errors))
            np.savez_compressed(output/'fields'/f'{key}.npz',**fields)
    for p,row in zip(paths,before):
        if sha(p)!=row['sha256']:raise ValueError('Input modified during analysis')
    csv_write(output/'results/motion_field_comparison.csv',rows)
    csv_write(output/'results/near_far_analysis.csv',region_rows)
    csv_write(output/'results/neighbor_structure.csv',structure_rows)
    write(output/'results/spatial_error_locations.json',selection)
    write(output/'provenance/input_hashes.json',before)
    write(output/'provenance/mapping_checks.json',gates)
    write(output/'provenance/analysis.json',dict(source=str(source),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=[sys.executable,*sys.argv],device='cpu',network_loaded=False,optimizer_created=False,backward=False,
        coordinate_system='world metres; displacement relative to identical initial GS row; no GT identity used',
        direction_epsilon_m=EPS_M,direction_threshold_purpose='numerical undefined direction only',
        rbf='production float32 rebuild after inverse world normalization; all saved outputs gated at max 0.0001 mm',
        regions='original saved near/far indices: closest/farthest quarter to initial valid operating controller; middle remainder',
        neighbors='production initial 16 slots including self; exclude self and normalized distance <=1e-8, matching prior audit',
        anchor_warning='D controls are not FPS physical displacement: actual FPS movement is (W D)[FPS]',
        loss_selection='each saved best minimizes its original objective; both initializations and last included',
        points='all saved rows are valid initial template rows, verified by prior source export and region bounds',
        files=len(before),comparison_rows=len(rows),region_rows=len(region_rows)))
    print(json.dumps(dict(mapping_passed=True,rows=len(rows),region_rows=len(region_rows),max_mapping_error_mm=max(e['max_mapping_error_mm'] for g in gates for e in g['mapping']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.source,a.output)
