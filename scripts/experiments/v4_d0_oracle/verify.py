"""CPU evidence checks; never re-executes an oracle or V4 network."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image


def read(p):return json.loads(p.read_text(encoding='utf8'))


def verify(root,normal_reference):
    rows=read(root/'results/results.json');status=read(root/'provenance/completion.json');old=read(normal_reference)
    curves=list(csv.DictReader((root/'results/optimization_curves.csv').open(encoding='utf8')))
    assert len(rows)==60 and len(curves)==7224 and status['passed']
    assert status['network_updates']==0 and status['model_state_before']==status['model_state_after']
    checks=[]
    for ep,start in ((7,105),(3,135)):
        for h in (5,10,15):
            key=f'ep{ep}_start{start}_h{h:02}';rr=[r for r in rows if r['window']==key]
            assert len(rr)==10
            normal=next(r for r in rr if r['method']=='normal')
            prior=next(r for r in old if r['episode']==ep and r['start']==start)
            target=next(r for r in prior['metrics'] if r['mode']=='normal' and r['horizon']==h)
            errors={}
            for a,b,tol in [('cd_mm','cd_mm',1e-4),('rgb','rgb_mae',1e-6),('alpha','alpha_mae',1e-6)]:
                errors[a]=abs(normal[a]-target[b]);assert errors[a]<=tol
            a=np.load(root/'arrays'/f'{key}.npz')
            assert all(np.isfinite(a[k]).all() for k in a.files)
            for loss in ('cd','joint'):
                for init in ('zero','normal'):
                    cc=[r for r in curves if r['window']==key and r['loss']==loss and r['initialization']==init]
                    assert [int(r['step']) for r in cc]==list(range(301))
                    assert all(np.isfinite(float(r[field])) for r in cc for field in ('cd','rgb','alpha','joint','objective','previous_gradient_norm','previous_update_norm'))
                    for snapshot in ('best','last'):
                        x=next(r for r in rr if r['method']=='D0-'+loss and r['initialization']==init and r['snapshot']==snapshot)
                        if snapshot=='best':assert abs(x[loss]-min(float(c['objective']) for c in cc))<1e-6
                        else:assert x['selected_step']==300
            checks.append(dict(window=key,normal_regression_errors=errors))
    pngs=list((root/'images').glob('*.png'))+list((root/'figures').glob('*.png'))
    assert len(pngs)==426,(len(pngs),'expected 396 raw +12 sheets +18 figures')
    for p in pngs:
        with Image.open(p) as im:im.verify()
    for ep,start in ((7,105),(3,135)):
        assert (root/'figures'/f'ep{ep}_start{start}_h15_geometry.png').is_file()
    result=dict(passed=True,rows=60,optimization_rows=7224,independent_fits=24,updates_per_fit=300,png_count=len(pngs),normal_regression=checks,
                browser_interaction='Not performed; prior file:// browser policy refusal. Static assets and directly inspected figures only.')
    (root/'provenance/artifact_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    paths=[p for p in root.rglob('*') if p.is_file() and p.name!='file_manifest.json']
    manifest=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)]
    (root/'provenance/file_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--normal-reference',type=Path,required=True);a=p.parse_args();verify(a.root,a.normal_reference)
