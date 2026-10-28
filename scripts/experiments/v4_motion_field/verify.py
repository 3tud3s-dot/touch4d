"""Verify every offline page route, numerical identities, input immutability and PNGs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .analyze import sha,write


def verify(source,root):
    manifest=json.loads((root/'provenance/input_hashes.json').read_text())
    assert all(sha(source/x['path'])==x['sha256'] for x in manifest)
    a=list(csv.DictReader((root/'results/motion_field_comparison.csv').open()))
    b=list(csv.DictReader((root/'results/near_far_analysis.csv').open()))
    c=list(csv.DictReader((root/'results/neighbor_structure.csv').open()))
    assert (len(a),len(b),len(c))==(144,192,54)
    for row in a+b:
        total=float(row['difference_rms_mm'])**2
        decomposed=float(row['mean_translation_difference_mm'])**2+float(row['residual_difference_rms_mm'])**2
        assert np.isclose(total,decomposed,rtol=1e-8,atol=1e-8)
        assert np.isclose(float(row['direction_error_energy_fraction'])+float(row['magnitude_error_energy_fraction']),1)
    pngs=list((root/'figures').glob('*.png'));assert len(pngs)==40
    for p in pngs:
        with Image.open(p) as im:im.verify()
    states=0;best_last_differences=[]
    for window in ('ep7_start105','ep3_start135'):
        for h in (5,10,15):
            z=np.load(root/'fields'/f'{window}_h{h:02}.npz')
            for method in ('cd_zero','cd_normal','joint_zero','joint_normal'):
                raw=torch.load(source/'arrays'/f'{window}_h{h:02}_{method}.pt',map_location='cpu',weights_only=True)
                same=np.array_equal(z[method+'_best'],z[method+'_last'])
                assert same==torch.equal(raw['best_d'],raw['last_d'])
                if not same:
                    diff=z[method+'_best']-z[method+'_last']
                    best_last_differences.append(dict(window=window,horizon=h,method=method,best_step=raw['best_step'],rms_mm=float(np.sqrt(np.mean(np.sum(diff**2,axis=-1))))*1000))
            for snapshot in ('best','last'):
                states+=1
                for kind in ('fields','anchors','near_far'):
                    assert (root/'figures'/f'{window}_h{h:02}_{snapshot}_{kind}.png').exists()
            for kind in ('motion_field_summary','horizon_trends'):assert (root/'figures'/f'{window}_{kind}.png').exists()
    assert len(best_last_differences)==1 and best_last_differences[0]['best_step']==296
    page=(root/'report.html').read_text(encoding='utf8');assert 'https://' not in page and 'fetch(' not in page
    result=dict(passed=True,input_files_unchanged=len(manifest),png_count=len(pngs),html_selector_routes=states,
                rows=[len(a),len(b),len(c)],best_last_differences=best_last_differences,
                browser_interaction='Not claimed; offline paths and all selection combinations checked statically, representative PNGs inspected.')
    write(root/'provenance/verification.json',result)
    write(root/'provenance/output_manifest.json',[dict(path=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='output_manifest.json'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();verify(a.source,a.root)
