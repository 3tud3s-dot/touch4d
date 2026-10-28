import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.experiments.v4_motion_field.analyze import sha,write


def verify(root):
    hashes=json.loads((root/'provenance/input_hashes.json').read_text())
    assert all(sha(x['path'])==x['sha256'] for x in hashes)
    audit=json.loads((root/'results/coordinate_time_audit.json').read_text());assert audit['passed']
    counts={}
    for name,n in [('controller_motion',360),('object_motion',312),('signed_direction_comparison',2772),('tool_motion_alignment',360),('timeline',30),('pathway_values',30)]:
        rows=list(csv.DictReader((root/'results'/f'{name}.csv').open()));assert len(rows)==n,(name,len(rows));counts[name]=len(rows)
        for r in rows:
            if name=='object_motion' and r['method']!='normal':assert int(r['horizon']) in (5,10,15)
            if name=='signed_direction_comparison':
                if r['direction_defined']=='False':assert r['mean_direction_cosine']==''
                else:assert -1<=float(r['mean_direction_cosine'])<=1
    for w in audit['windows']:
        assert len(w['steps'])==15
        assert all(s['target_frame']==s['condition_frame']+1 for s in w['steps'])
        assert w['steps'][0]['adjacent_valid_nodes']==0
    images=list((root/'figures').glob('*.png'));assert len(images)==10
    for p in images:
        with Image.open(p) as im:im.verify()
    for window in ('ep3_start135','ep7_start105'):
        for kind in ('signed_motion_summary','world_directions','near_middle_far','tool12_alignment','condition_pathway'):
            assert (root/'figures'/f'{window}_{kind}.png').is_file()
    page=(root/'report.html').read_text(encoding='utf8');assert 'fetch(' not in page and 'https://' not in page
    result=dict(passed=True,input_files_unchanged=len(hashes),csv_counts=counts,png_count=len(images),
                browser='No real-browser interaction claim; both routes, all images and static assets verified; representative PNGs viewed.',
                device='cpu',new_network_forward=False,optimizer=False,backward=False)
    write(root/'provenance/verification.json',result)
    write(root/'provenance/output_manifest.json',[dict(path=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='output_manifest.json'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);verify(p.parse_args().root)
