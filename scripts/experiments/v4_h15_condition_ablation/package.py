"""Package complete evidence, full tracked source and hash-verified offline report."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def package(root,destination):
    if destination.exists():raise FileExistsError(destination)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Commit source first')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    records=json.loads((root/'results.json').read_text(encoding='utf8'))
    if [(r['split'],r['episode'],r['start']) for r in records]!=[('train',7,105),('val',3,135),('val',6,110)]:raise ValueError('Window drift')
    checks=[]
    for r in records:
        d=root/'windows'/f"ep{r['episode']}_start{r['start']}";g=np.load(d/'geometry.npz')
        for mode in ('normal','freeze_action','zero_tool','copy'):
            rows=[x for x in r['metrics'] if x['mode']==mode]
            if [x['horizon'] for x in rows]!=list(range(1,16)):raise ValueError('Missing horizon')
            if not np.isfinite(g[mode]).all() or len(g[mode])!=15:raise ValueError('Invalid geometry')
            distinct=len({hashlib.sha256(x.tobytes()).hexdigest() for x in g[mode]})
            if mode!='copy' and distinct!=15:raise ValueError('Repeated predicted horizon')
            checks.append(dict(window=d.name,mode=mode,distinct_states=distinct))
        for h in (1,5,10,15):
            if not (root/'figures'/f'{d.name}_h{h:02}_geometry.png').is_file():raise FileNotFoundError('Missing geometry')
            for mode in ('gt','normal','freeze_action','zero_tool','copy'):
                for v in range(3):
                    for kind in ('rgb','alpha'):
                        if not (d/'images'/f'{mode}_h{h:02}_v{v}_{kind}.png').is_file():raise FileNotFoundError('Missing image')
    source=root/'source';source.mkdir(exist_ok=True)
    archive=source/f'touch4d-source-{head[:7]}.zip'
    if archive.exists():raise FileExistsError(archive)
    subprocess.run(['git','archive','--format=zip','--output',str(archive),head],check=True)
    inference=json.loads((root/'status/infer.json').read_text(encoding='utf8'))['execution']['source_sha']
    (source/'versions.json').write_text(json.dumps(dict(final_source_sha=head,inference_source_sha=inference,
        checkpoint_training_sha='60b7e6044a139a92bf7fdad29c262bde437ae2f9',
        base_to_final_diff=subprocess.check_output(['git','diff','--stat','60b7e604',head],text=True),
        note='Inference source frozen during GPU execution. Later commits only add report/packaging tools; source archive contains complete tracked project.'),indent=2),encoding='utf8')
    (root/'status/artifact_checks.json').write_text(json.dumps(dict(windows=3,conditions=3,metrics_rows=180,expected_rgb_alpha_pngs=360,all_predictions_finite=True,states=checks),indent=2),encoding='utf8')
    files=[p for p in root.rglob('*') if p.is_file() and p.name!='file_manifest.json']
    manifest=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    mp=root/'file_manifest.json';mp.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*sorted(files),mp]:z.write(p,root.name+'/'+p.relative_to(root).as_posix())
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        for entry in [*manifest,dict(path='file_manifest.json',sha256=sha(mp))]:
            h=hashlib.sha256()
            with z.open(root.name+'/'+entry['path']) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            if h.hexdigest()!=entry['sha256']:raise ValueError('Decompressed SHA mismatch')
    result=dict(zip=str(destination),bytes=destination.stat().st_size,sha256=sha(destination),entries=len(files)+1,
        zip_crc_pass=True,all_decompressed_members_sha256_pass=True,source_sha=head)
    receipt=destination.with_suffix('.verification.json');receipt.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--zip',type=Path,required=True);a=p.parse_args();package(a.root,a.zip)
