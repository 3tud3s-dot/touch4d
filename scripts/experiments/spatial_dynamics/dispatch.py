"""Launch exactly one frozen configuration from a Slurm array, no retries."""
import argparse,json,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--configs');p.add_argument('--index',type=int);p.add_argument('--source');p.add_argument('--sha');p.add_argument('--manifest');p.add_argument('--run-root');a=p.parse_args()
    config=json.loads(Path(a.configs).read_text())[a.index]
    command=[sys.executable,'-B','-m','scripts.experiments.spatial_dynamics.runner',config.get('command','run'),
        '--root','/data1/userdata/tcweng/projects/touch4d/datasets/touch4d-v1',
        '--output',str(Path(a.run_root)/config['id']), '--manifest',a.manifest,'--expected-sha',a.sha]
    for key in ('mode','decoder','feedback','regularization','horizon','seed','updates'):
        if key in config:command.extend(['--'+key,str(config[key])])
    print(json.dumps(dict(config=config,command=command,sha=a.sha)),flush=True)
    subprocess.run(command,check=True,cwd=a.source)
if __name__=='__main__':main()
