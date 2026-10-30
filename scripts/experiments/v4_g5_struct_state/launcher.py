"""Persistent single-allocation launcher. No sbatch/new GPU, at most two resumes."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from scripts.training.train import atomic_json
from .protocol import (BUCKETS,JOB_ID,source_guard,resource_guard,calibration_read,
                       configure_runtime,graph_review_read)

VARIANTS=tuple(BUCKETS)


def stamp():return datetime.now(timezone.utc).isoformat()


def process_start(pid):
    try:
        status=Path(f'/proc/{pid}/status').read_text()
        if int(next(v for v in status.splitlines() if v.startswith('Uid:')).split()[1])!=os.getuid():return None
        values=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return values[19] if values[0]!='Z' else None
    except (OSError,IndexError,StopIteration,ValueError):return None


def live(entry):
    return bool(entry.get('pid') and entry.get('start_ticks') and process_start(entry['pid'])==entry['start_ticks'])


class Launcher:
    def __init__(self,args):
        self.a=args;self.root=Path(args.common);self.control=self.root/'launcher'
        identity=dict(source_sha=args.expected_sha,source=str(Path.cwd().resolve()),common=str(self.root.resolve()),
            a_output=str(Path(args.a_output).resolve()),b_output=str(Path(args.b_output).resolve()),
            root=str(Path(args.root).resolve()),depth_root=str(Path(args.depth_root).resolve()),
            g0_reference=str(Path(args.g0_reference).resolve()))
        self.control.mkdir(parents=True,exist_ok=True);self.logs=self.root/'logs';self.logs.mkdir(exist_ok=True)
        self.path=self.control/'state.json';self.handles={};self.log_handles=[]
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(source_sha=args.expected_sha,launch_identity=identity,
            phase='NEW',variants={v:dict(status='PENDING',attempts=0) for v in VARIANTS},recoveries=[])
        if self.state['source_sha']!=args.expected_sha:raise ValueError('Launcher source changed; refusing cross-source recovery')
        if self.state.get('launch_identity')!=identity:raise ValueError('Launcher dataset/Run/output identity changed')
        self.runs={v:Path(args.a_output if v==VARIANTS[0] else args.b_output) for v in VARIANTS}
        self.calibration=self.root/'calibration'/'calibration.json'

    def save(self,**values):
        self.state.update(values);self.state.update(launcher_pid=os.getpid(),launcher_start=process_start(os.getpid()),updated_utc=stamp())
        atomic_json(self.path,self.state)

    def event(self,event_name,**data):
        row=dict(data);row.setdefault('time_utc',stamp());row['event']=event_name
        with (self.control/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def base(self,module):
        return [sys.executable,'-B','-m','scripts.experiments.v4_g5_struct_state.'+module,
            '--root',self.a.root,'--depth-root',self.a.depth_root,'--g0-reference',self.a.g0_reference,'--expected-sha',self.a.expected_sha]

    def one(self,module,extra,label):
        resource_guard(require_idle=True)
        path=self.logs/(label+'.log')
        if path.exists():path=self.logs/(label+'-'+str(time.time_ns())+'.log')
        command=self.base(module)+extra
        self.save(phase=label);self.event('PHASE_START',label=label,command=command,log=str(path))
        with path.open('xb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            self.save(active_phase_process=dict(pid=child.pid,start_ticks=process_start(child.pid),command=command,log=str(path)))
            code=child.wait()
        self.event('PHASE_END',label=label,exit_code=code)
        if code:raise RuntimeError(f'{label} failed with exit {code}; evidence retained at {path}')
        self.save(active_phase_process=None)

    def gates(self):
        initial=self.root/'engineering'/'preflight.json'
        if not initial.exists():
            self.one('train',['--variant',VARIANTS[0],'--output',str(self.runs[VARIANTS[0]]),
                '--common',str(self.root),'--preflight-only'],'PREFLIGHT')
        proof=json.loads(initial.read_text())
        if not proof['passed'] or proof['source_sha']!=self.a.expected_sha:raise ValueError('Wrong paired preflight')
        if not self.calibration.exists():
            self.one('calibrate',['--output',str(self.calibration.parent)],'CALIBRATION')
        calibration_read(self.calibration)
        for v in VARIANTS:
            out=self.root/'smoke'/v;done=out/'completion.json'
            if not done.exists():
                self.one('smoke',['--variant',v,'--calibration',str(self.calibration),'--output',str(out)],'SMOKE_'+v)
            d=json.loads(done.read_text())
            if not d.get('passed') or d.get('source_sha')!=self.a.expected_sha or d.get('successful_updates')!=4:
                raise ValueError('Same-source four-update Smoke not passed')
        self.save(phase='WAIT_GRAPH_REVIEW')
        deadline=time.monotonic()+1800
        while not (self.root/'graph_review.json').exists():
            resource_guard(require_idle=True)
            if time.monotonic()>deadline:raise RuntimeError('Graph review pending; restart launcher after CPU image inspection')
            time.sleep(5)
        graph_review_read(self.root,self.calibration)

    def parallel_probe(self):
        if 'scheduling' in self.state:return self.state['scheduling']['mode']=='parallel'
        r=resource_guard(require_idle=True)
        evidence=[json.loads((self.root/'smoke'/v/'completion.json').read_text()) for v in VARIANTS]
        estimate=sum(d['peak_memory_reserved'] for d in evidence)*1.25/(1024**2)+1536
        rss=sum(d['process_max_rss'] for d in evidence)
        plan=dict(mode='serial',estimated_mib=estimate,reserve_mib=r['reserve_mib'],smoke_rss_sum=rss,
                  reason='Measured Smoke peaks plus25percent and context headroom; CPU allocation16GiB')
        if estimate>r['free_mib']-r['reserve_mib'] or rss>int(16*1024**3*.8):
            self.save(scheduling=plan);return False
        probe=self.root/'probes'/'probe-001'
        if probe.exists():
            self.save(scheduling=dict(plan,reason='Interrupted probe evidence retained; conservative serial, no repeated probe'))
            return False
        probe.mkdir(parents=True,exist_ok=False)
        barrier=probe/'barrier';barrier.mkdir();registry=probe/'peers.json';children={};outputs={v:probe/v for v in VARIANTS}
        for v in VARIANTS:
            log=(probe/(v+'.log')).open('xb');self.log_handles.append(log)
            command=self.base('smoke')+['--variant',v,'--calibration',str(self.calibration),'--output',str(outputs[v]),
                '--probe','--smoke-checkpoint',str(self.root/'smoke'/v/'smoke_4updates.pt'),'--peer-registry',str(registry)]
            children[v]=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        self.save(active_probe={v:dict(pid=p.pid,start_ticks=process_start(p.pid)) for v,p in children.items()},phase='PARALLEL_PROBE')
        atomic_json(registry,dict(source_sha=self.a.expected_sha,parent_pid=os.getpid(),pids=[p.pid for p in children.values()],
            outputs={v:str(p) for v,p in outputs.items()},barrier=str(barrier)))
        peak=0;released=False;reason=None;start=time.monotonic()
        while any(p.poll() is None for p in children.values()):
            info=resource_guard(allowed_pids=[p.pid for p in children.values()]);peak=max(peak,info['used_mib'])
            if info['foreign_pids'] or info['free_mib']<info['reserve_mib']:
                reason='foreign activity or safety reserve breached'
            if any(p.poll() not in (None,0) for p in children.values()):reason=reason or 'probe child failed'
            ready=all((barrier/(v+'-ready.json')).exists() for v in VARIANTS)
            if not released and (ready or reason):
                atomic_json(barrier/'go.json',dict(source_sha=self.a.expected_sha,parent_pid=os.getpid(),go=not bool(reason),
                    abort=bool(reason),reason=reason));released=True
            if time.monotonic()-start>600:
                reason='concurrent probe timed out'
                for p in children.values():
                    if p.poll() is None:p.terminate()  # Only this launcher's disposable zero-update probes.
            time.sleep(.5)
        codes={v:p.wait() for v,p in children.items()}
        for h in self.log_handles:h.close()
        self.log_handles=[]
        passed=all(code==0 for code in codes.values()) and released and not reason
        if not passed:
            traces='\n'.join((probe/(v+'.log')).read_text(errors='replace') for v in VARIANTS)
            if reason is None or 'child failed' in str(reason):
                if 'out of memory' not in traces.lower():raise RuntimeError('Non-OOM concurrent probe failed; inspect logs')
            plan.update(reason=reason or 'concurrent OOM',probe_exit_codes=codes,actual_peak_mib=peak)
        else:plan.update(mode='parallel',reason='Both live H15 graphs overlapped and backward succeeded with reserve',actual_peak_mib=peak,probe_exit_codes=codes)
        self.save(scheduling=plan,active_probe=None);return plan['mode']=='parallel'

    def outputs_complete(self,v):
        p=self.runs[v]/'completion.json'
        if not p.exists():return False
        d=json.loads(p.read_text())
        if d.get('source_sha')!=self.a.expected_sha or type(d.get('completed_epochs')) is not int or d.get('completed_epochs')!=8 or d.get('variant')!=v:raise ValueError('Wrong formal completion identity')
        return True

    def stop_own_workers(self):
        """Keep the controlling Slurm task alive until its own safe stops finish."""
        for entry in self.state['variants'].values():
            if live(entry):os.kill(entry['pid'],signal.SIGUSR1)
        probes=self.state.get('active_probe') or {}
        for entry in probes.values():
            if live(entry):os.kill(entry['pid'],signal.SIGTERM)  # Disposable probes never optimize or save formal snapshots.
        started=time.monotonic()
        while any(live(e) for e in self.state['variants'].values()) or any(live(e) for e in probes.values()):
            self.save(stop_requested=True,stop_wait_seconds=time.monotonic()-started)
            time.sleep(2)
        for entry in self.state['variants'].values():
            if entry.get('status') in ('RUNNING','STARTING'):
                entry.update(status='INTERRUPTED',failure_reason='Stopped safely after peer/controller blocking failure')
        self.save(stop_requested=True,own_workers_drained=True)

    def launch_workers(self,variants):
        count=1+len(list(self.control.glob('formal-peers-*.json')));registry=self.control/f'formal-peers-{count:02d}.json'
        children={}
        concurrent=len(variants)+sum(live(e) for v,e in self.state['variants'].items() if v not in variants)>1
        for v in variants:
            entry=self.state['variants'][v]
            if live(entry):continue
            root=self.runs[v];resume=(root/'initial.pt').exists()
            if root.exists() and not resume:raise RuntimeError('Partial fresh Run has no durable initial; do not overwrite')
            if entry['attempts']>0:
                if len(self.state['recoveries'])>=2:raise RuntimeError('Two automatic recovery limit reached')
                counters=[]
                committed={p.name for p in root.glob('epoch_*.pt')}
                for f in sorted((root/'logs').glob('*-counter.json')):
                    d=json.loads(f.read_text())
                    if f"epoch_{d['epoch']:02d}.pt" not in committed:
                        counters.append(dict(path=str(f),**d,known_replayed_updates_min=d['successful_updates'],
                            replayed_updates_max=d['successful_updates']+int(d.get('pending_update',False))))
                recovery=dict(variant=v,time_utc=stamp(),reason=entry.get('failure_reason','prior worker interrupted'),uncommitted_attempts=counters)
                self.state['recoveries'].append(recovery);self.event('RECOVERY',**recovery)
            attempt=entry['attempts']+1;logpath=self.logs/f'{v}-formal-attempt-{attempt:02d}.log'
            command=self.base('train')+['--variant',v,'--calibration',str(self.calibration),'--common',str(self.root),
                '--output',str(root),'--peer-registry',str(registry)]+(['--resume'] if resume else [])
            log=logpath.open('xb');self.log_handles.append(log)
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);children[v]=child;self.handles[v]=child
            entry.update(status='STARTING',attempts=attempt,pid=child.pid,start_ticks=process_start(child.pid),log=str(logpath),command=command,
                launch_mode='parallel' if concurrent else 'serial')
            self.save(phase='FORMAL')
        allowed=[e['pid'] for e in self.state['variants'].values() if live(e)]
        atomic_json(registry,dict(source_sha=self.a.expected_sha,parent_pid=os.getpid(),pids=allowed))
        for v in children:self.state['variants'][v]['status']='RUNNING'
        self.save(phase='FORMAL')

    def formal(self,parallel):
        while True:
            source_guard(self.a.expected_sha)
            if any(e.get('status')=='FAILED' for e in self.state['variants'].values()):
                raise RuntimeError('Persisted deterministic FAILED state; no automatic restart')
            active=[e['pid'] for e in self.state['variants'].values() if live(e)]
            resource=resource_guard(allowed_pids=active)
            pressure=bool(resource['foreign_pids']) or resource['free_mib']<resource['reserve_mib']
            if pressure and active:
                for entry in self.state['variants'].values():
                    if live(entry):os.kill(entry['pid'],signal.SIGUSR1)
                if parallel:
                    parallel=False;self.state['scheduling']['mode']='serial_after_reserve_pressure'
                elif not resource['foreign_pids']:
                    self.save(status='RESOURCE_BLOCKED',resource=resource)
                    raise RuntimeError('Single-worker GPU safety reserve cannot be maintained')
                self.event('SAFE_WINDOW_STOP_REQUESTED',resource=resource)
            for v in VARIANTS:
                e=self.state['variants'][v]
                if self.outputs_complete(v):e['status']='COMPLETED';continue
                if e.get('status') in ('RUNNING','STARTING') and not live(e):
                    trace=Path(e['log']).read_text(errors='replace') if e.get('log') else ''
                    failure=self.runs[v]/f"failure-{e.get('pid')}.json"
                    failure_trace=failure.read_text() if failure.exists() else ''
                    combined=trace+'\n'+failure_trace
                    is_oom='out of memory' in combined.lower()
                    boundary=self.runs[v]/'boundary_stop.json'
                    stopped=(boundary.exists() and json.loads(boundary.read_text()).get('pid')==e.get('pid')) or (self.root/'worker_stops'/f"{e.get('pid')}.json").exists()
                    handle=self.handles.get(v);exit_code=handle.poll() if handle is not None else None
                    if 'Traceback' in combined and not is_oom:
                        e.update(status='FAILED',failure_reason='deterministic numerical/identity/causality failure');self.save();raise RuntimeError(e['failure_reason'])
                    if exit_code==0 and not stopped:
                        e.update(status='FAILED',failure_reason='Exited without eight-Epoch completion or valid boundary stop');self.save();raise RuntimeError(e['failure_reason'])
                    if handle is not None:handle.wait()
                    e.update(status='INTERRUPTED',failure_reason=trace[-3000:])
                    if is_oom and e.get('launch_mode')=='parallel':
                        parallel=False;self.state['scheduling']['mode']='serial_after_concurrent_oom'
                        for other in self.state['variants'].values():
                            if live(other):os.kill(other['pid'],signal.SIGUSR1)
                        self.save();self.event('CONCURRENT_OOM_SERIAL_FALLBACK')
                    elif is_oom:
                        e.update(status='FAILED',failure_reason='Single-worker OOM; preserve data/model/protocol');self.save();raise RuntimeError(e['failure_reason'])
            if all(e['status']=='COMPLETED' for e in self.state['variants'].values()):
                # Completion is atomically published before Python releases CUDA.
                # Wait for our own children to exit before idle-only evaluation.
                while any(live(e) for e in self.state['variants'].values()):time.sleep(1)
                for child in self.handles.values():child.wait()
                self.save(phase='TRAINING_COMPLETED');return
            running=[v for v,e in self.state['variants'].items() if live(e)]
            todo=[v for v,e in self.state['variants'].items() if e['status']!='COMPLETED' and not live(e)]
            if pressure:
                self.save(phase='WAIT_RESOURCE',resource=resource);time.sleep(10);continue
            # Never overlap recovery with a still-stopping serial peer.
            if parallel and todo:self.launch_workers(todo)
            elif not parallel and not running and todo:self.launch_workers(todo[:1])
            self.save();time.sleep(10)

    def run(self):
        source_guard(self.a.expected_sha)
        active=[e['pid'] for e in self.state['variants'].values() if live(e)]
        phase=self.state.get('active_phase_process') or {}
        probes=self.state.get('active_probe') or {}
        active += [phase['pid']] if live(phase) else []
        active += [e['pid'] for e in probes.values() if live(e)]
        resource=resource_guard(require_idle=True,allowed_pids=active);self.save(resource=resource)
        if live(phase):
            self.event('ADOPT_EXISTING_PHASE',pid=phase['pid'])
            while live(phase):
                resource_guard(require_idle=True,allowed_pids=[phase['pid']]);time.sleep(5)
            self.save(active_phase_process=None)
        if probes:
            # A new parent never forges the old parent's barrier release.
            while any(live(e) for e in probes.values()):
                resource_guard(require_idle=True,allowed_pids=[e['pid'] for e in probes.values() if live(e)]);time.sleep(5)
            self.save(active_probe=None,scheduling=dict(mode='serial',reason='Orphaned probe retained; no duplicate probe or updates'))
        active=[e['pid'] for e in self.state['variants'].values() if live(e)]
        if not active and self.state.get('phase') not in ('FORMAL','TRAINING_COMPLETED','EVALUATION_COMPLETED'):
            self.gates();parallel=self.parallel_probe()
        else:parallel=self.state.get('scheduling',{}).get('mode')=='parallel'
        self.formal(parallel)
        acceptance=self.root/'acceptance'
        if not (acceptance/'completion.json').exists():
            if acceptance.exists():raise RuntimeError('Partial evaluation retained; use a new acceptance directory after diagnosis')
            self.one('evaluate',['--a-run',str(self.runs[VARIANTS[0]]),'--b-run',str(self.runs[VARIANTS[1]]),
                '--calibration',str(self.calibration),'--output',str(acceptance)],'EVALUATION')
        self.save(phase='EVALUATION_COMPLETED',status='GPU_COMPLETE_CPU_REPORT_PENDING')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','depth-root','g0-reference','common','a-output','b-output','expected-sha'):p.add_argument('--'+k,required=True)
    a=p.parse_args();configure_runtime();launch=Launcher(a)
    import fcntl
    with (launch.control/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:launch.run()
        except BaseException:
            launch.save(status='BLOCKED_OR_FAILED',failure=__import__('traceback').format_exc())
            launch.stop_own_workers()
            raise


if __name__=='__main__':main()
