"""Mocked persistent controller QA; never starts a process or touches a GPU."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from scripts.data_process import DataProcessor
from scripts.training.rendering import GaussianRenderer
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.tests.training.test_v4_g5_training import first
from scripts.tests.training.test_v4_struct_state_training import raster
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_g5_struct_state import launcher,runner,protocol
from scripts.experiments.v4_g5_struct_state.model import fresh_model


class StopLoop(RuntimeError):pass


class FakeChild:
    def __init__(self,pid,owner):self.pid=pid;self.owner=owner;self.code=None
    def poll(self):return self.code
    def wait(self,timeout=None):
        if self.code is None:self.code=0
        self.owner.alive.pop(self.pid,None)
        return self.code
    def terminate(self):self.code=-15;self.owner.alive.pop(self.pid,None)
    def send_signal(self,number):self.owner.signals.append((self.pid,number))


class LauncherControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.args=argparse.Namespace(root=str(self.root/'data'),depth_root=str(self.root/'depth'),
            g0_reference=str(self.root/'reference'),common=str(self.root/'comparison'),
            a_output=str(self.root/protocol.BUCKETS['g5_struct']/'run-001'),
            b_output=str(self.root/protocol.BUCKETS['g6_state_struct']/'run-001'),expected_sha='a'*40)
        self.alive={os.getpid():'launcher-ticks'};self.children=[];self.signals=[];self.instances=[]
        self.info=dict(foreign_pids=[],free_mib=30000,reserve_mib=4916,used_mib=2000,total_mib=32768)
        self.patches=[patch.object(launcher,'process_start',side_effect=lambda pid:self.alive.get(pid)),
            patch.object(launcher,'source_guard',return_value=(self.args.expected_sha,{})),
            patch.object(launcher,'resource_guard',side_effect=lambda **kwargs:copy.deepcopy(self.info)),
            patch.object(launcher.subprocess,'Popen',side_effect=self.spawn),
            patch.object(launcher.os,'kill',side_effect=lambda pid,sig:self.signals.append((pid,sig))),
            patch.object(launcher.os,'getuid',return_value=1025,create=True),
            patch.object(signal,'SIGUSR1',10,create=True)]
        for p in self.patches:p.start()
        self.launch=self.new_launcher()

    def tearDown(self):
        for inst in self.instances:
            for handle in inst.log_handles:
                if not handle.closed:handle.close()
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def new_launcher(self,args=None):
        result=launcher.Launcher(args or self.args);self.instances.append(result);return result

    def spawn(self,command,**kwargs):
        pid=10000+len(self.children)
        child=FakeChild(pid,self);child.command=list(command)
        self.children.append(child);self.alive[pid]='ticks-'+str(pid)
        return child

    def finish(self,variant,code=-9):
        entry=self.launch.state['variants'][variant];pid=entry['pid']
        self.alive.pop(pid,None)
        for child in self.children:
            if child.pid==pid:child.code=code

    def completed(self,target,**updates):
        root=self.launch.runs[target];root.mkdir(parents=True,exist_ok=True)
        value=dict(source_sha=self.args.expected_sha,variant=target,completed_epochs=8,successful_updates=12112)
        value.update(updates);(root/'completion.json').write_text(json.dumps(value))

    def active_entry(self,variant,pid=20001,ticks='ticks-active'):
        log=self.launch.logs/(variant+'-existing.log');log.write_text('')
        entry=self.launch.state['variants'][variant]
        entry.update(status='RUNNING',attempts=1,pid=pid,start_ticks=ticks,log=str(log))
        self.alive[pid]=ticks
        return entry

    def test_process_stat_zombie_malformed_and_pid_reuse_excluded(self):
        # Bypass the fixture's process_start mock to exercise the actual parser.
        self.patches[0].stop()
        try:
            def stat(state,ticks):
                values=[state]+['0']*18+[ticks]
                return '42 (python worker) '+' '.join(values)
            def content(state='R',uid=1025):
                return lambda path,*a,**k: f'Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n' if path.name=='status' else stat(state,'99')
            with patch.object(Path,'read_text',new=content('R')):
                self.assertEqual(launcher.process_start(42),'99')
                self.assertTrue(launcher.live(dict(pid=42,start_ticks='99')))
                self.assertFalse(launcher.live(dict(pid=42,start_ticks='old')))
            with patch.object(Path,'read_text',new=content('Z')):
                self.assertIsNone(launcher.process_start(42));self.assertFalse(launcher.live(dict(pid=42,start_ticks='99')))
            with patch.object(Path,'read_text',new=content('R',1026)):
                self.assertIsNone(launcher.process_start(42))
            with patch.object(Path,'read_text',side_effect=OSError):self.assertIsNone(launcher.process_start(42))
            with patch.object(Path,'read_text',return_value='malformed'):self.assertIsNone(launcher.process_start(42))
        finally:self.patches[0].start()

    def test_live_worker_is_not_duplicated_and_registry_has_current_parent_and_peers(self):
        self.launch.launch_workers(['g5_struct'])
        count=len(self.children);entry=copy.deepcopy(self.launch.state['variants']['g5_struct'])
        self.launch.launch_workers(['g5_struct'])
        self.assertEqual(len(self.children),count)
        self.assertEqual(self.launch.state['variants']['g5_struct']['attempts'],entry['attempts'])
        self.assertEqual(self.launch.state['recoveries'],[])
        path=sorted(self.launch.control.glob('formal-peers-*.json'))[-1]
        peers=json.loads(path.read_text())
        self.assertEqual(peers['parent_pid'],os.getpid())
        self.assertEqual(peers['source_sha'],self.args.expected_sha)
        self.assertEqual(peers['pids'],[entry['pid']])

    def test_gate_phase_logs_and_recovery_timestamp_do_not_collide_with_event_fields(self):
        self.launch.one('train',['--preflight-only'],'PREFLIGHT')
        self.assertEqual(len(self.children),1)
        self.assertIsNone(self.launch.state['active_phase_process'])
        self.launch.event('RECOVERY',time_utc='recorded-time',variant='g5_struct')
        events=[json.loads(line) for line in (self.launch.control/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['event'] for e in events],['PHASE_START','PHASE_END','RECOVERY'])
        self.assertEqual(events[0]['label'],'PREFLIGHT')
        self.assertIsInstance(events[-1]['time_utc'],str)

    def test_global_two_recovery_limit_includes_preinitial_failures_across_variants(self):
        for v in ('g5_struct','g6_state_struct','g5_struct','g6_state_struct'):
            self.launch.launch_workers([v]);self.finish(v)
            self.launch.state['variants'][v]['failure_reason']='Transient worker loss before initial snapshot'
        self.assertEqual(len(self.children),4)
        self.assertEqual(len(self.launch.state['recoveries']),2)
        self.assertTrue(all('--resume' not in c.command for c in self.children))
        with self.assertRaisesRegex(RuntimeError,'recovery limit'):self.launch.launch_workers(['g5_struct'])
        self.assertEqual(len(self.children),4)

    def test_partial_fresh_output_preserved_and_legal_initial_uses_resume(self):
        root=self.launch.runs['g5_struct'];root.mkdir(parents=True)
        marker=root/'user-evidence.txt';marker.write_text('preserve')
        with self.assertRaisesRegex(RuntimeError,'no durable initial'):self.launch.launch_workers(['g5_struct'])
        self.assertEqual(marker.read_text(),'preserve');self.assertFalse(self.children)
        initial=root/'initial.pt';initial.write_bytes(b'unit durable boundary placeholder')
        self.launch.launch_workers(['g5_struct'])
        self.assertIn('--resume',self.children[-1].command)
        self.assertEqual(initial.read_bytes(),b'unit durable boundary placeholder')

    def test_completion_requires_exact_source_variant_and_eight_integer_epochs(self):
        self.assertFalse(self.launch.outputs_complete('g5_struct'))
        self.completed('g5_struct');self.assertTrue(self.launch.outputs_complete('g5_struct'))
        for fields in (dict(source_sha='wrong'),dict(variant='g6_state_struct'),dict(completed_epochs=7),
                       dict(completed_epochs='8'),dict(completed_epochs=8.0)):
            self.completed('g5_struct',**fields)
            with self.assertRaises(ValueError):self.launch.outputs_complete('g5_struct')

    def test_completed_A_does_not_block_serial_B(self):
        self.completed('g5_struct');self.launch.state['variants']['g5_struct']['status']='COMPLETED'
        self.launch.state['scheduling']=dict(mode='serial')
        def tick(_):
            if self.children:
                self.completed('g6_state_struct');self.finish('g6_state_struct',0)
        with patch.object(launcher.time,'sleep',side_effect=tick):self.launch.formal(False)
        self.assertEqual(len(self.children),1)
        self.assertEqual(self.children[0].command[self.children[0].command.index('--variant')+1],'g6_state_struct')
        self.assertTrue(all(e['status']=='COMPLETED' for e in self.launch.state['variants'].values()))
        self.assertEqual(self.launch.state['phase'],'TRAINING_COMPLETED')

    def test_existing_live_phase_is_adopted_before_gates_without_duplicate_child(self):
        phase=dict(pid=22001,start_ticks='phase-ticks',command=['existing'])
        self.alive[phase['pid']]=phase['start_ticks'];self.launch.state['active_phase_process']=phase
        acceptance=self.launch.root/'acceptance';acceptance.mkdir();(acceptance/'completion.json').write_text('{}')
        def tick(_):self.alive.pop(phase['pid'],None)
        def gates():self.assertFalse(launcher.live(phase))
        with (patch.object(launcher.time,'sleep',side_effect=tick),
              patch.object(self.launch,'gates',side_effect=gates) as g,
              patch.object(self.launch,'parallel_probe',return_value=False),
              patch.object(self.launch,'formal')):
            self.launch.run()
        g.assert_called_once();self.assertFalse(self.children)
        self.assertIsNone(self.launch.state['active_phase_process'])

    def test_orphaned_probe_adoption_never_forges_old_parent_release_or_relaunches(self):
        entries={v:dict(pid=23001+i,start_ticks='probe-'+str(i)) for i,v in enumerate(launcher.VARIANTS)}
        for e in entries.values():self.alive[e['pid']]=e['start_ticks']
        self.launch.state['active_probe']=entries;self.launch.state['phase']='PARALLEL_PROBE'
        probe=self.launch.root/'probes'/'probe-001';probe.mkdir(parents=True)
        registry=probe/'peers.json';old=dict(source_sha=self.args.expected_sha,parent_pid=999,pids=[e['pid'] for e in entries.values()])
        registry.write_text(json.dumps(old));before=registry.read_bytes()
        barrier=probe/'barrier';barrier.mkdir()
        acceptance=self.launch.root/'acceptance';acceptance.mkdir();(acceptance/'completion.json').write_text('{}')
        def tick(_):
            for e in entries.values():self.alive.pop(e['pid'],None)
        with patch.object(launcher.time,'sleep',side_effect=tick),patch.object(self.launch,'gates'),patch.object(self.launch,'formal'):
            self.launch.run()
        self.assertFalse(self.children)
        self.assertFalse((barrier/'go.json').exists());self.assertEqual(registry.read_bytes(),before)
        self.assertIsNone(self.launch.state['active_probe']);self.assertEqual(self.launch.state['scheduling']['mode'],'serial')

    def test_resource_pressure_signals_only_live_registered_workers_not_foreign_or_reused_pid(self):
        self.active_entry('g5_struct',pid=24001)
        self.active_entry('g6_state_struct',pid=24002,ticks='old-ticks')
        self.alive[24002]='new-reused-process'
        self.info['foreign_pids']=[55555];self.launch.state['scheduling']=dict(mode='parallel')
        with patch.object(launcher.time,'sleep',side_effect=StopLoop):
            with self.assertRaises(StopLoop):self.launch.formal(True)
        self.assertEqual(self.signals,[(24001,signal.SIGUSR1)])
        self.assertEqual(self.launch.state['scheduling']['mode'],'serial_after_reserve_pressure')

    def test_low_memory_reserve_also_uses_safe_worker_signal(self):
        self.active_entry('g5_struct',pid=25001);self.info.update(free_mib=4000,foreign_pids=[])
        self.launch.state['scheduling']=dict(mode='parallel')
        with patch.object(launcher.time,'sleep',side_effect=StopLoop):
            with self.assertRaises(StopLoop):self.launch.formal(True)
        self.assertEqual(self.signals,[(25001,signal.SIGUSR1)])

    def test_reopen_rejects_source_and_launch_destination_changes(self):
        self.launch.save();before=self.launch.path.read_bytes()
        wrong=copy.deepcopy(self.args);wrong.expected_sha='b'*40
        with self.assertRaises(ValueError):self.new_launcher(wrong)
        self.assertEqual(self.launch.path.read_bytes(),before)
        wrong=copy.deepcopy(self.args);wrong.b_output=str(self.root/'different-run')
        with self.assertRaises(ValueError):self.new_launcher(wrong)
        self.assertEqual(self.launch.path.read_bytes(),before)

    def test_terminal_stop_drains_only_live_own_worker_and_disposable_probe(self):
        self.active_entry('g6_state_struct',pid=26002)
        self.launch.state['variants']['g5_struct'].update(status='FAILED',pid=26001,start_ticks='dead')
        self.launch.state['active_probe']={
            'current':dict(pid=26003,start_ticks='probe'),
            'reused':dict(pid=26004,start_ticks='old-probe')}
        self.alive.update({26003:'probe',26004:'new-unrelated-process'})
        def stopped(_):self.alive.pop(26002,None);self.alive.pop(26003,None)
        with patch.object(launcher.time,'sleep',side_effect=stopped):self.launch.stop_own_workers()
        self.assertEqual(self.signals,[(26002,signal.SIGUSR1),(26003,signal.SIGTERM)])
        self.assertTrue(self.launch.state['stop_requested'])
        self.assertNotIn(26002,self.alive);self.assertNotIn(26003,self.alive)
        self.assertIn(26004,self.alive)

    def test_main_terminal_failure_calls_safe_stop_before_raising(self):
        self.active_entry('g6_state_struct',pid=27002)
        fake_fcntl=types.SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=lambda *a:None)
        def stopped(_):self.alive.pop(27002,None)
        with (patch.object(launcher.argparse.ArgumentParser,'parse_args',return_value=self.args),
              patch.object(launcher,'Launcher',return_value=self.launch),
              patch.object(launcher,'configure_runtime'),
              patch.object(self.launch,'run',side_effect=RuntimeError('A deterministic failure')),
              patch.object(launcher.time,'sleep',side_effect=stopped),
              patch.dict(sys.modules,{'fcntl':fake_fcntl})):
            with self.assertRaisesRegex(RuntimeError,'A deterministic failure'):launcher.main()
        self.assertEqual(self.signals,[(27002,signal.SIGUSR1)])
        self.assertEqual(self.launch.state['status'],'BLOCKED_OR_FAILED')
        self.assertNotIn(27002,self.alive)

    def test_persisted_deterministic_failed_group_is_not_automatically_restarted(self):
        self.launch.state['variants']['g5_struct']['status']='FAILED'
        with self.assertRaisesRegex(RuntimeError,'FAILED'):self.launch.formal(False)
        self.assertFalse(self.children)

    def test_simultaneous_parallel_ooms_both_fall_back_to_serial(self):
        for index,v in enumerate(launcher.VARIANTS):
            self.active_entry(v,pid=28001+index)
            entry=self.launch.state['variants'][v];entry['launch_mode']='parallel'
            Path(entry['log']).write_text('torch.cuda.OutOfMemoryError: CUDA out of memory\n')
            self.alive.pop(entry['pid'])
        self.launch.state['scheduling']=dict(mode='parallel')
        with patch.object(launcher.time,'sleep',side_effect=StopLoop):
            with self.assertRaises(StopLoop):self.launch.formal(True)
        self.assertEqual(self.launch.state['scheduling']['mode'],'serial_after_concurrent_oom')
        self.assertTrue(all(e['status']!='FAILED' for e in self.launch.state['variants'].values()))
        self.assertEqual(len(self.children),1)
        self.assertEqual(self.launch.state['variants']['g5_struct']['launch_mode'],'serial')
        self.assertEqual(len(self.launch.state['recoveries']),1)

    def test_single_worker_oom_is_terminal_without_model_or_budget_change(self):
        self.active_entry('g5_struct',pid=29001)
        entry=self.launch.state['variants']['g5_struct'];entry['launch_mode']='serial'
        Path(entry['log']).write_text('CUDA out of memory\n');self.alive.pop(29001)
        self.launch.state['scheduling']=dict(mode='serial')
        with self.assertRaisesRegex(RuntimeError,'Single-worker OOM'):self.launch.formal(False)
        self.assertEqual(entry['status'],'FAILED');self.assertFalse(self.children)

    def test_runner_stop_after_complete_window_retains_attempt_counts_without_next_visit(self):
        model=fresh_model('g5_struct');processor=DataProcessor('001-rope')
        config=protocol.configuration('data','depth','source','g5_struct',.01)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        raw=first(synthetic_batch(24,15,1));items=[];pending=[];visits=[]
        def stream():
            for index in range(2):visits.append(index);yield raw
        before=state_hash(model.state_dict())
        with self.assertRaises(runner.WindowBoundaryStop):
            runner.train_epoch(model,optimizer,stream(),processor,runner.CachedRenderer(GaussianRenderer(raster)),
                config,'cpu',[[2,73],[2,73]],record=items.append,pending=lambda *v:pending.append(v),stop_requested=lambda:True)
        self.assertEqual(visits,[0]);self.assertEqual(len(items),1);self.assertTrue(items[0]['successful_update'])
        self.assertEqual(pending,[(1,0,True),(1,1,False)])
        self.assertNotEqual(before,state_hash(model.state_dict()))


if __name__=='__main__':unittest.main()
