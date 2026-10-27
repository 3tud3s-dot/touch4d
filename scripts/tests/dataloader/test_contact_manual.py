import json
from pathlib import Path
import shutil
import subprocess
import unittest

from scripts.experiments.contact_manual.collect import boundary_stats,template


class ManualContactTest(unittest.TestCase):
    def test_h10_half_open(self):
        t=list(range(0,20000000,100000))
        a=boundary_stats(71,145,t)
        self.assertEqual((a['before'],a['contact'],a['after'],a['h10_windows']),(71,74,55,64))
        self.assertAlmostEqual(a['boundary_duration_s'],7.4)
        self.assertEqual(boundary_stats(0,10,t)['h10_windows'],0)
        self.assertEqual(boundary_stats(0,11,t)['h10_windows'],1)

    def test_terminal_duration_is_not_fabricated(self):
        s=boundary_stats(1,3,[0,100000,350000])
        self.assertIsNone(s['boundary_duration_s'])
        self.assertEqual(s['observed_span_s'],.25)

    def test_invalid_boundaries(self):
        for s,e in [(-1,2),(0,4),(2,2),(2,1),(1.2,2),(True,2)]:
            with self.assertRaises(ValueError):boundary_stats(s,e,[1,2,3])

    def test_template_has_no_suggested_labels(self):
        a=template(dict(episodes=[dict(episode=i,frames=191) for i in range(10)]))
        for e in a['episodes'].values():
            self.assertEqual(e['review_status'],'UNREVIEWED')
            self.assertIsNone(e['contact_start']);self.assertIsNone(e['contact_end_exclusive'])

    @unittest.skipUnless(shutil.which('node'),'Node required for shared browser core tests')
    def test_shared_javascript_export_validation_and_restore(self):
        core=Path(__file__).resolve().parents[2]/'experiments/contact_manual/core.js'
        js='''const assert=require('assert'),C=require(CORE);
const m={episodes:Array.from({length:10},(_,episode)=>({episode,frames:191,status:'READY'}))};
let s=Object.fromEntries(m.episodes.map(e=>[e.episode,C.empty()]));
assert.equal(C.serialize(s,m).episodes[1].review_status,'UNREVIEWED');
s[1]={start:71,end:145,confirmed:false,notes:'a,"b"\\nc'};
let d=C.serialize(s,m);assert.equal(d.episodes[1].contact_start,null);assert(C.text(d).includes('ep1: UNREVIEWED'));
s[1].confirmed=true;d=C.serialize(s,m);assert(C.text(d).includes('ep1: 71, 145'));assert(C.csv(d).includes('"1","71","145","MANUAL_CONFIRMED"'));
assert.deepStrictEqual(C.restore(d,m),s);assert.equal(C.stats(71,145,Array.from({length:191},(_,i)=>i*1000)).h10,64);
s[1].end=191;assert.equal(C.stats(71,191,Array.from({length:191},(_,i)=>i*1000)).boundaryDuration,null);
for(const [a,b] of [[71,71],[-1,145],[71,192],[71.5,145]]){s[1].start=a;s[1].end=b;assert.equal(C.serialize(s,m).episodes[1].review_status,'UNREVIEWED');}
d.episodes[1].contact_end_exclusive=999;assert.throws(()=>C.restore(d,m));
console.log('JS core passed');'''.replace('CORE',json.dumps(str(core)))
        subprocess.run(['node','-e',js],check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()
