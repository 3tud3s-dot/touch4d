"""Frozen manual boundaries and deterministic experiment sampling."""
import hashlib,json,random
from pathlib import Path
from scripts.dataloader.splits import SPLITS,window_starts
BOUNDARIES=((82,200),(70,140),(53,240),(62,252),(151,303),(104,263),(105,193),(60,181),(101,181),(83,242))
COUNTS={'train':709,'val':52,'test':51}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def labels(path):
 a=json.loads(Path(path).read_text(encoding='utf-8'))
 if a['scene']!='001-rope' or a['boundary_convention']!='half_open':raise ValueError('Wrong annotation contract')
 for i,(s,e) in enumerate(BOUNDARIES):
  row=a['episodes'][str(i)]
  if (row['review_status']!='MANUAL_CONFIRMED' or (row['contact_start'],row['contact_end_exclusive'])!=(s,e)
      or not 0<=s<e<=row['total_frames']):raise ValueError('Frozen annotation mismatch')
 return a

def keys(annotation,split):
 out=[]
 for ep in SPLITS['001-rope'][split]:
  a=annotation['episodes'][str(ep)];s,e=BOUNDARIES[ep]
  out.extend((ep,t) for t in window_starts(a['total_frames'],split,10,1) if s<=t and t+10<e)
 if len(out)!=COUNTS[split]:raise ValueError('Contact window count mismatch')
 return out

def representative(windows):
 return [sorted(k for k in windows if k[0]==ep)[len([k for k in windows if k[0]==ep])//2] for ep in sorted({k[0] for k in windows})]

def sequence(windows,sampling,updates,seed=42):
 r=random.Random(seed);windows=sorted(windows);out=[]
 if sampling=='balanced':
  groups={ep:[k for k in windows if k[0]==ep] for ep in sorted({k[0] for k in windows})}
  for _ in range(updates):out.append(r.choice(groups[r.choice(list(groups))]))
 elif sampling=='window':
  while len(out)<updates:
   epoch=windows.copy();r.shuffle(epoch);out.extend(epoch)
 else:raise ValueError('Unknown sampling')
 return out[:updates]

def matrix():
 return {f'S{i}':dict(loss='cd' if i%2 else 'squared',sampling='balanced' if i>=7 else 'window',episodes=[1] if i<3 else [5] if i<5 else [1,2,4,5,7,8],updates=800,eval_every=200,seed=42) for i in range(1,9)}

def select_candidate(scores):
 # Preregistered: both episode CDs and common render MAE no worse than S5;
 # rank eligible by episode-equal CD then common joint. Otherwise fixed S6.
 base=scores['S5'];eligible=[]
 for name in ('S6','S7','S8'):
  x=scores[name]
  if (all(x['episodes'][e]['mean_cd_mm']<=base['episodes'][e]['mean_cd_mm'] for e in ('3','6'))
      and x['equal']['mean_cd_mm']<base['equal']['mean_cd_mm']
      and x['equal']['render_score']<=base['equal']['render_score']):eligible.append(name)
 winner=min(eligible,key=lambda n:(scores[n]['equal']['mean_cd_mm'],scores[n]['equal']['common_joint'])) if eligible else 'S6'
 return dict(winner=winner,eligible=eligible,fallback=not eligible,rule='No worse ep3/ep6 CD and render score than S5; lowest equal CD, tie common joint. Else S6 exploratory. No seed43 repeats.')


def validate_resume(state,config,source_sha,run_id):
 old=state.get('run_state') or {};steps=state.get('successful_steps')
 if state.get('source_sha')!=source_sha or old.get('config')!=config or old.get('run_id')!=run_id:raise ValueError('Unsafe resume source/config/run conflict')
 if type(steps) is not int or not 0<=steps<=config['updates']:raise ValueError('Invalid step count')
 if old.get('successful_updates')!=steps:raise ValueError('Checkpoint and ledger update counts disagree')
 if [r['update'] for r in old.get('updates',[])]!=list(range(1,steps+1)):raise ValueError('Invalid successful step ledger')
 return old
