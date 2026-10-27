/* Pure boundary/export logic, shared by the offline UI and Node unit tests. */
(function(root){
  'use strict';
  const valid=(s,e,n)=>Number.isInteger(s)&&Number.isInteger(e)&&Number.isInteger(n)&&0<=s&&s<e&&e<=n;
  const empty=()=>({start:null,end:null,confirmed:false,notes:''});
  function stats(s,e,ts){
    if(!valid(s,e,ts.length))return null;
    return {before:s,contact:e-s,after:ts.length-e,h10:Math.max(0,e-s-10),
      observedSpan:(ts[e-1]-ts[s])/1e6,boundaryDuration:e<ts.length?(ts[e]-ts[s])/1e6:null};
  }
  function serialize(state,manifest){
    const episodes={};
    for(const item of manifest.episodes){
      const a=state[item.episode]||empty();const ok=a.confirmed===true&&item.status==='READY'&&valid(a.start,a.end,item.frames);
      episodes[item.episode]={contact_start:ok?a.start:null,contact_end_exclusive:ok?a.end:null,
        review_status:ok?'MANUAL_CONFIRMED':'UNREVIEWED',notes:a.notes||'',total_frames:item.frames,
        draft:ok?null:{contact_start:a.start,contact_end_exclusive:a.end}};
    }
    return {scene:'001-rope',boundary_convention:'half_open',annotation_method:'manual_two_cuts',episodes};
  }
  function text(data){return '001-rope contact boundaries\nFormat: episode: contact_start, contact_end_exclusive\n\n'+Object.entries(data.episodes).map(([ep,a])=>`ep${ep}: `+(a.review_status==='MANUAL_CONFIRMED'?`${a.contact_start}, ${a.contact_end_exclusive}`:'UNREVIEWED')).join('\n');}
  const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
  function csv(data){return '\ufeffepisode,contact_start,contact_end_exclusive,review_status,total_frames,notes\r\n'+Object.entries(data.episodes).map(([ep,a])=>[ep,a.contact_start,a.contact_end_exclusive,a.review_status,a.total_frames,a.notes].map(quote).join(',')).join('\r\n');}
  function restore(data,manifest){
    if(data.scene!=='001-rope'||data.boundary_convention!=='half_open'||data.annotation_method!=='manual_two_cuts'||!data.episodes)throw Error('标注来源或半开区间格式不匹配');
    const state={};
    for(const item of manifest.episodes){
      const a=data.episodes[item.episode];if(!a){state[item.episode]=empty();continue;}
      if(a.total_frames!=null&&a.total_frames!==item.frames)throw Error(`ep${item.episode} 总帧数不匹配`);
      if(!['MANUAL_CONFIRMED','UNREVIEWED'].includes(a.review_status))throw Error('未知 review_status');
      const confirmed=a.review_status==='MANUAL_CONFIRMED';
      const s=confirmed?a.contact_start:(a.draft?.contact_start??null),e=confirmed?a.contact_end_exclusive:(a.draft?.contact_end_exclusive??null);
      if(confirmed&&(!valid(s,e,item.frames)||item.status!=='READY'))throw Error(`ep${item.episode} 确认边界不合法`);
      for(const v of [s,e])if(v!==null&&(!Number.isInteger(v)||v<0||v>item.frames))throw Error('草稿帧号不合法');
      state[item.episode]={start:s,end:e,confirmed,notes:typeof a.notes==='string'?a.notes:''};
    }
    return state;
  }
  const api={valid,empty,stats,serialize,text,csv,restore};
  if(typeof module!=='undefined')module.exports=api;else root.ContactCore=api;
})(typeof window==='undefined'?{}:window);
