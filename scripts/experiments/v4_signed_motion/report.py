import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np


def make(root):
    audit=json.loads((root/'results/coordinate_time_audit.json').read_text())
    objects=list(csv.DictReader((root/'results/object_motion.csv').open(encoding='utf8')))
    cs=list(csv.DictReader((root/'results/controller_motion.csv').open(encoding='utf8')))
    tools=list(csv.DictReader((root/'results/tool_motion_alignment.csv').open(encoding='utf8')))
    lines=['# Controller–Object Signed Motion Audit V1','',
    '范围：仅Train ep7/start105和Val ep3/start135。冻结H15 Epoch6 best_joint；使用之前保存的原始Batch、逐步Trace与Motion Decoder实际输入，不运行模型前向，不创建优化器、不反向、不重新拟合Oracle。只在CPU重放DataProcessor及确定性features()，分析保存张量。',
    '', '## Q1：是否发现确定性的输入、时间、坐标或符号错误？',
    '**在已检查的数据链路中未发现确定性实现Bug。** 这不是对外部物理标定绝对正确的证明。',
    '', '|核验项|实际证据|结论|','|---|---|---|',
    '|原始Controller与Action|用源controller.npz的T_worlds/openings重新执行controller_arrays；两窗口全部15个条件帧点坐标/Mask bitwise一致。action_conditions与保存Batch一致|通过|',
    '|手与指索引|固定网络left/right；ep3原槽right=0、left不存在；ep7原槽right=0、left=1。手内finger0=0..14、finger1=15..29|通过|',
    '|世界→归一化|统一正标量scale，(XYZ−center)/scale；位移乘scale恢复、不加center、不交换轴|通过|',
    '|真实输入重放|prepare_batch输出的所有输入与既有zero-tool输入快照bitwise一致；该旧干预只在Motion Decoder hook置零，不改输入|通过|',
    '|A0与12D区别|features13[3:6]为当前−初态；12D为最后一层四头权重×当前−上一已观测Controller，不取头平均、不丢弃null质量|通过|',
    '|Mask|action/controller有效性、操作手、finite与端点有效性均核对；无效token为0，无效工具attention为0。ep3右30点，ep7双手60点；真实Batch没有GS padding|通过，padding另有专项测试|',
    '|Decoder实际输入|151D中0:128为Attention输出、128:140为12D、140:143为Anchor历史、143:146为前次累计D、146:150为头历史质量、150为历史有效标志|逐步核验通过|',
    '|时间|H1读t预测t+1；H15读t+14预测t+15；原始帧号、timestamps、触觉过去最近采样/age全部核对|通过|',
    '|外部物理校准|没有独立重新测量机器人世界轴方向、设备时钟同步或实体标定|未验证，不能由软件一致性替代|',
    '', '## 数值与时间证据',
    f"归一化参数：center={audit['processor']['center']} m，scale={audit['processor']['scale']} m。XYZ轴不置换；旋转矩阵不进行位置归一化；开口除同一scale。",
    '', '|窗口|条件帧→目标帧|Controller往返最大误差mm|13D最大差|12D最大差（归一化）|预测步时长ms|',
    '|---|---|---:|---:|---:|---|']
    for w in audit['windows']:
        s=w['steps'];lines.append(f"|{w['window']}|{s[0]['condition_frame']}–{s[-1]['condition_frame']} → {s[0]['target_frame']}–{s[-1]['target_frame']}|{w['roundtrip_error_mm']:.8f}|{max(q['feature_error'] for q in s):.3g}|{max(q['tool_error_normalized'] for q in s):.3g}|{w['prediction_dt_ms_min']:.3f}–{w['prediction_dt_ms_max']:.3f}|")
    lines += ['', '输入/输出都是世界空间相关量，但不是同一物理对象。Controller是夹爪采样点，物体可弯曲/旋转/滑移，不能强制同向。诊断余弦比较各手/各指的平均位移与区域平均位移，任一模长≤0.001 mm留空，不当作反向。节点位移RMS另外报告，避免平均抵消。',
    '', '### 既有设计限制：开口裁剪',
    'ep7：右手开口30.008–31.982 mm，全部15帧低于几何映射40 mm下限；左手33.280–41.215 mm，其中10帧低于下限。Controller几何夹到[40,112] mm，Action State保留原始开口。此行为与生产实现一致，不是此次发现的代码错配。它限制了Controller几何对超范围开口的表达；本轮不改变。ep3右手46.833–48.305 mm，无裁剪，因此不能把ep3反向归因于这项限制。',
    '12D使用相邻帧的位移，不除时间间隔，不能称为速度。实际步时长约32–36 ms，有轻微不均匀；本轮没有实验去判断它对预测的影响。',
    '', '## Q2：ep3的正Y预测与Controller负Y是什么关系？',
    'ep3右手观测早期曾有小幅正Y：H2累计+0.707 mm，H3累计+1.084 mm。相邻Y从H4开始转负；累计Y到H5为−0.046 mm、H10为−7.427 mm、H15为−14.687 mm。对应条件帧是135–149，而目标是136–150。',
    'Normal全场平均Y在H1已经为+0.221 mm，之后始终为正，H5/H10/H15为+3.001/+4.048/+4.795 mm；增长不是单调的。H1缺少运动历史，只能说明存在初态/时间驱动输出，不能据此判错；最早Oracle方向对照仍只有H5。',
    '**负向信息并未在显式12D通路丢失或变号。** 四头Tool Motion的Anchor平均Y从H4也均为负。H15分别约−0.3432、−0.2311、−0.0348、−0.2448 mm，与实际Motion Decoder输入一致。它们是被Attention加权、未移除null的位移，不应与14步累计Controller位移直接比大小。',
    'H15物体Normal全场平均向量与右手累计位移余弦−0.961；Joint/Zero与Joint/Normal Oracle分别+0.837/+0.779。四个Oracle的全场Y为−9.795、−9.872、−15.623、−14.010 mm，均为负。Oracle不是材料轨迹，这只说明存在低几何误差、方向分配不同的可行解。',
    '现象与“早期正Y预测未充分随后续负向动作更新”相容，但没有证明模型记住早期方向。Normal仍能通过绝对Controller、A0、Hand State及128D Attention特征接收条件；当前分析没有分离这些通路的因果作用。',
    '', '## Q3：ep7与ep3有何不同？',
    'ep7两手都持续上移，但X不同，必须分手比较。H15左手Controller累计(−16.330,−1.281,+28.757) mm，右手(+2.523,−1.941,+29.066) mm；Normal Near平均(−6.110,−1.293,+24.280) mm。Near与两手累计方向余弦分别0.964/0.945，强局部上移与两手主Z方向及Oracle场相容。不能由此认定已学会物理因果。',
    'ep3只有右手，H15累计(+1.819,−14.687,−0.510) mm。Normal Near=(+3.108,+3.847,−9.599)，Middle=(−4.574,+5.910,+1.718)，Far=(+0.814,+3.512,+1.998) mm：三个区域均保持正Y，同时Near偏向下、其余偏向上。它并非一个统一的坐标负号错误，而是输出空间分配与可行Oracle场不一致。',
    'H2时ep3 Near的累计方向余弦已经可以为负，但Controller与物体本来不必同向，不能把首次负余弦当作确定错误起点。需要与已有Oracle和GT几何误差一起解释；本轮没有H1–H4 Oracle。',
    '', '## Q4：优先条件编码、Attention、Motion Head还是AR？',
    '**工程优先级：优先追踪“已收到正确负向显式运动后，Motion Head/状态反馈为何仍输出正向累计位移”。这是下一步定位位置，不是已证实的根因。** 坐标/索引/掩码和A0构造已有直接通过证据；12D在最后Attention之后仍保留负Y，故不支持先修复假想的轴符号翻转。',
    '仍不能区分：①128D Attention语义特征是否有效表达条件；②Motion MLP是否忽略/抵消12D或受到时间/GS特征偏置；③前次累计D和Anchor历史是否通过AR放大偏差；④训练优化是否形成此映射。Head输入中还包含其他项，12D符号正确不能证明整个Attention已正常利用动作。Null权重是退让质量，不是接触概率，也不能单凭较大就认定故障。',
    'H1已有非零输出、H5已有较明显方向差、随后扩大会使“纯末期AR累积”不足以单独解释现象，但仍不能排除早期映射误差与AR共同作用。本轮不进行干预或新架构实验。',
    '', '## H15逐手/区域数值（mm）',
    '|窗口|对象/区域|X|Y|Z|向量RMS|','|---|---|---:|---:|---:|---:|']
    for w in ('ep3_start135','ep7_start105'):
        rr=[r for r in cs if r['window']==w and r['horizon']=='15' and r['finger']=='all' and r['kind']=='cumulative' and r['count']!='0']
        for r in rr:lines.append('|'+ '|'.join([w,'Controller '+r['hand']]+[f'{float(r[k]):.3f}' for k in ['mean_x_mm','mean_y_mm','mean_z_mm','rms_mm']])+'|')
        for r in objects:
            if r['window']==w and r['horizon']=='15' and (r['method']=='normal' or r['region']=='all') and not r['method'].endswith('_last'):
                lines.append('|'+ '|'.join([w,r['method']+'/'+r['region']]+[f'{float(r[k]):.3f}' for k in ['mean_x_mm','mean_y_mm','mean_z_mm','rms_mm']])+'|')
    lines += ['', '## 源码位置与复核入口']
    for file,data in audit['source_locations'].items():
        lines.append('- '+file+': '+', '.join(str(q['line']) for q in data['locations']))
    lines += ['', '## 文件、测试和限制',
    '- controller_motion.csv：360行，各手/各指、累计/相邻均保留；object_motion.csv：312行，Normal H1–15，Oracle仅H5/H10/H15且保留两初值、best/last。signed_direction_comparison.csv：2772行，无定义方向留空。',
    '- tool_motion_alignment.csv保留四头与左右手贡献，不用双手平均构造所谓主要方向；pathway_values.csv提供各实际输入段的数值范数。Norm在不同语义段/维数间不可直接用作重要性结论。',
    '- coordinate_time_audit.json包含逐H帧号、Mask、数值误差、源码位置；provenance保存读取文件哈希、源metadata/Controller/timestamps和checkpoint索引。',
    '- 真实Batch无GS padding；专项合成测试检查无效Controller端点、整手屏蔽、后续条件局部性、坐标符号、槽位重排、首帧零历史、四头/null及帧号。仅使用无梯度CPU操作。',
    '- 初次审计工具错误日志保留：移除深度字段却保留元信息造成契约拒绝，现已保留完整Batch；将两次float32导出误要求bitwise相等，现按已有0.0001 mm往返门槛核对；新刚体测试按1 ULP世界位置检查并严格核对符号。均非生产模型修复。',
    '- 图片为保存的世界坐标几何/统计，非新增GS渲染；三维箭头真实长度×1。Oracle散点不连线，不构造虚假的连续轨迹。全局残差仍可包含刚性旋转，不应全部称作材料形变。',
    '- 物理世界标定、原始硬件时钟同步和真正接触状态不由本轮软件一致性证明。浏览器交互未声称实测，离线页面及图像路径由verification.json记录。']
    text='\n'.join(lines)+'\n';(root/'final_report.md').write_text(text,encoding='utf8')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Controller–Object Signed Motion Audit</title><style>body{font:16px system-ui;color:#253443;background:#f4f6f8;margin:24px}main{max-width:1500px;margin:auto}select{font-size:17px;padding:8px}img{width:100%;background:white}pre{white-space:pre-wrap;line-height:1.65;background:white;padding:24px}.toolbar{position:sticky;top:0;background:#eaf0f7;padding:12px}a{color:#246c9f}</style><main><h1>Controller–Object 有符号运动审计</h1><p>只读既有Trace，未训练、未前向推理。Oracle仅显示已有H5/H10/H15。双手分别报告。</p><div class="toolbar"><select id="window"><option>ep3_start135</option><option>ep7_start105</option></select></div>'''
    for kind,title in [('signed_motion_summary','有符号运动主图'),('world_directions','世界坐标方向对照'),('near_middle_far','Near / Middle / Far'),('tool12_alignment','实际12D与时间对齐'),('condition_pathway','生产条件路径')]:page+=f'<h2>{title}</h2><a id="{kind}Link"><img id="{kind}"></a>'
    page+='<p>'+ ' · '.join(f'<a href="results/{n}.csv">{n}.csv</a>' for n in ('controller_motion','object_motion','signed_direction_comparison','tool_motion_alignment','pathway_values'))+'</p><pre>'+html.escape(text)+'</pre>'
    page+='''<script>function update(){const w=document.getElementById('window').value;for(const k of ['signed_motion_summary','world_directions','near_middle_far','tool12_alignment','condition_pathway']){const p=`figures/${w}_${k}.png`;document.getElementById(k).src=p;document.getElementById(k+'Link').href=p;}}document.getElementById('window').addEventListener('change',update);update();</script></main></html>'''
    (root/'report.html').write_text(page,encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);make(p.parse_args().root)
