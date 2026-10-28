"""Offline HTML and evidence-linked Chinese interpretation; no external assets."""
import argparse
import csv
import html
import json
from pathlib import Path


def generate(root):
    rows=list(csv.DictReader((root/'results/motion_field_comparison.csv').open(encoding='utf8')))
    region=list(csv.DictReader((root/'results/near_far_analysis.csv').open(encoding='utf8')))
    gates=json.loads((root/'provenance/mapping_checks.json').read_text())
    lines=['# V4 Normal–Oracle 运动场审计 V1','',
    '本轮只读取既有 Oracle 数组、Anchor 控制变量和元数据；CPU 分析，无网络实例化、无优化器、无反向、无新拟合。仅 ep7/start105（Train）与 ep3/start135（Val诊断），H5/H10/H15。',
    '', '## 数据与等价性',
    '- 源结果：v4-d0-oracle-capacity-v1/run-v2，源码基准753aaac；H15网络训练源码60b7e60，Epoch6 best_joint。checkpoint SHA256：1fa35ddcc4c62ca359d8491d77bdd7888152a795ffae49b07a8480e190ccd05f。',
    '- 六份NPZ的初态/Copy完全一致，同一窗口的初态点序和Near/Far索引跨H一致；所有被读取文件先后SHA256不变。原始有效点导出过程由基准run.py核对，未使用GT数组行号构造运动。',
    f'- 生产FPS/RBF由保存的世界坐标逆归一化后重建；54个Normal/Oracle映射核验最大误差 {max(e["max_mapping_error_mm"] for g in gates for e in g["mapping"]):.8f} mm。核验门槛事先固定0.0001 mm，仅容纳float32导出往返误差，非研究质量阈值；并非声称浮点bitwise一致。',
    '- Oracle控制量来自PT文件，转换为世界米；比较用相同初始GS行身份。所有Oracle都是各H独立使用未来GT拟合的可行解，不是唯一材料运动真值。',
    '', '## 指标定义',
    '- GS位移u=X−X0；向量RMS=sqrt(mean(||u||²))。幅度比=Normal RMS/Oracle RMS；方向余弦逐点等权，两个模长均大于1e-6 m才定义；未定义值在CSV留空、热图灰色。该0.001 mm仅为数值方向阈值，不是接触或有效运动标签。',
    '- 平移项为mean(u)；非刚体残差u−mean(u)。差异平方精确分解为平移差平方与残差差平方。残差仍可包含刚性旋转，因此不能全部解释为柔性形变。',
    '- 幅度/方向误差分解：||n−o||²=(||n||−||o||)²+2(||n||||o||−n·o)。报告各项占总体平方差的比例；这不是因果贡献率。',
    '- Near/Far直接使用上一实验固定初态到有效操作手Controller距离最小/最大各25%的索引，其余Middle；不是接触标签，也不是按当前运动重新分组。',
    '- 初态16-NN沿生产实现（含self槽，之后排除self及归一化长度≤1e-8的边）。局部运动差为u_i−u_j，距离比为||X_i−X_j||/||X0_i−X0_j||。它们是结构失真代理，不是真实材料应变。',
    '- 三个层级分别报告：Anchor累计控制变量D、固定W映射后的全部GS位移、映射后FPS位置的实际位移。D通常不等于(WD)[FPS]。',
    '', '## 完整主要对照（best，全部初始化）',
    'best按各Oracle原始优化目标选取，CD与Joint不互换选模指标；last在CSV/HTML中独立保留。23/24组best在step300；ep7/H10 Joint/Zero的best在step296，与last的GS位移差RMS约0.0236 mm（最大0.0944 mm）。不能据此证明收敛。',
    '', '|窗口|H|Oracle/初始化|差异RMS mm|方向中位角°|平移差mm|残差差mm|方向项占平方差|',
    '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['level']=='gs' and r['snapshot']=='best':
            keys=['difference_rms_mm','direction_angle_median_deg','mean_translation_difference_mm','residual_difference_rms_mm','direction_error_energy_fraction']
            lines.append('|'+ '|'.join([r['window'],r['horizon'],r['oracle']+'/'+r['initialization']]+[f'{float(r[k]):.3f}' for k in keys])+'|')
    lines += ['', '## Q1：ep3差异在哪里，是方向还是幅度？',
    'H15全部四个Oracle都显示大方向差异：Near中位角100.6–106.2°，Far120.8–124.8°。Near Normal RMS15.10 mm，Oracle16.97–17.61 mm，幅度近似却方向不同；Far Normal6.70 mm，Oracle15.99–18.89 mm，方向与幅度都不匹配。全场方向项占平方差87.1%–88.8%，说明相对这些Oracle解，主要不是单纯运动放大/缩小。',
    'H15最大差异的前10%点主要位于初态高X/Z的弯曲段与Near—Middle过渡：各解该区域质心约世界X=301–318、Y=87–107、Z=215–224 mm；不是一个独立材料部位标注。该Top10%中Near占13.7%–50.7%，Far占0%–1.2%，其余大多是Middle。空间位置随Oracle初始化/目标有所差异，不能只以一张热图命名唯一病灶。',
    '以Joint/Zero H15为例，Near平均运动Normal=(+3.11,+3.85,−9.60)mm，Oracle=(−5.36,−16.27,−2.75)mm；Far从(+0.81,+3.51,+2.00)变为(−9.33,−15.34,−4.25)mm。Normal的Near向下、Far略向上及整体Y方向，与该Oracle协调的负Y/负X运动不同。',
    '这既有整体平移偏差，也有局部空间差：四个Oracle对应的平均平移差15.64–21.42 mm；扣除各自平均平移后仍有12.60–17.40 mm差异。初态邻边运动差RMS由Normal2.340 mm降为Oracle0.369–1.261 mm；P95 |距离比−1|由1.090降至0.112–0.559。',
    '', '## Q2：为什么ep7可以有很强的Near运动？',
    'ep7的“大幅Near、小幅Far”分配在两类Oracle及两种初始化中均保留：H15 Normal Near29.93/Far2.97 mm，Oracle Near31.76–32.62/Far3.32–4.09 mm。Near方向中位差仅8.0–11.3°；同样强的近端运动在这个窗口能与较低几何误差并存。Far角度29–41°仍存在，但对应低幅度，不能与ep3的大幅反向运动等量解释。',
    'ep3的Oracle则要求Far运动接近Near的量级，且总体方向不同。故不能规定Near一律减小、Far一律静止，也不能把“强近端响应”直接视为错误。ep7仍有局部修正需要：H15全场差异6.15–6.55 mm，去平移残差5.57–5.90 mm，Normal P95邻边失真0.514，Oracle0.225–0.327。',
    '这是两个固定窗口的可行运动场比较，不是物理接触机制的因果证明；Near不等于真实夹持位置，Oracle也不等于材料轨迹。',
    '', '## Q3：H5→H10→H15如何变化？',
    'ep3全场Normal–Oracle差异RMS：H5为8.96–9.22 mm，H10为17.02–18.30 mm，H15为21.88–25.50 mm。H5方向中位角已为104–117°、方向项占平方差71%–76%；说明在最早保存的H5已存在明显错误空间分配，之后差距扩大。不能臆测H1–H4的Oracle场，也不能只由增长就认定自回归是根因。',
    'ep7相应为2.71–3.07、4.63–5.59、6.15–6.55 mm；H15全场方向中位角11.7–14.4°，主运动方向相对更接近。每个Oracle horizon独立拟合，没有受约束的连续Oracle时间动力学。',
    '', '## Q4：对下一轮设计的具体约束与未知项',
    '1. 后续验收需同时检查有符号XYZ、幅度、Near/Middle/Far分配与局部残差；不能只追求位移更大、近端Attention更强或更平滑。保留ep7的局部大运动能力，同时避免ep3的跨区域方向相反和Far不足。',
    '2. 关注区域间协调而非强迫邻点同动：ep7真实任务可需要强运动梯度，ep3可行Oracle更接近共同方向。当前证据不足以指定新的Attention形式、距离带宽、阈值或物理loss。',
    '3. Motion Head必须区分累计D控制量与实际FPS移动。H15 Joint/Zero：ep7控制量差RMS15.81 mm、GS差6.40 mm；ep3分别34.44/25.50 mm。固定RBF对控制差的映射依赖空间模式；不能用控制量差直接声称某Anchor对应材料粒子错误。',
    '4. 差异可定位输出端空间分配问题，却不能区分Attention输入、Motion Head函数、递推反馈或优化各自的因果作用。需要未来受控干预才能定位，当前不执行新实验。不能把Oracle当唯一监督标签或跨episode泛化证据。',
    '', '## 图像与可复核文件',
    '- 两张主图：figures/ep3_start135_motion_field_summary.png、figures/ep7_start105_motion_field_summary.png。三行H5/H10/H15，五列Normal与全部四个Oracle；3D固定视角。',
    '- 每个H和best/last：*_fields.png（三维向量/差值热图/角度热图）、*_anchors.png（控制量与RBF映射）、*_near_far.png（幅度ECDF、角度ECDF、带符号分量）；*_horizon_trends.png保留两种初始化。',
    '- 全部图使用同一窗口跨H/解的世界坐标范围、差值颜色尺度和真实长度箭头×1；角度范围0–180°。不同窗口的坐标框分别记录于provenance/plot_scales.json。图中文字用英文避免字体缺字。',
    '- results/motion_field_comparison.csv：144行，三个层级、两窗口、三H、两Loss、两初值、best/last；near_far_analysis.csv：192行；neighbor_structure.csv：54行。',
    '- fields/保存派生向量、固定索引与重建RBF，可复核；provenance/保存输入SHA256、映射误差、坐标及指标约定；logs/保存命令和测试。',
    '', '## 限制',
    '不是训练，不重新优化，不调用模型。未来GT只存在于既有Oracle结果中。Oracle未必充分收敛或唯一，独立重建和固定模板亦会影响解；对照差不能直接等同于真实运动误差。原checkpoint及Oracle文件未改动。浏览器交互与文件完整性验收以provenance/verification.json的实际记录为准。']
    text='\n'.join(lines)+'\n';(root/'final_report.md').write_text(text,encoding='utf8')
    # Plain selectable report text plus full offline image navigation, no fetch/CDN.
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Normal–Oracle Motion Field</title>
<style>body{font:16px system-ui;background:#f4f6f8;color:#263443;margin:24px}main{max-width:1500px;margin:auto}select{font-size:16px;padding:8px}img{width:100%;background:white;margin:12px 0}pre{white-space:pre-wrap;line-height:1.65;background:white;padding:24px}a{color:#236e9b}.tools{position:sticky;top:0;background:#edf3f8;padding:12px;z-index:2}</style><main>
<h1>V4 Normal–Oracle 运动场审计</h1><p>仅离线CPU分析。Normal + CD/Joint × Zero/Normal初始化；所有图采用固定坐标与真实长度箭头。Oracle不是唯一真实运动标签。</p>
<div class="tools"><select id="window"><option>ep3_start135</option><option>ep7_start105</option></select>
<select id="h"><option>5</option><option>10</option><option selected>15</option></select>
<select id="snapshot"><option>best</option><option>last</option></select></div>
<h2>H5 → H10 → H15 主图（best）</h2><a id="summaryLink"><img id="summary"></a><h2>选定 Horizon：三维向量／差值／方向角</h2><a id="fieldsLink"><img id="fields"></a>
<h2>Anchor控制变量与映射后GS位移</h2><a id="anchorsLink"><img id="anchors"></a><h2>Near/Far分布</h2><a id="near_farLink"><img id="near_far"></a><h2>跨H变化</h2><a id="trendsLink"><img id="trends"></a>
<p><a href="results/motion_field_comparison.csv">全量运动场CSV</a> · <a href="results/near_far_analysis.csv">Near/Far CSV</a> · <a href="results/neighbor_structure.csv">邻域结构CSV</a> · <a href="final_report.md">Markdown报告</a></p><h2>研究报告</h2><pre>'''+html.escape(text)+'''</pre>
<script>function update(){const w=document.getElementById('window').value,h=document.getElementById('h').value.padStart(2,'0'),s=document.getElementById('snapshot').value;
const paths={summary:`figures/${w}_motion_field_summary.png`,trends:`figures/${w}_horizon_trends.png`};for(const k of ['fields','anchors','near_far'])paths[k]=`figures/${w}_h${h}_${s}_${k}.png`;
for(const [id,p] of Object.entries(paths)){document.getElementById(id).src=p;document.getElementById(id+'Link').href=p;}}
for(const id of ['window','h','snapshot'])document.getElementById(id).addEventListener('change',update);update();</script></main></html>'''
    (root/'report.html').write_text(page,encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);generate(p.parse_args().root)
