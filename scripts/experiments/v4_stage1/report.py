"""Offline report from one immutable V4 run. No inference or training here."""
import html
import json
from pathlib import Path


def curve(rows, key, title):
    values = [float(row[key]) for row in rows]
    low, high = min(values), max(values)
    span = max(high-low, 1e-12)
    points = ' '.join(f'{45+row["update"]/400*650:.2f},{190-(value-low)/span*150:.2f}' for row, value in zip(rows, values))
    return f'<figure><figcaption>{html.escape(title)} ({low:.5g}–{high:.5g})</figcaption><svg viewBox="0 0 720 220"><path d="M45 25V190H695" fill="none" stroke="#999"/><polyline points="{points}" fill="none" stroke="#1766ba" stroke-width="2"/><text x="40" y="212">0</text><text x="660" y="212">400 updates</text></svg></figure>'


def write_report(directory):
    directory = Path(directory)
    data = json.loads((directory/'metrics.json').read_text())
    evaluations = data['evaluations']
    copy = data['copy_initial']
    best = next(r for r in evaluations if r['update'] == data['lowest_cd_update'])
    last = next(r for r in evaluations if r['update'] == 400)
    rows = ''
    for label, result in (('Copy Initial', copy), (f'Lowest CD / update {data["lowest_cd_update"]}', best), ('Last 400', last)):
        rows += f'<tr><td>{label}</td><td>{result["mean_cd_mm"]:.6f}</td><td>{result["mean_rgb"]:.6f}</td><td>{result["mean_alpha"]:.6f}</td>'
        rows += ''.join(f'<td>{result["cd_mm"][h-1]:.5f}</td>' for h in (1, 5, 10))+'</tr>'
    charts = ''.join(curve(evaluations, key, title) for key, title in
                     (('mean_cd_mm', 'CD (mm), evaluated every 20 updates'), ('mean_rgb', 'RGB MAE'),
                      ('mean_alpha', 'Alpha MAE'), ('loss', 'Joint objective')))
    config = html.escape(json.dumps(data['config'], indent=2))
    report = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V4 H10 单窗口验收</title>
<style>body{{font:15px system-ui;background:#f4f6f9;color:#182c42;margin:24px}}table{{border-collapse:collapse;background:white}}td,th{{padding:9px;border:1px solid #cdd5df}}.charts{{display:grid;grid-template-columns:1fr 1fr;max-width:1250px}}figure{{background:white;padding:12px;margin:8px}}svg{{width:100%}}iframe{{width:100%;height:830px;border:1px solid #bdc9d5;background:white}}button,select{{padding:9px;margin:6px}}pre{{white-space:pre-wrap}}small{{color:#465b6f}}</style>
<h1>V4 · 001-rope ep2 · initial 73 → targets 74–83</h1>
<p>H10 / Δ1，条件 73–82，右夹爪。单窗口 400 次成功更新；这是训练窗口拟合，不是留出集泛化或动作因果性证明。</p>
<table><tr><th>结果</th><th>Mean CD mm</th><th>RGB MAE</th><th>Alpha MAE</th><th>h1 CD</th><th>h5 CD</th><th>h10 CD</th></tr>{rows}</table>
<p>最低 CD 为诊断快照；联合指标最优 update {data['best_joint_update']} 单独保存为 best_joint.pt。</p>
<div class="charts">{charts}</div>
<h2>三相机 RGB / Alpha 与三维几何</h2>
<select id="snapshot"><option value="update_000">Initial / 0</option><option value="update_200">Update 200</option><option value="update_400">Update 400</option><option value="lowest_cd">Lowest CD</option></select>
<select id="kind"><option value="index.html">RGB / Alpha / Depth</option><option value="geometry.html">Centers / Ellipsoids / Trajectories</option></select>
<a id="direct" target="_blank">单独打开</a><iframe id="viewer" title="同步十帧诊断"></iframe>
<script>function show(){{let u='visualizations/'+document.getElementById('snapshot').value+'/'+document.getElementById('kind').value;document.getElementById('viewer').src=u;document.getElementById('direct').href=u}}document.getElementById('snapshot').onchange=show;document.getElementById('kind').onchange=show;show();</script>
<p>GT 为独立重建点集，没有跨帧固定 Gaussian ID；质心变化及 Copy CD 不等同于材料粒子位移。渲染监督只在原有有效深度区域比较。</p>
<details><summary>冻结配置与源码</summary><pre>{config}</pre></details>
<p><a href="metrics.json">完整数值、逐步 CD、Attention 和条件扰动</a> · <a href="training_metrics.csv">400 次训练日志 CSV</a> · <a href="manifest.json">窗口与数据指纹</a></p></html>'''
    (directory/'report.html').write_text(report, encoding='utf-8')
    summary = dict(copy=copy, lowest_cd=best, last_400=last, lowest_cd_update=data['lowest_cd_update'],
                   best_joint_update=data['best_joint_update'], successful_updates=data['successful_updates'],
                   condition_diagnostics=data['condition_diagnostics'])
    (directory/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory')
    write_report(parser.parse_args().directory)
