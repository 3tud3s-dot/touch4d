"""Build the current self-contained HTML from one verified presentation bundle.

python -B -m scripts.visualization.build_demo --bundle assets/current/demo_bundle.zip --output /path/comparison.html
No historical worktree, renderer HTML, network connection or training is needed.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import zipfile


def canonical_html_bytes(html):
    """Match the original delivered UTF-8/CRLF HTML on every host platform."""
    return html.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\r\n').encode('utf-8')


def build(bundle, output, *, overwrite=False):
    bundle, output = Path(bundle), Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    source = Path(__file__).resolve().parent
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['schema'] != 'touch4d-demo-bundle-v1':
            raise ValueError('Unsupported presentation bundle')
        for name, expected in manifest['files_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'Presentation payload hash mismatch: {name}')

        def read(name):
            return archive.read(name).decode('utf-8').replace('\r\n', '\n')

        html = (source / 'template.html').read_text(encoding='utf-8')
        html = html.replace('<link rel="stylesheet" href="styles.css">',
                            '<style>' + (source / 'styles.css').read_text(encoding='utf-8') + '</style>')
        html = re.sub(r'\s*<script defer src="[^"]+"></script>', '', html)
        for name in re.findall(r"--rgb-image:url\('([^']+)'\)", html):
            encoded = base64.b64encode(archive.read(name)).decode('ascii')
            html = html.replace(f"--rgb-image:url('{name}')",
                                f"--rgb-image:url('data:image/png;base64,{encoded}')")
        rgb = json.loads(read('rgb_sequences.js').split('=', 1)[1].rstrip(';\r\n'))
        for case in rgb['cases']:
            for view in case['views']:
                for frame in view['frames']:
                    encoded = base64.b64encode(archive.read(frame['path'])).decode('ascii')
                    frame['path'] = 'data:image/webp;base64,' + encoded
        scripts = ['<script>' + read('metrics.js') + '</script>',
                   '<script>window.DEMO_RGB=' + json.dumps(rgb, separators=(',', ':')) + ';</script>',
                   '<script>' + (source / 'gaussian_viewer.js').read_text(encoding='utf-8') + '</script>']
        for key in ['train', 'generalization']:
            data = json.loads(read(f'assets/3d/{key}.js').split(']=', 1)[1][:-1])
            scripts.append(f'<script id="scene-data-{key}" type="application/json">'
                           + json.dumps(data, separators=(',', ':')) + '</script>')
        scripts.append('<script>' + (source / 'app.js').read_text(encoding='utf-8') + '</script>')
        html = html.replace('</body>', '\n'.join(scripts) + '\n</body>')
        encoded = base64.b64encode(archive.read('provenance.json')).decode('ascii')
        html = html.replace('href="provenance.json"',
                            f'download="provenance.json" href="data:application/json;base64,{encoded}"')
    if re.search(r'<(?:script|link)[^>]+(?:src|href)=', html):
        raise ValueError('Standalone HTML still depends on an external script or stylesheet')
    if any(not value.startswith('data:') for value in re.findall(r"--rgb-image:url\('([^']+)'\)", html)):
        raise ValueError('Standalone HTML still depends on external images')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_html_bytes(html))
    return {'output': str(output.resolve()), 'bytes': output.stat().st_size,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.bundle, args.output, overwrite=args.overwrite), indent=2))


if __name__ == '__main__':
    main()
