"""Add the project's Chinese annotations to an official HTML replay."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
START = '<!-- kaggriculture-zh:start -->'
END = '<!-- kaggriculture-zh:end -->'


def localize_html(html):
    """Preserve the official renderer and replay data; replace only our own addon."""
    html = re.sub(re.escape(START) + r'.*?' + re.escape(END), '', html, flags=re.S)
    css = (ROOT / 'viewer_zh.css').read_text(encoding='utf-8')
    js = (ROOT / 'viewer_zh.js').read_text(encoding='utf-8')
    addon = f'{START}<style>{css}</style><script>{js}</script>{END}'
    if '</body>' not in html:
        raise ValueError('Not an official HTML replay: missing body')
    return html.replace('</body>', addon + '</body>').replace('<html lang="en">', '<html lang="zh-CN">').replace(
        '<title>Kaggriculture Visualizer</title>', '<title>Kaggriculture 农场对战 · 中文回放</title>')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='汉化已有回放，默认另存为 replay.zh-CN.html')
    parser.add_argument('replay', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.output or args.replay.with_name('replay.zh-CN.html')
    target.write_text(localize_html(args.replay.read_text(encoding='utf-8')), encoding='utf-8')
    print(target.resolve())
