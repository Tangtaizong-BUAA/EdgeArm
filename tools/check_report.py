"""Check public report links, aggregate values and evaluation-media hashes."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    links = 0
    for name in ['README.md', 'README.en.md', 'README.zh-CN.md', 'docs/media/README.md']:
        source = ROOT / name
        for link in re.findall(r'\]\(([^)]+)\)', source.read_text()):
            if link.startswith(('https://', 'http://', '#')):
                continue
            target = (source.parent / link.split('#')[0]).resolve()
            assert target.exists(), (name, link)
            links += 1
    media = ROOT / 'docs/media'
    data = json.loads((media / 'training_data.values.json').read_text())
    assert sum(data['action_rows_by_route']) == data['action_total'] == 46532
    benchmark = json.loads((ROOT / 'provenance/benchmark.json').read_text())
    assert sum(benchmark['per_route_successes']) == benchmark['successes'] == 51
    assert benchmark['successes'] + benchmark['timeouts'] + benchmark['out_of_bounds'] == 72
    metadata = list(media.glob('run102-*.json'))
    assert len(metadata) == 2
    assert {json.loads(p.read_text())['route'] for p in metadata} == {2, 6}
    for path in metadata:
        row = json.loads(path.read_text())
        assert row['outcome_from_archived_result'] == 'success'
        assert row['new_evaluation'] is False and row['training_data_public'] is False
        assert row['playback_fps'] == row['control_hz'] / row['archived_frame_stride']
        for name, expected in row['files'].items():
            if name.endswith('.mp4'):  # hosted as a Release asset, not in Git
                continue
            file = media / name
            assert hashlib.sha256(file.read_bytes()).hexdigest() == expected['sha256']
    for path in media.glob('*.svg'):
        ET.parse(path)
    print(json.dumps({'local_links': links, 'figures': len(list(media.glob('*.svg'))),
                      'evaluation_videos': len(metadata), 'aggregate_checks': 'passed'}))


if __name__ == '__main__':
    main()
