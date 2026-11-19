"""Per-clip summary of completed sampled object checks, stored beside recordings."""
import json
import logging
import os
import tempfile
from pathlib import Path


def metadata_path(clip_path):
    return Path(str(clip_path) + '.detections.json')


def load_metadata(clip_path):
    try:
        data = json.loads(metadata_path(clip_path).read_text())
        if data.get('version') != 1 or not isinstance(data.get('objects'), list):
            return None
        return data
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def write_metadata(clip_path, data):
    path = metadata_path(clip_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.motion_metadata_', delete=False) as out:
            temporary = out.name
            json.dump(data, out)
            out.flush()
        os.replace(temporary, path)
    except OSError:
        logging.exception('Could not save motion clip detection summary')
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def start_metadata(clip_path):
    write_metadata(clip_path, {'version': 1, 'recording': True, 'samples': 0, 'objects': []})


def update_metadata(clip_path, labels, evidence=None):
    data = load_metadata(clip_path)
    if data is None or not data.get('recording'):
        return
    data['samples'] += 1
    data['objects'] = sorted(set(data['objects']) | {str(label) for label in labels if label})
    if labels:
        if evidence:
            name = Path(evidence).name
            data.setdefault('evidence', []).append(name)
        else:
            data['evidence_unavailable'] = True
    write_metadata(clip_path, data)


def finish_metadata(clip_path):
    data = load_metadata(clip_path)
    if data is not None:
        data['recording'] = False
        write_metadata(clip_path, data)


def title_for_metadata(data):
    if data is None:
        return 'Object detection not recorded'
    objects = data.get('objects', [])
    if objects:
        return 'Object detected: ' + ', '.join(str(x) for x in objects) + (' (some images unavailable)' if data.get('evidence_unavailable') else '')
    if data.get('recording'):
        return 'Recording; object detection in progress'
    if data.get('samples', 0):
        return 'No objects detected in sampled frames'
    return 'Object detection not checked'
