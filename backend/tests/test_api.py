import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from transformer_observatory.api import create_app
from transformer_observatory.storage import Store


@pytest.fixture
def client(tmp_path):
    app = create_app(Store(tmp_path / 'data', tmp_path / 'traces'), testing=True)
    with TestClient(app) as c:
        token = c.get('/api/session').json()['token']
        c.headers['x-observatory-token'] = token
        yield c


def wait_job(client, trace_id):
    for _ in range(300):
        job = client.get(f'/api/analysis/jobs/{trace_id}').json()
        if job['state'] in ['completed', 'cancelled', 'failed']:
            assert job['state'] == 'completed', job
            return job
        time.sleep(0.025)
    pytest.fail('Job timeout')


def run_job(client, **kwargs):
    response = client.post('/api/analysis/jobs', json=dict(prompt='Hello', device='cpu', **kwargs))
    assert response.status_code == 202, response.text
    trace_id = response.json()['id']
    wait_job(client, trace_id)
    return trace_id


def test_csrf_host_and_origin(client):
    assert client.get('/api/models', headers={'host': 'evil.test'}).status_code == 403
    assert client.get('/api/models', headers={'origin': 'https://evil.test'}).status_code == 403
    assert client.post('/api/analysis/preflight', json={'prompt': 'x'},
                       headers={'x-observatory-token': 'wrong'}).status_code == 403


def test_full_job_reconnect_export_import_and_comparison(client):
    trace_id = run_job(client, level=4, attention=True)
    m = client.get(f'/api/traces/{trace_id}').json()
    assert m['status'] == 'completed'
    assert m['integrity']['all_passed']
    response = client.get(f'/api/traces/{trace_id}/coordinates?view=displacement')
    coords = np.frombuffer(response.content, '<f4').reshape(6, 9, 3)
    assert np.all(coords[:, 0] == 0)
    details = client.get(f'/api/traces/{trace_id}/tensor?kind=tensor&layer=0&token=1').json()
    assert details['status'] == '取得済み'
    assert any(d['name'] == 'head_contribution' for d in details['tensors'])
    assert client.get(f'/api/traces/{trace_id}/verify').json()['verified']
    exported = client.get(f'/api/traces/{trace_id}/export')
    imported = client.post('/api/storage/import', content=exported.content).json()['id']
    assert client.get(f'/api/traces/{imported}').json()['retained']
    comparison = client.post('/api/traces/compare', json={'left': trace_id, 'right': imported})
    assert comparison.status_code == 200, comparison.text
    assert comparison.json()['common_prefix_max_distance'] == 0
    assert client.delete(f'/api/traces/{trace_id}').status_code == 400
    assert client.delete(f'/api/traces/{trace_id}', headers={'x-confirm-trace': trace_id}).status_code == 200


def test_generation_positions_and_level_one(client):
    trace_id = run_job(client, level=1, generate=True, max_new_tokens=3)
    m = client.get(f'/api/traces/{trace_id}').json()
    assert all(t['processed'] for t in m['tokens'])
    assert len(m['tokens']) > 6
    assert all(g['processed_at_step'] == g['predicted_at_step'] + 1 for g in m['generation'])
    assert not any(r['kind'] == 'residual' for r in m['tensors'])
    assert client.get(f'/api/traces/{trace_id}/tensor').json()['status'] == '未取得'


def test_causal_prefix_invariance(client):
    left = run_job(client, level=2)
    response = client.post('/api/analysis/jobs', json={'prompt': 'Helpx', 'device': 'cpu'})
    right = response.json()['id']
    wait_job(client, right)
    c = client.post('/api/traces/compare', json={'left': left, 'right': right}).json()
    assert c['common_prefix_tokens'] == 4
    assert c['common_prefix_max_distance'] < 1e-5


def test_invalid_preflight_and_zip(client):
    assert client.post('/api/analysis/preflight', json={'prompt': 'x', 'layers': [99]}).status_code == 400
    assert client.post('/api/analysis/preflight', json={'prompt': ''}).status_code == 422
    assert client.post('/api/storage/import', content=b'not a zip').status_code == 400
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('manifest.json', json.dumps({'schema_version': 1, 'status': 'completed', 'id': 'a'*32,
                                               'tensors': [{'file': '../escape.npy'}]}))
        z.writestr('../escape.npy', b'x')
    assert client.post('/api/storage/import', content=archive.getvalue()).status_code == 400
