import threading

import numpy as np
import pytest

from transformer_observatory.storage import Store, TraceWriter


def make_trace(tmp_path, level=2):
    store = Store(tmp_path / 'data', tmp_path / 'traces')
    writer = TraceWriter(store, 'a' * 32, dict(title='fixture', level=level, model={}, tokens=[]))
    return store, writer


def test_level_one_discards_raw_only_after_success(tmp_path):
    store, writer = make_trace(tmp_path, 1)
    r = writer.write(np.ones((2, 4)), kind='residual', stage='embedding')
    path = store.tensor_path(writer.trace_id, r)
    assert path.exists()
    writer.write(np.zeros((2, 3, 3), dtype=np.float32), kind='coordinates', stage='all')
    writer.finalize(dict(status='verified'))
    assert not path.exists()
    assert [r['kind'] for r in store.manifest(writer.trace_id)['tensors']] == ['coordinates']


def test_storage_budget_interrupt_and_recovery(tmp_path):
    store, writer = make_trace(tmp_path)
    writer.limit = 32
    with pytest.raises(OSError, match='容量上限'):
        writer.write(np.zeros((20, 5)), kind='residual', stage='embedding')
    store.put('jobs', writer.trace_id, dict(id=writer.trace_id, state='capturing'))
    reopened = Store(store.data_root, store.trace_root)
    assert reopened.get('jobs', writer.trace_id)['state'] == 'failed'
    assert reopened.manifest(writer.trace_id)['status'] == 'incomplete'


def test_cleanup_does_not_delete_retained_or_running(tmp_path):
    store, writer = make_trace(tmp_path)
    writer.manifest['created_at'] = 0
    writer.manifest['retained'] = True
    store.save_manifest(writer.manifest)
    assert store.cleanup_preview()['targets'] == []
    with pytest.raises(ValueError):
        store.delete(writer.trace_id)
    writer.manifest['retained'] = False
    store.save_manifest(writer.manifest)
    store.put('jobs', writer.trace_id, dict(id=writer.trace_id, state='capturing'))
    assert store.cleanup_preview()['targets'] == []
    store.put('jobs', writer.trace_id, dict(id=writer.trace_id, state='cancelled'))
    preview = store.cleanup_preview()
    assert len(preview['targets']) == 1
    # Retaining after preview must still prevent deletion at confirmation.
    writer.manifest['retained'] = True
    store.save_manifest(writer.manifest)
    assert store.cleanup_confirm(preview['token']) == []
    assert store.path(writer.trace_id).exists()


def test_unowned_and_traversal_paths_rejected(tmp_path):
    store, writer = make_trace(tmp_path)
    for trace_id in ['..', '../outside', 'C:\\', 'x' * 32]:
        with pytest.raises(ValueError):
            store.path(trace_id)
    (store.path(writer.trace_id) / '.trace-owner').unlink()
    with pytest.raises(ValueError):
        store.delete(writer.trace_id)
