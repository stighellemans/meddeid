"""Exercise the release comparator against incomplete and divergent API responses."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('parity_validation', ROOT / 'deploy/validate_triton_parity.py')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def documents(n=4):
    return [{'document_id': str(i), 'text': 'synthetic text', 'metadata': {}} for i in range(n)]


def response(document):
    return {'document_id': document['document_id'], 'deid_text': 'synthetic text', 'spans': [],
            'provenance': {'software': {'version': 'test'}, 'model': {'bundle_sha256': 'abc'}}}


def run(monkeypatch, tmp_path, mutate=lambda side, rows: rows, **kwargs):
    calls = []

    def request(url, *, api_key, payload=None):
        if payload is None:
            return {'status': 'ok', 'ready': True}
        calls.append((url, [d['document_id'] for d in payload['documents']]))
        rows = [response(d) for d in payload['documents']]
        return {'documents': mutate(url.split('/')[2], rows)}

    monkeypatch.setattr(parity, 'request_json', request)
    path = tmp_path / 'report.json'
    report = parity.compare(documents(), reference_url='http://reference', candidate_url='http://candidate',
                            api_key='test', batch_size=2, output=path, **kwargs)
    assert json.loads(path.read_text()) == report
    return report, calls


def test_complete_equality_passes(monkeypatch, tmp_path):
    report, calls = run(monkeypatch, tmp_path)
    assert report['passed'] and report['checked_documents'] == 4
    assert len(calls) == 4


def test_fixture_reader_forwards_only_supported_api_metadata(tmp_path):
    fixture = tmp_path / 'benchmark.jsonl'
    fixture.write_text(
        json.dumps({
            'document_id': 'english-1',
            'text': 'Synthetic English note',
            'metadata': {
                'lang': 'en-GB',
                'patient': {'given_name': 'Ada'},
                'generation_method': 'benchmark-only',
                'synthetic': True,
            },
        }) + '\n',
        encoding='utf-8',
    )

    assert parity.read_documents(fixture) == [{
        'document_id': 'english-1',
        'text': 'Synthetic English note',
        'metadata': {'lang': 'en-GB', 'patient': {'given_name': 'Ada'}},
    }]


@pytest.mark.parametrize('side', ['reference', 'candidate'])
@pytest.mark.parametrize('bad', ['empty', 'duplicate', 'missing', 'unexpected', 'no_identity'])
def test_incomplete_outputs_never_pass(monkeypatch, tmp_path, side, bad):
    def mutate(actual_side, rows):
        if actual_side != side:
            return rows
        if bad == 'empty': return []
        if bad == 'duplicate': return [rows[0], rows[0]]
        if bad == 'missing': return rows[:1]
        if bad == 'unexpected': rows[0]['document_id'] = 'unknown'
        if bad == 'no_identity': rows[0]['provenance'] = {}
        return rows
    report, _ = run(monkeypatch, tmp_path, mutate)
    assert not report['passed'] and report['errors']


def test_identity_drift_after_first_document_fails(monkeypatch, tmp_path):
    def mutate(side, rows):
        if rows[0]['document_id'] == '2':
            # Even identical drift on both servers must be rejected.
            rows[0]['provenance']['model']['bundle_sha256'] = 'wrong'
        return rows
    report, _ = run(monkeypatch, tmp_path, mutate)
    assert not report['passed'] and not report['model_identity_matches']


def test_first_failure_checkpoints_and_stops_before_next_batch(monkeypatch, tmp_path):
    def mutate(side, rows):
        if side == 'candidate': rows[0]['deid_text'] = '[Organization:Healthcare]'
        return rows
    report, calls = run(monkeypatch, tmp_path, mutate, fail_fast=True)
    assert not report['passed'] and not report['complete']
    assert report['checked_documents'] == 2 and len(calls) == 2
    assert report['semantic_differences'][0]['changed_fields'] == ['deid_text']
    assert report['semantic_differences'][0]['batch_document_ids'] == ['0', '1']


def test_diagnostic_selection_preserves_original_batch(monkeypatch, tmp_path):
    report, calls = run(monkeypatch, tmp_path, document_id='3')
    assert report['passed'] and report['documents'] == 2 and report['fixture_documents'] == 4
    assert [ids for _, ids in calls] == [['2', '3'], ['2', '3']]


def test_http_error_leaves_failed_report(monkeypatch, tmp_path):
    def broken(*args, **kwargs): raise RuntimeError('connection lost')
    monkeypatch.setattr(parity, 'request_json', broken)
    path = tmp_path / 'report.json'
    report = parity.compare(documents(), reference_url='ref', candidate_url='candidate', api_key='test',
                            batch_size=2, output=path)
    assert not report['passed'] and report['errors'] == ['RuntimeError: connection lost']
    assert json.loads(path.read_text())['passed'] is False


def test_duplicate_fixture_and_unknown_replay_id_rejected(tmp_path):
    with pytest.raises(ValueError, match='duplicate'):
        parity.compare(documents() * 2, reference_url='ref', candidate_url='candidate', api_key='test',
                       batch_size=2, output=tmp_path/'report.json')
    with pytest.raises(ValueError, match='not found'):
        parity.selected_batches(documents(), 2, 'unknown')


def test_report_only_records_missed_masking_and_finishes_every_batch(monkeypatch, tmp_path):
    def mutate(side, rows):
        if side == 'reference':
            for row in rows:
                row['spans'] = [{'begin': 0, 'end': 3, 'label': 'Name'}]
                row['deid_text'] = '[Name]thetic text'
        return rows
    report, calls = run(monkeypatch, tmp_path, mutate, semantic_policy='report-only', fail_fast=True)
    assert report['passed'] and not report['strict_passed']
    assert report['complete'] and report['checked_documents'] == 4 and len(calls) == 4
    assert len(report['semantic_differences']) == 4
    assert report['semantic_summary']['unmasked_reference_characters'] == 12
    assert report['semantic_summary']['documents_with_reduced_mask_coverage'] == 4
    assert report['semantic_summary']['affected_document_fraction'] == 1.0
    assert 'Reference-masked characters unmasked by candidate: 12' in (tmp_path/'report.md').read_text()


def test_report_only_never_accepts_identity_mismatch(monkeypatch, tmp_path):
    def mutate(side, rows):
        if side == 'candidate': rows[0]['provenance']['model']['bundle_sha256'] = 'different'
        return rows
    report, _ = run(monkeypatch, tmp_path, mutate, semantic_policy='report-only')
    assert not report['passed'] and not report['model_identity_matches']


@pytest.mark.parametrize('bad', ['missing', 'offsets'])
def test_report_only_never_accepts_broken_responses(monkeypatch, tmp_path, bad):
    def mutate(side, rows):
        if side == 'candidate':
            if bad == 'missing': return []
            rows[0]['spans'] = [{'begin': 0, 'end': 999999999, 'label': 'Name'}]
        return rows
    report, _ = run(monkeypatch, tmp_path, mutate, semantic_policy='report-only')
    assert not report['passed'] and report['errors']


def test_report_distinguishes_label_changes_from_reduced_masking():
    ref = {'spans': [{'begin': 0, 'end': 3, 'label': 'Name'}]}
    cand = {'spans': [{'begin': 0, 'end': 3, 'label': 'Organization'},
                      {'begin': 5, 'end': 6, 'label': 'Organization'}]}
    delta = parity.semantic_delta(ref, cand)
    assert len(delta['label_changes']) == 1
    assert delta['unmasked_reference_characters'] == 0
    assert delta['additional_masked_characters'] == 1


def test_default_policy_still_rejects_semantic_changes(monkeypatch, tmp_path):
    def mutate(side, rows):
        if side == 'candidate': rows[0]['deid_text'] = 'different'
        return rows
    report, _ = run(monkeypatch, tmp_path, mutate)
    assert report['semantic_policy'] == 'strict' and not report['passed']
