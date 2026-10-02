"""Validate fixture integrity, not model quality. Python standard library only."""
from pathlib import Path
from datetime import date
from collections import Counter
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    documents = manifest['documents']
    ids = [d['document_id'] for d in documents]
    require(len(ids) == len(set(ids)) == 14, 'Document IDs/count invalid')
    require(manifest['splits'] == {'development': ['A', 'B'], 'holdout': ['C']}, 'Invalid split')
    texts = {}
    for d in documents:
        p = ROOT / d['path']
        data = p.read_bytes()
        require(hashlib.sha256(data).hexdigest() == d['sha256'], f'Checksum: {p}')
        text = data.decode('utf-8')
        require(f"Case: {d['case_id']}\n" in text, 'Case metadata mismatch')
        require(f"Document ID: {d['document_id']}\n" in text, 'Document metadata mismatch')
        date.fromisoformat(d['record_created'])
        texts[d['document_id']] = text.splitlines()
    require(set(ROOT.glob('inputs/*/*.md')) == {ROOT/d['path'] for d in documents}, 'Unlisted input files')
    coverage = Counter()
    def source_ok(s, case):
        require(s['document_id'] in texts, 'Unknown source document')
        require(s['document_id'][0] == case, 'Cross-case citation')
        a, b = s['line_start'], s['line_end']
        lines = texts[s['document_id']]
        require(8 <= a <= b <= len(lines), 'Invalid narrative line range')
        require('\n'.join(lines[a-1:b]) == s['quote'], 'Quote/location mismatch')
        if s['date_text'] is not None:
            require(s['date_text'] in s['quote'], 'Original date text missing from quote')
        for n in range(a,b+1):
            coverage[s['document_id'],n] += 1
    all_events = []
    excluded = 0
    for case in 'ABC':
        gold = json.loads((ROOT/'gold'/f'case_{case}.json').read_text())
        require(gold['case_id'] == case and len(gold['events']) == 8, 'Case/count mismatch')
        for e in gold['events']:
            require(e['case_id'] == case and e['event_id'].startswith(case+'-E'), 'Event case mismatch')
            require(e['event_type'] in {'visit','procedure','medication_start'}, 'Bad event type')
            require(e['status'] in {'planned','completed'}, 'Bad status')
            require(e['sources'], 'Missing evidence')
            require(e['needs_review'] == bool(e['review_reasons']), 'Review flag mismatch')
            require('confidence' not in e, 'Unexpected confidence score')
            for s in e['sources']:
                source_ok(s,case)
            precision = e['date_precision']
            if precision == 'day':
                date.fromisoformat(e['date'])
                require(e['date_interval'] is None and not e['alternative_dates'], 'Invalid exact date')
            elif precision == 'month':
                require(e['date'] is None and not e['alternative_dates'], 'False precision')
                start,end = [date.fromisoformat(e['date_interval'][k]) for k in ('start','end')]
                require(start <= end and start.strftime('%Y-%m') == end.strftime('%Y-%m'), 'Invalid interval')
                require('approximate_date' in e['review_reasons'], 'Unflagged uncertainty')
            elif precision == 'conflicting':
                require(e['date'] is None and e['date_interval'] is None, 'Silently resolved conflict')
                require(len(set(e['alternative_dates'])) >= 2, 'Missing conflict alternatives')
                for value in e['alternative_dates']:
                    date.fromisoformat(value)
                require('conflicting_dates' in e['review_reasons'], 'Unflagged conflict')
            else:
                raise ValueError('Unknown date precision')
            all_events.append(e)
        for x in gold['exclusions']:
            require(x['case_id'] == case and x['reason'], 'Invalid exclusion')
            source_ok(x['source'],case)
            excluded += 1
        expected_zero = ['C05'] if case == 'C' else []
        require(gold['zero_event_documents'] == expected_zero, 'Zero-event annotation mismatch')
        for doc_id in expected_zero:
            require(not any(s['document_id']==doc_id for e in gold['events'] for s in e['sources']), 'Zero-event doc has event')
    event_ids = [e['event_id'] for e in all_events]
    require(len(event_ids) == len(set(event_ids)) == 24, 'Event IDs/count invalid')
    for doc_id, lines in texts.items():
        for n,line in enumerate(lines,1):
            if n >= 8 and line.strip():
                require(coverage[doc_id,n] == 1, f'Unaccounted or multiply labelled narrative: {doc_id}:{n}')
    by_id = {e['event_id']:e for e in all_events}
    require({e['event_id'] for e in all_events if e['status']=='planned'} == {'C-E03','C-E08'}, 'Planned status changed')
    require(by_id['C-E05']['date_interval']=={'start':'2025-03-01','end':'2025-03-31'}, 'Month bounds changed')
    require(by_id['C-E06']['alternative_dates']==['2025-04-14','2025-04-15'], 'Conflict dates changed')
    repeated = [e for e in all_events if len(e['sources'])>1]
    require({e['event_id'] for e in repeated} == {'B-E02','B-E03','B-E04','B-E07','C-E03','C-E06'}, 'Repeated mention groups changed')
    return dict(status='passed', dataset_version=manifest['dataset_version'], documents=len(documents),
        events=len(all_events), completed_events=sum(e['status']=='completed' for e in all_events),
        planned_events=2, evidence_mentions=sum(len(e['sources']) for e in all_events),
        events_with_multiple_sources=len(repeated), explicit_exclusions=excluded,
        accounted_narrative_lines=len(coverage), conflicting_date_events=1, approximate_date_events=1,
        zero_event_documents=['C05'],
        checks=['input checksums and inventory','unique IDs and case isolation','exact quotes and line locations',
                'complete narrative annotation coverage','date shapes and uncertainty flags',
                'event counts, status and repeated-mention regression checks'],
        limitations='Structural and annotation-consistency checks; not independent clinical validation or LLM extraction evaluation.')

if __name__ == '__main__':
    report = validate()
    output = json.dumps(report,indent=2)+'\n'
    if '--write-report' in sys.argv:
        (ROOT/'VALIDATION_REPORT.json').write_text(output)
    print(output,end='')
