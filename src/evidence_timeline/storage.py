"""Save a finished run to PostgreSQL.

The whole run is written in one transaction, so a crash never leaves half a run behind.
Every row is an UPSERT on its primary key, so a repeated save overwrites the same rows
instead of adding new ones.

The pipeline does not know about the database and runs fine without it.
"""

from typing import Any

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from evidence_timeline.models import TimelineRun
from evidence_timeline.schema import batches, documents, event_sources, events, metadata, runs


async def save_run(run: TimelineRun, database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        await create_tables(engine)
        # Parents before children, so the foreign keys to runs.run_id always point at a row.
        tables_and_rows = [
            (runs, [run_row(run)]),
            (documents, document_rows(run)),
            (batches, batch_rows(run)),
            (events, event_rows(run)),
            (event_sources, event_source_rows(run)),
        ]
        async with engine.begin() as connection:  # one transaction for the whole run
            for table, rows in tables_and_rows:
                if rows:
                    await connection.execute(upsert(table, rows))
    finally:
        await engine.dispose()


async def create_tables(engine: AsyncEngine) -> None:
    """Create any missing table. There are no migrations, so changing a column means dropping the database volume."""
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)


def upsert(table: Table, rows: list[dict[str, Any]]) -> Insert:
    """Insert the rows, or overwrite the ones whose primary key is already there."""
    statement = insert(table).values(rows)
    key_columns = [column.name for column in table.primary_key]
    # `excluded` is the row we tried to insert, so this replaces the stored row with the new one.
    updates = {name: statement.excluded[name] for name in rows[0] if name not in key_columns}
    return statement.on_conflict_do_update(index_elements=key_columns, set_=updates)


def run_row(run: TimelineRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "created_at": run.created_at,
        "case_id": run.case_id,
        "status": run.status,
        "extractor_kind": run.extractor.kind,
        "model": run.extractor.model,
        "settings": run.extractor.settings,
        "max_chars_per_batch": run.max_chars_per_batch,
        "prompt_sha256": run.prompt_sha256,
        "total_lines": run.total_lines,
        "lines_in_successful_batches": run.lines_in_successful_batches,
    }


def document_rows(run: TimelineRun) -> list[dict[str, Any]]:
    return [
        {"run_id": run.run_id, "document_id": document_id, "sha256": sha256}
        for document_id, sha256 in run.document_sha256.items()
    ]


def batch_rows(run: TimelineRun) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run.run_id,
            "batch_id": batch.batch_id,
            "status": batch.status,
            "attempts": batch.attempts,
            "lines": batch.lines,
            "model": batch.model,
            "usage": batch.usage,
            "error": batch.error,
        }
        for batch in run.batches
    ]


def event_rows(run: TimelineRun) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run.run_id,
            "event_id": event.event_id,
            "batch_id": event.batch_id,
            "event_type": event.event_type,
            "status": event.status,
            "description": event.description,
            "date": event.date,
            "date_precision": event.date_precision,
            "date_earliest": event.date_earliest,
            "date_latest": event.date_latest,
            # A JSON column cannot hold date objects, so these keep the dataset's YYYY-MM-DD wording.
            "alternative_dates": [date.isoformat() for date in event.alternative_dates],
            "review_reasons": event.review_reasons,
            "citation_errors": event.citation_errors,
            "needs_review": event.needs_review,
        }
        for event in run.events
    ]


def event_source_rows(run: TimelineRun) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run.run_id,
            "event_id": event.event_id,
            "position": position,
            "document_id": quote.document_id,
            "line_start": quote.line_start,
            "line_end": quote.line_end,
            "quote": quote.quote,
            "date_text": quote.date_text,
        }
        for event in run.events
        for position, quote in enumerate(event.evidence, start=1)
    ]
