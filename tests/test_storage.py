"""Saving a run to PostgreSQL, and saving it twice.

These tests need a running database: `docker compose up -d`, then set DATABASE_URL
(see .env.example). Without it they are skipped, so the rest of the suite stays offline.
"""

import asyncio
import os

import pytest
from helpers import REPO_ROOT
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import create_async_engine

from evidence_timeline.evaluation import load_run
from evidence_timeline.schema import batches, documents, event_sources, events, runs
from evidence_timeline.storage import save_run

DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="needs DATABASE_URL and a running database")

CASE_B_RUN = REPO_ROOT / "runs" / "case_B_llm_20260917T141026Z.json"
# Case B: one run, five documents, one batch, eight events and their thirteen quotes.
EXPECTED_ROWS = {runs: 1, documents: 5, batches: 1, events: 8, event_sources: 13}


async def count_rows(run_id: str) -> dict:
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.connect() as connection:
            counts = {}
            for table in EXPECTED_ROWS:
                query = select(func.count()).select_from(table).where(table.c.run_id == run_id)
                counts[table] = (await connection.execute(query)).scalar_one()
            return counts
    finally:
        await engine.dispose()


async def remove_run(run_id: str) -> None:
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.begin() as connection:
            for table in [event_sources, events, batches, documents, runs]:  # children before parents
                await connection.execute(delete(table).where(table.c.run_id == run_id))
    finally:
        await engine.dispose()


def test_saving_the_same_run_twice_does_not_duplicate_rows():
    run = load_run(CASE_B_RUN)
    try:
        asyncio.run(save_run(run, DATABASE_URL))
        assert asyncio.run(count_rows(run.run_id)) == EXPECTED_ROWS

        asyncio.run(save_run(run, DATABASE_URL))  # the retry Temporal would do after a crash
        assert asyncio.run(count_rows(run.run_id)) == EXPECTED_ROWS
    finally:
        asyncio.run(remove_run(run.run_id))


def test_saved_event_keeps_its_quotes_and_their_order():
    run = load_run(CASE_B_RUN)
    event = run.events[1]  # the X-ray, cited by B01, B02 and B03
    try:
        asyncio.run(save_run(run, DATABASE_URL))
        rows = asyncio.run(read_sources(run.run_id, event.event_id))
    finally:
        asyncio.run(remove_run(run.run_id))

    assert [row.document_id for row in rows] == [quote.document_id for quote in event.evidence]
    assert [row.position for row in rows] == [1, 2, 3]
    assert rows[0].quote == event.evidence[0].quote


async def read_sources(run_id: str, event_id: str) -> list:
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.connect() as connection:
            query = (
                select(event_sources)
                .where(event_sources.c.run_id == run_id, event_sources.c.event_id == event_id)
                .order_by(event_sources.c.position)
            )
            return list(await connection.execute(query))
    finally:
        await engine.dispose()
