"""The database tables a saved run is written into.

SQLAlchemy Core, not the ORM: the tables are described once here and the data
shapes stay in models.py, so there is no second set of classes to keep in step.

Every primary key is an identifier the pipeline already produces. A run's rows
are therefore always the same rows, which is what lets a repeated save overwrite
instead of adding duplicates.
"""

from sqlalchemy import JSON, Boolean, Column, Date, DateTime, ForeignKey, Integer, MetaData, String, Table, Text

metadata = MetaData()

# One row per `extract` command: the model, the settings and the prompt hash that produced the events.
runs = Table(
    "runs",
    metadata,
    Column("run_id", String, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("case_id", String, nullable=False),
    Column("status", String, nullable=False),
    Column("extractor_kind", String, nullable=False),
    Column("model", String),
    Column("settings", JSON, nullable=False),
    Column("max_chars_per_batch", Integer, nullable=False),
    Column("prompt_sha256", String, nullable=False),
    Column("total_lines", Integer, nullable=False),
    Column("lines_in_successful_batches", Integer, nullable=False),
)

# One row per source file. The hash says which exact text the quotes were taken from.
documents = Table(
    "documents",
    metadata,
    Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
    Column("document_id", String, primary_key=True),
    Column("sha256", String, nullable=False),
)

# One row per batch. A batch is a unit of work, so its id only means something inside its own run.
batches = Table(
    "batches",
    metadata,
    Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
    Column("batch_id", String, primary_key=True),
    Column("status", String, nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("lines", JSON, nullable=False),
    Column("model", String),
    Column("usage", JSON),
    Column("error", Text),
)

events = Table(
    "events",
    metadata,
    Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
    Column("event_id", String, primary_key=True),
    Column("batch_id", String, nullable=False),
    Column("event_type", String, nullable=False),
    Column("status", String, nullable=False),
    Column("description", Text, nullable=False),
    Column("date", Date),
    Column("date_precision", String, nullable=False),
    Column("date_earliest", Date),
    Column("date_latest", Date),
    Column("alternative_dates", JSON, nullable=False),
    Column("review_reasons", JSON, nullable=False),
    Column("citation_errors", JSON, nullable=False),
    Column("needs_review", Boolean, nullable=False),
)

# One row per evidence quote. Provenance gets its own table so that "which events cite B02?"
# is an ordinary query instead of a search through JSON.
event_sources = Table(
    "event_sources",
    metadata,
    Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
    Column("event_id", String, primary_key=True),
    Column("position", Integer, primary_key=True),  # the quote's place in the event's evidence list
    Column("document_id", String, nullable=False),
    Column("line_start", Integer, nullable=False),
    Column("line_end", Integer, nullable=False),
    Column("quote", Text, nullable=False),
    Column("date_text", Text),
)
