"""TRD 23-25: Data Models + Alert History (FR-09).

SQLite for MVP (Sec. 20), swappable for Postgres later via DATABASE_URL.
"""
from __future__ import annotations

import datetime as dt
import os

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

from src.config_loader import load_config


class Base(DeclarativeBase):
    pass


class MetricRecord(Base):
    __tablename__ = "metric_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    revenue = Column(Float)
    orders = Column(Float)
    conversion_rate = Column(Float)
    traffic = Column(Float)
    cost = Column(Float)
    refunds = Column(Float)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric = Column(String, nullable=False)
    date = Column(String, nullable=False)
    current_value = Column(Float)
    baseline_value = Column(Float)
    change_percent = Column(Float)
    z_score = Column(Float)
    severity = Column(String, nullable=False)
    explanation = Column(Text)  # JSON-serialized AnomalyExplanation
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    alerts = relationship("Alert", back_populates="anomaly")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    anomaly_id = Column(Integer, ForeignKey("anomalies.id"))
    recipient = Column(String)
    subject = Column(String)
    message = Column(Text)
    status = Column(String)  # "sent" | "failed" | "skipped"
    sent_at = Column(DateTime)

    anomaly = relationship("Anomaly", back_populates="alerts")


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        cfg = load_config()
        url = os.environ.get("DATABASE_URL", cfg["database"]["url"])
        # Ensure the sqlite directory exists
        if url.startswith("sqlite:///") and "/" in url.replace("sqlite:///", ""):
            db_path = url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _engine = create_engine(url)
        Base.metadata.create_all(_engine)
    return _engine


def get_session() -> Session:
    return Session(get_engine())


def save_metric_records(df) -> None:
    """Upsert-free simple append of cleaned metric rows (MVP - no dedupe UI)."""
    cfg = load_config()
    col_map = cfg["metrics"]
    with get_session() as session:
        for _, row in df.iterrows():
            session.add(
                MetricRecord(
                    date=row["Date"],
                    revenue=row.get(col_map["revenue"]),
                    orders=row.get(col_map["orders"]),
                    conversion_rate=row.get(col_map["conversion_rate"]),
                    traffic=row.get(col_map["traffic"]),
                    cost=row.get(col_map["cost"]),
                    refunds=row.get(col_map["refunds"]),
                )
            )
        session.commit()


def save_anomaly(anomaly: dict, explanation_json: str) -> int:
    with get_session() as session:
        record = Anomaly(
            metric=anomaly["metric"],
            date=anomaly["date"],
            current_value=anomaly.get("current_value"),
            baseline_value=anomaly.get("baseline_value"),
            change_percent=anomaly.get("change_percent"),
            z_score=anomaly.get("z_score"),
            severity=anomaly["severity"],
            explanation=explanation_json,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id


def save_alert(anomaly_id: int, recipient: str, subject: str, message: str, status: str) -> None:
    with get_session() as session:
        session.add(
            Alert(
                anomaly_id=anomaly_id,
                recipient=recipient,
                subject=subject,
                message=message,
                status=status,
                sent_at=dt.datetime.utcnow(),
            )
        )
        session.commit()


def get_recent_anomalies(limit: int = 50) -> list[Anomaly]:
    with get_session() as session:
        return (
            session.query(Anomaly)
            .order_by(Anomaly.created_at.desc())
            .limit(limit)
            .all()
        )
