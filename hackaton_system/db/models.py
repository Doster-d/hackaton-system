from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Возвращает timestamp в UTC с информацией о таймзоне."""

    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Базовый класс SQLAlchemy для всех моделей проекта."""


class Video(Base):
    """Модель для исходных видеороликов и их метаданных."""

    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    fps: Mapped[float] = mapped_column(Float, default=25.0)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    persons: Mapped[list["Person"]] = relationship(
        "Person", back_populates="video", cascade="all, delete-orphan"
    )
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="video", cascade="all, delete-orphan"
    )
    activities: Mapped[list["Activity"]] = relationship(
        "Activity", back_populates="video", cascade="all, delete-orphan"
    )
    poses: Mapped[list["PoseKeypoints"]] = relationship(
        "PoseKeypoints", back_populates="video", cascade="all, delete-orphan"
    )
    train_events: Mapped[list["TrainEvent"]] = relationship(
        "TrainEvent", back_populates="video", cascade="all, delete-orphan"
    )


class Person(Base):
    """Уникальный человек (трек) в пределах конкретного видео."""

    __tablename__ = "persons"
    __table_args__ = (
        UniqueConstraint("video_id", "track_id", name="uq_person_video_track"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    person_type: Mapped[str] = mapped_column(String(50), default="unknown")
    person_type_conf: Mapped[float] = mapped_column(Float, default=0.0)
    reid_descriptor: Mapped[str | None] = mapped_column(String, nullable=True)

    video: Mapped["Video"] = relationship("Video", back_populates="persons")
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="person", cascade="all, delete-orphan"
    )
    activities: Mapped[list["Activity"]] = relationship(
        "Activity", back_populates="person", cascade="all, delete-orphan"
    )
    poses: Mapped[list["PoseKeypoints"]] = relationship(
        "PoseKeypoints", back_populates="person", cascade="all, delete-orphan"
    )


class Detection(Base):
    """Покадровые рамки для каждого человека."""

    __tablename__ = "detections"
    __table_args__ = (Index("ix_detections_video_time", "video_id", "time_sec"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"), nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    time_sec: Mapped[float] = mapped_column(Float, nullable=False)
    x_min: Mapped[int] = mapped_column(Integer, nullable=False)
    y_min: Mapped[int] = mapped_column(Integer, nullable=False)
    x_max: Mapped[int] = mapped_column(Integer, nullable=False)
    y_max: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    video: Mapped["Video"] = relationship("Video", back_populates="detections")
    person: Mapped["Person"] = relationship("Person", back_populates="detections")


class Activity(Base):
    """Интервалы активностей для треков."""

    __tablename__ = "activities"
    __table_args__ = (Index("ix_activities_video_person", "video_id", "person_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"), nullable=False)
    activity_class: Mapped[str] = mapped_column(String(50), nullable=False)
    t_start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    t_end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    activity_conf: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="heuristic")
    review_status: Mapped[str | None] = mapped_column(String(50), nullable=True, default=None)
    reviewer: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)
    review_notes: Mapped[str | None] = mapped_column(String(500), nullable=True, default=None)

    video: Mapped["Video"] = relationship("Video", back_populates="activities")
    person: Mapped["Person"] = relationship("Person", back_populates="activities")


class PoseKeypoints(Base):
    """Позы и скелеты, связанные с конкретными треками."""

    __tablename__ = "pose_keypoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"), nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    time_sec: Mapped[float] = mapped_column(Float, nullable=False)
    keypoints: Mapped[list[dict[str, float]]] = mapped_column(JSON, nullable=False)
    pose_conf: Mapped[float] = mapped_column(Float, default=0.0)

    video: Mapped["Video"] = relationship("Video", back_populates="poses")
    person: Mapped["Person"] = relationship("Person", back_populates="poses")


class TrainEvent(Base):
    """События прихода, ухода и присутствия поездов на станции."""

    __tablename__ = "train_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # Возможные значения: arrival, departure или presence.
    time_sec: Mapped[float] = mapped_column(Float, nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_x_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_y_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_x_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_y_max: Mapped[int | None] = mapped_column(Integer, nullable=True)

    video: Mapped["Video"] = relationship("Video", back_populates="train_events")
