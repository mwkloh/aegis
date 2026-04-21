"""Append-only structured event stream (JSONL under ~/.aegis/workspace/sessions/)."""

from .stream import EventStream, EventType

__all__ = ["EventStream", "EventType"]
