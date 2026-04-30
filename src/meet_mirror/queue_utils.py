from __future__ import annotations

import queue

from loguru import logger


def put_drop_oldest(q: queue.Queue, item: object, name: str = "queue") -> bool:
    """Best-effort enqueue with drop-oldest fallback on queue.Full.

    Returns True if `item` was enqueued, False if even after dropping
    the oldest entry the queue is still full (logged as a warning).
    The dropped-oldest path also logs a warning so backpressure is
    visible without flooding when the queue is healthy.
    """
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        pass

    try:
        q.get_nowait()
        logger.warning(f"{name} full; dropped oldest entry")
    except queue.Empty:
        # Race: someone else drained between Full and now. Just retry put.
        pass

    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        logger.warning(f"{name} still full after drop; dropping new item")
        return False
