"""Minimal retry helper for outbound LLM calls: retry on any failure with a fixed sleep."""
import asyncio
import time
from helpers import logger

def retry_call(fn, *args, max_retries=3, retry_delay=15.0, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries:
                raise
            logger.warning("LLM call failed (attempt %d/%d), retrying in %.0fs: %s",
                           attempt, max_retries, retry_delay, exc)
            time.sleep(retry_delay)


async def aretry_call(fn, *args, max_retries=3, retry_delay=15.0, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries:
                raise
            logger.warning("LLM call failed (attempt %d/%d), retrying in %.0fs: %s",
                           attempt, max_retries, retry_delay, exc)
            await asyncio.sleep(retry_delay)
