import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request

logger = logging.getLogger("research_rag.http")


def register_http_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request_failed request_id=%s method=%s path=%s latency_ms=%.2f",
                request_id, request.method, request.url.path, latency_ms,
            )
            raise

        latency_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
        logger.info(
            "request_done request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
            request_id, request.method, request.url.path, response.status_code, latency_ms,
        )
        return response