"""Payload-free OTel instrumentation; SDK/exporters are optional.

Like TinySearch, library callers own their OTel providers. Standalone server
entry points configure OTLP from the environment and shut down only providers
created here. The API alone is a no-op without a configured SDK.
"""

from __future__ import annotations

import atexit
import importlib
import importlib.metadata
import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Iterator

from opentelemetry import metrics, trace

_lock = threading.Lock()
_runtime: Any = None
_configured = False
_owned_providers: list[Any] = []
_current: ContextVar[Any] = ContextVar("tinycontext_telemetry", default=None)
_logger = logging.getLogger(__name__)


def _disabled() -> bool:
    return os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in {"true", "1", "yes", "on"}


def _signal_enabled(signal: str) -> bool:
    exporter = os.getenv(f"OTEL_{signal}_EXPORTER", "").strip().lower()
    if exporter:
        values = {part.strip() for part in exporter.split(",")}
        return "otlp" in values and "none" not in values
    return bool(os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_ENDPOINT", "").strip() or
                os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())


def _exporter(signal: str) -> Any:
    protocol = (
        os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_PROTOCOL", "").strip().lower()
        or os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "").strip().lower()
        or "http/protobuf"
    )
    if protocol not in {"http/protobuf", "grpc"}:
        raise ValueError("Unsupported OTLP protocol")
    transport = "http" if protocol == "http/protobuf" else "grpc"
    module = "trace_exporter" if signal == "TRACES" else "metric_exporter"
    cls = "OTLPSpanExporter" if signal == "TRACES" else "OTLPMetricExporter"
    # The SDK implements endpoint/header/TLS/compression/timeout precedence.
    return getattr(importlib.import_module(
        f"opentelemetry.exporter.otlp.proto.{transport}.{module}"
    ), cls)()


def _version() -> str:
    try:
        return importlib.metadata.version("tinysuite-context")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _resource() -> Any:
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create()
    attributes = dict(resource.attributes)
    if not attributes.get("service.name") or attributes["service.name"] == "unknown_service":
        attributes["service.name"] = "tinycontext"
    attributes.setdefault("service.version", _version())
    return Resource(attributes)


def _build_provider(signal: str) -> Any:
    resource = _resource()
    exporter = _exporter(signal)
    try:
        if signal == "TRACES":
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(resource=resource, shutdown_on_exit=False)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            return provider
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        return MeterProvider(
            resource=resource,
            metric_readers=[PeriodicExportingMetricReader(exporter)],
            shutdown_on_exit=False,
        )
    except Exception:
        exporter.shutdown()
        raise


def configure_from_environment() -> None:
    """Opt standalone servers into OTLP; preserve application-owned providers.

    Setup failures disable only the affected signal and never block memory.
    Restart the process after changing export configuration.
    """
    global _configured
    if _disabled():
        return
    with _lock:
        if _configured:
            return
        _configured = True
        for signal, getter, setter, proxy_type in (
            ("TRACES", trace.get_tracer_provider, trace.set_tracer_provider, "ProxyTracerProvider"),
            ("METRICS", metrics.get_meter_provider, metrics.set_meter_provider, "_ProxyMeterProvider"),
        ):
            if not _signal_enabled(signal) or type(getter()).__name__ != proxy_type:
                continue
            try:
                provider = _build_provider(signal)
                setter(provider)
                _owned_providers.append(provider)
            except Exception:
                # Exception text can contain URLs, headers or credentials.
                _logger.warning(
                    "TinyContext %s telemetry unavailable; install the telemetry extra "
                    "and check OTEL configuration. Continuing without this export.", signal.lower(),
                )
        if _owned_providers:
            atexit.register(shutdown)


def shutdown() -> None:
    """Flush and stop owned providers, leaving application providers alone."""
    with _lock:
        providers = list(reversed(_owned_providers))
        _owned_providers.clear()
    for provider in providers:
        try:
            provider.force_flush()
        except Exception:
            _logger.warning("TinyContext telemetry flush failed")
        finally:
            try:
                provider.shutdown()
            except Exception:
                _logger.warning("TinyContext telemetry shutdown failed")


class _Runtime:
    def __init__(self, trace_provider: Any, meter_provider: Any) -> None:
        self.trace_provider = trace_provider
        self.meter_provider = meter_provider
        self.tracer = trace_provider.get_tracer("tinycontext", _version())
        meter = meter_provider.get_meter("tinycontext", _version())
        self.duration = meter.create_histogram(
            "tinycontext.operation.duration", unit="s",
            description="Memory operation and internal stage duration",
        )
        self.results = meter.create_histogram(
            "tinycontext.operation.result.count", unit="{result}",
            description="Items returned or saved by an operation",
        )


def _get_runtime() -> Any:
    global _runtime
    if _disabled():
        return None
    trace_provider, meter_provider = trace.get_tracer_provider(), metrics.get_meter_provider()
    if (type(trace_provider).__name__ in {"ProxyTracerProvider", "NoOpTracerProvider"}
            and type(meter_provider).__name__ in {"_ProxyMeterProvider", "NoOpMeterProvider"}):
        return None
    with _lock:
        if (_runtime is None or _runtime.trace_provider is not trace_provider
                or _runtime.meter_provider is not meter_provider):
            _runtime = _Runtime(trace_provider, meter_provider)
    return _runtime


def set_attributes(**attributes: int | str) -> None:
    """Record only explicitly selected operational attributes at call sites."""
    span = _current.get()
    if span is not None:
        try:
            span.set_attributes(attributes)
        except Exception:
            pass


def _record_result_count(runtime: Any, name: str, count: int) -> None:
    if runtime is None:
        return
    try:
        if runtime.results is not None:
            runtime.results.record(count, {"tinycontext.operation.name": name})
    except Exception:
        pass


@contextmanager
def operation(name: str, **attributes: int | str) -> Iterator[None]:
    try:
        runtime = _get_runtime()
    except Exception:
        runtime = None
    if runtime is None:
        yield
        return
    from opentelemetry.trace import StatusCode

    labels = {"tinycontext.operation.name": name}
    started = perf_counter()
    try:
        span = runtime.tracer.start_span("tinycontext." + name, attributes={**labels, **attributes})
    except Exception:
        yield
        return
    from opentelemetry.trace import use_span

    # Automatic exception recording includes messages and stack traces.
    with use_span(span, end_on_exit=False, record_exception=False, set_status_on_exception=False):
        token = _current.set(span)
        try:
            yield
        except BaseException as exc:
            error_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
            labels["error.type"] = error_type
            try:
                span.set_attribute("error.type", error_type)
                span.set_status(StatusCode.ERROR)
            except Exception:
                pass
            raise
        finally:
            _current.reset(token)
            try:
                if runtime.duration is not None:
                    runtime.duration.record(perf_counter() - started, labels)
            except Exception:
                pass
            finally:
                try:
                    span.end()
                except Exception:
                    pass


def instrument(name: str, *, result: str | None = None, **attributes: str) -> Callable:
    """Wrap synchronous boundaries without inspecting arguments or payload text."""
    def decorate(func: Callable) -> Callable:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            runtime = None
            with operation(name, **attributes):
                try:
                    runtime = _get_runtime()
                except Exception:
                    runtime = None
                value = func(*args, **kwargs)
                if _current.get() is not None and result is not None:
                    count = None
                    if result == "length":
                        count = len(value)
                    elif result == "memory":
                        items = value.get("memories", value.get("saved"))
                        if items is not None:
                            count = len(items)
                        for key in ("total_tokens", "matched_count"):
                            if type(value.get(key)) is int:
                                set_attributes(**{f"tinycontext.{key}": value[key]})
                    if count is not None:
                        if attributes.get("gen_ai.operation.name") == "search_memory":
                            set_attributes(**{"gen_ai.memory.record.count": count})
                        else:
                            set_attributes(**{"tinycontext.result.count": count})
                        _record_result_count(runtime, name, count)
                return value
        return wrapped
    return decorate
