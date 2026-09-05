from __future__ import annotations

import asyncio
import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from tinycontext import core, telemetry, MemoryInput
from tinycontext.services.memory_store_service import close_connection
from tests.embedding_fakes import fake_embed_texts

_HAS_SDK = importlib.util.find_spec('opentelemetry.sdk') is not None
if _HAS_SDK:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

_ROOT = Path(__file__).resolve().parents[1]


def _isolated(code, extra_env=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith('OTEL_')}
    env['PYTHONPATH'] = str(_ROOT / 'src')
    env.update(extra_env or {})
    return subprocess.run([sys.executable, '-c', code], env=env, cwd=_ROOT,
                          capture_output=True, text=True, timeout=30)


class DisabledTelemetryTests(unittest.TestCase):
    def test_core_works_without_sdk_or_configuration(self):
        process = _isolated('''
import sys
class NoSDK:
    def find_spec(self, fullname, *args):
        if fullname.startswith(('opentelemetry.sdk', 'opentelemetry.exporter')):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, NoSDK())
from tinycontext import recall_memories
from tinycontext.telemetry import configure_from_environment
import tempfile
from pathlib import Path
configure_from_environment()
with tempfile.TemporaryDirectory() as directory:
    assert recall_memories(config={'memory_db_path': str(Path(directory) / 'm.db')})['memories'] == []
assert not any(name.startswith('opentelemetry.sdk') for name in sys.modules)
''')
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_disabled_and_signal_selection(self):
        for env in ({}, {'OTEL_SDK_DISABLED': 'true', 'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://unused'},
                    {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://unused', 'OTEL_TRACES_EXPORTER': 'none',
                     'OTEL_METRICS_EXPORTER': 'none'}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), \
                    patch.object(telemetry, '_configured', False), \
                    patch.object(telemetry, '_build_provider') as build:
                telemetry.configure_from_environment()
                build.assert_not_called()
        with patch.dict(os.environ, {'OTEL_TRACES_EXPORTER': 'console,otlp'}, clear=True):
            self.assertTrue(telemetry._signal_enabled('TRACES'))

    def test_missing_extra_does_not_break_server_configuration(self):
        process = _isolated('''
import sys
class NoSDK:
    def find_spec(self, fullname, *args):
        if fullname.startswith('opentelemetry.sdk'):
            raise ModuleNotFoundError('sensitive-detail')
sys.meta_path.insert(0, NoSDK())
from tinycontext.telemetry import configure_from_environment
configure_from_environment()
''', {'OTEL_TRACES_EXPORTER': 'otlp'})
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('telemetry extra', process.stderr)
        self.assertNotIn('sensitive-detail', process.stderr)


@unittest.skipUnless(_HAS_SDK, 'requires telemetry extra')
class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider(shutdown_on_exit=False)
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(metric_readers=[self.reader], shutdown_on_exit=False)
        self.addCleanup(self.tracer_provider.shutdown)
        self.addCleanup(self.meter_provider.shutdown)
        for patcher in (
            patch.dict(os.environ, {}, clear=True),
            patch.object(telemetry.trace, 'get_tracer_provider', return_value=self.tracer_provider),
            patch.object(telemetry.metrics, 'get_meter_provider', return_value=self.meter_provider),
            patch.object(telemetry, '_runtime', None),
            patch('tinycontext.services.embedding_service._embed_texts_onnx', side_effect=fake_embed_texts),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Path(directory.name) / 'private-store.db'
        self.addCleanup(close_connection, self.db)
        self.config = {'memory_db_path': str(self.db), 'embedding_model': 'fast'}

    def spans(self):
        return self.exporter.get_finished_spans()

    def test_lifecycle_hierarchy_metrics_and_privacy(self):
        secret = 'private-memory-payload Python backend'
        parent_tracer = self.tracer_provider.get_tracer('application')
        with parent_tracer.start_as_current_span('application') as parent:
            saved = core.save_memories([MemoryInput(content=secret)], session_id='secret-session', config=self.config)
            recalled = core.recall_memories('private-query Python backend', config=self.config)
            core.recall_memories(config=self.config)
            updated = core.update_memory(saved['saved'][0]['id'], 'private-updated SQLite data', config=self.config)
            core.get_memory(updated['id'], config=self.config)
            core.list_memories(config=self.config)
            core.delete_memory(updated['id'], config=self.config)
        spans = self.spans()
        names = {span.name for span in spans}
        self.assertTrue({'tinycontext.save_memories', 'tinycontext.recall_memories',
                         'tinycontext.update_memory', 'tinycontext.delete_memory',
                         'tinycontext.get_memory', 'tinycontext.list_memories',
                         'tinycontext.embed_texts', 'tinycontext.memory_recall', 'tinycontext.rank',
                         'tinycontext.insert_memories', 'tinycontext.fetch_dense_scores',
                         'tinycontext.fetch_sparse_scores', 'tinycontext.fetch_memories_by_ids'} <= names)
        save = next(s for s in spans if s.name == 'tinycontext.save_memories')
        embed = next(s for s in spans if s.name == 'tinycontext.embed_texts')
        self.assertEqual(save.parent.span_id, parent.get_span_context().span_id)
        self.assertEqual(embed.parent.span_id, save.context.span_id)
        search = next(s for s in spans if s.name == 'tinycontext.memory_recall')
        rank = next(s for s in spans if s.name == 'tinycontext.rank')
        self.assertEqual(rank.parent.span_id, search.context.span_id)
        self.assertEqual(search.attributes['gen_ai.memory.record.count'], len(recalled['memories']))
        self.assertEqual(save.attributes['gen_ai.memory.record.count'], 1)
        self.assertTrue(all(s.status.status_code == StatusCode.UNSET for s in spans))
        serialized = '\n'.join(s.to_json() for s in spans) + repr(self.reader.get_metrics_data())
        for forbidden in (secret, 'secret-session', 'private-query', 'private-updated',
                          str(self.db), saved['saved'][0]['id'], updated['id']):
            self.assertNotIn(forbidden, serialized)
        data = self.reader.get_metrics_data()
        metrics = [m for r in data.resource_metrics for s in r.scope_metrics for m in s.metrics]
        self.assertEqual({m.name for m in metrics}, {'tinycontext.operation.duration',
                                                   'tinycontext.operation.result.count'})
        duration = next(m for m in metrics if m.name.endswith('duration'))
        self.assertEqual(duration.unit, 's')
        self.assertTrue(all(p.sum >= 0 for p in duration.data.data_points))

    def test_errors_propagate_without_sensitive_events_or_status_descriptions(self):
        failure = ValueError('secret memory and https://private/?api_key=secret')
        with patch('tinycontext.services.embedding_service._embed_texts_onnx', side_effect=failure):
            with self.assertRaises(ValueError) as caught:
                core.save_memories([MemoryInput(content='private input')], config=self.config)
        self.assertIs(caught.exception, failure)
        for span in self.spans():
            if span.name in {'tinycontext.save_memories', 'tinycontext.embed_texts'}:
                self.assertEqual(span.status.status_code, StatusCode.ERROR)
                self.assertEqual(span.attributes['error.type'], 'builtins.ValueError')
                self.assertIsNone(span.status.description)
                self.assertEqual(span.events, ())
            self.assertNotIn('secret', span.to_json())
        data = self.reader.get_metrics_data()
        points = [p for r in data.resource_metrics for s in r.scope_metrics for m in s.metrics
                  if m.name.endswith('duration') for p in m.data.data_points]
        self.assertEqual(sum(p.count for p in points if p.attributes.get('error.type') == 'builtins.ValueError'), 2)

    def test_sdk_disabled_suppresses_application_provider(self):
        with patch.dict(os.environ, {'OTEL_SDK_DISABLED': 'true'}):
            core.recall_memories(config=self.config)
        self.assertEqual(self.spans(), ())
        self.assertIsNone(self.reader.get_metrics_data())

    def test_application_providers_preserved_and_not_shutdown(self):
        with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://unused'}), \
                patch.object(telemetry, '_configured', False), \
                patch.object(telemetry, '_owned_providers', []), \
                patch.object(telemetry, '_build_provider') as build, \
                patch.object(self.tracer_provider, 'shutdown') as stop:
            telemetry.configure_from_environment()
            telemetry.shutdown()
            build.assert_not_called()
            stop.assert_not_called()

    def test_background_reindex_inherits_trace(self):
        from tinycontext.services.embedding_reindex_service import ensure_background_reindex, wait_for_reindex
        core.save_memories([MemoryInput(content='Python backend')], config=self.config)
        with telemetry.operation('trigger'):
            ensure_background_reindex(self.db, embedding_model='balanced', models_dir=self.db.parent,
                                      embedding_batch_size=16, document_prefix='')
            wait_for_reindex(self.db, timeout=5)
        worker = next(s for s in self.spans() if s.name == 'tinycontext.reindex')
        trigger = next(s for s in self.spans() if s.name == 'tinycontext.trigger')
        self.assertEqual(worker.parent.span_id, trigger.context.span_id)
        self.assertEqual(worker.context.trace_id, trigger.context.trace_id)

    def test_both_transports_emit_core_spans(self):
        from tinycontext.servers import fastapi_server, mcp_server
        async def exercise():
            with patch.object(fastapi_server, 'load_context_config', return_value=self.config):
                await fastapi_server.save_memories_endpoint(fastapi_server.SaveMemoriesRequest(
                    memories=[fastapi_server.MemoryInputModel(content='Python backend')]))
            with patch.object(mcp_server, 'load_context_config', return_value=self.config):
                function = getattr(mcp_server.recall_memories_tool, 'fn', mcp_server.recall_memories_tool)
                await function('Python backend')
        asyncio.run(exercise())
        self.assertIn('tinycontext.save_memories', {s.name for s in self.spans()})
        self.assertIn('tinycontext.recall_memories', {s.name for s in self.spans()})

    def test_telemetry_failure_does_not_change_memory_result(self):
        with patch.object(self.tracer_provider, 'get_tracer', side_effect=RuntimeError('broken SDK')):
            result = core.recall_memories(config=self.config)
        self.assertEqual(result['memories'], [])


class _Receiver(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.received.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def log_message(self, *_args):
        pass


@unittest.skipUnless(_HAS_SDK, 'requires telemetry extra')
class BootstrapTests(unittest.TestCase):
    def test_otlp_http_exports_decodable_traces_metrics_and_resources(self):
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
        _Receiver.received = []
        receiver = ThreadingHTTPServer(('127.0.0.1', 0), _Receiver)
        thread = threading.Thread(target=receiver.serve_forever, daemon=True)
        thread.start()
        try:
            process = _isolated('''
from tinycontext.telemetry import configure_from_environment, shutdown, instrument
configure_from_environment()
@instrument('test', result='length')
def test():
    return ['private-body']
test()
shutdown()
shutdown()
''', {'OTEL_EXPORTER_OTLP_ENDPOINT': f'http://127.0.0.1:{receiver.server_port}',
      'OTEL_SERVICE_NAME': 'tinycontext-test', 'OTEL_RESOURCE_ATTRIBUTES': 'deployment.environment.name=test',
      'OTEL_EXPORTER_OTLP_HEADERS': 'authorization=collector-secret',
      'OTEL_METRIC_EXPORT_INTERVAL': '60000'})
        finally:
            receiver.shutdown()
            receiver.server_close()
            thread.join(5)
        self.assertEqual(process.returncode, 0, process.stderr)
        paths = {path for path, _, _ in _Receiver.received}
        self.assertEqual(paths, {'/v1/traces', '/v1/metrics'})
        for path, headers, body in _Receiver.received:
            self.assertEqual(headers.get('authorization'), 'collector-secret')
            message = (ExportTraceServiceRequest() if path.endswith('traces') else ExportMetricsServiceRequest())
            message.ParseFromString(body)
            self.assertIn('tinycontext-test', str(message))
            self.assertIn('deployment.environment.name', str(message))
            self.assertNotIn('private-body', str(message))
            self.assertNotIn('collector-secret', str(message))

    def test_protocol_and_signal_specific_precedence(self):
        for signal in ('TRACES', 'METRICS'):
            for protocol, transport in [('http/protobuf', 'http'), ('grpc', 'grpc')]:
                with self.subTest(signal=signal, protocol=protocol), patch.dict(os.environ, {
                    'OTEL_EXPORTER_OTLP_PROTOCOL': 'invalid-common',
                    f'OTEL_EXPORTER_OTLP_{signal}_PROTOCOL': protocol,
                }, clear=True):
                    exporter = telemetry._exporter(signal)
                    self.assertIn(f'.proto.{transport}.', type(exporter).__module__)
                    exporter.shutdown()

    def test_invalid_protocol_does_not_disable_other_signal_or_leak(self):
        process = _isolated('''
from tinycontext import telemetry
telemetry.configure_from_environment()
assert len(telemetry._owned_providers) == 1
assert type(telemetry._owned_providers[0]).__name__ == 'MeterProvider'
telemetry.shutdown()
''', {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://private.invalid',
      'OTEL_EXPORTER_OTLP_TRACES_PROTOCOL': 'secret-unsupported'})
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertNotIn('private.invalid', process.stderr)
        self.assertNotIn('secret-unsupported', process.stderr)

    def test_server_lifecycle_flushes_even_on_failure(self):
        from tinycontext.servers import fastapi_server, mcp_server
        async def exercise():
            async with fastapi_server._lifespan(fastapi_server.app):
                raise ValueError('test')
        with patch.object(fastapi_server, 'configure_from_environment') as configure, \
                patch.object(fastapi_server, 'shutdown_telemetry') as shutdown, \
                patch.object(fastapi_server, '_prepare_embedding_model'):
            with self.assertRaises(ValueError):
                asyncio.run(exercise())
            configure.assert_called_once()
            shutdown.assert_called_once()
        with patch.object(mcp_server, 'configure_from_environment') as configure, \
                patch.object(mcp_server, 'shutdown_telemetry') as shutdown, \
                patch.object(mcp_server, '_run_server', side_effect=ValueError('test')):
            with self.assertRaises(ValueError):
                mcp_server.main()
            configure.assert_called_once()
            shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()
