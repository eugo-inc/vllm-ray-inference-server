"""The scale-out settings contract: the keys and their defaults, the rules both the values check and the charts
apply, the rhvp-scale log line, the per-server record and the comparison's mode. None of them need Ray, a GPU
or a cluster; the render tests use the one render of tests/conftest.py and are skipped the same way.

These tests hold only what stays true once the scale-out work is built: what the comparison refuses until then,
and whether a launch must carry `servers`, are tested in test_compare_runner.py."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml

from chart_helpers import GLOBAL, REPO, docs, one

sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO / "scripts"))

import pattern_settings  # noqa: E402
from rhvp_compare import logs, plan as plans, servers  # noqa: E402

SERVER_VALUES = yaml.safe_load((REPO / "charts/all/ray-serve-llm/values.yaml").read_text())
SCHEMA = json.loads((REPO / "notebooks/rhvp_compare/results.schema.json").read_text())


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rhvp_scale = load_module("rhvp_scale", REPO / "inference-server/rhvp_scale.py")


def merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        return {**base, **{k: merge(base[k], v) if k in base else v for k, v in over.items()}}
    return over


def chart_values(**over) -> dict:
    """The server chart's values as Helm sees them with values-global.yaml, plus `over`."""
    return merge(merge(copy.deepcopy(SERVER_VALUES), GLOBAL), over)


# ---- keys and defaults ------------------------------------------------------------------------


def test_defaults_change_nothing():
    server = SERVER_VALUES["server"]
    scaling = server["autoscaling"]
    assert scaling["minReplicas"] == 1 and scaling["maxReplicas"] == 1
    assert "targetOngoingRequests" in scaling and scaling["targetOngoingRequests"] is None
    assert "minReplicas" not in server and "maxReplicas" not in server
    assert server["gpuPerReplica"] == 1 and server["gpuCount"] == 1
    assert server["metricsPort"] == 8080
    assert server["streamBatchingIntervalMs"] == 0
    assert SERVER_VALUES["compileCache"]["mode"] == "none"
    assert SERVER_VALUES["modelStorage"]["accessMode"] == "ReadWriteOnce"
    assert set(SERVER_VALUES["scaleOut"]) >= {"minPods", "maxPods", "waitingRequestsPerPod", "pollingIntervalSeconds",
                                              "cooldownSeconds", "thanosUrl"}
    assert SERVER_VALUES["global"]["scaleOut"]["mode"] == "none"
    assert GLOBAL["global"]["scaleOut"]["mode"] == "none"
    assert GLOBAL["global"]["compare"]["mode"] == "levels"
    assert pattern_settings.check(chart_values()) == []


def test_values_check_names_every_mode():
    assert pattern_settings.SCALE_OUT_MODES == ("none", "keda")
    assert pattern_settings.COMPARE_MODES == plans.MODES == ("levels", "scaleout")
    assert pattern_settings.COMPILE_CACHE_MODES == ("none", "node", "shared")


# ---- the rules ------------------------------------------------------------------------------


STREAM_INTERVAL = ("server.streamBatchingIntervalMs must be null, 0, or a number of milliseconds from 0.001 to "
                   "60000")


@pytest.mark.parametrize("over, message", [
    ({"global": {"scaleOut": {"mode": "hpa"}}}, "global.scaleOut.mode must be none or keda; got hpa"),
    ({"global": {"scaleOut": {"mode": False}}}, "global.scaleOut.mode must be none or keda"),
    ({"global": {"compare": {"mode": "surge"}}},
     "global.compare.mode must be levels or scaleout; got surge"),
    ({"server": {"minReplicas": 2}}, pattern_settings.MOVED),
    ({"server": {"autoscaling": {"minReplicas": 0}}}, "server.autoscaling.minReplicas must be a whole number"),
    ({"server": {"autoscaling": {"maxReplicas": 1.5}}}, "server.autoscaling.maxReplicas must be a whole number"),
    ({"server": {"autoscaling": {"minReplicas": 2}}}, "server.autoscaling.maxReplicas must be a whole number"),
    ({"server": {"autoscaling": {"maxReplicas": 2}}}, pattern_settings.NO_TARGET),
    ({"server": {"autoscaling": {"targetOngoingRequests": 0}}}, "targetOngoingRequests must be a number above 0"),
    ({"server": {"autoscaling": {"maxReplicas": 2, "targetOngoingRequests": 4}}},
     "server.autoscaling.maxReplicas (2) times server.gpuPerReplica (1) is more than server.gpuCount (1)"),
    ({"server": {"autoscaling": {"maxReplicas": 2, "targetOngoingRequests": 4}, "gpuPerReplica": 0.5}},
     pattern_settings.SHARED_GPU_MEMORY.format(memory="0.85", share="0.5", high="2")),
    ({"server": {"autoscaling": {"maxReplicas": 2, "targetOngoingRequests": 4}, "gpuPerReplica": 0.5,
                 "engineKwargs": {"gpu_memory_utilization": 0.5}}},
     pattern_settings.SHARED_GPU_MEMORY.format(memory="0.5", share="0.5", high="2")),
    ({"server": {"gpuPerReplica": 2, "gpuCount": 2}}, pattern_settings.ONE_WORKER_SHARE.format(share="2")),
    ({"server": {"gpuPerReplica": 0.5, "engineKwargs": {"tensor_parallel_size": 2}, "gpuCount": 2}},
     pattern_settings.GPU_PER_WORKERS.format(share="0.5", workers="2")),
    ({"server": {"gpuPerReplica": 0}}, "server.gpuPerReplica must be a number above 0"),
    ({"server": {"gpuPerReplica": "0.5"}}, "server.gpuPerReplica must be a number above 0; got 0.5 (string)"),
    ({"server": {"metricsPort": 8000}}, "server.metricsPort must be a port number"),
    ({"server": {"streamBatchingIntervalMs": -1}}, STREAM_INTERVAL + "; got -1 (float64)"),
    ({"server": {"streamBatchingIntervalMs": -0.5}}, STREAM_INTERVAL + "; got -0.5 (float64)"),
    ({"server": {"streamBatchingIntervalMs": "50"}}, STREAM_INTERVAL + "; got 50 (string)"),
    ({"server": {"streamBatchingIntervalMs": True}}, STREAM_INTERVAL + "; got true (bool)"),
    ({"server": {"streamBatchingIntervalMs": float("nan")}}, STREAM_INTERVAL + "; got nan (float64)"),
    ({"server": {"streamBatchingIntervalMs": float("inf")}}, STREAM_INTERVAL + "; got inf (float64)"),
    # Outside 0.001 to 60000: the chart's toJson writes 1e-7 and 1e+21 in exponent form, read back as text.
    ({"server": {"streamBatchingIntervalMs": 1e-7}}, STREAM_INTERVAL + "; got 1e-07 (float64)"),
    ({"server": {"streamBatchingIntervalMs": 0.0005}}, STREAM_INTERVAL + "; got 0.0005 (float64)"),
    ({"server": {"streamBatchingIntervalMs": 60000.5}}, STREAM_INTERVAL + "; got 60000.5 (float64)"),
    ({"server": {"streamBatchingIntervalMs": 1e21}}, STREAM_INTERVAL + "; got 1e+21 (float64)"),
    ({"server": {"streamBatchingIntervalMs": 10**400}}, STREAM_INTERVAL + "; got "),
    ({"compileCache": {"mode": "disk"}}, "compileCache.mode must be none, node or shared"),
    ({"modelStorage": {"accessMode": "ReadOnlyMany"}}, "modelStorage.accessMode must be ReadWriteOnce or ReadWriteMany"),
    ({"global": {"scaleOut": {"mode": "keda"}}, "scaleOut": {"maxPods": 0}}, "scaleOut.maxPods must be a whole number"),
    ({"global": {"scaleOut": {"mode": "keda"}}, "scaleOut": {"waitingRequestsPerPod": 0}},
     "scaleOut.waitingRequestsPerPod must be a number above 0; got 0"),
    ({"global": {"scaleOut": {"mode": "keda"}}, "scaleOut": {"pollingIntervalSeconds": 0.5}},
     "scaleOut.pollingIntervalSeconds must be a whole number of at least 1; got 0.5"),
    ({"global": {"scaleOut": {"mode": "keda"}}, "scaleOut": {"cooldownSeconds": 7200}},
     "scaleOut.cooldownSeconds must be a whole number from 0 to 3600; got 7200"),
    ({"global": {"scaleOut": {"mode": "keda"}}, "scaleOut": {"thanosUrl": "thanos-querier:9092"}},
     "scaleOut.thanosUrl must be an http:// or https:// address; got thanos-querier:9092"),
])
def test_values_check_refuses(over, message):
    problems = pattern_settings.check(chart_values(**over))
    assert len(problems) == 1 and message in problems[0], problems


def test_the_shared_gpu_reserve_is_the_same_everywhere():
    """The chart, the values check, the server and the comparison's plan all leave 0.025 of a shared GPU's memory
    per replica outside its share."""
    sys.path.insert(0, str(REPO / "inference-server"))  # serve_app imports rhvp_scale from its own folder
    serve_app = load_module("serve_app", REPO / "inference-server/serve_app.py")
    assert pattern_settings.SHARED_GPU_RESERVE == serve_app.SHARED_GPU_RESERVE == plans.SHARED_GPU_RESERVE == 0.025
    template = (REPO / "charts/all/ray-serve-llm/templates/_settings.tpl").read_text()
    assert "(subf $gpr 0.025)" in template and "minus 0.025 when up to" in template
    assert "minus 0.025 when up to" in pattern_settings.SHARED_GPU_MEMORY


@pytest.mark.parametrize("replicas, share, memory, refused", [
    (2, 0.5, 0.475, False),
    (2, 0.5, 0.476, True),
    (4, 0.25, 0.225, False),
    (4, 0.25, 0.2375, True),  # 0.95 of the share: the room left would not grow with the replica count
    (10, 0.1, 0.075, False),
    (10, 0.1, 0.095, True),
])
def test_the_shared_gpu_reserve_grows_with_the_replica_count(replicas, share, memory, refused):
    over = {"server": {"autoscaling": {"maxReplicas": replicas, "targetOngoingRequests": 2}, "gpuPerReplica": share,
                       "engineKwargs": {"gpu_memory_utilization": memory}}}
    problems = pattern_settings.check(chart_values(**over))
    assert (problems != []) == refused, problems
    if refused:
        assert problems[0].startswith(f"server.engineKwargs.gpu_memory_utilization ({memory:g}) must be at most "
                                      f"server.gpuPerReplica ({share:g}) minus 0.025 when up to {replicas} replicas")
    # The free memory the last of `replicas` replicas finds at vLLM's check, for each replica's own share, is then
    # enough for one CUDA context per replica of up to 0.025 of the GPU: 1 - k * gmu >= k * 0.025.
    fits = 1 - replicas * memory >= replicas * pattern_settings.SHARED_GPU_RESERVE - 1e-9
    assert fits != refused


@pytest.mark.parametrize("over", [
    {"server": {"autoscaling": {"maxReplicas": 2, "targetOngoingRequests": 4}, "gpuPerReplica": 0.5,
                "engineKwargs": {"gpu_memory_utilization": 0.45}}},
    {"server": {"autoscaling": {"maxReplicas": 4, "targetOngoingRequests": 8}, "gpuCount": 4}},
    {"server": {"autoscaling": {"maxReplicas": 3, "targetOngoingRequests": 2}, "gpuPerReplica": 1 / 3,
                "engineKwargs": {"gpu_memory_utilization": 0.3}}},
    {"server": {"autoscaling": {"maxReplicas": 2, "targetOngoingRequests": 4}, "gpuPerReplica": 0.5,
                "engineKwargs": {"gpu_memory_utilization": 0.475}}},
    {"server": {"gpuPerReplica": 2, "gpuCount": 2, "engineKwargs": {"tensor_parallel_size": 2}}},
    {"server": {"gpuCount": 2, "engineKwargs": {"tensor_parallel_size": 2}}},
    {"global": {"scaleOut": {"mode": "keda"}}, "modelStorage": {"accessMode": "ReadWriteMany"}},
    {"compileCache": {"mode": "shared"}},
    {"global": {"compare": {"mode": "scaleout"}}},
    {"server": {"streamBatchingIntervalMs": 0.0}},
    {"server": {"streamBatchingIntervalMs": 12.5}},
    {"server": {"streamBatchingIntervalMs": 50}},
    {"server": {"streamBatchingIntervalMs": 0.001}},
    {"server": {"streamBatchingIntervalMs": 60000}},
    {"server": {"streamBatchingIntervalMs": None}},
])
def test_values_check_accepts(over):
    assert pattern_settings.check(chart_values(**over)) == []


@pytest.mark.parametrize("value, shown", [
    # Helm's %v for each, from a render with it as server.streamBatchingIntervalMs or server.metricsPort.
    (1e6, "1e+06"), (1000000, "1e+06"), (1234567, "1.234567e+06"), (-1e6, "-1e+06"),
    (123456789.5, "1.234567895e+08"), (1e21, "1e+21"), (999999.5, "999999.5"), (70000, "70000"),
    (60000.5, "60000.5"), (12.5, "12.5"), (0.0001, "0.0001"), (0.00001, "1e-05"), (1e-7, "1e-07"),
])
def test_values_check_shows_numbers_as_helm_does(value, shown):
    assert pattern_settings._show(value) == shown


HELM_LOADER = pattern_settings.helm_loader(yaml)


@pytest.mark.parametrize("text, value", [
    # Helm reads each of these as a number (a render with it as server.streamBatchingIntervalMs); PyYAML alone as text.
    ("5e1", 50.0), ("1e-3", 0.001), ("1.0e3", 1000.0), ("1e+3", 1000.0), (".5e2", 50.0), ("1_0e1", 100.0),
    ("-1e6", -1e6), ("1e21", 1e21), ("1e-7", 1e-7), ("1e-999", 0.0), ("1.5_e1", 15.0),
    ("-.5e+1", -5.0), (".5_5e1", 5.5),
    # Text in Helm too: beyond float64's range, or an underscore where a digit must be.
    ("1e999", "1e999"), ("-1e999", "-1e999"), ("1.0e+400", "1.0e+400"), ("._e1", "._e1"), ("._5e1", "._5e1"),
    (".5_e1", ".5_e1"), (".5__5e1", ".5__5e1"), (".inf", float("inf")),
    # Read as before.
    ("'5e1'", "5e1"), ("1e", "1e"), ("e5", "e5"), ("1e5x", "1e5x"), ("1.2.3", "1.2.3"),
    ("\u0665e\u0661", "\u0665e\u0661"), ("1.5e-7", 1.5e-7), ("12.5", 12.5), ("50", 50), ("0x10", 16), ("null", None),
    ("true", True),
])
def test_exponent_numbers_are_read_as_helm_reads_them(text, value):
    got = yaml.load(f"x: {text}\n", Loader=HELM_LOADER)["x"]
    assert got == value and type(got) is type(value)


def test_the_helm_loader_reads_every_values_file_as_before():
    """Only a number in exponent form reads differently, and no values file in the repo has one."""
    files = [REPO / "values-global.yaml", *REPO.glob("variants/*/*.yaml"), *REPO.glob("overrides/*.yaml"),
             *REPO.glob("charts/all/*/values.yaml")]
    assert len(files) > 5
    for path in files:
        text = path.read_text()
        assert yaml.load(text, Loader=HELM_LOADER) == yaml.safe_load(text), path


def validate_values(tmp_path, rel: str, old: str, new: str) -> subprocess.CompletedProcess:
    """scripts/validate-values.sh on a copy of the repo with `old` replaced by `new` in `rel` (needs its pinned
    schemas from the network: skips without them)."""
    copy_root = tmp_path / "repo"
    for name in subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                               check=True).stdout.split():
        if (REPO / name).is_file():
            (copy_root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / name, copy_root / name)
    for name in ("scripts/pattern_settings.py", "scripts/validate-values.sh", "values-global.yaml",
                 "charts/all/ray-serve-llm/values.yaml"):
        shutil.copy(REPO / name, copy_root / name)
    text = (copy_root / rel).read_text()
    assert text.count(old) == 1
    (copy_root / rel).write_text(text.replace(old, new))
    proc = subprocess.run([str(copy_root / "scripts/validate-values.sh")], capture_output=True, text=True,
                          env={**os.environ, "PYTHON": sys.executable}, timeout=300)
    if "sha256" in proc.stderr or "urlopen" in proc.stderr or "URLError" in proc.stderr:
        pytest.skip("validate-values.sh could not download its schemas")
    return proc


def test_values_check_runs_in_validate_values(tmp_path):
    """validate-values.sh refuses an unknown scale-out mode in values-global.yaml."""
    proc = validate_values(tmp_path, "values-global.yaml", "  scaleOut:\n    mode: none\n",
                           "  scaleOut:\n    mode: hpa\n")
    assert proc.returncode == 1
    assert "scale-out settings, AWS: global.scaleOut.mode must be none or keda; got hpa" in proc.stderr


@pytest.mark.parametrize("written, refusal", [
    ("5e1", None),
    ("1.0e3", None),
    ("1e-7", STREAM_INTERVAL + "; got 1e-07 (float64)"),
])
def test_validate_values_reads_exponent_numbers_as_helm_does(tmp_path, written, refusal):
    """A number in exponent form in a values file is a number in validate-values.sh, as in the chart: 5e1 passes, and
    1e-7 is refused as a number, with the chart's message."""
    proc = validate_values(tmp_path, "charts/all/ray-serve-llm/values.yaml", "  streamBatchingIntervalMs: 0\n",
                           f"  streamBatchingIntervalMs: {written}\n")
    if refusal is None:
        assert proc.returncode == 0, proc.stderr
        assert "ok   scale-out settings, AWS\n" in proc.stdout
    else:
        assert proc.returncode == 1
        assert f"scale-out settings, AWS: {refusal}" in proc.stderr


# ---- the charts refuse the same settings --------------------------------------------------------


@pytest.mark.parametrize("case, message", [
    ("scaleout-unknown", "global.scaleOut.mode must be none or keda; got hpa"),
    ("autoscaling-no-target", pattern_settings.NO_TARGET),
    ("autoscaling-over-gpus",
     "server.autoscaling.maxReplicas (2) times server.gpuPerReplica (1) is more than server.gpuCount (1)"),
    ("autoscaling-shared-gpu-memory", pattern_settings.SHARED_GPU_MEMORY.format(memory="0.85", share="0.5", high="2")),
    ("autoscaling-shared-gpu-full-share",
     pattern_settings.SHARED_GPU_MEMORY.format(memory="0.5", share="0.5", high="2")),
    ("gpu-per-replica-one-worker", pattern_settings.ONE_WORKER_SHARE.format(share="2")),
    ("gpu-per-replica-workers", pattern_settings.GPU_PER_WORKERS.format(share="0.5", workers="2")),
    ("keda-threshold", "scaleOut.waitingRequestsPerPod must be a number above 0; got 0"),
    ("keda-thanos-url", "scaleOut.thanosUrl must be an http:// or https:// address; got thanos-querier:9092"),
    ("keda-cooldown", "scaleOut.cooldownSeconds must be a whole number from 0 to 3600; got 7200"),
    ("replicas-moved", pattern_settings.MOVED),
    ("stream-interval-string", STREAM_INTERVAL + "; got 50 (string)"),
    ("stream-interval-negative", STREAM_INTERVAL + "; got -5 (float64)"),
    ("stream-interval-tiny", STREAM_INTERVAL + "; got 1e-07 (float64)"),
    ("stream-interval-huge", STREAM_INTERVAL + "; got 1e+21 (float64)"),
    ("compare-mode-unknown",
     "global.compare.mode must be levels or scaleout; got surge"),
])
def test_charts_refuse_with_the_values_check_message(render, case, message):
    assert message in (render / f"refused/{case}.txt").read_text()


def test_replica_counts_come_from_server_autoscaling(server, render):
    llm = yaml.safe_load(one(server, "ConfigMap", "ray-serve-llm-config")["data"]["llm.yaml"])
    assert (llm["min_replicas"], llm["max_replicas"]) == (1, 1)
    scaling = docs(render / "charts/ray-serve-llm-scaling.yaml")
    llm = yaml.safe_load(one(scaling, "ConfigMap", "ray-serve-llm-config")["data"]["llm.yaml"])
    assert (llm["min_replicas"], llm["max_replicas"]) == (1, 2)


# ---- the rhvp-scale line ---------------------------------------------------------------------


def test_scale_line_shape_is_shared():
    assert logs.SCALE_EVENT == rhvp_scale.EVENT == "rhvp-scale"
    assert logs.SCALE_FIELDS == rhvp_scale.FIELDS == ("event", "ts", "app", "deployment", "from", "to", "reason")
    assert logs.SCALE_REASONS == rhvp_scale.REASONS == ("target", "running")
    assert logs.SCALE_MARKER == rhvp_scale.MARKER


def test_scale_line_round_trip():
    line = rhvp_scale.scale_event("llm", "LLMServer:qwen", 1, 2, "target", ts=1794551388.12345)
    assert line == ('{"event":"rhvp-scale","ts":1794551388.123,"app":"llm","deployment":"LLMServer:qwen",'
                    '"from":1,"to":2,"reason":"target"}')
    assert list(json.loads(line)) == list(rhvp_scale.FIELDS)
    log = "\n".join([
        "2026-11-13T06:10:00.000000000Z INFO starting",
        f"2026-11-13T06:10:01.500000000Z 2026-11-13 06:10:01,500 INFO rhvp: {line}",
        "2026-11-13T06:10:30.250Z (ServeController pid=12) "
        + rhvp_scale.scale_event("llm", "LLMServer:qwen", 1, 2, "running", ts=1794551430.25) + " trailing text",
        rhvp_scale.scale_event("llm", "LLMServer:qwen", 2, 1, "target", ts=1794551500),
    ])
    events = logs.scale_events(log)
    assert [(e["from"], e["to"], e["reason"]) for e in events] == [(1, 2, "target"), (1, 2, "running"),
                                                                   (2, 1, "target")]
    def epoch(text: str) -> float:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()

    assert events[0]["ts"] == 1794551388.123 and events[0]["log_ts"] == pytest.approx(epoch("2026-11-13T06:10:01.5"))
    assert events[1]["log_ts"] == pytest.approx(epoch("2026-11-13T06:10:30.25"))
    assert events[2]["log_ts"] is None and isinstance(events[2]["ts"], float)
    assert set(events[0]) == set(rhvp_scale.FIELDS) | {"log_ts"}


def test_scale_parser_reads_the_event_inside_a_ray_json_log_record():
    # The shape of a real line from the server's pod log on OpenShift (Ray's JSON log encoding): the event is quoted
    # inside "message". Ids and the node address are made up.
    inner = rhvp_scale.scale_event("llm", "LLMServer:qwen", 1, 2, "target", ts=1795288958.132)
    record = {"asctime": "2026-11-21 19:22:38,131", "levelname": "INFO", "message": inner,
              "filename": "serve_app.py", "lineno": 883, "process": 2, "timestamp_ns": 1795288958131943615}
    line = json.dumps(record)
    assert '\\"event\\":\\"rhvp-scale\\"' in line and logs.SCALE_MARKER not in line
    events = logs.scale_events("\n".join([line, f"2026-11-21T19:24:03.050Z {json.dumps(dict(record, message=rhvp_scale.scale_event('llm', 'LLMServer:qwen', 1, 2, 'running', ts=1795289043.05)))}"]))
    assert [(e["from"], e["to"], e["reason"]) for e in events] == [(1, 2, "target"), (1, 2, "running")]
    assert events[0]["log_ts"] == pytest.approx(1795288958.131943615)
    assert events[1]["log_ts"] == pytest.approx(datetime(2026, 11, 21, 19, 24, 3, 50000, tzinfo=timezone.utc).timestamp())
    assert logs.scale_events(json.dumps(dict(record, message="rhvp-scale is a word here"))) == []
    assert logs.scale_events(json.dumps(dict(record, message=inner[:-5]))) == []
    assert logs.scale_events('{"message": 3, "x": "rhvp-scale"}') == []


@pytest.mark.parametrize("text", [
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","from":1,"to":2}',
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","from":1,"to":2,"reason":"load"}',
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","from":true,"to":2,"reason":"target"}',
    '{"event":"rhvp-scale","ts":"now","app":"llm","deployment":"d","from":1,"to":2,"reason":"target"}',
    '{"event":"rhvp-scale","ts":1.0,"app":"","deployment":"d","from":1,"to":2,"reason":"target"}',
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","from":-1,"to":2,"reason":"target"}',
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","fr',
    '{"event": "rhvp-scale", "ts": 1.0, "app": "llm", "deployment": "d", "from": 1, "to": 2, "reason": "target"}',
])
def test_scale_parser_leaves_out_lines_outside_the_shape(text):
    assert logs.scale_events(f"2026-11-13T06:10:00Z {text}") == []


@pytest.mark.parametrize("args", [
    ("", "d", 1, 2, "target"), ("llm", "", 1, 2, "target"), ("llm", "d", -1, 2, "target"),
    ("llm", "d", 1, True, "target"), ("llm", "d", 1, 2, "load"), ("llm", "d", 1.0, 2, "target"),
])
def test_scale_line_refuses_values_outside_the_shape(args):
    with pytest.raises(ValueError):
        rhvp_scale.scale_event(*args, ts=1.0)
    with pytest.raises(ValueError):
        rhvp_scale.scale_event("llm", "d", 1, 2, "target", ts=float("nan"))


# ---- the per-server record ---------------------------------------------------------------------


def test_server_env_keeps_the_allow_list_only():
    env = {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=0", "CUDA_VISIBLE_DEVICES": "0", "VLLM_CACHE_ROOT": "/cache/vllm",
           "TORCH_CUDA_ARCH_LIST": "8.9", "HF_HOME": "/models/hf", "HF_TOKEN": "hf_x", "HF_TOKEN_PATH": "/x",
           "VLLM_API_KEY": "k", "VLLM_TOKENIZER_POOL_SIZE": "0", "HUGGING_FACE_HUB_TOKEN": "hf_y", "PATH": "/usr/bin",
           "LD_PRELOAD": "x.so", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HOME": "/cache/home"}
    kept, dropped = servers.recorded_env(env)
    assert kept == {"CUDA_VISIBLE_DEVICES": "0", "GLIBC_TUNABLES": "glibc.malloc.hugetlb=0", "HF_HOME": "/models/hf",
                    "TORCH_CUDA_ARCH_LIST": "8.9", "VLLM_CACHE_ROOT": "/cache/vllm",
                    "VLLM_TOKENIZER_POOL_SIZE": "0"}
    assert dropped == ["HF_TOKEN", "HF_TOKEN_PATH", "VLLM_API_KEY"]
    assert "hf_x" not in json.dumps(servers.record(1, None, [], env, "process"))


def test_server_record_matches_the_schema():
    override = servers.record(1, ["/opt/app-root/src/entrypoint.sh"], ["--x"], {"VLLM_FOO": "1"}, "container",
                              started_at="2026-11-13T06:10:00Z", image_entrypoint=["/usr/bin/tini"],
                              image_cmd=None)
    assert override["entrypoint"] == "override" and override["env"] == {"VLLM_FOO": "1"}
    image = servers.record(2, None, ["--model", "m"], None, "process")
    assert image["entrypoint"] == "image" and image["env_source"] == "none" and image["env"] == {}
    # Preset ray-serve-llm sets the server image's own ENTRYPOINT as its command: that runs the image's entrypoint.
    own = servers.record(1, ["/opt/app-root/src/entrypoint.sh"], [], None, "none",
                         image_entrypoint=["/opt/app-root/src/entrypoint.sh"])
    assert own["entrypoint"] == "image"
    unknown = servers.record(1, ["/opt/app-root/src/entrypoint.sh"], [], None, "none")
    assert unknown["entrypoint"] == "override"
    validator = jsonschema.Draft202012Validator({"$defs": SCHEMA["$defs"], "$ref": "#/$defs/server"})
    for item in (override, image):
        validator.validate(item)
    assert set(SCHEMA["$defs"]["server"]["required"]) <= set(override)
    assert set(SCHEMA["$defs"]["server"]["properties"]) == set(override)
    assert SCHEMA["$defs"]["launch"]["properties"]["servers"]["items"] == {"$ref": "#/$defs/server"}
    with pytest.raises(ValueError):
        servers.record(0, None, [], None, "none")
    with pytest.raises(ValueError):
        servers.record(1, None, [], {}, "proc")


def test_side_extra_env_is_in_the_plan():
    config = json.loads((REPO / "tests/fixtures/compare-local-mock.json").read_text())
    config["sides"][1]["env"] = {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=0"}
    built = plans.build(config, "docker")
    assert built["sides"][1]["env"] == {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=0"}
    assert plans.side_spec(built, built["sides"][1])["env"]["GLIBC_TUNABLES"] == "glibc.malloc.hugetlb=0"


# ---- the comparison's mode --------------------------------------------------------------------


def test_compare_mode_levels_keeps_the_run_id():
    config = json.loads((REPO / "tests/fixtures/compare-local-mock.json").read_text())
    assert "mode" not in config
    assert plans.build(config, "docker")["run_id"] == plans.build({**config, "mode": "levels"}, "docker")["run_id"]


def test_compare_mode_unknown_refused():
    config = json.loads((REPO / "tests/fixtures/compare-local-mock.json").read_text())
    with pytest.raises(plans.PlanError, match="mode must be one of levels, scaleout"):
        plans.build({**config, "mode": "surge"}, "docker")
