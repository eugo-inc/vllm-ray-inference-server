# SPDX-License-Identifier: Apache-2.0
"""rhvp_compare's readers: the GuideLLM report and command line, the server log, GPU power (nvidia-smi and DCGM
through the Thanos Querier), and the platform helpers of the two backends.

The GuideLLM fixture is a trimmed report from GuideLLM v0.8.1 against its own mock server (two levels, 8 s each);
the log lines follow vLLM's own wording.

Run with: python -m pytest -q tests/test_compare_parsers.py
"""

import base64
import json
import os
import re
import signal
import struct
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from rhvp_compare import dockerapi, energy, guidellm, kube, logs, plan as plans, runner, servers  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
REPORT = json.loads((FIXTURES / "guidellm-report-v0.8.1.json").read_text())


def _plan(**load) -> dict:
    config = {"runName": "t", "model": {"id": "m", "source": "org/m", "revision": "abc"},
              "sides": [{"name": "a", "image": "x/a:1"}, {"name": "b", "image": "x/b:1"}],
              "load": {"concurrencies": [1, 4], "secondsPerLevel": 8, **load}}
    return plans.build(config, "docker")


# ---- GuideLLM ----

def test_levels_from_the_report():
    levels = guidellm.levels(REPORT)
    assert [lv["concurrency"] for lv in levels] == [1, 4]
    first = levels[0]
    assert first["requests"] == {"successful": 26, "errored": 0, "incomplete": 1}
    assert first["usable"] and first["note"] is None
    assert first["ttft_ms"]["median"] == pytest.approx(43.40648651123047)
    assert first["itl_ms"]["median"] == pytest.approx(8.25753519611974)
    assert first["output_tokens_per_s"] == pytest.approx(105.125)
    assert first["duration_s"] == pytest.approx(first["end_epoch"] - first["start_epoch"], abs=0.5)
    assert first["started_at"].endswith("Z")
    assert guidellm.version(REPORT) == "0.8.1"


def test_a_level_with_errors_stays_out_of_the_comparison():
    report = json.loads(json.dumps(REPORT))
    report["benchmarks"][1]["metrics"]["request_totals"]["errored"] = 2
    levels = guidellm.levels(report)
    assert not levels[1]["usable"] and "2 requests failed" in levels[1]["note"]
    flat = guidellm.flat_values(levels)
    assert "c1/itl_ms_median" in flat and not any(k.startswith("c4/") for k in flat)


def test_flat_values_take_tokens_per_joule_from_the_energy_entry():
    levels = guidellm.levels(REPORT)
    levels[0]["energy"] = {"tokens_per_joule": 2.5}
    flat = guidellm.flat_values(levels)
    assert flat["c1/tokens_per_joule[gpu]"] == 2.5 and "c4/tokens_per_joule[gpu]" not in flat
    ids = {c["id"] for c in _plan()["cases"]}
    assert set(guidellm.flat_values(guidellm.levels(REPORT))) <= ids  # the ids of the plan's cases


@pytest.mark.parametrize("report", [{}, {"benchmarks": []}, {"benchmarks": [{"metrics": {}}]}])
def test_a_broken_report_is_a_report_error(report):
    with pytest.raises(guidellm.ReportError):
        guidellm.levels(report)


def test_command_line():
    args = guidellm.command(_plan(), "http://side:8000", "/results/r/guidellm.json")
    text = " ".join(args)
    assert args[0] == "run"
    assert ("kind=openai_http,target=http://side:8000,model=m,request_format=/v1/completions,"
            "validate_backend=false") in args
    assert "--override profile.streams 1,4" in text
    assert "kind=max_duration,seconds=8" in args
    assert "kind=huggingface_auto,model=org/m,load_kwargs.revision=abc" in args
    assert "kind=static,value=42" in args
    assert "kind=json,path=/results/r/guidellm.json" in args
    one = guidellm.command(plans.build({**_plan_config(), "load": {"concurrencies": [8]}}, "docker"), "t", "o")
    assert "--override" not in one and "kind=concurrent,streams=8" in one


def _plan_config() -> dict:
    return {"runName": "t", "model": {"id": "m", "source": "org/m"},
            "sides": [{"name": "a", "image": "x/a:1"}, {"name": "b", "image": "x/b:1"}]}


def test_run_stops_a_hung_load_and_reports_a_bad_exit(tmp_path):
    plan = _plan(timeoutSeconds=30)
    plan["load"]["timeout_s"] = 1
    hang = tmp_path / "hang.sh"
    hang.write_text("#!/bin/sh\nsleep 30\n")
    hang.chmod(0o755)
    with pytest.raises(guidellm.LoadError) as caught:
        guidellm.run(plan, "http://side:8000", tmp_path / "out", executable=str(hang))
    assert caught.value.timed_out
    fail = tmp_path / "fail.sh"
    fail.write_text("#!/bin/sh\necho no\nexit 3\n")
    fail.chmod(0o755)
    with pytest.raises(guidellm.LoadError) as caught:
        guidellm.run(plan, "http://side:8000", tmp_path / "out", executable=str(fail))
    assert not caught.value.timed_out and "code 3" in str(caught.value)
    assert (tmp_path / "out" / "guidellm.log").read_text() == "no\n"


class _Halt(BaseException):
    pass


def test_a_stop_while_a_timed_out_guidellm_is_taken_down_kills_it_and_keeps_the_time_out(tmp_path):
    # GuideLLM ignores SIGTERM; the runner's stop lands while run() waits for it to exit after the time-out.
    plan = _plan(timeoutSeconds=30)
    plan["load"]["timeout_s"] = 1
    stubborn = tmp_path / "stubborn.sh"
    stubborn.write_text("#!/bin/sh\necho $$ > \"$(dirname \"$0\")/pid\"\ntrap '' TERM\nwhile :; do sleep 1; done\n")
    stubborn.chmod(0o755)

    def halt(signum, frame):
        raise _Halt()

    previous = signal.signal(signal.SIGALRM, halt)
    signal.setitimer(signal.ITIMER_REAL, 2.0)  # 1 s after the time-out, inside the 15 s wait for SIGTERM to work
    try:
        with pytest.raises(_Halt) as caught:
            guidellm.run(plan, "http://side:8000", tmp_path / "out", executable=str(stubborn))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert caught.value.load_timed_out  # the runner counts the time-out all the same
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # and GuideLLM was killed, not left running


def test_run_returns_the_report(tmp_path):
    fake = tmp_path / "fake.sh"
    fake.write_text('#!/bin/sh\nfor a in "$@"; do case "$a" in kind=json,path=*) echo "{}" > "${a#kind=json,path=}";; '
                    'esac; done\n')
    fake.chmod(0o755)
    path = guidellm.run(_plan(), "http://side:8000", tmp_path / "out", executable=str(fake))
    assert path == tmp_path / "out" / "guidellm.json" and path.read_text() == "{}\n"


# ---- the server log ----

VLLM_LOG = """\
2026-01-01T00:00:03.500000000Z (APIServer pid=1) INFO 01-01 00:00:03 [api_server.py:1] vLLM API server version 0.33.0
2026-01-01T00:00:05.000000000Z (EngineCore pid=7) INFO 01-01 00:00:05 [core.py:1] Initializing a V1 LLM engine (v0.33.0) with config: model='org/m'
2026-01-01T00:00:20.250000Z (EngineCore pid=7) INFO 01-01 00:00:20 [loader.py:1] Loading weights took 12.25 seconds
2026-01-01T00:00:21Z (EngineCore pid=7) INFO 01-01 00:00:21 [runner.py:1] Model loading took 0.93 GiB memory and 13.10 seconds
2026-01-01T00:00:40.000000000Z (EngineCore pid=7) INFO 01-01 00:00:40 [backends.py:1] torch.compile took 15.20 s in total
2026-01-01T00:00:50.000000000Z (EngineCore pid=7) INFO 01-01 00:00:50 [runner.py:2] Graph capturing finished in 4 secs, took 0.10 GiB
2026-01-01T00:00:55.000000000Z (EngineCore pid=7) INFO 01-01 00:00:55 [runner.py:2] Graph capturing finished in 3 secs, took 0.05 GiB
2026-01-01T00:00:56.000000000Z (EngineCore pid=7) INFO 01-01 00:00:56 [core.py:2] init engine (profile, create kv cache, warmup model) took 30.40 seconds
2026-01-01T00:01:00.000000000Z (APIServer pid=1) INFO:     Application startup complete.
"""


def test_breakdown_of_a_vllm_start():
    started = logs.parse_time("2026-01-01T00:00:00Z")
    steps = {entry["phase"]: entry for entry in logs.breakdown(VLLM_LOG, started)}
    assert steps["first_log"]["at_s"] == pytest.approx(3.5)
    assert steps["api_server_start"]["at_s"] == pytest.approx(3.5)
    assert steps["engine_start"]["at_s"] == pytest.approx(5.0)
    assert steps["weights_load"]["seconds"] == pytest.approx(12.25)
    assert steps["model_load"]["seconds"] == pytest.approx(13.10)
    assert steps["torch_compile"]["seconds"] == pytest.approx(15.2)
    assert steps["cuda_graphs"]["seconds"] == pytest.approx(7.0) and steps["cuda_graphs"]["count"] == 2
    assert steps["engine_init"]["seconds"] == pytest.approx(30.4)
    assert steps["app_startup_complete"]["at_s"] == pytest.approx(60.0)
    assert all(entry["source"] == "log" for entry in steps.values())
    assert logs.server_version(logs.lines(VLLM_LOG)) == "0.33.0"


def test_log_lines_without_time_and_with_colour():
    parsed = logs.lines("plain line\n\x1b[32m2026-01-01T00:00:00.123456789+00:00 green\x1b[0m\n")
    assert parsed[0] == (None, "plain line")
    assert parsed[1][1] == "green" and parsed[1][0] == pytest.approx(logs.parse_time("2026-01-01T00:00:00.123456Z"))
    assert logs.breakdown("", 0.0) == []
    assert logs.server_version([(None, "nothing here")]) is None


# ---- engine restarts inside one start of the pattern's Ray Serve LLM server ----
#
# The fixtures are rebuilt from the line shapes of stock vLLM 0.33.0 / Ray 2.60.0 server logs of the pattern's app on a
# GPU whose memory is the machine's own. Process ids, replica ids, times, durations, memory figures and the memory
# setting are made up, apart from the model's weight size (0.93 GiB, the same in every one of those logs), the pattern's
# default model length (8192) and vLLM's own source line numbers; the only path and address are the pattern's ready file
# (/tmp/r/rhvp-ready) and listen address (http://0.0.0.0:8000/); no line is a copy of a line of those logs. One start
# with no restart, one where Ray Serve replaced the replica once after vLLM's start-up
# memory check stopped it (the shape seen in those logs), and two put together from the same shapes: two such
# replacements in a row, and a replica stopped at vLLM's KV-cache check (another error) before one stopped at the
# memory check. Ray relays a late copy of a stopped replica's failure ("[repeated 2x across cluster]", under that
# replica's prefix): in the first two it comes after the new replica's first line, as in most of those logs, and in
# the last one before it, as in the others.

def _ray_log(name: str) -> str:
    return (FIXTURES / f"ray-serve-start-{name}.log").read_text(encoding="utf-8")


@pytest.mark.parametrize(("name", "restarts"), [("no-restart", 0), ("memory-check-restart", 1),
                                                ("memory-check-restarts-2", 2), ("other-error-then-memory-check", 1)])
def test_engine_restarts_after_the_memory_check_in_a_ray_serve_log(name, restarts):
    text = _ray_log(name)
    assert logs.memory_check_restarts(text) == restarts
    assert logs.memory_profiling_failed(text) == (restarts > 0)
    # The failure shows on several lines of one replica (Ray relays it again seconds later) and counts once.
    if name == "memory-check-restart":
        failing = [line for _, line in logs.lines(text) if logs.MEMORY_PROFILING_ERROR in line]
        assert len(failing) == 4 and sum(1 for line in failing if logs.SERVE_REPLICA.match(line)) == 3


def test_a_ray_serve_replica_stopped_at_the_memory_check_counts_only_when_another_replica_follows():
    text = _ray_log("memory-check-restart")
    stopped = text.index("the replica will be stopped.")
    last = text[:text.index("\n", stopped) + 1]  # the server gave up there: no replica after it
    assert logs.memory_profiling_failed(last) and logs.memory_check_restarts(last) == 0
    # A replica of another deployment of the app starting after it is not a restart of the engine.
    ingress = "2026-01-01T00:03:30Z \x1b[36m(ServeReplica:llm:OpenAiIngress pid=5000)\x1b[0m INFO starting\n"
    assert logs.memory_check_restarts(last + ingress) == 0
    # A replica on another node carries ", ip=..." in its prefix; it is the same replica.
    remote = text.replace("pid=1520)", "pid=1520, ip=10.0.0.7)").replace("pid=2470)", "pid=2470, ip=10.0.0.8)")
    assert logs.memory_check_restarts(remote) == 1


@pytest.mark.parametrize(("name", "restarts", "late"), [("memory-check-restart", 1, True),
                                                        ("memory-check-restarts-2", 2, True),
                                                        ("other-error-then-memory-check", 1, False)])
def test_a_stopped_ray_serve_replica_counts_from_its_first_line_that_shows_the_check_failing(name, restarts, late):
    """Ray's late copy of a stopped replica's failure carries that replica's prefix. Where it comes after the new
    replica's first line (`late`), a count keyed on the stopped replica's last failing line would miss the restart;
    the replica is timed from its first such line, so it counts in both orders."""
    parsed = [text for _, text in logs.lines(_ray_log(name))]
    first: dict[tuple[str, str], int] = {}
    for number, text in enumerate(parsed):
        match = logs.SERVE_REPLICA.match(text)
        if match:
            first.setdefault((match.group("deployment"), match.group("pid")), number)
    copies = [(number, logs.SERVE_REPLICA.match(text)) for number, text in enumerate(parsed)
              if "across cluster]" in text and logs.MEMORY_PROFILING_ERROR in text]
    assert len(copies) == restarts
    for number, match in copies:
        stopped = (match.group("deployment"), match.group("pid"))
        new = min(at for replica, at in first.items() if replica[0] == stopped[0] and at > first[stopped])
        assert (number > new) == late
    assert logs.memory_check_restarts(_ray_log(name)) == restarts


def test_a_vllm_serve_log_that_stops_at_the_memory_check_gives_no_engine_restart():
    # `vllm serve` exits at the check (the runner's own start again counts it), and its lines have no replica prefix.
    stopped = VLLM_LOG + ("2026-01-01T00:01:01Z (EngineCore pid=7) ERROR 01-01 00:01:01 [core.py:3] AssertionError: "
                          "Error in memory profiling. Initial free memory 20.00 GiB, current free memory 21.25 GiB.\n")
    assert logs.memory_profiling_failed(stopped) and logs.memory_check_restarts(stopped) == 0
    assert logs.memory_check_restarts(VLLM_LOG) == 0 and logs.memory_check_restarts("") == 0


def test_the_breakdown_of_a_restarted_ray_serve_start_adds_up_both_engine_starts():
    steps = {entry["phase"]: entry for entry in logs.breakdown(_ray_log("memory-check-restart"),
                                                               logs.parse_time("2026-01-01T00:00:00Z"))}
    assert steps["weights_load"]["count"] == 2 and steps["weights_load"]["seconds"] == pytest.approx(142.8)
    assert steps["engine_init"]["count"] == 1  # only the engine that came up finished its start
    assert steps["ray_app_start"]["seconds"] == pytest.approx(353.1)  # Ray's own figure, both engine starts in it
    # A moment is its first line in the log: "the engine starts" is the stopped engine's, not the serving one's.
    assert steps["engine_start"]["at_s"] == pytest.approx(49.6)


# ---- GPU power ----

def test_nvidia_smi_samples():
    log = ("2026-01-01T00:00:00.000Z 41.20\n2026-01-01T00:00:00.500Z [N/A]\n2026-01-01T00:00:01.000Z 55.5, 60\n"
           "no time 99\n")
    samples = energy.parse_nvidia_smi(log)
    assert [s.value for s in samples] == [41.2, 55.5]
    assert samples[1].t - samples[0].t == pytest.approx(1.0)


def test_window_power_prefers_the_energy_counter():
    power = [energy.Sample(t, 100.0) for t in range(0, 11)]
    counter = [energy.Sample(2, 1_000_000), energy.Sample(8, 1_000_000 + 6 * 150_000)]  # mJ: 150 W over 6 s
    by_counter = energy.window_power(power, 1, 9, 2, counter)
    assert by_counter.method == "energy_counter" and by_counter.mean_w == pytest.approx(150.0)
    by_power = energy.window_power(power, 1, 9, 2)
    assert by_power.method == "power_mean" and by_power.mean_w == pytest.approx(100.0) and by_power.samples == 9
    too_few = energy.window_power(power[:1], 0, 9, 2)
    assert too_few.mean_w is None and "at least 2" in too_few.note
    # The plan's minimum holds for the counter too: a DCGM exporter at its default 30 s gives a 60 s level 2 readings
    # of each, and no figure.
    slow_power = [energy.Sample(0, 200.0), energy.Sample(30, 200.0)]
    slow_counter = [energy.Sample(0, 0), energy.Sample(30, 6_000_000)]
    refused = energy.window_power(slow_power, 0, 60, 10, slow_counter)
    assert refused.mean_w is None and refused.method is None and "at least 10 are needed" in refused.note
    ten = [energy.Sample(6 * i, 200.0) for i in range(10)]
    stale = energy.window_power(ten, 0, 60, 10, slow_counter)  # 2 counter readings: the power samples give the mean
    assert stale.method == "power_mean" and stale.mean_w == pytest.approx(200.0) and stale.samples == 10
    counter = [energy.Sample(6 * i, 6 * i * 150_000) for i in range(10)]  # ten distinct readings: 150 W
    by_counter = energy.window_power(ten, 0, 60, 10, counter)
    assert by_counter.method == "energy_counter" and by_counter.mean_w == pytest.approx(150.0)
    # The thresholds exactly: 9 power samples are one short of 10; 9 distinct counter readings are too, also when
    # repeats of one value (a scrape faster than the exporter's update) make it 10 samples.
    short = energy.window_power(ten[:9], 0, 60, 10, counter)
    assert short.mean_w is None and short.samples == 9 and "at least 10 are needed" in short.note
    nine = energy.window_power(ten, 0, 60, 10, counter[:9])
    assert nine.method == "power_mean" and nine.mean_w == pytest.approx(200.0)
    repeated = counter[:9] + [energy.Sample(57, counter[8].value)]
    assert len(repeated) == 10 and len(energy.collapse(repeated)) == 9
    assert energy.window_power(ten, 0, 60, 10, repeated).method == "power_mean"


def test_flat_values_keep_a_zero():
    levels = guidellm.levels(REPORT)
    levels[0]["output_tokens_per_s"] = 0.0
    levels[0]["energy"] = energy.level_energy(levels[0], energy.Power(150.0, 20, "power_mean", None), "nvidia-smi")
    flat = guidellm.flat_values(levels)
    c = levels[0]["concurrency"]
    assert flat[f"c{c}/output_tokens_per_s"] == 0.0 and flat[f"c{c}/tokens_per_joule[gpu]"] == 0.0
    assert levels[0]["energy"]["joules_per_token"] is None


def test_level_energy():
    level = {"output_tokens_per_s": 300.0, "duration_s": 10.0}
    entry = energy.level_energy(level, energy.Power(150.0, 20, "power_mean", None), "nvidia-smi")
    assert entry["tokens_per_joule"] == pytest.approx(2.0) and entry["joules_per_token"] == pytest.approx(0.5)
    assert entry["energy_j"] == pytest.approx(1500.0) and entry["label"] == "GPU-only power"
    empty = energy.level_energy(level, energy.Power(None, 0, None, "none"), "dcgm")
    assert empty["tokens_per_joule"] is None and empty["note"] == "none"


def _matrix(*series) -> dict:
    return {"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": labels, "values": [[t, str(v)] for t, v in values]} for labels, values in series]}}


def test_parse_matrix():
    parsed = energy.parse_matrix(_matrix(({"pod": "p"}, [(1, 10), (2, "NaN"), (3, 30)])))
    assert parsed[0][0] == {"pod": "p"} and [s.value for s in parsed[0][1]] == [10.0, 30.0]
    with pytest.raises(ValueError):
        energy.parse_matrix({"status": "error", "error": "bad"})
    with pytest.raises(ValueError):
        energy.parse_matrix({"status": "success", "data": {"resultType": "vector", "result": []}})


def _thanos(answers: dict) -> tuple[energy.Thanos, list]:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        seen.append((query, request.headers.get("authorization")))
        for needle, answer in answers.items():
            if needle in query:
                return httpx.Response(200, json=answer)
        return httpx.Response(200, json=_matrix())

    return energy.Thanos("https://thanos.example:9091", transport=httpx.MockTransport(handler), token="t0k",
                         verify=False), seen


def test_dcgm_samples_by_exported_pod_labels():
    thanos, seen = _thanos({
        'DCGM_FI_DEV_POWER_USAGE{exported_pod="side",exported_namespace="ns"}': _matrix(({}, [(10, 200), (11, 210)])),
        'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{exported_pod="side"': _matrix(({}, [(10, 5000), (11, 5205)])),
    })
    power, counter, matched, note = energy.dcgm_samples(thanos, "side", "ns", "node-1", 10, 11)
    assert [s.value for s in power] == [200, 210] and [s.value for s in counter] == [5000, 5205]
    assert matched == "pod (exported labels), DCGM_FI_DEV_POWER_USAGE" and note is None
    assert seen[0][0].endswith("[2s]") and seen[0][1] == "Bearer t0k"


def test_dcgm_samples_under_the_exporters_new_names():
    # DCGM exporter 4.9.0 renames both series; its energy counter is in joules, not millijoules.
    thanos, seen = _thanos({
        'DCGM_FI_DEV_BOARD_POWER_WATTS{exported_pod="side"': _matrix(({}, [(10, 200), (11, 210)])),
        'DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL{exported_pod="side"': _matrix(({}, [(10, 1000.0), (11, 1205.0)])),
    })
    power, counter, matched, note = energy.dcgm_samples(thanos, "side", "ns", None, 10, 11)
    assert [s.value for s in power] == [200, 210] and matched.endswith("DCGM_FI_DEV_BOARD_POWER_WATTS")
    assert [s.value for s in counter] == pytest.approx([1_000_000, 1_205_000])  # millijoules
    assert energy.window_power(power, 10, 11, 2, counter).mean_w == pytest.approx(205.0)
    assert seen[0][0].startswith("DCGM_FI_DEV_POWER_USAGE{")  # the older name is tried first


def test_repeated_counter_values_are_dropped_before_the_division():
    # A scrape faster than the exporter's update repeats the last value: 1000 J from t=10 to t=12, not to t=13.
    counter = [energy.Sample(10, 0), energy.Sample(11, 500_000), energy.Sample(12, 1_000_000),
               energy.Sample(13, 1_000_000)]
    assert energy.collapse(counter) == counter[:3]
    power = [energy.Sample(t, 400.0) for t in (10, 11, 12, 13)]
    assert energy.window_power(power, 10, 13, 2, counter).mean_w == pytest.approx(500.0)
    flat = [energy.Sample(10, 5000), energy.Sample(11, 5000)]
    power = [energy.Sample(10, 90), energy.Sample(11, 110)]
    result = energy.window_power(power, 10, 11, 2, flat)  # one distinct value: fall back to the power samples
    assert result.method == "power_mean" and result.mean_w == pytest.approx(100.0)


def test_dcgm_node_series_only_when_there_is_one_gpu():
    one = _matrix(({"gpu": "0"}, [(10, 90), (11, 95)]))
    two = _matrix(({"gpu": "0"}, [(10, 90)]), ({"gpu": "1"}, [(10, 300)]))
    thanos, _ = _thanos({'Hostname="node-1"': one})
    assert energy.dcgm_samples(thanos, "side", "ns", "node-1", 10, 11)[2].startswith("node, ")
    thanos, _ = _thanos({'Hostname="node-1"': two})
    power, _, matched, note = energy.dcgm_samples(thanos, "side", "ns", "node-1", 10, 11)
    assert power == [] and matched is None and "2 GPU series" in note
    thanos, _ = _thanos({})
    assert "no DCGM power series" in energy.dcgm_samples(thanos, "side", "ns", None, 10, 11)[3]


def test_dcgm_query_errors_carry_no_url():
    thanos = energy.Thanos("https://thanos.example:9091", verify=False, token="t",
                           transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    power, _, _, note = energy.dcgm_samples(thanos, "side", "ns", None, 1, 2)
    assert power == [] and "HTTP 403" in note and "thanos.example" not in note


# ---- the OpenShift backend's helpers ----

GFD_LABELS = {"nvidia.com/gpu.product": "NVIDIA-L4", "nvidia.com/gpu.memory": "23034", "nvidia.com/gpu.count": "1",
              "nvidia.com/cuda.driver-version.full": "580.95.05", "nvidia.com/cuda.runtime-version.full": "13.0"}


def test_gpu_and_node_facts_from_labels():
    gpu = kube.gpu_facts(GFD_LABELS)
    assert gpu["product"] == "NVIDIA-L4" and gpu["memory_mib"] == 23034 and gpu["driver"] == "580.95.05"
    older = kube.gpu_facts({"nvidia.com/gpu.product": "X", "nvidia.com/cuda.driver.major": "570",
                            "nvidia.com/cuda.driver.minor": "1", "nvidia.com/cuda.driver.rev": "2"})
    assert older["driver"] == "570.1.2" and older["memory_mib"] is None
    assert kube.gpu_facts({}) is None
    node = kube.node_facts({"metadata": {"name": "n1", "labels": {"kubernetes.io/arch": "amd64",
                                                                  "node.kubernetes.io/instance-type": "g6.2xlarge"}},
                            "status": {"nodeInfo": {"osImage": "RHCOS", "kernelVersion": "5.14",
                                                    "containerRuntimeVersion": "cri-o://1.33"}}})
    assert node == {"name": "n1", "arch": "amd64", "instance_type": "g6.2xlarge", "os_image": "RHCOS",
                    "kernel": "5.14", "container_runtime": "cri-o://1.33"}


def test_image_digest_from_the_pod():
    pod = {"status": {"containerStatuses": [{"name": "server", "imageID": "docker.io/x/a@sha256:abc"}]}}
    assert kube.image_digest(pod) == "docker.io/x/a@sha256:abc"
    pod["status"]["containerStatuses"][0]["imageID"] = "docker-pullable://x/a@sha256:def"
    assert kube.image_digest(pod) == "x/a@sha256:def"
    # A runtime that reports only the image's own id: the image the pod asked for, when it is pinned by digest.
    pod["status"]["containerStatuses"][0]["imageID"] = "sha256:0123"
    assert kube.image_digest(pod) is None
    pod["spec"] = {"containers": [{"name": "server", "image": "quay.io/x/a:1@sha256:feed"}]}
    assert kube.image_digest(pod) == "quay.io/x/a:1@sha256:feed"
    pod["spec"]["containers"][0]["image"] = "quay.io/x/a:1"
    assert kube.image_digest(pod) is None
    assert kube.image_digest({}) is None


def _side_backend(secret: str | None = None, server_replicas: int | None = None,
                  configmaps: dict | None = None, scale_fails: int = 0, runner_pods: list | None = None,
                  clock=None, scaled_object: dict | None = None, env_extra: dict | None = None,
                  autoscaler_replicas: int | None = None) -> tuple[kube.KubeBackend, list]:
    """A KubeBackend on a fake API. `scale_fails`: how many scale-ups answer HTTP 503 before one goes through.
    `scaled_object`: the server's ScaledObject (its metadata.annotations are patched in place), None for none;
    `autoscaler_replicas`: once it is resumed, the autoscaler sets the server to this many replicas."""
    calls = []
    state = {"replicas": server_replicas, "fails": scale_fails}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/apis/keda.sh/v1alpha1/namespaces/vllm-inference/scaledobjects/ray-serve-llm":
            if scaled_object is None:
                return httpx.Response(404, json={"message": "not found"})
            assert request.headers["content-type"] == "application/merge-patch+json"
            notes = scaled_object.setdefault("metadata", {}).setdefault("annotations", {})
            notes.update(body["metadata"]["annotations"])
            for key in [k for k, v in notes.items() if v is None]:
                del notes[key]
            if not notes and autoscaler_replicas is not None:
                state["autoscaler"] = autoscaler_replicas
            return httpx.Response(200, json=scaled_object)
        if request.url.path.endswith("/deployments/ray-serve-llm/scale"):
            if state["replicas"] is None:
                return httpx.Response(404, json={"message": "not found"})
            if request.method == "PATCH":
                assert request.headers["content-type"] == "application/merge-patch+json"
                if body["spec"]["replicas"] > 0 and state["fails"] > 0:
                    state["fails"] -= 1
                    return httpx.Response(503, json={"message": "unavailable"})
                state["replicas"] = body["spec"]["replicas"]
            if request.method == "GET" and state.get("autoscaler") is not None and state["replicas"]:
                state["replicas"] = state["autoscaler"]
            return httpx.Response(200, json={"spec": {"replicas": state["replicas"]}})
        if configmaps is not None and "/configmaps" in request.url.path:
            name = request.url.path.rsplit("/", 1)[1]
            if request.method == "PATCH":
                if name not in configmaps:
                    return httpx.Response(404, json={"message": "not found"})
                configmaps[name]["data"].update(body["data"])
                configmaps[name]["data"] = {k: v for k, v in configmaps[name]["data"].items() if v is not None}
                return httpx.Response(200, json=configmaps[name])
            configmaps[body["metadata"]["name"]] = body
            return httpx.Response(201, json=body)
        if request.url.path.endswith("/secrets/huggingface-secret"):
            if secret is None:
                return httpx.Response(404, json={"message": "not found"})
            return httpx.Response(200, json={"data": {"hftoken": base64.b64encode(secret.encode()).decode()}})
        if request.url.path == "/api/v1/namespaces/nb/pods" and request.method == "GET":
            assert request.url.params["labelSelector"] == "app.kubernetes.io/name=image-compare"
            return httpx.Response(200, json={"items": runner_pods or []})
        if request.url.path.endswith("/pods") and request.method == "GET":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(404, json={})

    client = kube.Kube(base_url="https://api.example:6443", token="t", verify=False,
                       transport=httpx.MockTransport(handler))
    env = {"RHVP_COMPARE_ARCH": "arm64", "RHVP_COMPARE_STATUS_NAMESPACE": "nb", "RHVP_COMPARE_POD_NAME": "me",
           "RHVP_COMPARE_SIDE_RESOURCES": json.dumps({"requests": {"cpu": "2", "memory": "12Gi"},
                                                      "limits": {"memory": "24Gi"}})}
    env.update(env_extra or {})
    extra = {"clock": clock} if clock else {}
    return kube.KubeBackend(kube=client, env=env, sleep=lambda s: None, **extra), calls


SCALED = {"RHVP_COMPARE_SERVER_SCALED_OBJECT": "ray-serve-llm"}
KEDA_PATH = "/apis/keda.sh/v1alpha1/namespaces/vllm-inference/scaledobjects/ray-serve-llm"


def test_the_servers_autoscaler_is_paused_for_the_run_and_resumed_first():
    scaled = {"metadata": {"name": "ray-serve-llm", "annotations": {"other": "x"}}}
    backend, calls = _side_backend(server_replicas=1, scaled_object=scaled, env_extra=SCALED, autoscaler_replicas=2)
    backend.prepare(_plan())
    steps = [(m, p.rsplit("/", 1)[1], b) for m, p, b in calls if p == KEDA_PATH or p.endswith("/scale")]
    assert steps == [("GET", "scale", None),
                     ("PATCH", "ray-serve-llm", {"metadata": {"annotations": {kube.KEDA_PAUSED: "0"}}}),
                     ("PATCH", "scale", {"spec": {"replicas": 0}})]
    assert scaled["metadata"]["annotations"] == {"other": "x", kube.KEDA_PAUSED: "0"}
    calls.clear()
    assert backend.restore_server() is True  # the autoscaler chose 2: at least the 1 asked for, so back
    steps = [(m, p.rsplit("/", 1)[1], b) for m, p, b in calls if p == KEDA_PATH or p.endswith("/scale")]
    assert steps == [("PATCH", "ray-serve-llm", {"metadata": {"annotations": {kube.KEDA_PAUSED: None}}}),
                     ("PATCH", "scale", {"spec": {"replicas": 1}}), ("GET", "scale", None)]
    assert scaled["metadata"]["annotations"] == {"other": "x"}


def test_a_missing_autoscaler_is_left_alone():
    backend, calls = _side_backend(server_replicas=1, scaled_object=None, env_extra=SCALED)
    backend.prepare(_plan())
    assert backend.restore_server() is True
    assert [m for m, p, b in calls if p == KEDA_PATH] == ["PATCH"]  # one look at prepare, none at the restore
    plain, calls = _side_backend(server_replicas=1)
    plain.prepare(_plan())
    assert plain.restore_server() is True and not [c for c in calls if "keda" in c[1]]


def test_a_paused_autoscaler_that_cannot_be_resumed_says_how():
    now = [0.0]

    def clock():
        now[0] += 5.0
        return now[0]
    scaled = {"metadata": {"name": "ray-serve-llm"}}
    backend, calls = _side_backend(server_replicas=1, scaled_object=scaled, env_extra=SCALED, clock=clock)
    backend.prepare(_plan())
    real = backend.kube.pause_scaled_object

    def fails(namespace, name, paused):
        if not paused:
            raise runner.RoundError("the Kubernetes API answered HTTP 403 while trying to resume the autoscaler")
        return real(namespace, name, paused)
    backend.kube.pause_scaled_object = fails
    with pytest.raises(runner.RoundError) as caught:
        backend.restore_server()
    assert ("oc annotate scaledobject/ray-serve-llm -n vllm-inference autoscaling.keda.sh/paused-replicas- && "
            "oc scale deployment/ray-serve-llm --replicas=1 -n vllm-inference") in str(caught.value)


def test_the_openshift_job_refuses_a_scale_out_plan_no_gpu_node_can_hold_before_it_changes_anything(tmp_path):
    # The mode runs on OpenShift (see "mode scaleout on OpenShift" below); a cluster whose GPU node cannot hold a
    # side's two replicas as scaleOut.sharedGpu declares is refused after one look at the nodes, and nothing else.
    config = _plan_config()
    config.update(mode="scaleout", rounds={"count": 7, "warmup": False})
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(config))
    # One L4, time-slicing off, from a device plugin without the sharing-strategy label, so the shares alone refuse
    # it. Any call that would change something fails the test at once: a check that wrongly passed would otherwise
    # run the whole scale round against the fake.
    cluster = FakeCluster(labels={kube.GPU_REPLICAS_LABEL: "1", kube.GPU_SHARING_LABEL: None}, read_only=True)
    backend = cluster.backend()
    with pytest.raises(plans.PlanError, match="scaleOut.sharedGpu is true: both replicas of a side run on one node"):
        runner.main_run(str(path), str(tmp_path / "results"), backend,
                        env={"RHVP_COMPARE_TERMINATION_LOG": str(tmp_path / "term")})
    assert cluster.calls == [("GET", "/api/v1/nodes")]
    assert json.loads((tmp_path / "term").read_text())["error_code"] == "PLAN_INVALID"
    assert not (tmp_path / "results").exists()


def test_the_live_server_is_scaled_down_for_the_run_and_back_after():
    backend, calls = _side_backend(server_replicas=1)
    backend.prepare(_plan())
    scale = [(method, body) for method, path, body in calls if path.endswith("/scale")]
    assert scale == [("GET", None), ("PATCH", {"spec": {"replicas": 0}})]
    calls.clear()
    assert backend.restore_server() is True
    # scaled back, then read back to check
    assert [(m, b) for m, p, b in calls if p.endswith("/scale")] == [("PATCH", {"spec": {"replicas": 1}}),
                                                                   ("GET", None)]
    calls.clear()
    assert backend.restore_server() is True  # once only
    assert not [c for c in calls if c[1].endswith("/scale")]
    absent, calls = _side_backend(server_replicas=None)  # no server Deployment: nothing to scale
    absent.prepare(_plan())
    assert absent.restore_server() is None
    assert [m for m, p, b in calls if p.endswith("/scale")] == ["GET"]


def test_a_scale_back_that_fails_is_tried_again():
    backend, calls = _side_backend(server_replicas=1, scale_fails=3)
    backend.prepare(_plan())
    assert backend.restore_server() is True
    assert [b for m, p, b in calls if p.endswith("/scale") and m == "PATCH"][-4:] == [{"spec": {"replicas": 1}}] * 4


def test_a_scale_back_that_keeps_failing_says_how_to_bring_the_server_back():
    now = [0.0]

    def clock():
        now[0] += 5.0  # each look at the clock is 5 s later: the minute of tries runs out
        return now[0]
    backend, calls = _side_backend(server_replicas=1, scale_fails=99, clock=clock)
    backend.prepare(_plan())
    with pytest.raises(runner.RoundError) as caught:
        backend.restore_server()
    text = str(caught.value)
    assert "HTTP 503" in text and "oc scale deployment/ray-serve-llm --replicas=1 -n vllm-inference" in text
    assert "oc delete pod -n vllm-inference -l app.kubernetes.io/name=image-compare-side" in text
    tries = len([1 for m, p, b in calls if p.endswith("/scale") and m == "PATCH" and b["spec"]["replicas"] == 1])
    assert 2 <= tries <= 5
    with pytest.raises(runner.RoundError):  # asked again at the end: one more try
        backend.restore_server()


def test_cleanup_deletes_the_run_pods_only():
    backend, calls = _side_backend(server_replicas=1)
    backend.prepare(_plan())
    calls.clear()
    backend.cleanup()
    assert not [c for c in calls if c[1].endswith("/scale")]
    assert [(m, p) for m, p, b in calls] == [("GET", "/api/v1/namespaces/vllm-inference/pods")]


def _runner_pod(name: str, created: str, phase: str = "Running") -> dict:
    return {"metadata": {"name": name, "creationTimestamp": created, "labels": {"job-name": f"job-{name}"}},
            "status": {"phase": phase}}


@pytest.mark.parametrize("others, refused", [
    ([], False),
    ([_runner_pod("old", "2026-01-01T00:00:00Z")], True),
    ([_runner_pod("old", "2026-01-01T00:00:00Z", "Pending")], True),
    ([_runner_pod("old", "2026-01-01T00:00:00Z", "Succeeded")], False),  # ended: no longer a run
    ([_runner_pod("old", "2026-01-01T00:00:00Z", "Failed")], False),
    ([_runner_pod("young", "2026-01-03T00:00:00Z")], False),  # the older one (this one) goes on; the other refuses
    ([_runner_pod("a-same-age", "2026-01-02T00:00:00Z")], True),  # same age: the name that sorts first wins
])
def test_one_run_at_a_time(others, refused):
    pods = [_runner_pod("me", "2026-01-02T00:00:00Z")] + others
    backend, calls = _side_backend(runner_pods=pods)
    if refused:
        with pytest.raises(runner.RunActive, match="another comparison is running: runner pod"):
            backend.check_alone()
    else:
        backend.check_alone()
    assert calls == [("GET", "/api/v1/namespaces/nb/pods", None)]  # only looked


def test_later_sides_run_on_the_first_sides_node():
    backend, _ = _side_backend()
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][0]
    first = backend.pod_body(plan, plan["launches"][0], side, plans.side_spec(plan, side))
    assert "kubernetes.io/hostname" not in first["spec"]["nodeSelector"]
    backend.kube.node = lambda name: {"metadata": {"name": name, "labels": {"kubernetes.io/hostname": "gpu-1"}}}
    backend._pod = lambda handle, fresh=False: {"spec": {"nodeName": "gpu-1.example"}}
    assert backend.facts({"name": "p", "gpu": True})["node_name"] == "gpu-1.example"
    later = backend.pod_body(plan, plan["launches"][2], side, plans.side_spec(plan, side))
    assert later["spec"]["nodeSelector"] == {"kubernetes.io/arch": "arm64", "kubernetes.io/hostname": "gpu-1"}


def test_a_stop_shortens_every_wait():
    backend, _ = _side_backend()
    assert (backend.kube.timeout_s, backend.kube.log_timeout_s, backend.delete_wait_s) == (30.0, 60.0, 180.0)
    backend.on_stop()
    assert (backend.kube.timeout_s, backend.kube.log_timeout_s) == (kube.HURRY_TIMEOUT_S, kube.HURRY_TIMEOUT_S)
    assert backend.delete_wait_s == kube.HURRY_DELETE_WAIT_S
    assert kube.stop_budget_seconds() < 300  # the chart's grace period (tests/test_charts.py checks the chart)


def test_the_token_is_read_again_for_every_request(tmp_path):
    token = tmp_path / "token"
    token.write_text("first\n")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"items": []})
    client = kube.Kube(base_url="https://api.example:6443", verify=False, token_path=token,
                       transport=httpx.MockTransport(handler))
    client.pods("ns")
    token.write_text("second\n")  # the kubelet rotated it
    client.pods("ns")
    assert seen == ["Bearer first", "Bearer second"]
    thanos = energy.Thanos("https://thanos.example:9091", verify=False, token_path=token,
                           transport=httpx.MockTransport(handler))
    thanos._client.get("/x")
    assert seen[-1] == "Bearer second"


def test_the_status_configmap_is_created_then_merged():
    store: dict = {}
    backend, _ = _side_backend(configmaps=store)
    backend.publish("run-1", '{"phase":"running"}', '{"big":1}')
    made = store["compare-run-1"]
    assert made["metadata"]["labels"] == {"app.kubernetes.io/component": "compare-run", "compare-run-id": "run-1",
                                          "app.kubernetes.io/part-of": "vllm-ray-inference-server"}
    assert made["data"] == {"status.json": '{"phase":"running"}', "result.json": '{"big":1}'}
    backend.publish("run-1", '{"phase":"complete"}', None)  # too big this time: result.json is dropped
    assert store["compare-run-1"]["data"] == {"status.json": '{"phase":"complete"}'}


def test_a_side_pod_has_its_own_time_limit():
    backend, _ = _side_backend()
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][0]
    warm = backend.pod_body(plan, plan["launches"][0], side, plans.side_spec(plan, side))
    scored = backend.pod_body(plan, plan["launches"][2], side, plans.side_spec(plan, side))
    load = 2 * plan["load"]["timeout_s"] + plan["energy"]["settle_s"] + 600
    assert warm["spec"]["activeDeadlineSeconds"] == int(plan["server"]["warmup_timeout_s"] + load)
    assert scored["spec"]["activeDeadlineSeconds"] == int(plan["server"]["ready_timeout_s"] + load)


@pytest.mark.parametrize("secret, passed", [("hf_abc", True), ("none", False)])
def test_guidellm_gets_the_token_only_when_real(secret, passed, monkeypatch, tmp_path):
    backend, _ = _side_backend(secret)
    plan = _plan()
    backend.prepare(plan)
    seen = {}
    monkeypatch.setattr(guidellm, "run", lambda plan, target, out_dir, env=None: seen.setdefault("env", env))
    backend.run_load(plan, plan["launches"][2], "http://side:8000", tmp_path)
    assert seen["env"] == ({"HF_TOKEN": "hf_abc"} if passed else None)


def test_side_pod_like_the_patterns_server():
    backend, _ = _side_backend()
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][1]
    twin = plan["launches"][5]  # launch 6: side B's twin in round 1, which launches side B
    assert (twin["index"], twin["side"], twin["of"], twin["round"]) == (6, "B~same", "B", 1)
    body = backend.pod_body(plan, twin, side, plans.side_spec(plan, side))
    spec = body["spec"]
    [container] = spec["containers"]
    assert body["metadata"]["name"] == f"cmp-{plan['run_id'][-8:]}-06-b-same"
    labels = body["metadata"]["labels"]
    assert labels[kube.LABEL_RUN] == plan["run_id"]
    assert (labels["rhvp-compare/side"], labels["rhvp-compare/launch"], labels["rhvp-compare/round"]) == (
        "B-same", "6", "1")
    warm = backend.pod_body(plan, plan["launches"][1], side, plans.side_spec(plan, side))
    assert warm["metadata"]["name"].endswith("-02-b") and warm["metadata"]["labels"]["rhvp-compare/round"] == "warmup"
    assert spec["restartPolicy"] == "Never" and spec["nodeSelector"] == {"kubernetes.io/arch": "arm64"}
    assert spec["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == container["resources"]["limits"]["nvidia.com/gpu"]
    assert container["resources"]["requests"]["memory"] == "12Gi"
    assert container["securityContext"] == {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}
    assert "runAsUser" not in json.dumps(body)
    assert {m["mountPath"] for m in container["volumeMounts"]} == {"/models", "/cache", "/tmp/r", "/dev/shm"}
    dshm = next(v for v in spec["volumes"] if v["name"] == "dshm")
    assert dshm["emptyDir"] == {"medium": "Memory", "sizeLimit": "4Gi"}
    assert next(v for v in spec["volumes"] if v["name"] == "models")["persistentVolumeClaim"] == {
        "claimName": "ray-serve-llm-models"}
    assert "HF_TOKEN" not in {e["name"] for e in container["env"]}
    assert container["args"][:2] == ["--model", "org/m"] and "command" not in container


@pytest.mark.parametrize("secret, referenced", [("hf_abc", True), ("none", False), (None, False)])
def test_the_token_goes_in_by_reference_and_only_when_real(secret, referenced):
    backend, _ = _side_backend(secret)
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][0]
    body = backend.pod_body(plan, plan["launches"][0], side, plans.side_spec(plan, side))
    env = {e["name"]: e for e in body["spec"]["containers"][0]["env"]}
    assert ("HF_TOKEN" in env) is referenced
    if referenced:
        assert env["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {"name": "huggingface-secret", "key": "hftoken",
                                                                "optional": True}
        assert "hf_abc" not in json.dumps(body)


def test_a_side_without_a_gpu_and_the_ray_preset():
    backend, _ = _side_backend()
    config = _plan_config()
    config["sides"] = [{"name": "a", "image": "x/a:1", "preset": "guidellm-mock"},
                       {"name": "b", "image": "x/b:1", "preset": "ray-serve-llm", "gpu": False}]
    plan = plans.build(config, "openshift")
    backend.prepare(plan)
    mock = backend.pod_body(plan, plan["launches"][0], plan["sides"][0], plans.side_spec(plan, plan["sides"][0]))
    assert mock["spec"]["tolerations"] == []
    assert "nvidia.com/gpu" not in json.dumps(mock["spec"]["containers"][0]["resources"])
    assert mock["spec"]["containers"][0]["command"] == ["guidellm", "mock-server"]
    ray = backend.pod_body(plan, plan["launches"][1], plan["sides"][1], plans.side_spec(plan, plan["sides"][1]))
    assert {"name": "config", "configMap": {"name": "ray-serve-llm-config"}} in ray["spec"]["volumes"]
    assert plan["energy"]["source"] == "none"


STREAM_VAR = "RAYLLM_MODEL_RESPONSE_BATCH_TIMEOUT_MS"
INTERVAL_VAR = "RHVP_STREAM_BATCHING_INTERVAL_MS"


def test_the_ray_preset_sends_each_streamed_token_at_once_unless_a_side_says_otherwise(tmp_path):
    config = _plan_config()
    config["sides"] = [{"name": "a", "image": "x/a:1", "preset": "ray-serve-llm"},
                       {"name": "b", "image": "x/b:1", "preset": "ray-serve-llm", "env": {STREAM_VAR: "50"}}]
    plan = plans.build(config, "docker")
    a, b = (plans.side_spec(plan, side) for side in plan["sides"])
    assert (a["env"][STREAM_VAR], b["env"][STREAM_VAR]) == ("0", "50")
    assert a["env"][INTERVAL_VAR] == b["env"][INTERVAL_VAR] == "null"
    for preset in ("vllm-openai", "guidellm-mock"):
        other = plans.build({**_plan_config(), "sides": [{"name": "a", "image": "x/a:1", "preset": preset},
                                                         {"name": "b", "image": "x/b:1", "preset": preset}]},
                            "docker")
        assert STREAM_VAR not in plans.side_spec(other, other["sides"][0])["env"], preset
        assert INTERVAL_VAR not in plans.side_spec(other, other["sides"][0])["env"], preset
    # llm.yaml leaves the key null, so the side's variable sets the interval on every image of the server,
    # including images built before the key existed.
    llm = json.loads(plans.llm_yaml(plan))
    assert "stream_batching_interval_ms" in llm and llm["stream_batching_interval_ms"] is None
    sys.path.insert(0, str(ROOT / "inference-server"))
    import serve_app

    path = tmp_path / "llm.yaml"
    path.write_text(plans.llm_yaml(plan))
    cfg = serve_app.load_config(str(path), a["env"])
    assert cfg["stream_batching_interval_ms"] is None
    assert "experimental_configs" not in serve_app.build_llm_config_kwargs(cfg)
    assert serve_app.stream_interval_text(cfg, a["env"]) == (
        f"stream_batching_interval_ms=null: Ray Serve LLM's own setting ({STREAM_VAR}=0)")
    assert serve_app.stream_interval_note(cfg, a["env"]) is None


def test_an_openshift_ray_side_gets_the_stream_variables_in_its_pod():
    """On OpenShift a side mounts the live server's ConfigMap, not llm_yaml, so the interval comes from the pod's
    env: RHVP_STREAM_BATCHING_INTERVAL_MS=null and Ray Serve LLM's variable at 0, or what the side's env says."""
    backend, _ = _side_backend()
    config = _plan_config()
    config["sides"] = [{"name": "a", "image": "x/a:1", "preset": "ray-serve-llm"},
                       {"name": "b", "image": "x/b:1", "preset": "ray-serve-llm", "env": {STREAM_VAR: "50"}}]
    plan = plans.build(config, "openshift")
    backend.prepare(plan)
    envs = []
    for side, launch in zip(plan["sides"], plan["launches"][:2]):
        body = backend.pod_body(plan, launch, side, plans.side_spec(plan, side))
        assert {"name": "config", "configMap": {"name": "ray-serve-llm-config"}} in body["spec"]["volumes"]
        envs.append({e["name"]: e.get("value") for e in body["spec"]["containers"][0]["env"]})
    assert [(env[INTERVAL_VAR], env[STREAM_VAR]) for env in envs] == [("null", "0"), ("null", "50")]


@pytest.mark.parametrize("configmap_value", [0, 12.5, 50, None])
@pytest.mark.parametrize("side_env, expected, kwargs", [
    ({}, None, None),
    ({STREAM_VAR: "50"}, None, None),
    ({INTERVAL_VAR: "12.5"}, 12.5, {"stream_batching_interval_ms": 12.5}),
])
def test_a_ray_side_sets_the_interval_whatever_the_live_configmap_holds(tmp_path, configmap_value, side_env,
                                                                         expected, kwargs):
    """The live server's ConfigMap may hold any interval; the side's env decides: null by default (Ray Serve LLM's
    variable then sets it), or the side's own RHVP_STREAM_BATCHING_INTERVAL_MS."""
    config = _plan_config()
    config["sides"] = [{"name": "a", "image": "x/a:1", "preset": "ray-serve-llm", "env": side_env},
                       {"name": "b", "image": "x/b:1", "preset": "ray-serve-llm"}]
    plan = plans.build(config, "openshift")
    env = plans.side_spec(plan, plan["sides"][0])["env"]
    live = json.loads(plans.llm_yaml(plan))
    live["stream_batching_interval_ms"] = configmap_value
    path = tmp_path / "llm.yaml"
    path.write_text(json.dumps(live))
    sys.path.insert(0, str(ROOT / "inference-server"))
    import serve_app

    cfg = serve_app.load_config(str(path), env)
    assert cfg["stream_batching_interval_ms"] == expected
    assert serve_app.build_llm_config_kwargs(cfg).get("experimental_configs") == kwargs


def _pod(phase="Pending", waiting=None, terminated=None, scheduled=None, ip=None) -> dict:
    state = {}
    if waiting:
        state["waiting"] = {"reason": waiting, "message": "detail"}
    if terminated:
        state["terminated"] = {"reason": terminated, "exitCode": 1}
    status = {"phase": phase, "containerStatuses": [{"name": "server", "state": state}] if state else [],
              "conditions": [{"type": "PodScheduled", "status": scheduled[0], "message": scheduled[1]}]
              if scheduled else []}
    if ip:
        status["podIP"] = ip
    return {"status": status, "spec": {}}


def test_problems_a_side_can_have():
    backend, _ = _side_backend()
    now = [1000.0]
    backend.clock = lambda: now[0]
    handle = {"name": "p", "created": 1000.0, "pull_since": None}
    assert backend._problem(handle, _pod("Running")) is None
    assert "exit code 1" in backend._problem(handle, _pod("Failed", terminated="OOMKilled"))
    assert "InvalidImageName" in backend._problem(handle, _pod(waiting="InvalidImageName"))
    assert backend._problem(handle, _pod(waiting="ImagePullBackOff")) is None
    now[0] += 901
    assert "could not be pulled" in backend._problem(handle, _pod(waiting="ImagePullBackOff"))
    late = {"name": "p", "created": 0.0, "pull_since": None}
    assert "Insufficient nvidia.com/gpu" in backend._problem(late, _pod(scheduled=("False",
                                                                                   "0/3 nodes: Insufficient nvidia.com/gpu")))
    assert backend._problem(handle, None) == "the side's pod is gone"


def test_endpoint_is_the_pod_ip():
    backend, _ = _side_backend()
    handle = {"name": "p", "port": 8000, "created": backend.clock(), "pull_since": None, "last": None,
              "last_at": -1e9}
    backend.kube.pod = lambda ns, name: _pod("Running", ip="10.0.0.5")
    assert backend.endpoint(handle, backend.clock() + 10) == "http://10.0.0.5:8000"
    backend.kube.pod = lambda ns, name: _pod("Running", ip="fd00::5")
    assert backend.endpoint(handle, backend.clock() + 10) == "http://[fd00::5]:8000"


def test_kube_errors_carry_no_url_or_token():
    client = kube.Kube(base_url="https://api.example:6443", token="secret-token", verify=False,
                       transport=httpx.MockTransport(lambda r: httpx.Response(403, json={"message": "forbidden"})))
    with pytest.raises(Exception) as caught:
        client.pods("ns")
    text = str(caught.value)
    assert "HTTP 403 forbidden" in text and "api.example" not in text and "secret-token" not in text


# ---- the docker backend's helpers ----

def test_docker_log_frames():
    frame = lambda stream, text: struct.pack(">BxxxI", stream, len(text)) + text  # noqa: E731
    raw = frame(1, b"2026-01-01T00:00:00Z out\n") + frame(2, b"2026-01-01T00:00:01Z err\n")
    assert dockerapi.demux(raw) == "2026-01-01T00:00:00Z out\n2026-01-01T00:00:01Z err\n"
    assert dockerapi.demux(b"plain tty text\n") == "plain tty text\n"


def test_repository_of_an_image_reference():
    assert dockerapi.repository("docker.io/vllm/vllm-openai:v0.33.0@sha256:ab") == "docker.io/vllm/vllm-openai"
    assert dockerapi.repository("localhost:5000/x") == "localhost:5000/x"
    assert dockerapi.repository("x:1") == "x"


class FakeDocker:
    """The Engine API calls DockerBackend makes for the run lock and the server container."""

    def __init__(self, running=None, server_state=None, start_fails: int = 0):
        self.running_list = running or []
        self.server_state = server_state
        self.start_fails = start_fails
        self.calls = []

    def running(self, label):
        self.calls.append(("running", label))
        return self.running_list

    def inspect(self, cid):
        self.calls.append(("inspect", cid))
        return None if self.server_state is None else {"State": dict(self.server_state)}

    def stop(self, cid, grace_s=30):
        self.calls.append(("stop", cid, grace_s))
        self.server_state = {"Running": False, "Status": "exited"}

    def start(self, cid):
        self.calls.append(("start", cid))
        if self.start_fails:
            self.start_fails -= 1
            raise runner.RoundError("docker could not start a container: HTTP 500")
        self.server_state = {"Running": True, "Status": "running"}

    def network(self, name):
        pass

    def volume(self, name):
        pass

    def pull(self, ref):
        self.calls.append(("pull", ref))

    def run(self, name, body, timeout_s=300.0, log_to=None):
        self.calls.append(("run", name, body))
        return 0, getattr(self, "run_log", "")

    def info(self):
        return {}

    def by_label(self, value):
        return []


def _stops(docker) -> list[tuple]:
    """(container, grace in seconds) of every stop call."""
    return [c[1:] for c in docker.calls if c[0] == "stop"]


def _docker_backend(docker, results_root: Path = Path("/results"), **env) -> dockerapi.DockerBackend:
    base = {"RHVP_COMPARE_NETWORK": "n", "RHVP_COMPARE_HOST_RESULTS": "/tmp/r", "RHVP_COMPARE_RUNNER_NAME": "me"}
    return dockerapi.DockerBackend(docker=docker, env={**base, **env}, results_root=results_root)


@pytest.mark.parametrize("text, expected", [("", None), ("32g", 32 * 2**30), ("512m", 512 * 2**20),
                                            ("1073741824", 2**30)])
def test_side_memory_limit(text, expected):
    assert dockerapi.memory_bytes(text) == expected
    with pytest.raises(runner.RoundError):
        dockerapi.memory_bytes("lots")
    backend = _docker_backend(FakeDocker(), RHVP_COMPARE_MEMORY=text)
    assert backend.memory == expected


def _container(name: str, created: int) -> dict:
    return {"Id": f"{name}-id", "Names": [f"/{name}"], "Created": created}


@pytest.mark.parametrize("others, refused", [
    ([], False),
    ([_container("old", 100)], True),
    ([_container("young", 300)], False),
])
def test_one_local_run_at_a_time(others, refused):
    docker = FakeDocker(running=[_container("me", 200)] + others)
    backend = _docker_backend(docker)
    if refused:
        with pytest.raises(runner.RunActive, match="runner container old"):
            backend.check_alone()
    else:
        backend.check_alone()
    assert docker.calls == [("running", dockerapi.RUNNER_LABEL)]


def test_the_server_container_is_stopped_for_the_run_and_started_after(monkeypatch):
    monkeypatch.setattr(dockerapi.time, "sleep", lambda s: None)
    docker = FakeDocker(server_state={"Running": True}, start_fails=1)
    backend = _docker_backend(docker, RHVP_COMPARE_SERVER_CONTAINER="live")
    backend.stop_grace_s = 17  # not docker's default of 30, so the stop is seen to pass the backend's own grace
    backend.prepare(_plan())
    assert _stops(docker) == [("live", 17)] and backend.server_found_down is None
    kinds = [c[0] for c in docker.calls]
    # The server stops only after every pull and the volume helper: a stop during a long pull leaves it running.
    assert kinds.index("stop") > max(i for i, k in enumerate(kinds) if k in ("pull", "run"))
    assert backend.restore_server() is True  # the first start failed, the second went through
    assert [c for c in docker.calls if c[0] == "start"] == [("start", "live")] * 2
    assert backend.restore_server() is True and len([c for c in docker.calls if c[0] == "start"]) == 2
    stuck = FakeDocker(server_state={"Running": True}, start_fails=99)
    backend = _docker_backend(stuck, RHVP_COMPARE_SERVER_CONTAINER="live")
    backend.prepare(_plan())
    with pytest.raises(runner.RoundError, match="docker start live"):
        backend.restore_server()
    idle = _docker_backend(FakeDocker(server_state={"Running": False, "Status": "exited"}),
                           RHVP_COMPARE_SERVER_CONTAINER="live")
    idle.prepare(_plan())
    assert idle.restore_server() is None  # it was not running: left alone
    assert idle.server_found_down is None  # and it was not running when the run began either: nothing to record
    missing = FakeDocker()
    with pytest.raises(runner.RoundError, match="names no container"):
        _docker_backend(missing, RHVP_COMPARE_SERVER_CONTAINER="missing").prepare(_plan())
    assert not [c for c in missing.calls if c[0] == "pull"]  # a wrong name fails before any pull


class HookDocker(FakeDocker):
    """A FakeDocker that runs a callback once: during the first pull (`at="pull"`), or during the second read of the
    server's state, the one just before its stop (`at="re-read"`). It stands in for a signal or a state change that
    lands then."""

    def __init__(self, hook, at: str = "pull", **kwargs):
        super().__init__(**kwargs)
        self.hook, self.at, self.inspects = hook, at, 0

    def _fire(self):
        hook, self.hook = self.hook, None
        if hook:
            hook()

    def pull(self, ref):
        super().pull(ref)
        if self.at == "pull":
            self._fire()

    def inspect(self, cid):
        self.inspects += 1
        if self.at == "re-read" and self.inspects == 2:
            self._fire()
        return super().inspect(cid)


def test_a_stop_during_the_pulls_leaves_the_server_running():
    holder = {}
    docker = HookDocker(lambda: holder["backend"].on_stop(), server_state={"Running": True})
    backend = holder["backend"] = _docker_backend(docker, RHVP_COMPARE_SERVER_CONTAINER="live")
    backend.prepare(_plan())
    assert _stops(docker) == [] and docker.server_state["Running"]
    assert backend.restore_server() is None  # nothing was stopped, so nothing to bring back
    assert backend.server_found_down is None  # it is still running: nothing to record


@pytest.mark.parametrize("now, found", [({"Running": False, "Status": "exited"}, "exited"), (None, "gone")])
def test_the_server_state_is_read_again_just_before_the_stop(now, found):
    docker = HookDocker(lambda: setattr(docker, "server_state", now), server_state={"Running": True})
    backend = _docker_backend(docker, RHVP_COMPARE_SERVER_CONTAINER="live")
    backend.prepare(_plan())
    assert _stops(docker) == []  # it had stopped by itself (or been removed) during the pulls: left alone
    assert backend.restore_server() is None
    assert backend.server_found_down == found  # and the runner records it (run.server_found_down)


@pytest.mark.parametrize("at", ["pull", "re-read"])
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_before_the_server_stop_leaves_it_running(tmp_path, sig, at):
    """The whole path, with a real signal sent during the image pulls or during the server's second state read:
    Runner._on_term hands the stop to the backend at once (on_stop), so prepare() does not stop the server, and the
    run's end has nothing to start again. SIGINT is Ctrl-C on scripts/compare-local.sh."""
    held = {}

    def send():
        # Only once the runner's handler is in place; otherwise the signal would end the test process itself.
        assert signal.getsignal(sig) == held["runner"]._on_term
        os.kill(os.getpid(), sig)

    docker = HookDocker(send, at=at, server_state={"Running": True, "Status": "running"})
    backend = _docker_backend(docker, results_root=tmp_path, RHVP_COMPARE_SERVER_CONTAINER="live")
    the_runner = held["runner"] = runner.Runner(_plan(), backend, tmp_path, sleep=lambda s: None)
    results = the_runner.run()
    assert docker.hook is None  # the signal was sent
    assert _stops(docker) == [] and not [c for c in docker.calls if c[0] == "start"]
    assert docker.server_state["Running"]
    run = results["run"]
    assert (run["status"], run["error_code"], run["server_restored"]) == ("partial", runner.STOPPED, None)
    assert results["launches"] == [] and run["server_found_down"] is None
    assert run["errors"] == ["stopped by SIGTERM"]  # the runner's words for any stop; no cleanup error
    assert signal.getsignal(sig) != the_runner._on_term  # the earlier handler is back


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_server_that_stopped_by_itself_is_recorded_also_in_a_run_stopped_during_the_pulls(tmp_path, sig):
    """The live server exits by itself during a pull, and a stop lands in the same pull: the run still records it
    (run.server_found_down, a note, results.md), and neither stops nor starts it."""
    held = {}

    def exit_then_signal():
        docker.server_state = {"Running": False, "Status": "exited"}
        assert signal.getsignal(sig) == held["runner"]._on_term
        os.kill(os.getpid(), sig)

    docker = HookDocker(exit_then_signal, at="pull", server_state={"Running": True, "Status": "running"})
    backend = _docker_backend(docker, results_root=tmp_path, RHVP_COMPARE_SERVER_CONTAINER="live")
    the_runner = held["runner"] = runner.Runner(_plan(), backend, tmp_path, sleep=lambda s: None)
    results = the_runner.run()
    assert docker.hook is None  # the signal was sent
    assert _stops(docker) == [] and not [c for c in docker.calls if c[0] == "start"]
    run = results["run"]
    assert (run["status"], run["error_code"], run["server_restored"]) == ("partial", runner.STOPPED, None)
    assert run["server_found_down"] == "exited"
    assert any(note.startswith(runner.FOUND_DOWN) for note in run["notes"])
    assert "**The live server had stopped by itself**" in (the_runner.dir / "results.md").read_text()


class FailingReread(HookDocker):
    """The second read of the server's state (after the pulls) fails, as a Docker API error would."""

    def inspect(self, cid):
        if self.inspects == 1:
            self.inspects += 1
            raise runner.RoundError("docker could not inspect a container: HTTP 500")
        return super().inspect(cid)


def test_a_state_read_that_fails_during_a_stop_leaves_the_run_partial(tmp_path):
    """A stop lands during the pulls, and the state read after them fails: the server is left as it is either way,
    so the run is stopped (partial, exit 1) with a note, not failed (exit 3). Without a stop the same error fails
    the run before it has changed anything."""
    held = {}
    docker = FailingReread(lambda: os.kill(os.getpid(), signal.SIGTERM), at="pull",
                           server_state={"Running": True, "Status": "running"})
    backend = _docker_backend(docker, results_root=tmp_path, RHVP_COMPARE_SERVER_CONTAINER="live")
    the_runner = held["runner"] = runner.Runner(_plan(), backend, tmp_path, sleep=lambda s: None)
    results = the_runner.run()
    run = results["run"]
    assert (run["status"], run["error_code"], run["server_found_down"]) == ("partial", runner.STOPPED, None)
    assert runner.exit_code(results) == runner.EXIT_PARTIAL and _stops(docker) == []
    assert "the live server's state was not read again after the stop: RoundError" in run["notes"]
    quiet = FailingReread(None, server_state={"Running": True, "Status": "running"})
    with pytest.raises(runner.RoundError, match="inspect"):
        _docker_backend(quiet, results_root=tmp_path, RHVP_COMPARE_SERVER_CONTAINER="live").prepare(_plan())
    assert _stops(quiet) == []


def test_the_local_gpu_check_sees_the_hosts_gpu_programs(monkeypatch):
    monkeypatch.setattr(dockerapi.time, "sleep", lambda s: None)
    docker = FakeDocker()
    backend = _docker_backend(docker)
    backend.run_id = "t-12345678"
    side = {"image": "x/a:1", "gpu": True}
    backend.gpu_free({}, side)
    [(_, _, body)] = [c for c in docker.calls if c[0] == "run"]
    assert body["HostConfig"]["PidMode"] == "host" and body["HostConfig"]["Memory"] == dockerapi.HELPER_MEMORY
    docker.run_log = "2026-01-01T00:00:00Z 4242\n"  # another program on the GPU
    with pytest.raises(runner.RoundError, match="another program holds the GPU") as caught:
        backend.gpu_free({}, side, wait_s=0)
    assert caught.value.code == runner.GPU_NOT_FREE


# ---- mode scaleout: the surge, the scale-ups the server logged, the docker replicas ----

SLO_REPORT = json.loads((FIXTURES / "guidellm-report-v0.8.1-slo.json").read_text())


def _scale_plan(**scale) -> dict:
    config = {**_plan_config(), "mode": "scaleout", "rounds": {"count": 7, "warmup": True},
              "scaleOut": {"requestRate": 2.5, "surgeSeconds": 60, "stepAfterSeconds": 10, "ttftMs": 300,
                           "tpotMs": 25, **scale}}
    return plans.build(config, "docker")


def _constant(report: dict, rate: float = 2.5) -> dict:
    """The fixture is a concurrent run with latency goals; a surge's report differs in its strategy block only,
    written here by hand as GuideLLM v0.8.1 writes it for the constant profile."""
    out = json.loads(json.dumps(report))
    out["benchmarks"][0]["config"]["strategy"] = {"type_": "constant", "rate": rate, "max_concurrency": None,
                                                  "rampup_duration": 0.0}
    return out


def test_the_surge_command_runs_a_constant_rate_with_the_latency_goals():
    args = guidellm.command(_scale_plan(), "http://127.0.0.1:8787", "/results/r/guidellm.json")
    assert args[args.index("--profile") + 1] == "kind=constant,rate=2.5"
    assert args[args.index("--constraint") + 1] == "kind=max_duration,seconds=60"
    assert "--override" not in args and "kind=json,path=/results/r/guidellm.json" in args
    metrics = json.loads(args[args.index("--metrics") + 1])
    assert metrics == {"kind": "generative", "sample_size": 0, "slo": {"ttft_ms": 300, "tpot_ms": 25}}
    # the shape GuideLLM v0.8.1 records for the goals it was given (the fixture's own config)
    recorded = SLO_REPORT["config"]["spec"]["metrics"]
    assert set(metrics) <= set(recorded) and set(metrics["slo"]) <= set(recorded["slo"])
    one_goal = guidellm.command(_scale_plan(tpotMs=None), "t", "o")
    assert json.loads(one_goal[one_goal.index("--metrics") + 1])["slo"] == {"ttft_ms": 300}
    assert guidellm.command(_plan(), "t", "o")[guidellm.command(_plan(), "t", "o").index("--metrics") + 1] == (
        "kind=generative,sample_size=0")  # mode levels: unchanged


def test_guidellm_skips_its_own_health_check_in_both_modes():
    """GuideLLM v0.8.1's openai_http backend sends GET <target>/health before the load unless validate_backend is
    false, and Ray Serve LLM's OpenAI app has no /health route; the runner has had a streamed answer first."""
    for plan, target in ((_plan(), "http://side:8000"), (_scale_plan(), "http://127.0.0.1:8787")):
        args = guidellm.command(plan, target, "o")
        backend = args[args.index("--backend") + 1]
        fields = dict(part.split("=", 1) for part in backend.split(","))
        assert fields == {"kind": "openai_http", "target": target, "model": "m",
                          "request_format": plan["load"]["request_format"], "validate_backend": "false"}


def test_the_surge_level_from_a_report_with_latency_goals():
    level = guidellm.surge_level(_constant(SLO_REPORT))
    assert level["concurrency"] is None and level["rate"] == 2.5 and level["usable"] and level["note"] is None
    assert level["slo_attainment"] == pytest.approx(0.9574468085106383)
    assert level["slo_determined_requests"] == 47 and level["goodput_rps"] == pytest.approx(11.0)
    assert level["requests"] == {"successful": 47, "errored": 0, "incomplete": 1}
    flat = guidellm.surge_values(level)
    assert set(flat) == {"surge/slo_attainment", "surge/goodput_rps", "surge/ttft_ms_median", "surge/itl_ms_median",
                         "surge/output_tokens_per_s"}
    assert flat["surge/ttft_ms_median"] == level["ttft_ms"]["median"]


def test_a_surge_with_failed_requests_keeps_its_attainment_only():
    report = _constant(SLO_REPORT)
    metrics = report["benchmarks"][0]["metrics"]
    metrics["request_totals"]["errored"] = 5
    metrics["slo_attainment"] = 0.5
    level = guidellm.surge_level(report)
    assert not level["usable"] and "count as requests that missed the latency goals" in level["note"]
    assert set(guidellm.surge_values(level)) == {"surge/slo_attainment", "surge/goodput_rps"}


@pytest.mark.parametrize("successful, errored, attainment, goodput, expected", [
    (0, 0, None, None, {"surge/slo_attainment": 0.0, "surge/goodput_rps": 0.0}),  # nothing finished: a 0
    (0, 6, 0.0, None, {"surge/slo_attainment": 0.0, "surge/goodput_rps": 0.0}),   # all failed: GuideLLM's 0
    (10, 0, None, None, {}),  # judged none (a goal it cannot measure): missing, not 0
])
def test_a_surge_that_gives_no_attainment(successful, errored, attainment, goodput, expected):
    report = _constant(SLO_REPORT)
    metrics = report["benchmarks"][0]["metrics"]
    metrics["request_totals"].update(successful=successful, errored=errored)
    metrics["slo_attainment"] = attainment
    if goodput is None:
        del metrics["request_goodput"]
    level = guidellm.surge_level(report)
    got = {k: v for k, v in guidellm.surge_values(level).items() if k in ("surge/slo_attainment",
                                                                       "surge/goodput_rps")}
    assert got == expected
    if successful and attainment is None:
        assert "judged no request" in level["note"]


def test_start_runs_guidellm_in_the_background(tmp_path):
    fake = tmp_path / "guidellm"
    fake.write_text("#!/bin/sh\nsleep 1\nfor a in \"$@\"; do case \"$a\" in kind=json,path=*) "
                    "echo '{}' > \"${a#kind=json,path=}\";; esac; done\n")
    fake.chmod(0o755)
    running = guidellm.start(_scale_plan(), "http://127.0.0.1:8787", tmp_path / "out", executable=str(fake))
    assert running.poll() is None  # still running: start returned at once
    assert running.finish() == tmp_path / "out" / "guidellm.json"
    hang = tmp_path / "hang"
    hang.write_text("#!/bin/sh\nsleep 30\n")
    hang.chmod(0o755)
    started = guidellm.start(_scale_plan(), "t", tmp_path / "out2", executable=str(hang))
    started.stop()
    assert started.poll() is not None


def test_the_first_scale_up_the_server_logged():
    events = [{"ts": 90.0, "from": 1, "to": 2, "reason": "target"}, {"ts": 112.5, "from": 1, "to": 2, "reason": "running"},
              {"ts": 105.0, "from": 1, "to": 2, "reason": "target"}, {"ts": 130.0, "from": 2, "to": 1,
                                                                    "reason": "target"}]
    assert logs.first_scale_up(events, 100.0) == 5.0  # a decision before the surge and a scale-down do not count
    assert logs.first_scale_up(events[3:], 100.0) is None


class ReplicaDocker(FakeDocker):
    """FakeDocker with the calls a side's start, its server facts and the cache volumes make."""

    def __init__(self, exec_result=(0, "VLLM_X=1\0HF_TOKEN=hf_s\0PATH=/bin\0")):
        super().__init__()
        self.created, self.volumes_removed, self.exec_result = {}, [], exec_result

    def create(self, name, body):
        self.created[name] = body
        return f"{name}-id"

    def image(self, ref):
        return {"Config": {"Entrypoint": ["/usr/bin/tini", "--"], "Cmd": ["serve"]}}

    def exec(self, cid, cmd, timeout_s=30.0):
        self.calls.append(("exec", cid, cmd))
        if isinstance(self.exec_result, Exception):
            raise self.exec_result
        return self.exec_result

    def inspect(self, cid):
        return {"State": {"Running": True}, "Config": {"Env": ["VLLM_Y=2", "PATH=/usr/bin"]}}

    def by_label(self, value):
        return []

    def remove_volume(self, name):
        self.volumes_removed.append(name)


class ProbedDocker(ReplicaDocker):
    """nvidia-smi's answer per GPU probe: `answers` maps the probe's replica (1 or 2) to its index, UUID line; `mig`
    maps it to the MIG device UUID `nvidia-smi -L` lists under the parent GPU (none listed when left out)."""

    def __init__(self, answers: dict, mig: dict | None = None):
        super().__init__()
        self.answers, self.mig = answers, mig or {}

    def run(self, name, body, timeout_s=300.0, log_to=None):
        self.calls.append(("run", name, body))
        for replica, answer in self.answers.items():
            if f"gpu-probe-{replica}" in name:
                return 0, f"2026-01-01T00:00:00Z {answer}\n"
            if f"gpu-mig-{replica}" in name:
                listed = "2026-01-01T00:00:00Z GPU 0: Test GPU (UUID: GPU-0000)\n"
                if replica in self.mig:
                    listed += f"2026-01-01T00:00:00Z   MIG 1g.10gb Device  0: (UUID: {self.mig[replica]})\n"
                return 0, listed
        return 0, ""


def test_docker_replicas_get_their_own_gpu_the_cpu_set_and_the_warm_cache():
    docker = ProbedDocker({1: "0, GPU-0000", 2: "1, GPU-1111"})  # what nvidia-smi lists for each replica's GPU
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,1", RHVP_COMPARE_CPUSET="0-3",
                              RHVP_COMPARE_CACHE_VOLUME="cc")
    config = {**_plan_config(), "mode": "scaleout", "rounds": {"count": 7, "warmup": True},
              "scaleOut": {"sharedGpu": False},
              "sides": [{"name": "a", "image": "x/a:1", "compileCache": "warm"}, {"name": "b", "image": "x/b:1"}]}
    plan = plans.build(config, "docker")
    backend.prepare(plan)
    volume = f"cc-{plan['run_id'][-8:]}-a"
    assert backend.cache_volumes == {"A": volume} and docker.volumes_removed == [volume]  # a fresh one
    chmod = [body for kind, name, body in [c for c in docker.calls if c[0] == "run"] if "cache" in name]
    assert chmod[0]["Cmd"] == ["chmod 1777 /cache"] and chmod[0]["HostConfig"]["CpusetCpus"] == "0-3"
    side_a, side_b = plan["sides"]
    launch = plan["launches"][2]  # launch 3, side A
    first = backend.start(plan, launch, side_a, plans.side_spec(plan, side_a))
    second = backend.start(plan, launch, side_a, plans.side_spec(plan, side_a), replica=2)
    assert second["name"] == first["name"] + "-2" and second["replica"] == 2
    host1, host2 = docker.created[first["name"]]["HostConfig"], docker.created[second["name"]]["HostConfig"]
    assert host1["DeviceRequests"][0]["DeviceIDs"] == ["0"] and host2["DeviceRequests"][0]["DeviceIDs"] == ["1"]
    assert f"{volume}:/cache" in host2["Binds"] and "/cache" not in host2["Tmpfs"] and host2["CpusetCpus"] == "0-3"
    # The CUDA driver's own kernel cache goes on the warm volume too, as the server chart's node and shared modes do.
    for started in (first, second):
        env = docker.created[started["name"]]["Env"]
        assert "CUDA_CACHE_PATH=/cache/cuda" in env and "CUDA_CACHE_MAXSIZE=4294967296" in env
    cold = backend.start(plan, plan["launches"][4], side_b, plans.side_spec(plan, side_b))  # launch 5, side B
    cold_host = docker.created[cold["name"]]["HostConfig"]
    assert "/cache" in cold_host["Tmpfs"] and not any(b.endswith(":/cache") for b in cold_host["Binds"])
    assert not any(e.startswith("CUDA_CACHE_") for e in docker.created[cold["name"]]["Env"])
    # A side's own env wins over the warm defaults.
    spec = plans.side_spec(plan, side_a)
    spec["env"]["CUDA_CACHE_MAXSIZE"] = "1073741824"
    own = backend.start(plan, plan["launches"][6], side_a, spec)
    assert "CUDA_CACHE_MAXSIZE=1073741824" in docker.created[own["name"]]["Env"]
    assert "CUDA_CACHE_PATH=/cache/cuda" in docker.created[own["name"]]["Env"]
    with pytest.raises(runner.RoundError, match="replica 3 needs one more"):
        backend.start(plan, launch, side_a, plans.side_spec(plan, side_a), replica=3)
    backend.cleanup()
    assert docker.volumes_removed == [volume, volume]


def test_the_warm_cache_env_matches_the_server_charts_kept_cache_env():
    # dockerapi.WARM_CACHE_ENV and the server chart's compileCache node/shared env stay in step.
    template = (ROOT / "charts/all/ray-serve-llm/templates/deployment.yaml").read_text()
    blocks = [part.split("{{- end }}", 1)[0]
              for part in template.split('{{- if include "ray-serve-llm.compileCacheKept" . }}')[1:]]
    [kept] = [block for block in blocks if "CUDA_CACHE_" in block]  # the env the kept-cache modes add
    found = dict(re.findall(r"- name: (CUDA_CACHE_\w+)\s+(?:#[^\n]*\s+)*value: \"?([^\"\s]+)\"?", kept))
    assert found == dockerapi.WARM_CACHE_ENV
    assert "CUDA_CACHE_" not in template.replace(kept, "")  # and only those modes


def _scale_plan_for(shared: bool) -> dict:
    config = {**_plan_config(), "mode": "scaleout", "rounds": {"count": 7, "warmup": True},
              "scaleOut": {"sharedGpu": shared}}
    return plans.build(config, "docker")


@pytest.mark.parametrize("shared, replica_gpus, refused", [
    (True, "", None),
    (True, "0,0", None),
    (False, "0,1", None),
    (True, "0,1", "gives replica 1 GPU 0 and replica 2 GPU 1"),
    (False, "", "both replicas would get GPUs all"),
    (False, "1,1", "both replicas would get GPUs 1"),
    (False, "0", "needs one GPU id per replica"),
])
def test_docker_refuses_a_shared_gpu_setting_that_does_not_match_where_the_replicas_run(shared, replica_gpus,
                                                                                       refused):
    # scaleOut.sharedGpu sets the memory share and the notebook's sentence; REPLICA_GPUS (else GPUS) places the
    # replicas. A run whose two disagree is refused before anything changes, and prepare() refuses it too.
    plan = _scale_plan_for(shared)
    backend = _docker_backend(FakeDocker(), RHVP_COMPARE_REPLICA_GPUS=replica_gpus)
    if refused is None:
        backend.check_plan(plan)
        return
    with pytest.raises(plans.PlanError, match=refused):
        backend.check_plan(plan)
    docker = FakeDocker(server_state={"Running": True})
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS=replica_gpus, RHVP_COMPARE_SERVER_CONTAINER="live")
    with pytest.raises(plans.PlanError, match=refused):
        backend.prepare(plan)
    assert docker.calls == []  # nothing pulled, nothing stopped


def test_a_mismatched_shared_gpu_setting_ends_the_run_plan_invalid_before_anything_changes(tmp_path):
    config_path = tmp_path / "compare.json"
    config_path.write_text(json.dumps({**_plan_config(), "mode": "scaleout", "rounds": {"count": 7, "warmup": True}}))
    docker = FakeDocker()
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,1")
    termination = tmp_path / "termination"
    with pytest.raises(plans.PlanError, match="scaleOut.sharedGpu is true"):
        runner.main_run(str(config_path), str(tmp_path / "results"), backend,
                        env={"RHVP_COMPARE_TERMINATION_LOG": str(termination)})
    assert "PLAN_INVALID" in termination.read_text() and docker.calls == []
    assert not (tmp_path / "results").exists()


def test_docker_reads_each_replicas_gpus_before_the_first_launch_and_records_them():
    docker = ProbedDocker({1: "0, GPU-0000", 2: "1, GPU-1111"})
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,1")
    plan = _scale_plan_for(False)
    backend.prepare(plan)
    probes = [body for kind, name, body in [c for c in docker.calls if c[0] == "run"] if "gpu-probe" in name]
    assert [b["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] for b in probes] == [["0"], ["1"]]
    assert probes[0]["Cmd"] == ["--query-gpu=index,uuid", "--format=csv,noheader"]
    side_a = plan["sides"][0]
    second = backend.start(plan, plan["launches"][2], side_a, plans.side_spec(plan, side_a), replica=2)
    facts = backend.server_facts(second)
    assert facts["gpus"] == "1" and facts["gpu_uuids"] == ["GPU-1111"]
    record = servers.record(2, None, [], facts["env"], facts["env_source"], gpus=facts["gpus"],
                            gpu_uuids=facts["gpu_uuids"])
    assert (record["gpus"], record["gpu_uuids"]) == ("1", ["GPU-1111"])
    shared = _docker_backend(ReplicaDocker())  # sharedGpu true, no REPLICA_GPUS: one request, read once
    shared.docker.run_log = "2026-01-01T00:00:00Z 0, GPU-0000\n"
    shared.prepare(_scale_plan_for(True))
    assert len([c for c in shared.docker.calls if c[0] == "run" and "gpu-probe" in c[1]]) == 1


def test_two_requests_that_reach_one_gpu_fail_in_prepare_when_each_replica_needs_its_own():
    # An index and that GPU's UUID are different strings, so check_plan lets them through; nvidia-smi's UUIDs do not.
    docker = ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"})
    docker.server_state = {"Running": True}
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,GPU-0000", RHVP_COMPARE_SERVER_CONTAINER="live")
    plan = _scale_plan_for(False)
    backend.check_plan(plan)
    with pytest.raises(runner.RoundError, match="REPLICA_GPUS 0 and GPU-0000 both reach GPU GPU-0000"):
        backend.prepare(plan)
    assert _stops(docker) == []  # the server was never stopped
    shared = _docker_backend(ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"}),
                             RHVP_COMPARE_REPLICA_GPUS="0,GPU-0000")
    with pytest.raises(plans.PlanError, match="gives replica 1 GPU 0 and replica 2 GPU GPU-0000"):
        shared.check_plan(_scale_plan_for(True))  # sharedGpu true still compares the strings, before any probe


@pytest.mark.parametrize("request_, mig", [
    ("0:0", True), ("12:3", True), ("MIG-4f2a9c1e-0000-5000-8000-000000000001", True),
    ("MIG-GPU-4f2a9c1e-0000-5000-8000-000000000000/1/0", True), ("0:0,0:1", True),
    ("0", False), ("GPU-4f2a9c1e-0000-5000-8000-000000000000", False), ("all", False), ("0:0,1", False), ("", False),
    ("0:", False),
])
def test_mig_requests_are_told_from_whole_gpus(request_, mig):
    assert dockerapi.is_mig_request(request_) is mig


@pytest.mark.parametrize("replica_gpus", ["0:0,0:1", "MIG-aaaa,MIG-bbbb"])
def test_two_mig_devices_of_one_gpu_run_when_each_replica_needs_its_own(replica_gpus):
    # nvidia-smi's GPU query in a container given a MIG device lists the parent GPU's UUID, so both probes show
    # GPU-0000; nvidia-smi -L lists each request's own MIG device, and two different devices run. Their server
    # records name the parent GPU.
    docker = ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"}, mig={1: "MIG-aaaa", 2: "MIG-bbbb"})
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS=replica_gpus)
    plan = _scale_plan_for(False)
    backend.check_plan(plan)
    backend.prepare(plan)
    assert backend.notes == []
    listed = [body for kind, name, body in [c for c in docker.calls if c[0] == "run"] if "gpu-mig" in name]
    assert [b["Cmd"] for b in listed] == [["-L"], ["-L"]]
    assert [b["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] for b in listed] == [[r] for r in replica_gpus.split(",")]
    side_a = plan["sides"][0]
    second = backend.start(plan, plan["launches"][2], side_a, plans.side_spec(plan, side_a), replica=2)
    facts = backend.server_facts(second)
    assert facts["gpus"] == replica_gpus.split(",")[1] and facts["gpu_uuids"] == ["GPU-0000"]


def test_one_mig_device_named_two_ways_fails_in_prepare_when_each_replica_needs_its_own():
    # 0:0 and MIG-aaaa are two strings for one MIG device: check_plan lets them through, nvidia-smi -L does not.
    docker = ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"}, mig={1: "MIG-aaaa", 2: "MIG-aaaa"})
    docker.server_state = {"Running": True}
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0:0,MIG-aaaa", RHVP_COMPARE_SERVER_CONTAINER="live")
    plan = _scale_plan_for(False)
    backend.check_plan(plan)
    with pytest.raises(runner.RoundError, match="REPLICA_GPUS 0:0 and MIG-aaaa both reach MIG device MIG-aaaa"):
        backend.prepare(plan)
    assert _stops(docker) == []  # the server was never stopped


def test_mig_devices_whose_own_uuids_are_not_listed_run_with_a_note():
    docker = ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"})  # nvidia-smi -L lists the parent GPU only
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0:0,0:1")
    backend.prepare(_scale_plan_for(False))
    [note] = backend.notes
    assert "REPLICA_GPUS 0:0 and 0:1 are MIG devices whose own UUIDs nvidia-smi -L did not list" in note


def test_a_mig_device_and_its_whole_gpu_fail_in_prepare_when_each_replica_needs_its_own():
    docker = ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"})
    docker.server_state = {"Running": True}
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,0:1", RHVP_COMPARE_SERVER_CONTAINER="live")
    plan = _scale_plan_for(False)
    backend.check_plan(plan)
    with pytest.raises(runner.RoundError, match="REPLICA_GPUS 0 and 0:1 both reach GPU GPU-0000"):
        backend.prepare(plan)
    assert _stops(docker) == []  # the server was never stopped


def test_a_replica_gpu_this_machine_lacks_fails_in_prepare_before_the_server_stops():
    class NoSecondGpu(ReplicaDocker):
        def run(self, name, body, timeout_s=300.0, log_to=None):
            self.calls.append(("run", name, body))
            if "gpu-probe-2" in name:
                raise runner.RoundError("docker could not start a container: HTTP 500 unknown device id: 1")
            return 0, "2026-01-01T00:00:00Z 0, GPU-0000\n"

    docker = NoSecondGpu()
    docker.server_state = {"Running": True}
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,1", RHVP_COMPARE_SERVER_CONTAINER="live")
    with pytest.raises(runner.RoundError, match="the GPUs of replica 2 .REPLICA_GPUS 1. could not be reached"):
        backend.prepare(_scale_plan_for(False))
    assert _stops(docker) == []  # the server was never stopped


def test_docker_server_facts_read_the_server_process_then_the_container():
    docker = ReplicaDocker()
    backend = _docker_backend(docker)
    facts = backend.server_facts({"id": "c1", "image": "x/a:1"})
    assert facts == {"image_entrypoint": ["/usr/bin/tini", "--"], "image_cmd": ["serve"], "env_source": "process",
                     "env": {"VLLM_X": "1", "HF_TOKEN": "hf_s", "PATH": "/bin"}}
    assert docker.calls[-1] == ("exec", "c1", ["sh", "-c", dockerapi.SERVER_ENVIRON])
    no_shell = _docker_backend(ReplicaDocker(exec_result=runner.RoundError("docker could not start an exec")))
    facts = no_shell.server_facts({"id": "c1", "image": "x/a:1"})
    assert facts["env_source"] == "container" and facts["env"] == {"VLLM_Y": "2", "PATH": "/usr/bin"}
    assert dockerapi.environ("A=1\0B=x=y\0\0junk\0") == {"A": "1", "B": "x=y"}


def test_the_server_environ_script_picks_the_largest_process(tmp_path):
    # Run the script against this machine's own /proc: it must print some process's environment, NUL-separated.
    out = subprocess.run(["sh", "-c", dockerapi.SERVER_ENVIRON], capture_output=True, timeout=30)
    assert out.returncode == 0 and b"=" in out.stdout and b"\0" in out.stdout


# ---- mode scaleout on OpenShift: the second replica's pod, the GPU share, the router and the surge ----

L4_SHARED = {"kubernetes.io/hostname": "gpu-1", "kubernetes.io/arch": "amd64",
             "nvidia.com/gpu.product": "NVIDIA-L4-SHARED", kube.GPU_COUNT_LABEL: "1", kube.GPU_REPLICAS_LABEL: "2",
             kube.GPU_SHARING_LABEL: "time-slicing"}
#: KubeBackend.prepare in mode scaleout raises the runner's own open-file limit; the tests keep the test process's.
RAISE_OPEN_FILES = kube.raise_open_files


@pytest.fixture(autouse=True)
def _runner_open_files(monkeypatch):
    monkeypatch.setattr(kube, "raise_open_files", lambda: (
        "the runner: open-file limit 1048576 (hard limit 1048576), already at the hard limit", 1048576))


class FakeCluster:
    """A fake Kubernetes API for the scale round. One GPU node (`labels` over an L4 offered as two time-sliced
    shares, a label set to None left out; `allocatable` its status.allocatable); a pod is Running on it with an IP as
    soon as it is created, unless its name is in `unschedulable` (it then stays Pending with the scheduler's message).
    The live server's Deployment is at 1, the status ConfigMaps are kept, and every call is recorded as (method,
    path). `read_only`: any call but a GET fails the test at once (AssertionError, which the Kube client does not
    catch)."""

    def __init__(self, labels: dict | None = None, unschedulable: tuple = (), extra_nodes: list | None = None,
                 allocatable: dict | None = None, read_only: bool = False):
        self.labels = {k: v for k, v in {**L4_SHARED, **(labels or {})}.items() if v is not None}
        self.allocatable = {kube.WHOLE_GPU_RESOURCE: "2"} if allocatable is None else allocatable
        self.extra_nodes, self.unschedulable, self.read_only = extra_nodes or [], set(unschedulable), read_only
        self.pods: dict[str, dict] = {}
        self.created: list[dict] = []
        self.calls: list[tuple] = []
        self.replicas, self.configmaps = 1, {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else None
        self.calls.append((method, path))
        if self.read_only and method != "GET":
            raise AssertionError(f"nothing may change here, and the runner sent {method} {path}")
        pods = "/api/v1/namespaces/vllm-inference/pods"
        if path == "/api/v1/nodes":
            assert request.url.params["labelSelector"] == "kubernetes.io/arch=amd64"
            return httpx.Response(200, json={"items": [{"metadata": {"name": "gpu-1.example", "labels": self.labels},
                                                        "status": {"allocatable": self.allocatable}},
                                                       *self.extra_nodes]})
        if path.startswith("/api/v1/nodes/"):
            return httpx.Response(200, json={"metadata": {"name": path.rsplit("/", 1)[1], "labels": self.labels},
                                             "status": {"nodeInfo": {"osImage": "RHCOS"},
                                                        "allocatable": self.allocatable}})
        if path == pods and method == "POST":
            pod = json.loads(json.dumps(body))
            name = pod["metadata"]["name"]
            if name in self.unschedulable:
                pod["status"] = {"phase": "Pending", "conditions": [{
                    "type": "PodScheduled", "status": "False",
                    "message": "0/3 nodes are available: 1 Insufficient nvidia.com/gpu."}]}
            else:
                pod["spec"]["nodeName"] = "gpu-1.example"
                pod["status"] = {"phase": "Running", "podIP": f"10.128.0.{len(self.created) + 10}",
                                 "containerStatuses": [{"name": "server", "state": {"running": {}},
                                                        "imageID": "x/a@sha256:abc"}]}
            self.pods[name] = pod
            self.created.append(pod)
            return httpx.Response(201, json=pod)
        if path == pods and method == "GET":
            selector = request.url.params.get("labelSelector")
            items = list(self.pods.values())
            if selector:
                key, value = selector.split("=", 1)
                items = [p for p in items if p["metadata"]["labels"].get(key) == value]
            return httpx.Response(200, json={"items": items})
        if path.startswith(pods + "/"):
            name = path[len(pods) + 1:]
            if name.endswith("/log"):
                return httpx.Response(200, text="2026-01-01T00:00:00Z vLLM API server version 0.0.1\n")
            if method == "DELETE":
                self.pods.pop(name, None)
                return httpx.Response(200, json={})
            pod = self.pods.get(name)
            return httpx.Response(200, json=pod) if pod else httpx.Response(404, json={"message": "not found"})
        if path.endswith("/deployments/ray-serve-llm/scale"):
            if method == "PATCH":
                self.replicas = body["spec"]["replicas"]
            return httpx.Response(200, json={"spec": {"replicas": self.replicas}})
        if path == "/api/v1/namespaces/nb/pods":
            return httpx.Response(200, json={"items": []})
        if "/namespaces/nb/configmaps" in path:
            name = body["metadata"]["name"] if method == "POST" else path.rsplit("/", 1)[1]
            if method == "PATCH" and name not in self.configmaps:
                return httpx.Response(404, json={"message": "not found"})
            self.configmaps[name] = {**self.configmaps.get(name, {}), **(body.get("data") or {})}
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"message": "not found"})

    def backend(self, **env) -> kube.KubeBackend:
        client = kube.Kube(base_url="https://api.example:6443", token="t", verify=False,
                           transport=httpx.MockTransport(self.handler))
        return kube.KubeBackend(kube=client, env={"RHVP_COMPARE_STATUS_NAMESPACE": "nb", "RHVP_COMPARE_POD_NAME": "me",
                                                  **env}, sleep=lambda s: None)


def _kube_scale_plan(shared: bool = True, gpu: bool = True, **extra) -> dict:
    config = {**_plan_config(), "mode": "scaleout", "rounds": {"count": 2, "warmup": False, "twins": "none"},
              "scaleOut": {"requestRate": 2.5, "surgeSeconds": 60, "stepAfterSeconds": 10, "ttftMs": 300,
                           "tpotMs": 25, "sharedGpu": shared}, **extra}
    if not gpu:
        config["sides"] = [{**side, "gpu": False} for side in config["sides"]]
    return plans.build(config, "openshift")


@pytest.mark.parametrize("labels, shared, fits", [
    ({kube.GPU_COUNT_LABEL: "1", kube.GPU_REPLICAS_LABEL: "2"}, True, True),  # one L4 offered as two shares
    ({kube.GPU_COUNT_LABEL: "1", kube.GPU_REPLICAS_LABEL: "4"}, True, False),  # others could share it unseen
    ({kube.GPU_COUNT_LABEL: "2", kube.GPU_REPLICAS_LABEL: "2"}, True, False),  # the shares may be on two GPUs
    ({kube.GPU_COUNT_LABEL: "1"}, True, False),  # time-slicing off
    ({kube.GPU_COUNT_LABEL: "2"}, False, True),
    ({kube.GPU_COUNT_LABEL: "4", kube.GPU_REPLICAS_LABEL: "1"}, False, True),
    ({kube.GPU_COUNT_LABEL: "2", kube.GPU_REPLICAS_LABEL: "2"}, False, False),  # two pods may share one GPU
    ({kube.GPU_COUNT_LABEL: "1", kube.GPU_REPLICAS_LABEL: "1"}, False, False),
    ({}, True, False),  # not a GPU node, or no GPU feature discovery
    ({kube.GPU_COUNT_LABEL: "one", kube.GPU_REPLICAS_LABEL: "2"}, True, False),
    # the sharing strategy: time-sliced shares only, and none at all when not shared (absent: the other two decide)
    ({kube.GPU_COUNT_LABEL: "1", kube.GPU_REPLICAS_LABEL: "2", kube.GPU_SHARING_LABEL: "time-slicing"}, True, True),
    ({kube.GPU_COUNT_LABEL: "1", kube.GPU_REPLICAS_LABEL: "2", kube.GPU_SHARING_LABEL: "mps"}, True, False),
    ({kube.GPU_COUNT_LABEL: "1", kube.GPU_REPLICAS_LABEL: "2", kube.GPU_SHARING_LABEL: ""}, True, False),
    ({kube.GPU_COUNT_LABEL: "2", kube.GPU_REPLICAS_LABEL: "1", kube.GPU_SHARING_LABEL: "none"}, False, True),
    ({kube.GPU_COUNT_LABEL: "2", kube.GPU_SHARING_LABEL: "none"}, False, True),
    ({kube.GPU_COUNT_LABEL: "2", kube.GPU_REPLICAS_LABEL: "1", kube.GPU_SHARING_LABEL: "time-slicing"}, False, False),
    ({kube.GPU_COUNT_LABEL: "2", kube.GPU_REPLICAS_LABEL: "1", kube.GPU_SHARING_LABEL: "mps"}, False, False),
])
def test_which_gpu_nodes_hold_a_scale_launch(labels, shared, fits):
    assert kube.share_fits(labels, shared) is fits


@pytest.mark.parametrize("allocatable, shared, fits", [
    ({"nvidia.com/gpu": "2"}, True, True),
    ({"nvidia.com/gpu.shared": "2"}, True, False),  # the device plugin renamed the shares (renameByDefault: true)
    ({"nvidia.com/gpu": "1"}, True, False),
    ({}, True, False),
    (None, True, False),
    ({"nvidia.com/gpu": "4"}, False, True),
    ({"nvidia.com/gpu": "1"}, False, False),  # one of the two GPUs is not offered (unhealthy)
])
def test_a_gpu_node_must_offer_the_two_replicas_their_gpu_resource(allocatable, shared, fits):
    labels = ({kube.GPU_COUNT_LABEL: "1", kube.GPU_REPLICAS_LABEL: "2", kube.GPU_SHARING_LABEL: "time-slicing"}
              if shared else {kube.GPU_COUNT_LABEL: "4", kube.GPU_SHARING_LABEL: "none"})
    node = {"metadata": {"name": "gpu-1.example", "labels": labels},
            **({} if allocatable is None else {"status": {"allocatable": allocatable}})}
    assert kube.share_fits(labels, shared) is True  # the labels alone would let it through
    assert kube.node_fits(node, shared) is fits


def _matches(affinity: dict, labels: dict) -> bool:
    """Kubernetes' required node affinity, for the operators share_affinity uses."""
    def one(expr: dict) -> bool:
        value = labels.get(expr["key"])
        if expr["operator"] == "In":
            return value in expr["values"]
        if expr["operator"] == "DoesNotExist":
            return value is None
        if expr["operator"] == "Gt":
            return value is not None and value.isdigit() and int(value) > int(expr["values"][0])
        raise AssertionError(expr)
    terms = affinity["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    return any(all(one(e) for e in term["matchExpressions"]) for term in terms)


@pytest.mark.parametrize("count", ["1", "2", "4"])
@pytest.mark.parametrize("replicas", [None, "1", "2", "4"])
@pytest.mark.parametrize("strategy", [None, "none", "time-slicing", "mps"])
@pytest.mark.parametrize("shared", [True, False])
def test_the_node_affinity_asks_the_scheduler_for_the_nodes_the_check_accepts(count, replicas, strategy, shared):
    labels = {kube.GPU_COUNT_LABEL: count, **({kube.GPU_REPLICAS_LABEL: replicas} if replicas else {}),
              **({kube.GPU_SHARING_LABEL: strategy} if strategy else {})}
    assert _matches(kube.share_affinity(shared), labels) is kube.share_fits(labels, shared)


def test_the_plan_check_reads_the_gpu_nodes_once_and_names_what_they_offer():
    shared = _kube_scale_plan()
    FakeCluster().backend().check_plan(shared)  # the L4 offered as two shares
    cluster = FakeCluster(labels={kube.GPU_REPLICAS_LABEL: "1", kube.GPU_SHARING_LABEL: "none"},
                          extra_nodes=[{"metadata": {"name": "cpu-1", "labels": {"kubernetes.io/arch": "amd64"}}}])
    with pytest.raises(plans.PlanError) as caught:
        cluster.backend().check_plan(shared)
    text = str(caught.value)
    assert ("the amd64 GPU nodes offer gpu-1.example: 1 GPU, 1 share(s) each, sharing strategy none, 2 "
            "nvidia.com/gpu allocatable") in text and "cpu-1" not in text
    assert "time-slicing (not MPS) with 2 shares (timeSlicing.enabled: true, replicas: 2, in " in text
    assert "charts/all/nvidia-gpu-config/values.yaml" in text
    # With global.gpuOperators reuse the pattern's chart does nothing: the hint names the cluster's own ClusterPolicy.
    assert ("with global.gpuOperators reuse, in the cluster's own ClusterPolicy's devicePlugin.config, with "
            "renameByDefault: false") in text
    assert cluster.calls == [("GET", "/api/v1/nodes")]
    with pytest.raises(plans.PlanError, match="sharing strategy mps, 2 nvidia.com/gpu allocatable.*not MPS"):
        FakeCluster(labels={kube.GPU_SHARING_LABEL: "mps"}).backend().check_plan(shared)
    renamed = FakeCluster(labels={"nvidia.com/gpu.product": "NVIDIA-L4"}, allocatable={"nvidia.com/gpu.shared": "2"})
    with pytest.raises(plans.PlanError) as caught:
        renamed.backend().check_plan(shared)  # the labels fit, but the node offers no nvidia.com/gpu
    text = str(caught.value)
    assert "sharing strategy time-slicing, 0 nvidia.com/gpu allocatable" in text
    assert "renames time-sliced shares to nvidia.com/gpu.shared (renameByDefault: true" in text
    assert "or is still restarting after a time-slicing change" in text
    two_gpus = {kube.GPU_COUNT_LABEL: "2", kube.GPU_REPLICAS_LABEL: "1", kube.GPU_SHARING_LABEL: "none"}
    one_offered = FakeCluster(labels=two_gpus, allocatable={"nvidia.com/gpu": "1"})
    with pytest.raises(plans.PlanError) as caught:
        one_offered.backend().check_plan(_kube_scale_plan(shared=False))  # nothing shared: no rename hint
    text = str(caught.value)
    assert "2 GPU, 1 share(s) each, sharing strategy none, 1 nvidia.com/gpu allocatable" in text
    assert "use a node that offers at least 2 nvidia.com/gpu to pods" in text and "one of its GPUs is unhealthy" in text
    assert "renameByDefault" not in text and "time-slic" not in text.split("To run it, ")[1]
    with pytest.raises(plans.PlanError, match="use a node with two GPUs or more and no GPU sharing"):
        FakeCluster().backend().check_plan(_kube_scale_plan(shared=False))
    FakeCluster(labels={kube.GPU_COUNT_LABEL: "2", kube.GPU_REPLICAS_LABEL: "1", kube.GPU_SHARING_LABEL: "none"}
                ).backend().check_plan(_kube_scale_plan(shared=False))
    with pytest.raises(plans.PlanError, match="compare.side.gpuResourceName is nvidia.com/mig-1g.10gb"):
        FakeCluster().backend(RHVP_COMPARE_GPU_RESOURCE="nvidia.com/mig-1g.10gb").check_plan(shared)


@pytest.mark.parametrize("shared", [True, False])
def test_the_plan_check_names_gpu_feature_discovery_when_no_node_has_its_labels(shared):
    # A reused ClusterPolicy may leave GPU feature discovery off: the hint then names it, not time-slicing.
    no_gfd = FakeCluster(labels={kube.GPU_COUNT_LABEL: None, kube.GPU_REPLICAS_LABEL: None,
                                 kube.GPU_SHARING_LABEL: None})
    with pytest.raises(plans.PlanError) as caught:
        no_gfd.backend().check_plan(_kube_scale_plan(shared=shared))
    text = str(caught.value)
    assert "the amd64 GPU nodes offer no node with the label nvidia.com/gpu.count" in text
    to_run = text.split("To run it, ")[1]
    assert "NVIDIA's GPU feature discovery on" in to_run and "spec.gfd.enabled" in to_run
    assert "with global.gpuOperators reuse the cluster's own ClusterPolicy decides" in to_run
    assert "time-slic" not in to_run and "sharedGpu" not in to_run


def test_the_plan_check_looks_at_nothing_in_mode_levels_or_without_a_gpu():
    cluster = FakeCluster(labels={kube.GPU_REPLICAS_LABEL: "1"})
    backend = cluster.backend()
    backend.check_plan(plans.build(_plan_config(), "openshift"))
    backend.check_plan(_kube_scale_plan(gpu=False))  # mock sides: nothing to place on a GPU
    assert cluster.calls == []


def test_a_plan_check_the_api_refuses_says_which_permission_it_needs():
    def refuses(request):
        return httpx.Response(403, json={"message": "nodes is forbidden"})
    client = kube.Kube(base_url="https://api.example:6443", token="t", verify=False,
                       transport=httpx.MockTransport(refuses))
    with pytest.raises(plans.PlanError, match="HTTP 403 nodes is forbidden .*allows list on nodes"):
        kube.KubeBackend(kube=client, env={}, sleep=lambda s: None).check_plan(_kube_scale_plan())


def test_the_second_replicas_pod_is_the_sides_on_the_first_replicas_node():
    cluster = FakeCluster()
    backend = cluster.backend()
    plan = _kube_scale_plan()
    backend.prepare(plan)
    launch, side = plan["launches"][0], plan["sides"][0]
    spec = plans.side_spec(plan, side)
    first = backend.start(plan, launch, side, spec)
    second = backend.start(plan, launch, side, spec, replica=2)
    one, two = cluster.created
    run8 = plan["run_id"][-8:]
    assert (first["name"], second["name"]) == (f"cmp-{run8}-01-a", f"cmp-{run8}-01-a-2")
    assert one["metadata"]["labels"][kube.LABEL_REPLICA] == "1" and two["metadata"]["labels"][kube.LABEL_REPLICA] == "2"
    assert two["metadata"]["labels"][kube.LABEL_RUN] == plan["run_id"]  # the cleanup and the by-hand command find it
    assert "kubernetes.io/hostname" not in one["spec"]["nodeSelector"]  # the first launch: the scheduler picks
    assert two["spec"]["nodeSelector"] == {"kubernetes.io/arch": "amd64", "kubernetes.io/hostname": "gpu-1"}
    assert one["spec"]["affinity"] == two["spec"]["affinity"] == kube.share_affinity(True)
    for part in ("requests", "limits"):  # one share each
        assert two["spec"]["containers"][0]["resources"][part]["nvidia.com/gpu"] == 1
    same = ("image", "args", "env", "volumeMounts", "securityContext")
    assert [one["spec"]["containers"][0][k] for k in same] == [two["spec"]["containers"][0][k] for k in same]
    assert one["spec"]["volumes"] == two["spec"]["volumes"]  # the model PVC too: one node, so ReadWriteOnce holds
    assert second["pending_s"] == kube.REPLICA_PENDING_S and "pending_s" not in first
    # Its time limit covers the forced step and the second replica's whole start on top of a side's.
    assert one["spec"]["activeDeadlineSeconds"] == (kube.side_deadline(_plan_config_levels(plan), launch) + 10
                                                     + plan["server"]["ready_timeout_s"])
    assert two["spec"]["activeDeadlineSeconds"] == one["spec"]["activeDeadlineSeconds"]
    assert any("2 time-sliced shares" in note for note in backend.notes)
    assert any(note.startswith("the runner: open-file limit 1048576 ") and "GuideLLM, a child process of the runner, "
               "inherits it" in note for note in backend.notes)
    assert not any(note.startswith("warning:") for note in backend.notes)


@pytest.mark.parametrize("soft, hard, refuse, calls, after, line", [
    (1024, 524288, None, [(524288, 524288)], 524288,
     "the runner: open-file limit 1024 (hard limit 524288), raised to 524288"),
    (1024, 1024, None, [], 1024, "the runner: open-file limit 1024 (hard limit 1024), already at the hard limit"),
    (1024, "unlimited", ValueError("current limit exceeds maximum limit"), [("unlimited", "unlimited")], 1024,
     "the runner: open-file limit 1024 (hard limit unlimited); could not raise it (current limit exceeds maximum "
     "limit), left as it is"),
    ("unlimited", "unlimited", None, [], None,
     "the runner: open-file limit unlimited (hard limit unlimited), already at the hard limit"),
])
def test_the_runner_raises_its_open_file_limit_for_guidellm(monkeypatch, soft, hard, refuse, calls, after, line):
    # GuideLLM is a child process of the runner and inherits its soft limit; a pod spec cannot set one.
    resource = pytest.importorskip("resource")

    def unlimited(value):
        return resource.RLIM_INFINITY if value == "unlimited" else value
    limits, made = [unlimited(soft), unlimited(hard)], []

    def getrlimit(which):
        assert which == resource.RLIMIT_NOFILE
        return tuple(limits)

    def setrlimit(which, values):
        assert which == resource.RLIMIT_NOFILE
        made.append(values)
        if refuse:
            raise refuse
        limits[:] = values
    monkeypatch.setattr(resource, "getrlimit", getrlimit)
    monkeypatch.setattr(resource, "setrlimit", setrlimit)
    assert RAISE_OPEN_FILES() == (line, after)  # never raises, never above the hard limit
    assert made == [tuple(unlimited(v) for v in c) for c in calls]


@pytest.mark.parametrize("scale_out, soft, need, warned", [
    ({"requestRate": 2.5, "surgeSeconds": 60}, 1048576, 2 * 150 + 64, False),
    ({"requestRate": 2.5, "surgeSeconds": 60}, None, 2 * 150 + 64, False),  # unlimited, or not known
    ({"requestRate": 2.5, "surgeSeconds": 60}, 364, 2 * 150 + 64, False),
    ({"requestRate": 2.5, "surgeSeconds": 60}, 363, 2 * 150 + 64, True),
    ({}, 1024, 2 * 1200 + 64, True),  # the default surge, 4 requests a second for 300 s, at docker's usual soft limit
])
def test_the_runs_notes_warn_when_the_open_file_limit_is_below_what_the_surge_could_need(monkeypatch, scale_out, soft,
                                                                                          need, warned):
    plan = _kube_scale_plan(scaleOut={"stepAfterSeconds": 10, "ttftMs": 300, "tpotMs": 25, "sharedGpu": True,
                                      **scale_out})
    assert kube.open_files_needed(plan["scale_out"]) == need and kube.ROUTER_SPARE_FILES == 64
    raised = []
    monkeypatch.setattr(kube, "raise_open_files", lambda: raised.append(1) or ("the runner: open-file limit x", soft))
    backend = FakeCluster().backend()
    backend.prepare(plan)
    warnings = [n for n in backend.notes if n.startswith("warning:")]
    assert bool(warnings) is warned
    if warned:
        assert warnings[0].startswith(f"warning: the runner's open-file limit is {soft}, below {need}: two for each "
                                      f"of the surge's {(need - 64) // 2} requests, were all of them in flight")
    levels = FakeCluster().backend()
    levels.prepare(plans.build(_plan_config(), "openshift"))  # mode levels: no raise, no note
    assert levels.notes == [] and len(raised) == 1


def _plan_config_levels(plan: dict) -> dict:
    """`plan` as mode levels would time a side: the same server and load settings, no scale round."""
    return {k: v for k, v in plan.items() if k not in ("mode", "scale_out")}


def test_mode_levels_pods_are_as_before():
    # Nothing of the scale round reaches a levels run's pods: no replica label, no affinity, the same time limit.
    backend, _ = _side_backend()
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][0]
    body = backend.pod_body(plan, plan["launches"][2], side, plans.side_spec(plan, side))
    assert kube.LABEL_REPLICA not in body["metadata"]["labels"] and "affinity" not in body["spec"]
    assert body["metadata"]["name"].endswith("-03-a") and backend.notes == []
    server, load = plan["server"], plan["load"]
    assert body["spec"]["activeDeadlineSeconds"] == int(server["ready_timeout_s"] + 2 * load["timeout_s"]
                                                        + plan["energy"]["settle_s"] + 600)


def test_a_second_replica_that_gets_no_node_fails_its_launch_in_two_minutes():
    plan = _kube_scale_plan()
    run8 = plan["run_id"][-8:]
    cluster = FakeCluster(unschedulable=(f"cmp-{run8}-01-a-2",))
    now = [0.0]
    backend = cluster.backend()
    backend.clock = lambda: now[0]
    backend.prepare(plan)
    launch, side = plan["launches"][0], plan["sides"][0]
    spec = plans.side_spec(plan, side)
    backend.start(plan, launch, side, spec)
    second = backend.start(plan, launch, side, spec, replica=2)
    now[0] = kube.REPLICA_PENDING_S - 1
    assert backend.alive(second) is None
    now[0] = kube.REPLICA_PENDING_S + 1
    assert backend.alive(second) == ("replica 2 never got a node in 120 s: 0/3 nodes are available: 1 Insufficient "
                                     "nvidia.com/gpu.")
    with pytest.raises(runner.RoundError) as caught:
        backend.endpoint(second, deadline=1e9)
    assert caught.value.code == runner.SIDE_FAILED


def test_a_second_replica_is_not_placed_on_a_node_that_cannot_share_its_gpu():
    plan = _kube_scale_plan()
    cluster = FakeCluster()
    backend = cluster.backend()
    backend.prepare(plan)
    launch, side = plan["launches"][0], plan["sides"][0]
    spec = plans.side_spec(plan, side)
    backend.start(plan, launch, side, spec)
    cluster.labels[kube.GPU_REPLICAS_LABEL] = "1"  # time-slicing turned off under the run
    with pytest.raises(runner.RoundError, match="runs on node gpu-1.example: 1 GPU, 1 share.s. each"):
        backend.start(plan, launch, side, spec, replica=2)
    assert len(cluster.created) == 1
    cluster.labels[kube.GPU_REPLICAS_LABEL] = "2"
    cluster.allocatable = {"nvidia.com/gpu.shared": "2"}  # the shares renamed under the run
    with pytest.raises(runner.RoundError, match="sharing strategy time-slicing, 0 nvidia.com/gpu allocatable"):
        backend.start(plan, launch, side, spec, replica=2)
    assert len(cluster.created) == 1


def test_the_surge_runs_through_the_router_in_the_runner_pod(tmp_path, monkeypatch):
    from rhvp_compare import router as routers
    seen = {}
    monkeypatch.setattr(routers, "spawn", lambda targets, log: seen.setdefault("router", (targets, log.name)))
    monkeypatch.setattr(guidellm, "start", lambda plan, target, out, env=None: seen.setdefault("load", (target, env)))
    backend, _ = _side_backend(secret="hf_abc")
    plan = _kube_scale_plan()
    backend.prepare(plan)
    backend.router(plan, ["http://10.0.0.1:8000"], tmp_path)
    backend.start_load(plan, plan["launches"][0], "http://127.0.0.1:8787", tmp_path)
    assert seen == {"router": (["http://10.0.0.1:8000"], "router.log"),
                    "load": ("http://127.0.0.1:8787", {"HF_TOKEN": "hf_abc"})}

    def fails(targets, log):
        raise routers.RouterError("the router exited with code 1 at its start; see router.log")
    monkeypatch.setattr(routers, "spawn", fails)
    with pytest.raises(runner.RoundError, match="the router exited with code 1"):
        backend.router(plan, ["http://10.0.0.1:8000"], tmp_path)


def test_a_stop_in_a_scale_launch_fits_the_grace_period_of_its_job():
    assert kube.stop_budget_seconds(1) == kube.stop_budget_seconds() < 300  # mode levels: unchanged
    assert kube.stop_budget_seconds(1) < kube.stop_budget_seconds(2) < 420  # the chart's scaleout floor
    assert kube.stop_budget_seconds(2) - kube.stop_budget_seconds(1) == (
        kube.ROUTER_STOP_S + 3 * kube.HURRY_TIMEOUT_S + kube.HURRY_DELETE_WAIT_S)


class FakeSurge:
    """The router and GuideLLM of a scale launch, on the runner's fake wall clock: the router sees the first request
    at once; the replica added later gets a quarter of the requests; GuideLLM's report is a constant-rate surge."""

    def __init__(self, clock: list) -> None:
        self.clock, self.routers = clock, []

    def router(self, targets, log_path):
        surge = self

        class Router:
            url = "http://127.0.0.1:8787"

            def __init__(self):
                self.targets, self.first, self.joined, self.stopped = list(targets), None, None, False
                log_path.write_text("router: open-file limit 1048576 (hard limit 1048576), already at the hard limit\n")

            def stats(self):
                self.first = self.first or surge.clock[0]
                backends = [{"url": self.targets[0], "sent": 30, "finished": 30, "first_at": self.first,
                             "last_at": self.first + 50}]
                if self.joined:
                    backends.append({"url": self.targets[1], "sent": 10, "finished": 10, "first_at": self.joined + 1,
                                     "last_at": self.first + 50})
                return {"policy": "least-busy", "first_request_at": self.first, "backends": backends}

            def add(self, target):
                self.targets.append(target)
                self.joined = surge.clock[0]
                return {"url": target, "joined_at": self.joined}

            def stop(self):
                self.stopped = True
        self.routers.append(Router())
        return self.routers[-1]

    def start(self, plan, target, out_dir, env=None):
        assert target == "http://127.0.0.1:8787"

        class Running:
            def poll(self):
                return None

            def finish(self):
                report = out_dir / "guidellm.json"
                report.write_text(json.dumps(_constant(SLO_REPORT)))
                return report

            def stop(self):
                pass
        (out_dir / "guidellm.log").write_text("surge\n")
        return Running()


def _model_server(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "m"}]})
    if request.url.path in ("/v1/completions", "/v1/chat/completions"):
        return httpx.Response(200, text='data: {"choices":[{"text":"Hi"}]}\n\ndata: [DONE]\n\n',
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(404)


def test_a_scale_out_run_on_openshift(tmp_path, monkeypatch):
    # The runner's scale round through the OpenShift backend, on a fake API: per scored launch the side's pod, the
    # router in front of its pod IP, the surge, the second replica's pod on the same node at the forced step, its
    # first answer, and both pods taken down; the live server is scaled to 0 and back. The results carry what the
    # docker path's pass rule reads: every scored launch ok, with a second replica's server record and its time to
    # first answer.
    from rhvp_compare import router as routers
    clock = [1_794_000_000.0]
    surge = FakeSurge(clock)
    monkeypatch.setattr(routers, "spawn", surge.router)
    monkeypatch.setattr(guidellm, "start", surge.start)
    cluster = FakeCluster()
    backend = cluster.backend()
    plan = _kube_scale_plan()
    backend.check_plan(plan)

    def sleep(seconds):
        clock[0] += seconds
    the_runner = runner.Runner(plan, backend, tmp_path, wall=lambda: clock[0], sleep=sleep,
                               client=lambda url: httpx.Client(base_url=url,
                                                               transport=httpx.MockTransport(_model_server)))
    results = the_runner.run()
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(results, json.loads((ROOT / "notebooks/rhvp_compare/results.schema.json").read_text()))
    assert results["run"]["status"] == "complete" and results["run"]["server_restored"] is True
    assert cluster.replicas == 1 and cluster.pods == {}  # the server back, every side pod gone
    run8 = plan["run_id"][-8:]
    names = [p["metadata"]["name"] for p in cluster.created]
    assert names == [f"cmp-{run8}-{n:02d}-{s}{r}" for n, s in ((1, "a"), (2, "b"), (3, "b"), (4, "a"))
                     for r in ("", "-2")]
    assert [p["spec"]["nodeSelector"].get("kubernetes.io/hostname") for p in cluster.created] == [None] + ["gpu-1"] * 7
    scored = results["launches"]
    assert len(scored) == 4 and all(r["status"] == "ok" and r["kind"] == "scored" for r in scored)
    for record in scored:
        assert [s["replica"] for s in record["servers"]] == [1, 2] and not any(s.get("discarded")
                                                                               for s in record["servers"])
        assert record["scale"]["new_replica"]["first_answer_s"] is not None and record["starts"] == 2
        assert record["scale"]["new_replica_share"] == 0.25 and record["scale"]["new_replica"]["joined_during_surge"]
        assert record["artifacts"]["server_log_2"].endswith("/server-2.log") and record["node"] == "gpu-1.example"
        assert 10 <= record["scale"]["step_at_s"] < 11  # the forced step
    first_router = surge.routers[0]
    assert first_router.targets == ["http://10.128.0.10:8000", "http://10.128.0.11:8000"] and first_router.stopped
    assert results["provenance"]["gpu"]["product"] == "NVIDIA-L4-SHARED"
    notes = results["run"]["notes"]
    assert any("one GPU offered as 2 time-sliced shares" in n for n in notes)
    assert any(n.startswith("the runner: open-file limit ") for n in notes)
    assert ("POST", "/api/v1/namespaces/vllm-inference/pods") in cluster.calls
