# SPDX-License-Identifier: Apache-2.0
"""The scale-out settings, the GPU operator setting and their rules, for scripts/validate-values.sh and the tests.

The same rules fail a chart render: charts/all/ray-serve-llm/templates/_settings.tpl for the server chart's values,
global.scaleOut and global.gpuOperators (templates/scaledobject.yaml for the keda trigger's own settings; the server
chart refuses global.gpuOperators because it always renders, and `check_gpu_config` holds that rule here),
charts/all/notebook/templates/_helpers.tpl (notebook.compareConfig) for global.compare.mode, and
charts/all/nvidia-gpu-config/templates/time-slicing-config.yaml for time-slicing with global.gpuOperators reuse. Keep
them in step; tests/test_scaleout_contract.py checks that a render and this module give the same message for the same
mistake.

`check(values)` takes the server chart's values as Helm sees them: charts/all/ray-serve-llm/values.yaml with
values-global.yaml, the variant file and the override files merged over it. `check_gpu_config(values)` takes the
GPU chart's values the same way (charts/all/nvidia-gpu-config/values.yaml under the same files). Each returns one
message per broken rule, or an empty list. Standard library only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from decimal import Decimal

#: global.scaleOut.mode values; none is the default.
SCALE_OUT_MODES = ("none", "keda")
#: global.compare.mode values; levels is the default.
COMPARE_MODES = ("levels", "scaleout")
#: global.gpuOperators values; install is the one in values-global.yaml. Every other value, and none, names an override
#: file that does not exist, so the install gets neither GPU operator: there is no default.
GPU_OPERATORS = ("install", "reuse")
GPU_OPERATORS_REFUSED = ("global.gpuOperators must be install or reuse; got {value} (any other value, or none, "
                         "installs neither Node Feature Discovery nor the NVIDIA GPU Operator)")
#: GPU time-slicing is a setting of the pattern's own ClusterPolicy, which only global.gpuOperators install creates.
TIME_SLICING_REUSE = ("timeSlicing.enabled needs global.gpuOperators install: with reuse the pattern creates no "
                      "ClusterPolicy and does not change the cluster's own, so set up time-slicing in that "
                      "ClusterPolicy's devicePlugin.config instead")
#: compileCache.mode values; none is the default.
COMPILE_CACHE_MODES = ("none", "node", "shared")
#: modelStorage.accessMode values; ReadWriteOnce is the default.
ACCESS_MODES = ("ReadWriteOnce", "ReadWriteMany")
#: Why maxReplicas above 1 needs a target.
NO_TARGET = ("server.autoscaling.targetOngoingRequests must be set when maxReplicas is above 1: Ray Serve LLM's "
             "default target is 10^9 ongoing requests per replica, so the replica count would never grow")
MOVED = ("server.minReplicas and server.maxReplicas moved to server.autoscaling.minReplicas and "
         "server.autoscaling.maxReplicas")
#: Replicas that share a GPU each leave this part of the GPU's memory outside their share (gpu_memory_utilization at
#: most gpuPerReplica - 0.025): vLLM refuses to start when the GPU's free memory is below gpu_memory_utilization
#: times its total memory, and each running replica also holds memory outside that fraction (its CUDA context), so
#: the room left must grow with the number of replicas on the GPU. inference-server/serve_app.py and the chart
#: (_settings.tpl) refuse the same, and the comparison's plan.py (SHARED_GPU_RESERVE) for its two replicas.
SHARED_GPU_RESERVE = 0.025
SHARED_GPU_MEMORY = ("server.engineKwargs.gpu_memory_utilization ({memory}) must be at most server.gpuPerReplica "
                     "({share}) minus 0.025 when up to {high} replicas share a GPU: each vLLM replica claims that "
                     "fraction of the whole GPU's memory when it starts and also holds memory outside it (its CUDA "
                     "context), so each leaves 0.025 of the GPU's memory for that, and a later replica still fits")
#: The server's rules for gpu_per_replica against a replica's workers (tensor_parallel_size times
#: pipeline_parallel_size); gpuPerReplica 1 is the chart's default and is not written, so the server then uses the
#: worker count.
GPU_PER_WORKERS = ("server.gpuPerReplica ({share}) must be 1 or equal engineKwargs.tensor_parallel_size times "
                   "pipeline_parallel_size ({workers}): each of a replica's workers needs a whole GPU")
ONE_WORKER_SHARE = ("server.gpuPerReplica ({share}) must be at most 1 when a replica has one worker "
                    "(engineKwargs.tensor_parallel_size and pipeline_parallel_size 1)")
#: server.streamBatchingIntervalMs is null, 0, or within these bounds (milliseconds), as in
#: inference-server/serve_app.py (STREAM_INTERVAL_MIN_MS, STREAM_INTERVAL_MAX_MS) and the chart: the chart writes
#: the value with Go's JSON encoder, which uses exponent form below 1e-6 and from 1e21 up, and the server's YAML
#: reader takes such a number as text when it has no decimal point (1e-7, 1e+21).
STREAM_INTERVAL_MIN_MS = 0.001
STREAM_INTERVAL_MAX_MS = 60000
STREAM_INTERVAL = ("server.streamBatchingIntervalMs must be null, 0, or a number of milliseconds from 0.001 to "
                   "60000; got {value} ({kind})")
#: A plain scalar in exponent form that PyYAML's YAML 1.1 rules leave as text and Helm's YAML reader takes as a number
#: when it fits a float64: one without a decimal point (5e1, 1e-3), without a sign after the e (1.0e3), or a signed one
#: that starts with a dot (-.5e+1).
HELM_EXPONENT = re.compile(r"^[-+]?([0-9][0-9_]*(\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$")
_HELM_EXPONENT_TAG = "tag:rhvp,helm-exponent"


def helm_loader(yaml_module):
    """A PyYAML loader class for values files: yaml.SafeLoader, plus a number in exponent form that matches
    HELM_EXPONENT read as a float (5e1 is 50.0), as Helm reads it. A number beyond float64's range, in that form or in
    PyYAML's own (1e999, 1.0e+400), stays text, as in Helm, and so does one that starts with a dot and has an underscore
    that is not between two digits (.5_e1). Only those forms change: PyYAML and Helm still read some other scalars
    differently (Helm reads 0o17 as 15 and 1:30 as text, PyYAML reads 0o17 as text and 1:30 as 90).
    scripts/validate-values.sh reads the values files with it. Takes the yaml module as an argument, so this file stays
    standard library only."""

    class HelmLoader(yaml_module.SafeLoader):
        pass

    def exponent(loader, node):
        text = loader.construct_scalar(node)
        if text.startswith(".") and re.search(r"(?<![0-9])_|_(?![0-9])", text):
            return text  # Helm reads a number that starts with a dot as written: "_" only between two digits
        try:
            number = float(text.replace("_", ""))
        except ValueError:  # an underscore where a digit must be (._e1): text, as in Helm
            return text
        return number if math.isfinite(number) else text

    def finite_float(loader, node):  # PyYAML's own float form: one beyond float64's range (1.0e+400) is text in Helm
        number = loader.construct_yaml_float(node)
        text = loader.construct_scalar(node)
        return number if math.isfinite(number) or text.lower().lstrip("+-") in (".inf", ".nan") else text

    HelmLoader.add_constructor("tag:yaml.org,2002:float", finite_float)
    HelmLoader.add_implicit_resolver(_HELM_EXPONENT_TAG, HELM_EXPONENT, list("-+0123456789."))
    HelmLoader.add_constructor(_HELM_EXPONENT_TAG, exponent)
    return HelmLoader


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _whole(value) -> bool:
    return _number(value) and float(value).is_integer()


def _kind(value) -> str:
    """The value's kind in Helm's words, for messages."""
    if value is None:
        return "invalid"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "float64"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "map"
    if isinstance(value, list):
        return "slice"
    return type(value).__name__


def _show(value) -> str:
    """The value the way Helm's %v prints it. Helm reads every number in a values file as a float64, and %v writes a
    finite one with the fewest digits that read back the same, in exponent form below 1e-4 and from 1e6 up (1e+06,
    1.234567e+06)."""
    if value is None:
        return "<nil>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if _number(value):
        try:
            number = float(value)
        except OverflowError:  # a whole number beyond float64
            return str(value)
        if math.isfinite(number):
            return _show_float(number)
    return str(value)


def _show_float(number: float) -> str:
    digits = Decimal(repr(number)).normalize()
    if number == 0 or -4 <= digits.adjusted() < 6:
        return format(digits, "f")
    sign, figures, _ = digits.as_tuple()
    mantissa = str(figures[0]) + ("." + "".join(map(str, figures[1:])) if len(figures) > 1 else "")
    exponent = digits.adjusted()
    return f"{'-' if sign else ''}{mantissa}e{'+' if exponent >= 0 else '-'}{abs(exponent):02d}"


def scale_out_mode(values: Mapping) -> object:
    """global.scaleOut.mode, or none when it is not set."""
    scale = (values.get("global") or {}).get("scaleOut") or {}
    return scale["mode"] if "mode" in scale else "none"


def check_global(values: Mapping) -> list[str]:
    """The two mode switches in global: global.scaleOut.mode and global.compare.mode."""
    out = []
    mode = scale_out_mode(values)
    if not (isinstance(mode, str) and mode in SCALE_OUT_MODES):
        out.append(f"global.scaleOut.mode must be none or keda; got {_show(mode)}")
    compare = (values.get("global") or {}).get("compare") or {}
    if "mode" in compare and not (isinstance(compare["mode"], str) and compare["mode"] in COMPARE_MODES):
        out.append(f"global.compare.mode must be levels or scaleout; got {_show(compare['mode'])}")
    return out


def check_server(values: Mapping) -> list[str]:
    """The server chart's own settings: server.autoscaling, server.gpuPerReplica (and, when replicas share a GPU,
    server.engineKwargs.gpu_memory_utilization), server.metricsPort, server.streamBatchingIntervalMs, compileCache,
    modelStorage.accessMode and, in keda mode, scaleOut."""
    out = []
    server = values.get("server") or {}
    if "minReplicas" in server or "maxReplicas" in server:
        out.append(MOVED)
    scaling = server.get("autoscaling") or {}
    low, high = scaling.get("minReplicas"), scaling.get("maxReplicas")
    if not (_whole(low) and low >= 1):
        out.append(f"server.autoscaling.minReplicas must be a whole number of at least 1; got {_show(low)} "
                   f"({_kind(low)})")
        return out
    if not (_whole(high) and high >= low):
        out.append(f"server.autoscaling.maxReplicas must be a whole number of at least minReplicas ({_show(low)}); "
                   f"got {_show(high)} ({_kind(high)})")
        return out
    target = scaling.get("targetOngoingRequests")
    if target is not None and not (_number(target) and target > 0):
        out.append(f"server.autoscaling.targetOngoingRequests must be a number above 0, or null; got {_show(target)} "
                   f"({_kind(target)})")
        return out
    if high > 1 and target is None:
        out.append(NO_TARGET)
        return out
    share = server.get("gpuPerReplica")
    if not (_number(share) and share > 0):
        out.append(f"server.gpuPerReplica must be a number above 0; got {_show(share)} ({_kind(share)})")
        return out
    engine = server.get("engineKwargs") or {}
    workers = 1
    for key in ("tensor_parallel_size", "pipeline_parallel_size"):
        workers *= engine.get(key) if _number(engine.get(key)) else 1
    if share != 1 and workers > 1 and share != workers:
        out.append(GPU_PER_WORKERS.format(share=_show(share), workers=_show(workers)))
        return out
    if workers <= 1 and share > 1:
        out.append(ONE_WORKER_SHARE.format(share=_show(share)))
        return out
    gpus = server.get("gpuCount")
    if _number(gpus) and high * share > gpus + 1e-6:
        out.append(f"server.autoscaling.maxReplicas ({_show(high)}) times server.gpuPerReplica ({_show(share)}) is "
                   f"more than server.gpuCount ({_show(gpus)})")
        return out
    memory = (server.get("engineKwargs") or {}).get("gpu_memory_utilization")
    if share < 1 and high > 1 and _number(memory) and memory > share - SHARED_GPU_RESERVE + 1e-9:
        out.append(SHARED_GPU_MEMORY.format(memory=_show(memory), share=_show(share), high=_show(high)))
        return out
    port = server.get("metricsPort")
    if not (_whole(port) and 1 <= port <= 65535 and port != server.get("httpPort")):
        out.append(f"server.metricsPort must be a port number (1 to 65535) other than server.httpPort; got "
                   f"{_show(port)} ({_kind(port)})")
        return out
    interval = server.get("streamBatchingIntervalMs")
    if interval is not None and not (
        _number(interval) and (interval == 0 or STREAM_INTERVAL_MIN_MS <= interval <= STREAM_INTERVAL_MAX_MS)
    ):
        out.append(STREAM_INTERVAL.format(value=_show(interval), kind=_kind(interval)))
        return out
    cache = (values.get("compileCache") or {}).get("mode")
    if not (isinstance(cache, str) and cache in COMPILE_CACHE_MODES):
        out.append(f"compileCache.mode must be none, node or shared; got {_show(cache)}")
        return out
    access = (values.get("modelStorage") or {}).get("accessMode")
    if not (isinstance(access, str) and access in ACCESS_MODES):
        out.append(f"modelStorage.accessMode must be ReadWriteOnce or ReadWriteMany; got {_show(access)}")
        return out
    if scale_out_mode(values) == "keda":
        pods = values.get("scaleOut") or {}
        low_pods, high_pods = pods.get("minPods"), pods.get("maxPods")
        if not (_whole(low_pods) and low_pods >= 1):
            out.append(f"scaleOut.minPods must be a whole number of at least 1; got {_show(low_pods)}")
        elif not (_whole(high_pods) and high_pods >= low_pods):
            out.append(f"scaleOut.maxPods must be a whole number of at least scaleOut.minPods ({_show(low_pods)}); "
                       f"got {_show(high_pods)}")
        else:
            out.extend(_check_trigger(pods)[:1])
    return out


def _check_trigger(pods: Mapping) -> list[str]:
    """The keda trigger's own settings, with the messages of charts/all/ray-serve-llm/templates/scaledobject.yaml."""
    out = []
    threshold = pods.get("waitingRequestsPerPod")
    if not (_number(threshold) and threshold > 0):
        out.append(f"scaleOut.waitingRequestsPerPod must be a number above 0; got {_show(threshold)}")
    interval = pods.get("pollingIntervalSeconds")
    if not (_whole(interval) and interval >= 1):
        out.append(f"scaleOut.pollingIntervalSeconds must be a whole number of at least 1; got {_show(interval)}")
    cooldown = pods.get("cooldownSeconds")
    if not (_whole(cooldown) and 0 <= cooldown <= 3600):
        out.append(f"scaleOut.cooldownSeconds must be a whole number from 0 to 3600; got {_show(cooldown)}")
    url = pods.get("thanosUrl")
    if not (isinstance(url, str) and re.match(r"https?://[^/]+", url)):
        out.append(f"scaleOut.thanosUrl must be an http:// or https:// address; got {_show(url)}")
    return out


def check_gpu_config(values: Mapping) -> list[str]:
    """global.gpuOperators (install or reuse, as the server chart's render refuses any other value or none), and no
    time-slicing with reuse. `values` is charts/all/nvidia-gpu-config/values.yaml with the case's values files merged
    over it. With reuse the cluster renders no app for that chart, so this check and render-charts.sh are the only
    places that refuse time-slicing then."""
    operators = (values.get("global") or {}).get("gpuOperators")
    if not (isinstance(operators, str) and operators in GPU_OPERATORS):
        return [GPU_OPERATORS_REFUSED.format(value=_show(operators))]
    if operators == "reuse" and (values.get("timeSlicing") or {}).get("enabled"):
        return [TIME_SLICING_REUSE]
    return []


def check(values: Mapping) -> list[str]:
    """Every rule. The global switches are all reported; of the server chart's own settings only the first broken
    rule is, in the order its render checks them (a render stops at its first failure)."""
    return check_global(values) + check_server(values)
