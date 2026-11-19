# Copyright (c) 2026, Eugo. Licensed under the BSD 3-Clause License (see LICENSES/BSD-3-Clause.txt).
"""OpenAI-compatible LLM server: Ray Serve LLM with the vLLM engine, in one process.

What ``main()`` does, in order:

1. Drops ``HF_TOKEN`` (and the older ``HUGGING_FACE_HUB_TOKEN``) from the
   environment unless it looks like a Hugging Face token (starts with
   ``hf_``), so a placeholder such as ``none`` is never sent to the Hub.
2. Removes a stale ready file left by an earlier run in the same pod.
3. Reads the config file (``$RHVP_CONFIG``, default ``/config/llm.yaml``),
   the ``RHVP_*`` environment overrides and then any vLLM engine flags among
   its arguments (every one other than ``--config``, ``--print-config`` and
   ``--help``; ``engine_flags``: ``--attention-backend FLASHINFER`` sets
   ``engine_kwargs.attention_backend``), checked against the engine arguments
   of the vLLM in the image.
4. Starts a single-node Ray cluster in this container, with Ray's Prometheus
   metrics on ``metrics_port``; checks that Ray sees enough GPUs for
   ``max_replicas`` replicas; starts Ray Serve on ``http_host:http_port``; and
   deploys one app built by ``ray.serve.llm.build_openai_app``.
5. ``serve.run(..., blocking=False)`` returns only when the app is RUNNING,
   which in Ray 2.60.0 means the model replica has finished starting (weights
   downloaded and loaded into vLLM). Only then is the ready file
   ``$RAY_TMPDIR/rhvp-ready`` written. The Kubernetes startup and readiness
   probes test for that file. In Ray 2.60.0's source, ``/-/healthz`` and
   ``/v1/models`` do not wait for the model; ``scripts/local-smoke.sh`` logs
   when each first answers.
6. Keeps the ready file in step with the app status (removes it if the app
   stops being RUNNING) until SIGTERM or SIGINT, then shuts Serve and Ray down.
   Ray Serve's autoscaler adding or removing a replica leaves the app RUNNING,
   so the ready file stays. On every check it also logs one ``rhvp-scale``
   line (``rhvp_scale.py``) for each replica count that changed since the last
   check. The checks run every second while ``max_replicas`` is above
   ``min_replicas`` (so a line's ``ts`` is at most about a second after the
   change), and every 10 seconds otherwise.

``load_config`` and ``build_llm_config_kwargs`` import only the standard
library and PyYAML, so they can be tested without Ray installed.

Config file keys (all but ``model_id`` and ``model_source`` are optional)::

    app_name: llm
    model_id: qwen2.5-0.5b-instruct          # name clients send as "model"
    model_source: Qwen/Qwen2.5-0.5B-Instruct  # Hugging Face repo or local path
    model_revision: "<commit sha>"            # empty or omitted: latest
    engine_kwargs: {max_model_len: 8192, gpu_memory_utilization: 0.85,
                    tensor_parallel_size: 1}  # vLLM engine arguments
    accelerator_type: null
    min_replicas: 1
    max_replicas: 1
    object_store_memory_gb: 2
    http_host: 0.0.0.0
    http_port: 8000

Keys for Ray Serve's autoscaling inside the pod, and their defaults (the chart
writes each only when it differs from the default)::

    target_ongoing_requests: null  # ongoing requests per replica Ray Serve aims
                                   # for; required when max_replicas is above 1
    gpu_per_replica: null          # GPUs one replica asks Ray for; null: one
                                   # whole GPU per tensor- and pipeline-parallel
                                   # worker. Below 1 (one worker only), replicas
                                   # share a GPU.
    metrics_port: 8080             # Ray's Prometheus metrics (ray.init)

How Ray Serve LLM sends a streamed answer, and its default (the chart writes
it only when it differs from the default)::

    stream_batching_interval_ms: 0  # 0: each token as soon as vLLM gives it.
                                    # A number of milliseconds: the tokens
                                    # of that interval in one write. null:
                                    # Ray Serve LLM's own setting (variable
                                    # RAYLLM_MODEL_RESPONSE_BATCH_TIMEOUT_MS,
                                    # 50 when unset).

Environment overrides (a variable that is unset or empty changes nothing):
``RHVP_MODEL_ID``, ``RHVP_MODEL_SOURCE``, ``RHVP_MODEL_REVISION``,
``RHVP_GPU_MEMORY_UTILIZATION``, ``RHVP_MAX_MODEL_LEN``. If
``RHVP_MODEL_SOURCE`` names a different model than the file and
``RHVP_MODEL_REVISION`` is not set, the file's revision is dropped, because a
commit of one model repository means nothing in another.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import signal
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rhvp_scale  # noqa: E402  (next to this file, in the image and in the repo)

log = logging.getLogger("rhvp.serve")

DEFAULT_CONFIG_PATH = "/config/llm.yaml"
DEFAULT_RAY_TMPDIR = "/tmp/r"
READY_FILE_NAME = "rhvp-ready"
STATUS_POLL_INTERVAL_S = 10.0
# While Ray Serve may change the replica count: the resolution of the rhvp-scale lines' ts.
SCALE_POLL_INTERVAL_S = 1.0
# Replicas that share a GPU each leave this part of the GPU's memory outside their share (gpu_memory_utilization at
# most gpu_per_replica - 0.025), for what a replica holds outside gpu_memory_utilization, such as its CUDA context:
# the room left then grows with the number of replicas. The chart and scripts/pattern_settings.py refuse the same
# before a sync, and the comparison's plan.py (SHARED_GPU_RESERVE) for its two replicas.
SHARED_GPU_RESERVE = 0.025
# Ray Serve LLM's own setting for how often it sends a streamed answer's tokens (milliseconds, read once when Ray
# Serve LLM is imported; 50 when unset). LLMConfig.experimental_configs["stream_batching_interval_ms"] overrides it
# when that key is set; stream_batching_interval_ms: null leaves it in charge.
RAY_STREAM_INTERVAL_VAR = "RAYLLM_MODEL_RESPONSE_BATCH_TIMEOUT_MS"
# Replica states that count towards the target when Ray reports no target.
LIVE_REPLICA_STATES = ("STARTING", "UPDATING", "RECOVERING", "RUNNING", "PENDING_MIGRATION")

REQUIRED_KEYS = ("model_id", "model_source")
DEFAULTS: Dict[str, Any] = {
    "app_name": "llm",
    "model_revision": None,
    "engine_kwargs": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
    },
    "accelerator_type": None,
    "min_replicas": 1,
    "max_replicas": 1,
    "target_ongoing_requests": None,
    "gpu_per_replica": None,
    "object_store_memory_gb": 2,
    "http_host": "0.0.0.0",
    "http_port": 8000,
    "metrics_port": 8080,
    "stream_batching_interval_ms": 0,
}
KNOWN_KEYS = frozenset(DEFAULTS) | frozenset(REQUIRED_KEYS)

# Environment variable -> (path in the config, parser).
ENV_OVERRIDES = {
    "RHVP_MODEL_ID": (("model_id",), str),
    "RHVP_MODEL_SOURCE": (("model_source",), str),
    "RHVP_MODEL_REVISION": (("model_revision",), str),
    "RHVP_GPU_MEMORY_UTILIZATION": (("engine_kwargs", "gpu_memory_utilization"), float),
    "RHVP_MAX_MODEL_LEN": (("engine_kwargs", "max_model_len"), int),
}

# Keys Ray Serve LLM sets itself from model_loading_config; it rejects them
# in engine_kwargs.
FORBIDDEN_ENGINE_KWARGS = ("model", "served_model_name")

HF_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class ConfigError(ValueError):
    """The config file or an override has a missing or invalid value."""


# --------------------------------------------------------------------------
# Config: pure functions, no Ray import.
# --------------------------------------------------------------------------


# A vLLM engine flag's name after "--": lower-case words joined by "-" (or "_", which vLLM also takes).
_FLAG_NAME = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")
# Values of these keys stay text even where they look like JSON: a revision 1.10 is not the number 1.1. (Used where
# vLLM cannot be imported; with vLLM, every argument declared as text keeps its value as text.)
TEXT_ENGINE_KWARGS = ("revision", "tokenizer_revision", "code_revision")
# Fields of vLLM's AsyncEngineArgs that are also OpenAI server settings (FrontendArgs) in vLLM 0.29.0 and 0.33.0.
# Ray Serve LLM 2.60.0 gives a key that FrontendArgs has to the server settings only: the engine would not see it.
RAY_FRONTEND_ONLY_KWARGS = ("tokens_only",)
ENGINE_FLAGS_HELP = (
    "Every argument other than --config, --print-config and --help is a vLLM engine argument (a field of vLLM's "
    "AsyncEngineArgs, by its vllm serve flag name) and becomes a key of engine_kwargs, after llm.yaml and the "
    "RHVP_* variables. The syntax is this server's own, not vllm serve's parser: --name value or --name=value, the "
    "name in lower-case words; a boolean as --name (true) or --no-name (false), with no value, for every boolean; "
    "a whole number or a number as JSON (no 8k); a list or set as one JSON list (--name=[1,2], where vllm serve "
    "takes --name 1 2); text as given; anything else read as JSON when it is JSON (a mapping as --name={...}) and "
    "kept as text otherwise. A value that starts with '-' and is not a number needs --name=value. Before Ray "
    "starts, each name is checked against the vLLM in this image, and so is the value of a boolean, a whole "
    "number, a number or a list. Choices, enums, mappings, config objects and the items of a list go to the engine "
    "as given: they are not checked here, and vLLM's own config checks are expected to refuse most bad ones when "
    "the engine starts. Refused (exit 2): a name that is not a vLLM engine argument (vLLM's OpenAI server "
    "settings, such as --port or --api-key, included); a flag that needs a value and has none; a value of the "
    "wrong kind for a boolean, whole number, number or list; the values None, '' and null (they read as no "
    "value); short flags (-x); dotted names (--a.b); a flag given twice; a --no-name with a value; a value with no "
    "flag before it; --hf-token (use the HF_TOKEN variable); --tokens-only (Ray Serve LLM gives it to vLLM's "
    "OpenAI server settings only); --model and --served-model-name (llm.yaml's model_source and model_id); and two "
    "settings Ray Serve LLM refuses once Ray has started: --disable-log-stats and a --distributed-executor-backend "
    "other than ray."
)


def _looks_like_flag(token: str) -> bool:
    """True for "-x" or "--x"; False for a value, also a negative number such as "-1"."""
    if not token.startswith("-"):
        return False
    try:
        float(token)
    except ValueError:
        return True
    return False


def _no_constant(name: str) -> Any:
    raise ValueError(name)  # NaN and Infinity are not JSON numbers: such a value stays text


def _flag_value(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_no_constant)
    except ValueError:
        return text


def _type_name(declared: Any) -> str:
    """A field's declared type as compact text ("bool|None"), from a type, a union or a string annotation."""
    text = declared.__name__ if isinstance(declared, type) else str(declared)
    return text.replace("typing.", "").replace(" ", "")


_KINDS = {
    "bool": ("bool", "bool|None", "None|bool", "Optional[bool]"),
    "text": ("str", "str|None", "None|str", "Optional[str]"),
    "int": ("int", "int|None", "None|int", "Optional[int]"),
    "number": ("float", "float|None", "None|float", "Optional[float]"),
}


def _split_top(text: str, sep: str) -> List[str]:
    """``text`` split at each ``sep`` outside brackets."""
    parts, depth, start = [], 0, 0
    for at, char in enumerate(text):
        depth += {"[": 1, "]": -1}.get(char, 0)
        if char == sep and depth == 0:
            parts.append(text[start:at])
            start = at + 1
    return parts + [text[start:]]


def _union_members(name: str) -> List[str]:
    """The members of a compact type name's union: "str|list[str]|None" and "Optional[Union[str,list[str]]]" are
    ["str", "list[str]", "None"]."""
    members: List[str] = []
    for member in _split_top(name, "|"):
        if member.startswith("Optional[") and member.endswith("]"):
            members += _union_members(member[9:-1]) + ["None"]
        elif member.startswith("Union[") and member.endswith("]"):
            for inner in _split_top(member[6:-1], ","):
                members += _union_members(inner)
        else:
            members.append(member)
    return members


def value_kind(declared: Any) -> str:
    """What value an engine argument declared as ``declared`` takes here: "bool" (no value), "text", "int",
    "number", "list" (a JSON list: a list or set type, or a union with one, as "str|list[str]", which vllm serve
    also gives as a list), "mixed" (a boolean or a value, as --hf-token: refused) or "other" (JSON when it is JSON,
    else text: Literal choices, enums, mappings and config objects, which are not checked here)."""
    name = _type_name(declared)
    for kind, names in _KINDS.items():
        if name in names:
            return kind
    if any(member.startswith(("list[", "set[")) for member in _union_members(name)):
        return "list"
    if re.search(r"(^|[|\[,])bool([|\],]|$)", name):
        return "mixed"
    return "other"


def vllm_argument_types() -> Optional[Dict[str, Any]]:
    """The declared type of every vLLM engine argument (the fields of vLLM's ``AsyncEngineArgs``, the class Ray Serve
    LLM builds the engine from), or None where vLLM cannot be imported, as in the unit tests. In the image
    ``ray.serve.llm`` imports vLLM anyway, before ``ray.init``; this adds the import of that one class."""
    import dataclasses

    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except Exception:  # noqa: BLE001 - no vLLM here: the flags are checked by their form only
        return None
    return {field.name: field.type for field in dataclasses.fields(AsyncEngineArgs)}


def engine_flags(argv: List[str], types: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The vLLM engine flags among this program's arguments, as ``engine_kwargs`` (see ``ENGINE_FLAGS_HELP``):
    ``--attention-backend FLASHINFER`` is ``{"attention_backend": "FLASHINFER"}``, ``--max-num-seqs=64`` is
    ``{"max_num_seqs": 64}``, ``--enforce-eager`` is ``{"enforce_eager": True}`` and ``--no-enable-prefix-caching``
    is ``{"enable_prefix_caching": False}``. So the image comparison can give a ``ray-serve-llm`` side the engine
    flags of a ``vllm-openai`` side, in the forms listed there.

    ``types`` (from ``vllm_argument_types``) is the declared type of each engine argument: with it, a name it does not
    list is refused and each value must be of its argument's kind (``value_kind``). Without it only the form is
    checked, a flag with no value is true, and a value is read as JSON when it is JSON. Anything it cannot map is a
    ``ConfigError`` that names the argument; no argument is dropped."""
    out: Dict[str, Any] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        i += 1
        if not token.startswith("--"):
            what = "short flags are not taken" if _looks_like_flag(token) else "a value with no flag before it"
            raise ConfigError(
                f"engine flag {token!r}: {what}; give vLLM engine flags as --name value or --name=value, and a list "
                f"as one JSON list, --name=[...]"
            )
        name, eq, text = token[2:].partition("=")
        if not _FLAG_NAME.fullmatch(name):
            raise ConfigError(
                f"engine flag {token.partition('=')[0]!r}: the name must be lower-case words joined by '-' "
                f"(no dotted names; give a "
                f"mapping as JSON, --name={{...}})"
            )
        value: Optional[str] = text if eq else None
        if value is None and i < len(argv) and not _looks_like_flag(argv[i]):
            value = argv[i]
            i += 1
        key = name.replace("-", "_")
        # Before any later message that would repeat the token. The name check above comes first and quotes an
        # argument up to its first "=": a flag and its token given as one argument, "--hf-token <token>", whole.
        if key in ("hf_token", "no_hf_token"):
            raise ConfigError("engine flag --hf-token: give a Hugging Face token as the HF_TOKEN variable instead")
        # Messages name the flag. Of this function's messages, only these quote what was given: a value of the wrong
        # kind, and None, '' or null; a value with no flag before it, or a short flag (the whole argument); and a name
        # the check above refuses (the argument up to its first "=", so "--api-key <key>" given as one argument is
        # quoted whole). The others never do, nor does a --no- flag with a value: that is checked before the name, so
        # --no-api-key=<key> would otherwise put a key in the log. Some of load_config's later checks quote the value
        # too: distributed_executor_backend, a revision that disagrees with model_revision, and the range checks.
        flag = f"--{name}"
        negated = key.startswith("no_")
        if negated:
            if value is not None:
                raise ConfigError(f"engine flag {flag!r} takes no value; {flag} alone sets false")
            key = key[3:]
        if key in FORBIDDEN_ENGINE_KWARGS:
            raise ConfigError(
                f"engine flag {flag!r}: engine_kwargs.{key} is not allowed; the model comes from llm.yaml's "
                f"model_source and model_id"
            )
        if key in RAY_FRONTEND_ONLY_KWARGS:
            raise ConfigError(
                f"engine flag {flag!r}: {key} is not taken here; Ray Serve LLM gives it to vLLM's OpenAI server "
                f"settings only, not to the engine"
            )
        kind = None
        if types is not None:
            if key not in types:
                hint = "vLLM's OpenAI server settings are not taken here"
                for prefix, other in (("disable_", "enable_"), ("enable_", "disable_")):
                    opposite = other + key[len(prefix):]
                    if key.startswith(prefix) and opposite in types:
                        # --disable-x is enable_x false (give --no-enable-x); --no-disable-x is enable_x true
                        # (give --enable-x).
                        spelled = opposite.replace("_", "-")
                        hint = (f"for {opposite} true, give --{spelled}" if negated else
                                f"for {opposite} false, give --no-{spelled}")
                        if negated and opposite == "disable_log_stats":  # _validate refuses it: no hint to it
                            hint = ("a true disable_log_stats is not allowed: Ray Serve LLM's engine metrics need "
                                    "vLLM's stats")
                raise ConfigError(
                    f"engine flag {flag!r}: {key} is not a vLLM engine argument of the vLLM in this image ({hint})"
                )
            kind = value_kind(types[key])
        if kind == "bool" and value is not None:
            raise ConfigError(
                f"engine flag {flag!r}: {key} is a boolean and takes no value; give {flag} for true or "
                f"--no-{name} for false"
            )
        if kind not in (None, "bool") and value is None:
            raise ConfigError(f"engine flag {flag!r} needs a value: {key} is not a boolean")
        if kind == "mixed":
            raise ConfigError(f"engine flag {flag!r}: {key} takes a boolean or a value, a form not taken here")
        if value in ("None", ""):
            raise ConfigError(
                f"engine flag {flag!r}: the value {value!r} is not taken (vllm serve reads it as no value); leave "
                f"the flag out for vLLM's default"
            )
        if value is None:
            parsed: Any = not negated
        elif kind == "text" or (kind is None and key in TEXT_ENGINE_KWARGS):
            parsed = value
        else:
            parsed = _flag_value(value)
            if parsed is None:
                raise ConfigError(
                    f"engine flag {flag!r}: the value {value!r} is not taken (it reads as no value); leave the "
                    f"flag out for vLLM's default"
                )
            wrong = {
                "int": lambda v: not _is_int(v),
                "number": lambda v: not _is_number(v),
                "list": lambda v: not isinstance(v, list),
            }.get(kind)
            if wrong is not None and wrong(parsed):
                what = {"int": "a whole number", "number": "a number",
                        "list": "a list, given as one JSON list (--name=[...])"}[kind]
                raise ConfigError(f"engine flag {flag!r}: {key} takes {what}, got {value!r}")
        if key in out:
            raise ConfigError(f"engine flag {flag!r}: {key} is given more than once")
        out[key] = parsed
    return out


def load_config(
    path: str, env: Optional[Mapping[str, str]] = None, flags: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Read the YAML config at ``path``, apply ``RHVP_*`` overrides from
    ``env`` and then the engine flags ``flags`` (from ``engine_flags``), fill
    defaults, validate, and return a normalised dict.

    Raises ``ConfigError`` with a readable message on any problem.
    """
    import yaml  # PyYAML; in the image through Ray's dependencies.

    env = {} if env is None else env
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {path} is not valid YAML: {exc}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must hold a mapping, got {type(raw).__name__}")

    cfg = copy.deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key not in KNOWN_KEYS:
            log.warning("config: ignoring unknown key %r", key)
            continue
        if key == "engine_kwargs":
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ConfigError("engine_kwargs must be a mapping")
            cfg["engine_kwargs"].update(value)
        else:
            cfg[key] = value

    file_source = cfg.get("model_source")
    overridden = _apply_env_overrides(cfg, env)
    if (
        "RHVP_MODEL_SOURCE" in overridden
        and "RHVP_MODEL_REVISION" not in overridden
        and cfg.get("model_source") != file_source
        and cfg.get("model_revision")
    ):
        log.warning(
            "config: RHVP_MODEL_SOURCE names a different model than the file; "
            "dropping the file's model_revision (set RHVP_MODEL_REVISION to pin one)"
        )
        cfg["model_revision"] = None
    if flags:
        cfg["engine_kwargs"].update(flags)

    return _validate(cfg)


def _apply_env_overrides(cfg: Dict[str, Any], env: Mapping[str, str]) -> List[str]:
    applied = []
    for name, (path, parse) in ENV_OVERRIDES.items():
        raw = env.get(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            value = parse(raw.strip())
        except ValueError:
            raise ConfigError(f"{name}={raw!r} is not a valid {parse.__name__}") from None
        target = cfg
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        applied.append(name)
    return applied


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nonempty_str(cfg: Dict[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string, got {value!r}")
    return value.strip()


def _validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["app_name"] = _nonempty_str(cfg, "app_name")
    out["model_id"] = _nonempty_str(cfg, "model_id")
    out["model_source"] = _nonempty_str(cfg, "model_source")

    revision = cfg.get("model_revision")
    if revision is not None and not isinstance(revision, str):
        # An unquoted all-digit or digits-and-one-"e" commit id is read by
        # YAML as a number and cannot be recovered exactly.
        raise ConfigError(
            f"model_revision must be a string (quote it in YAML), got {revision!r}"
        )
    revision = (revision or "").strip() or None

    engine_kwargs = dict(cfg.get("engine_kwargs") or {})
    for key in engine_kwargs:
        if not isinstance(key, str):
            raise ConfigError(f"engine_kwargs keys must be strings, got {key!r}")
    for key in FORBIDDEN_ENGINE_KWARGS:
        if key in engine_kwargs:
            raise ConfigError(
                f"engine_kwargs.{key} is not allowed; set model_id / model_source instead"
            )
    if engine_kwargs.get("disable_log_stats"):
        # Ray Serve LLM 2.60.0 keeps log_engine_metrics on by default and then refuses disable_log_stats; keda
        # mode reads those metrics (ray_vllm_num_requests_waiting).
        raise ConfigError(
            "engine_kwargs.disable_log_stats is not allowed: Ray Serve LLM's engine metrics need vLLM's stats"
        )
    # Ray Serve LLM 2.60.0 refuses the key whenever it is present and not "ray" (a null too), inside the replica.
    if "distributed_executor_backend" in engine_kwargs and engine_kwargs["distributed_executor_backend"] != "ray":
        raise ConfigError(
            f"engine_kwargs.distributed_executor_backend {engine_kwargs['distributed_executor_backend']!r} is not "
            f"allowed: Ray Serve LLM runs the engine's workers with Ray only (\"ray\"); leave it out or set \"ray\""
        )
    if "revision" in engine_kwargs:
        kw_revision = engine_kwargs.pop("revision")
        kw_revision = str(kw_revision).strip() if kw_revision is not None else None
        if revision is None:
            revision = kw_revision or None
        elif kw_revision and kw_revision != revision:
            raise ConfigError(
                f"model_revision {revision!r} and engine_kwargs.revision "
                f"{kw_revision!r} disagree; set only model_revision"
            )
    out["model_revision"] = revision

    if engine_kwargs.get("max_model_len") is None:
        engine_kwargs.pop("max_model_len", None)  # null: let vLLM use the model's limit
    else:
        if not _is_int(engine_kwargs["max_model_len"]) or engine_kwargs["max_model_len"] <= 0:
            raise ConfigError(
                f"engine_kwargs.max_model_len must be a positive integer, "
                f"got {engine_kwargs['max_model_len']!r}"
            )
    if "gpu_memory_utilization" in engine_kwargs:
        gmu = engine_kwargs["gpu_memory_utilization"]
        if not _is_number(gmu) or not 0.0 < float(gmu) <= 1.0:
            raise ConfigError(
                f"engine_kwargs.gpu_memory_utilization must be a number in (0, 1], got {gmu!r}"
            )
        engine_kwargs["gpu_memory_utilization"] = float(gmu)
    if "tensor_parallel_size" in engine_kwargs:
        tps = engine_kwargs["tensor_parallel_size"]
        if not _is_int(tps) or tps < 1:
            raise ConfigError(
                f"engine_kwargs.tensor_parallel_size must be an integer >= 1, got {tps!r}"
            )
    out["engine_kwargs"] = engine_kwargs

    accel = cfg.get("accelerator_type")
    if accel is not None and (not isinstance(accel, str) or not accel.strip()):
        raise ConfigError(f"accelerator_type must be null or a non-empty string, got {accel!r}")
    out["accelerator_type"] = accel.strip() if isinstance(accel, str) else None

    min_r, max_r = cfg.get("min_replicas"), cfg.get("max_replicas")
    # At least 1: with 0 replicas the app is RUNNING before any model is loaded, so the ready file (and the pod's
    # Ready) would come before the model; the pod holds its GPU either way.
    if not _is_int(min_r) or min_r < 1:
        raise ConfigError(f"min_replicas must be an integer >= 1, got {min_r!r}")
    if not _is_int(max_r) or max_r < 1:
        raise ConfigError(f"max_replicas must be an integer >= 1, got {max_r!r}")
    if min_r > max_r:
        raise ConfigError(f"min_replicas ({min_r}) must not exceed max_replicas ({max_r})")
    out["min_replicas"], out["max_replicas"] = min_r, max_r

    target = cfg.get("target_ongoing_requests")
    if target is not None and (not _is_number(target) or target <= 0):
        raise ConfigError(f"target_ongoing_requests must be null or a number above 0, got {target!r}")
    if max_r > 1 and target is None:
        # Ray Serve LLM sets its own default target, 10^9 ongoing requests per replica (Ray 2.60.0,
        # DEFAULT_MAX_TARGET_ONGOING_REQUESTS), so the autoscaler would never add a replica.
        raise ConfigError(
            "target_ongoing_requests must be set when max_replicas is above 1: Ray Serve LLM's default "
            "target is 10^9 ongoing requests per replica, so the replica count would never grow"
        )
    out["target_ongoing_requests"] = target

    workers = _workers_per_replica(engine_kwargs)
    share = cfg.get("gpu_per_replica")
    if share is None:
        share = workers
    elif not _is_number(share) or share <= 0:
        raise ConfigError(f"gpu_per_replica must be null or a number above 0, got {share!r}")
    elif workers > 1 and share != workers:
        # Ray Serve LLM repeats one bundle per worker; two tensor-parallel workers on one GPU cannot run.
        raise ConfigError(
            f"gpu_per_replica ({share}) must equal tensor_parallel_size times pipeline_parallel_size "
            f"({workers}) when that is above 1: each worker needs a whole GPU"
        )
    elif workers == 1 and share > 1:
        raise ConfigError(f"gpu_per_replica ({share}) must be at most 1 for a replica with one worker")
    gmu = engine_kwargs.get("gpu_memory_utilization")
    if share < 1 and max_r > 1 and _is_number(gmu) and gmu > share - SHARED_GPU_RESERVE + 1e-9:
        # vLLM 0.29 refuses to start when the GPU's free memory is below gpu_memory_utilization times its total
        # memory, and each running replica holds its CUDA context outside that fraction: with k replicas on one
        # GPU the last one needs (1 - k * gpu_memory_utilization) of the GPU free for k contexts, so a fixed
        # fraction of the share would leave too little once k grows. A fixed reserve per replica does not.
        raise ConfigError(
            f"engine_kwargs.gpu_memory_utilization ({gmu}) must be at most gpu_per_replica ({share}) minus "
            f"{SHARED_GPU_RESERVE} when up to {max_r} replicas share a GPU: each vLLM replica claims that fraction "
            f"of the whole GPU's memory when it starts and also holds memory outside it (its CUDA context), so each "
            f"leaves {SHARED_GPU_RESERVE} of the GPU's memory for that, and a later replica still fits"
        )
    out["gpu_per_replica"] = share

    osm = cfg.get("object_store_memory_gb")
    if not _is_number(osm) or osm <= 0:
        raise ConfigError(f"object_store_memory_gb must be a positive number, got {osm!r}")
    out["object_store_memory_gb"] = osm

    out["http_host"] = _nonempty_str(cfg, "http_host")
    port = cfg.get("http_port")
    if not _is_int(port) or not 1 <= port <= 65535:
        raise ConfigError(f"http_port must be an integer in 1..65535, got {port!r}")
    out["http_port"] = port
    metrics = cfg.get("metrics_port")
    if not _is_int(metrics) or not 1 <= metrics <= 65535 or metrics == port:
        raise ConfigError(f"metrics_port must be an integer in 1..65535 other than http_port, got {metrics!r}")
    out["metrics_port"] = metrics
    interval = cfg.get("stream_batching_interval_ms")
    if interval is not None and not (_is_number(interval) and math.isfinite(interval) and interval >= 0):
        raise ConfigError(
            f"stream_batching_interval_ms must be null or a number of milliseconds of at least 0, got {interval!r}"
        )
    out["stream_batching_interval_ms"] = interval
    return out


def _workers_per_replica(engine_kwargs: Mapping[str, Any]) -> int:
    """vLLM workers in one replica: tensor_parallel_size times pipeline_parallel_size (each 1 when unset)."""
    count = 1
    for key in ("tensor_parallel_size", "pipeline_parallel_size"):
        value = engine_kwargs.get(key, 1)
        if value is None:
            continue
        if not _is_int(value) or value < 1:
            raise ConfigError(f"engine_kwargs.{key} must be an integer >= 1, got {value!r}")
        count *= value
    return count


def gpus_per_worker(cfg: Mapping[str, Any]) -> float:
    """The GPU share of each vLLM worker of one replica: 1 unless replicas share a GPU."""
    return cfg["gpu_per_replica"] / _workers_per_replica(cfg["engine_kwargs"])


def gpu_shortfall(cfg: Mapping[str, Any], ray_gpus: float) -> Optional[str]:
    """A message when ``max_replicas`` replicas need more GPUs than Ray sees in this container, else None."""
    need = cfg["max_replicas"] * cfg["gpu_per_replica"]
    if need <= ray_gpus + 1e-6:
        return None
    return (
        f"max_replicas ({cfg['max_replicas']}) times the GPUs of one replica ({cfg['gpu_per_replica']}) is "
        f"{need:g}, more than the {ray_gpus:g} GPUs Ray sees in this container"
    )


def build_llm_config_kwargs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for ``ray.serve.llm.LLMConfig`` from a config made
    by ``load_config``.

    The model revision goes into ``engine_kwargs["revision"]``: Ray Serve LLM
    2.60.0 passes every key that is a vLLM engine argument to vLLM, and vLLM
    0.29.0 uses the same revision for the tokenizer unless told otherwise.
    ``model_loading_config`` has no revision field (and rejects unknown keys).
    """
    engine_kwargs = dict(cfg["engine_kwargs"])
    if cfg.get("model_revision"):
        engine_kwargs["revision"] = cfg["model_revision"]
    autoscaling: Dict[str, Any] = {
        "min_replicas": cfg["min_replicas"],
        "max_replicas": cfg["max_replicas"],
    }
    if cfg.get("target_ongoing_requests") is not None:
        autoscaling["target_ongoing_requests"] = cfg["target_ongoing_requests"]
    kwargs: Dict[str, Any] = {
        "model_loading_config": {
            "model_id": cfg["model_id"],
            "model_source": cfg["model_source"],
        },
        "engine_kwargs": engine_kwargs,
        "deployment_config": {"autoscaling_config": autoscaling},
    }
    if cfg.get("accelerator_type"):
        kwargs["accelerator_type"] = cfg["accelerator_type"]
    if cfg.get("gpu_per_replica") is not None and gpus_per_worker(cfg) < 1:
        # Ray Serve LLM's default bundle is one whole GPU per worker; this asks for the share instead.
        share = gpus_per_worker(cfg)
        kwargs["placement_group_config"] = {"bundle_per_worker": {"GPU": share}}
        # vLLM 0.29.0 asks Ray for VLLM_RAY_PER_WORKER_GPUS GPUs per worker (default 1.0), read in its engine
        # process, which the replica starts as a child process. Ray Serve LLM 2.60.0 sets the variable only for
        # the workers (VLLMEngineConfig.get_runtime_env_with_local_env_vars), so the engine asked for a whole GPU,
        # which a bundle of the share cannot hold ("Cannot schedule RayWorkerProc with the placement group").
        # LLMConfig.runtime_env goes on the replica's own runtime_env (LLMServer.get_deployment_options), and the
        # engine process inherits it.
        kwargs["runtime_env"] = {"env_vars": {"VLLM_RAY_PER_WORKER_GPUS": str(share)}}
    if cfg.get("stream_batching_interval_ms") is not None:
        # Ray Serve LLM 2.60.0 (LLMServer._get_batch_interval_ms) uses this key when it is set, and
        # RAYLLM_MODEL_RESPONSE_BATCH_TIMEOUT_MS (default 50) otherwise. At 0 its Batcher passes each token on as
        # vLLM yields it; above 0 it waits that long before each write, the first one included.
        kwargs["experimental_configs"] = {"stream_batching_interval_ms": cfg["stream_batching_interval_ms"]}
    return kwargs


def stream_interval_text(cfg: Mapping[str, Any], env: Mapping[str, str]) -> str:
    """The start-up log's line for ``stream_batching_interval_ms``."""
    interval = cfg.get("stream_batching_interval_ms")
    if interval is None:
        raw = (env.get(RAY_STREAM_INTERVAL_VAR) or "").strip()
        return (f"stream_batching_interval_ms=null: Ray Serve LLM's own setting "
                f"({RAY_STREAM_INTERVAL_VAR}={raw or 'unset, so 50'})")
    return f"stream_batching_interval_ms={interval:g}"


def stream_interval_note(cfg: Mapping[str, Any], env: Mapping[str, str]) -> Optional[str]:
    """A warning when RAYLLM_MODEL_RESPONSE_BATCH_TIMEOUT_MS is set to another value than the
    ``stream_batching_interval_ms`` that overrides it, else None."""
    raw = (env.get(RAY_STREAM_INTERVAL_VAR) or "").strip()
    interval = cfg.get("stream_batching_interval_ms")
    if not raw or interval is None:
        return None
    try:
        if float(raw) == float(interval):
            return None
    except ValueError:
        pass
    return (
        f"{RAY_STREAM_INTERVAL_VAR}={raw} is set, but stream_batching_interval_ms ({interval:g}) overrides it; "
        f"set stream_batching_interval_ms to null to use the variable"
    )


def object_store_bytes(cfg: Mapping[str, Any]) -> int:
    return int(cfg["object_store_memory_gb"] * 1024**3)


def ray_init_kwargs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Arguments for ``ray.init``. ``address="local"`` always starts a new single-node cluster in this container,
    even if RAY_ADDRESS is set; ``_metrics_export_port`` fixes the port of Ray's Prometheus metrics (vLLM's
    gauges among them, as ``ray_vllm_*``), which is otherwise chosen at random."""
    return {
        "address": "local",
        "object_store_memory": object_store_bytes(cfg),
        "_metrics_export_port": cfg["metrics_port"],
    }


def poll_interval_s(cfg: Mapping[str, Any]) -> float:
    """How often the server checks the app: every second while Ray Serve may change the replica count."""
    return SCALE_POLL_INTERVAL_S if cfg["max_replicas"] > cfg["min_replicas"] else STATUS_POLL_INTERVAL_S


# --------------------------------------------------------------------------
# Replica counts and the rhvp-scale lines: pure functions, no Ray import.
# --------------------------------------------------------------------------


def _state_name(state: Any) -> str:
    return str(getattr(state, "value", state))


def replica_counts(
    serve_status: Any, details: Optional[Mapping[str, Any]], app_name: str
) -> Dict[str, Dict[str, int]]:
    """``{deployment: {"target": n, "running": n}}`` for the app, from a ``serve.status()`` result and, when it
    could be read, the Serve controller's details (``target_num_replicas``). Empty when the app is not listed.

    ``running`` counts the replicas in the RUNNING state. ``target`` is the number Ray Serve aims for; without the
    details it is the replicas that are starting or running (``LIVE_REPLICA_STATES``)."""
    apps = getattr(serve_status, "applications", None) or {}
    app = apps.get(app_name)
    if app is None:
        return {}
    targets: Dict[str, int] = {}
    if isinstance(details, Mapping):
        app_details = (details.get("applications") or {}).get(app_name) or {}
        for name, deployment in (app_details.get("deployments") or {}).items():
            value = deployment.get("target_num_replicas") if isinstance(deployment, Mapping) else None
            if _is_int(value) and value >= 0:
                targets[name] = value
    counts: Dict[str, Dict[str, int]] = {}
    for name, deployment in (getattr(app, "deployments", None) or {}).items():
        states = {
            _state_name(state): int(number)
            for state, number in (getattr(deployment, "replica_states", None) or {}).items()
        }
        live = sum(states.get(state, 0) for state in LIVE_REPLICA_STATES)
        counts[name] = {"target": targets.get(name, live), "running": states.get("RUNNING", 0)}
    return counts


def scale_lines(
    app_name: str,
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
    ts: Optional[float] = None,
) -> List[str]:
    """One ``rhvp-scale`` line per count that differs between ``before`` and ``after``: deployments in name
    order, and for each its target before its running count. A deployment missing on one side counts 0."""
    lines = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name) or {}, after.get(name) or {}
        for reason in rhvp_scale.REASONS:
            was, now = int(old.get(reason, 0)), int(new.get(reason, 0))
            if was != now:
                lines.append(rhvp_scale.scale_event(app_name, name, was, now, reason, ts=ts))
    return lines


class ScaleWatch:
    """Keeps the app's last replica counts and logs an ``rhvp-scale`` line for each change.

    The first counts seen are the starting point and log no ``rhvp-scale`` line. ``update`` never raises: a
    failure is logged once, until a later update works again, so it cannot change the ready file."""

    def __init__(self, app_name: str, emit: Optional[Callable[[str], None]] = None) -> None:
        self.app_name = app_name
        self.counts: Optional[Dict[str, Dict[str, int]]] = None
        self.emit = emit or (lambda line: log.info("%s", line))
        self.failing = False

    def update(self, serve_status: Any, details: Optional[Mapping[str, Any]], ts: Optional[float] = None) -> List[str]:
        try:
            counts = replica_counts(serve_status, details, self.app_name)
            lines: List[str] = []
            if counts:
                if self.counts is None:
                    log.info("replica counts at start: %s", json.dumps(counts, sort_keys=True))
                else:
                    lines = scale_lines(self.app_name, self.counts, counts, ts=time.time() if ts is None else ts)
                self.counts = counts
            for line in lines:
                self.emit(line)
        except Exception as exc:  # noqa: BLE001 - the counts must never stop the server or its ready file
            if not self.failing:
                log.warning("replica count check failed (rhvp-scale lines may be missing): %s", exc)
            self.failing = True
            return []
        self.failing = False
        return lines


# --------------------------------------------------------------------------
# Process helpers: environment, ready file, status.
# --------------------------------------------------------------------------


def sanitize_env(env: MutableMapping[str, str]) -> List[str]:
    """Remove Hugging Face token variables that do not hold a real token.

    Returns the names removed. A real token starts with ``hf_``; anything
    else (the pattern's placeholder secret is ``none``) would make every Hub
    request fail with 401 instead of downloading a public model anonymously.
    """
    removed = []
    for name in HF_TOKEN_VARS:
        value = env.get(name)
        if value is None:
            continue
        stripped = value.strip()
        if stripped.startswith("hf_"):
            if stripped != value:
                env[name] = stripped
        else:
            del env[name]
            removed.append(name)
    return removed


def ready_file_path(env: Mapping[str, str]) -> str:
    return os.path.join(env.get("RAY_TMPDIR") or DEFAULT_RAY_TMPDIR, READY_FILE_NAME)


def write_ready_file(path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rhvp-ready.", dir=directory)
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{time.time():.3f}\n")
    os.replace(tmp, path)


def remove_ready_file(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def app_status_name(serve_status: Any, app_name: str) -> str:
    """One word for the app's state, from a ``serve.status()`` result.

    ``NO_CONTROLLER`` when Serve has no proxies (Serve not running at all),
    ``NOT_STARTED`` when Serve runs but the app is not deployed, otherwise
    the app's Ray Serve status (``RUNNING``, ``DEPLOYING``, ``DEPLOY_FAILED``,
    ``UNHEALTHY``, ``DELETING``).
    """
    proxies = getattr(serve_status, "proxies", None) or {}
    if len(proxies) == 0:
        return "NO_CONTROLLER"
    apps = getattr(serve_status, "applications", None) or {}
    app = apps.get(app_name)
    if app is None:
        return "NOT_STARTED"
    status = getattr(app, "status", None)
    status = getattr(status, "value", status)  # ApplicationStatus is a str Enum
    return str(status) if status else "UNKNOWN"


def watch_ready_file(
    ready_path: str,
    stop: threading.Event,
    get_status: Callable[[], str],
    interval_s: float = STATUS_POLL_INTERVAL_S,
) -> None:
    """Until ``stop`` is set, keep the ready file present exactly while
    ``get_status()`` returns ``RUNNING``. A failed status call leaves the
    file as it is (the liveness probe covers a dead Serve proxy)."""
    last = "RUNNING"
    failing = False
    while not stop.wait(interval_s):
        try:
            state = get_status()
        except Exception as exc:  # noqa: BLE001 - keep serving; just report
            if not failing:
                log.warning("status check failed (ready file unchanged): %s", exc)
            failing = True
            continue
        failing = False
        if state == "RUNNING":
            if not os.path.exists(ready_path):
                write_ready_file(ready_path)
        else:
            remove_ready_file(ready_path)
        if state != last:
            log.warning("app status changed: %s -> %s", last, state)
            last = state


def serve_details() -> Optional[Dict[str, Any]]:
    """The Serve controller's details (the source of ``serve.status()``, plus each deployment's
    ``target_num_replicas``), or None when they cannot be read. It goes through Ray Serve's private client
    (Ray 2.60.0), so any failure only drops the targets back to the starting plus running replicas."""
    try:
        from ray.serve.context import _get_global_client

        client = _get_global_client(raise_if_no_controller_running=False)
        details = client.get_serve_details() if client is not None else None
        return details if isinstance(details, dict) else None
    except Exception:  # noqa: BLE001
        return None


def serve_and_watch(
    app_name: str,
    ready_path: str,
    stop: threading.Event,
    interval_s: float,
    get_status: Optional[Callable[[], Any]] = None,
    get_details: Optional[Callable[[], Optional[Mapping[str, Any]]]] = None,
) -> ScaleWatch:
    """Once ``serve.run`` has returned with the app RUNNING: take the first replica counts, write the ready
    file, then keep the ready file in step with the app status and log the replica count changes, every
    ``interval_s``, until ``stop`` is set. Each check reads ``serve.status()`` once and the controller's details
    once. ``get_status`` and ``get_details`` replace those two reads (tests, and checks of a stub app)."""
    if get_status is None:
        from ray import serve

        get_status = serve.status
    get_details = get_details or serve_details
    watch = ScaleWatch(app_name)

    def check() -> str:
        status = get_status()
        watch.update(status, get_details())
        return app_status_name(status, app_name)

    try:
        check()
    except Exception as exc:  # noqa: BLE001 - the first counts are taken again on the next check
        log.warning("first status check failed: %s", exc)
    write_ready_file(ready_path)
    log.info("ready file written: %s", ready_path)
    watch_ready_file(ready_path, stop, check, interval_s=interval_s)
    return watch


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    """This program's own options; every other argument is a vLLM engine flag (``engine_flags``), kept in
    ``engine_flags`` of the result. ``allow_abbrev=False``: an engine flag that is the start of one of these
    options (``--print``) stays an engine flag instead of being read as that option. No ``-h``: a short flag is
    refused as an engine flag, as every other one is; ``--help`` prints the help."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], allow_abbrev=False,
                                     epilog=ENGINE_FLAGS_HELP, add_help=False)
    parser.add_argument("--help", action="help", help="show this help and exit")
    parser.add_argument(
        "--config",
        default=os.environ.get("RHVP_CONFIG") or DEFAULT_CONFIG_PATH,
        help="path to llm.yaml (default: $RHVP_CONFIG or %(default)s)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved config and LLMConfig arguments as JSON and exit (no Ray)",
    )
    args, rest = parser.parse_known_args(argv)
    args.engine_flags = rest
    return args


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s rhvp: %(message)s"
    )
    args = _parse_args(argv)

    # Before Ray starts: Ray's processes inherit this environment, and Ray
    # Serve LLM forwards HF_TOKEN to the engine from it.
    for name in sanitize_env(os.environ):
        log.info("%s does not start with 'hf_'; unset it (public models need no token)", name)
    ready_path = ready_file_path(os.environ)
    remove_ready_file(ready_path)

    try:
        types = vllm_argument_types() if args.engine_flags else None
        if args.engine_flags and types is None:
            log.warning("vLLM cannot be imported here, so the engine flags are checked by their form only")
        flags = engine_flags(args.engine_flags, types)
        if flags:
            log.info("engine flags from the command line: %s", json.dumps(flags, sort_keys=True))
        cfg = load_config(args.config, os.environ, flags)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    llm_kwargs = build_llm_config_kwargs(cfg)
    if args.print_config:
        print(json.dumps({"config": cfg, "llm_config": llm_kwargs}, indent=2, sort_keys=True))
        return 0

    log.info(
        "model_id=%s model_source=%s revision=%s engine_kwargs=%s",
        cfg["model_id"],
        cfg["model_source"],
        cfg["model_revision"] or "(latest)",
        json.dumps(cfg["engine_kwargs"], sort_keys=True),
    )
    log.info(
        "replicas %d to %d, target_ongoing_requests=%s, gpu_per_replica=%s, metrics_port=%d",
        cfg["min_replicas"],
        cfg["max_replicas"],
        cfg["target_ongoing_requests"],
        cfg["gpu_per_replica"],
        cfg["metrics_port"],
    )
    log.info("%s", stream_interval_text(cfg, os.environ))
    note = stream_interval_note(cfg, os.environ)
    if note:
        log.warning("%s", note)

    import ray
    from ray import serve
    from ray.serve.llm import LLMConfig, build_openai_app

    # ray.init installs a SIGTERM handler that raises SystemExit; the finally
    # block below cleans up.
    ray.init(**ray_init_kwargs(cfg))
    stop = threading.Event()
    try:
        shortfall = gpu_shortfall(cfg, float(ray.cluster_resources().get("GPU", 0)))
        if shortfall and cfg["max_replicas"] > 1:
            log.error("%s; lower max_replicas or gpu_per_replica", shortfall)
            return 2
        if shortfall:
            log.warning("%s; the replica waits until Ray sees a GPU", shortfall)
        serve.start(http_options={"host": cfg["http_host"], "port": cfg["http_port"]})
        app = build_openai_app({"llm_configs": [LLMConfig(**llm_kwargs)]})
        started = time.monotonic()
        # Returns once the app is RUNNING; raises if it is DEPLOY_FAILED.
        serve.run(app, name=cfg["app_name"], blocking=False)
        log.info("app %r is RUNNING after %.1f s", cfg["app_name"], time.monotonic() - started)

        def _on_signal(signum: int, _frame: Any) -> None:
            log.info("received signal %d; shutting down", signum)
            stop.set()

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)
        serve_and_watch(cfg["app_name"], ready_path, stop, poll_interval_s(cfg))
        return 0
    except Exception:
        log.exception("the server failed")
        return 1
    finally:
        remove_ready_file(ready_path)
        try:
            serve.shutdown()
        except Exception as exc:  # noqa: BLE001
            log.warning("serve.shutdown failed: %s", exc)
        try:
            ray.shutdown()
        except Exception as exc:  # noqa: BLE001
            log.warning("ray.shutdown failed: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
