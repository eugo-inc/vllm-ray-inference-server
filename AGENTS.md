# AGENTS.md

Instructions for coding agents (and people) who change or deploy this repository. `CLAUDE.md` imports this file;
keep this file the only copy of these instructions.

## Read this first: the default install costs money

- On AWS, the default install creates **one billable GPU instance: `g6.2xlarge` with one NVIDIA L4**. An
  imperative job applies a MachineSet for it, and AWS bills the instance while it runs.
- To stop that, set `global.gpuMachineSet: none` in `values-global.yaml` **in your fork**, commit and push it,
  **before** you install. Install from a fork so this value can be changed first.
- Off AWS, nothing is created; bring your own GPU node.
- Uninstalling the pattern does **not** delete that MachineSet. Argo CD did not create it, so Argo CD does not
  remove it. See [Uninstall](#uninstall).

## What this pattern is

`vllm-ray-inference-server` is a community sandbox Validated Pattern by Eugo. It is not produced, supported or
validated by Red Hat. Red Hat OpenShift GitOps (Argo CD), driven by the Validated Patterns Operator, deploys it onto
one OpenShift cluster:

- **The inference server.** One pod runs a Ray head, Ray Serve and the vLLM engine through Ray Serve LLM
  (`ray.serve.llm.build_openai_app`). It serves an OpenAI-compatible API on port 8000. The cluster builds the image
  itself from `inference-server/`, using stock packages only: PyPI wheels on NVIDIA's CUDA UBI 9 image. There is no
  KubeRay and no OpenShift AI.
- **GPU enablement.** Node Feature Discovery and the NVIDIA GPU Operator, installed by the pattern
  (`global.gpuOperators: install`, the default) or the cluster's own (`reuse`, run on one test cluster so far;
  "Installing or reusing the GPU operators" below). On AWS, an imperative job also creates one GPU node (see above).
- **Secrets.** HashiCorp Vault plus the External Secrets Operator carry an optional Hugging Face token.
- **A notebook.** A small JupyterLab behind OpenShift login. Its notebook times the server.
- **An image comparison**, off by default (`global.compare`). A Job compares two inference-server images on the
  server's GPU and takes the server down while it runs (README, "Compare two images"; here, "How the image
  comparison runs").
- **Scale out**, off by default: more vLLM replicas inside the server pod (`server.autoscaling`, Ray Serve's own
  autoscaler), more server pods (`global.scaleOut.mode: keda`, Red Hat's Custom Metrics Autoscaler), and the
  comparison's mode `scaleout` (README, "Scale out"; here, "How scale out works").

### Names that must stay in sync

The charts, `inference-server/serve_app.py`, the notebook and these docs all rely on these names. If you change
one, change it everywhere and run the tests.

| Thing | Name |
|---|---|
| Pattern (`global.pattern`) | `vllm-ray-inference-server`; also `name` in `pattern-metadata.yaml`, the `app.kubernetes.io/part-of` label in both charts, both Containerfiles' labels, and the secrets file name |
| GitHub repository | `https://github.com/eugo-inc/vllm-ray-inference-server`: `pattern-metadata.yaml`, the `global.repoURL` stubs in both charts' `values.yaml`, `scripts/render-charts.sh` and both Containerfiles' labels |
| Cluster group / variant | `prod` (`variants/prod/values-prod.yaml`) |
| Server namespace | `vllm-inference` |
| Notebook namespace | `vllm-notebook` |
| Server Deployment, Service, ImageStream, BuildConfig | `ray-serve-llm` (label `app.kubernetes.io/name=ray-serve-llm`, container `server`) |
| Server Service port | `8000`, port name `http` |
| Server config | ConfigMap `ray-serve-llm-config`, key `llm.yaml`, mounted at `/config/llm.yaml` |
| Model cache | PVC `ray-serve-llm-models`, mounted at `/models` |
| Hugging Face token | Vault `secret/data/hub/hfmodel`, property `hftoken` → ExternalSecret and Secret `huggingface-secret`, key `hftoken` |
| Server build | BuildConfig `contextDir: inference-server`, `dockerfilePath: Containerfile` |
| Notebook Deployment, Service, Route | `notebook` (label `app.kubernetes.io/name=notebook`, containers `notebook` and `oauth-proxy`); Secrets `notebook-tls` (serving certificate) and `notebook-proxy` (cookie secret) |
| Notebook build | BuildConfig and ImageStream `notebook`, `contextDir: notebooks`, `dockerfilePath: Containerfile` |
| Notebook ServiceAccount | `notebook`, with a Role and RoleBinding `notebook-read-pods` in `vllm-inference` that allow get/list/watch on pods |
| Rebuild permission | Role and RoleBinding `rebuild-on-commit` in both namespaces, for ServiceAccount `imperative/imperative-sa`; the namespaces and BuildConfig names are also listed in `ansible/rebuild-on-commit.yaml` |
| Image comparison runner | Job `compare-<runName>-<hash of the pod template>` in `vllm-notebook` (pod label `app.kubernetes.io/name=image-compare`), ServiceAccount `image-compare`, config ConfigMap `image-compare-config-<hash of the config>` (key `compare.json`) |
| Image comparison permissions | Role and RoleBinding `image-compare-sides` in `vllm-inference` (pods, pods/log, Secret `huggingface-secret`, `deployments/scale` of `ray-serve-llm`); Role and RoleBinding `image-compare-status` in `vllm-notebook` (ConfigMaps; list pods, for one run at a time); ClusterRole and ClusterRoleBinding `vllm-notebook-image-compare` (get on nodes, plus list in mode `scaleout`; get on clusterversions); ClusterRoleBinding `vllm-notebook-image-compare-monitoring` (`cluster-monitoring-view`, unless energy is `none`); in keda mode also get and patch on ScaledObject `ray-serve-llm` (`scaledobjects.keda.sh`) in `image-compare-sides` |
| Image comparison results | PVC `image-compare-results` in `vllm-notebook`, mounted at `/results` in the runner and read-only in the notebook (`RHVP_COMPARE_RESULTS`); status ConfigMap `compare-<run id>` (labels `app.kubernetes.io/component=compare-run`, `compare-run-id=<run id>`) |
| Image comparison sides | Pods `cmp-<last 8 characters of the run id>-<NN, the launch>-<a, a-same, b or b-same>` in `vllm-inference`, labels `app.kubernetes.io/name=image-compare-side`, `rhvp-compare/run=<run id>`, `rhvp-compare/side` (`A`, `A-same`, `B`, `B-same`), `rhvp-compare/launch` and `rhvp-compare/round` (a number, or `warmup`) |
| DCGM metrics | namespace `nvidia-gpu-operator` labeled `openshift.io/cluster-monitoring: "true"` (`overrides/values-gpu-operators-install.yaml`); while `global.compare.enabled` is true and energy is not `none`, the ClusterPolicy sets `DCGM_EXPORTER_INTERVAL` to 2000 ms and the ServiceMonitor interval to `2s`; with `global.gpuOperators: reuse` the pattern sets none of it, and the cluster's ClusterPolicy and namespace must |
| GPU operators (`global.gpuOperators: install`) | namespaces `openshift-nfd` and `nvidia-gpu-operator` (each with an OperatorGroup for its own namespace), Subscription keys `nfd` (package `nfd`) and `nvidia` (package `gpu-operator-certified`, `certified-operators`), apps `nfd-config` (NodeFeatureDiscovery `nfd-instance`) and `nvidia-gpu-config` (ClusterPolicy `nvidia-gpu-cluster-policy`); all in `overrides/values-gpu-operators-install.yaml`, which `sharedValueFiles` includes as `/overrides/values-gpu-operators-<global.gpuOperators>.yaml`; `reuse` has no file |
| Server metrics | Ray's Prometheus metrics on port `8080` (`server.metricsPort`, `metrics_port` in `llm.yaml`, `ray.init(_metrics_export_port=...)`) in every install; container port and Service port named `metrics` only in keda mode; vLLM's waiting-request gauge is `ray_vllm_num_requests_waiting` |
| Replica count log line | one JSON line `{"event":"rhvp-scale",...}` per change (`inference-server/rhvp_scale.py` defines it; `notebooks/rhvp_compare/logs.py` keeps its own copy of the names); on OpenShift the pod log carried it inside a Ray JSON log record, as the record's `message`, which `logs.scale_events` also reads; deployments as Ray Serve LLM names them, `LLMServer:<model id, "/" as "--" and "." as "_">` and `OpenAiIngress` |
| Compile cache volume | PVC `ray-serve-llm-compile-cache` at `/cache` while `compileCache.mode` is `node` (ReadWriteOnce) or `shared` (ReadWriteMany); `none` keeps the emptyDir |
| Autoscaler install (keda mode) | namespace `openshift-keda` (OperatorGroup for all namespaces, labeled `openshift.io/cluster-monitoring: "true"`), Subscription key `keda` for package `openshift-custom-metrics-autoscaler-operator` (channel `stable`), app `keda-config` (`charts/all/keda-config`) with KedaController `keda`; all in `overrides/values-scaleout-keda.yaml`, which `sharedValueFiles` includes as `/overrides/values-scaleout-<global.scaleOut.mode>.yaml` |
| Autoscaler objects (keda mode) | in `vllm-inference`: ScaledObject `ray-serve-llm` (on Deployment `ray-serve-llm`), TriggerAuthentication `keda-trigger-auth-prometheus`, ServiceAccount `thanos`, Roles and RoleBindings `thanos-metrics-reader` and `keda-operator-token-creator` (for ServiceAccount `openshift-keda/keda-operator`), ServiceMonitor `ray-serve-llm` (port `metrics`, bearer token from Secret `ray-serve-llm-metrics-token`, key `token`) |
| Metrics token (keda mode) | Secret `ray-serve-llm-metrics-token`, key `token` (`templates/metrics-token.yaml`): Ray's cluster token, mounted read-only in the server container at `/etc/ray-auth/token` (volume `ray-auth`, `subPath: token`, named by `RAY_AUTH_TOKEN_PATH`) and sent by ServiceMonitor `ray-serve-llm`; Argo CD ignores its `/data/token` (`overrides/values-scaleout-keda.yaml`) |
| Autoscaler pause | annotation `autoscaling.keda.sh/paused-replicas: "0"` on ScaledObject `ray-serve-llm`, set by the comparison's runner (env `RHVP_COMPARE_SERVER_SCALED_OBJECT`, from `compare.serverScaledObject` in the notebook chart) and ignored by Argo CD (`overrides/values-scaleout-keda.yaml`) |
| GPU time-slicing | ConfigMap `time-slicing-config` (key `any`) in `nvidia-gpu-operator`, named by the ClusterPolicy's `devicePlugin.config` only while `timeSlicing.enabled` is true; `install` only (refused with `reuse`) |
| Comparison on OpenShift, scale-out mode | second replica pod `<first replica's pod name>-2` in `vllm-inference` (a twin's: `cmp-<id>-<NN>-a-same-2`), both pods labeled `rhvp-compare/replica` (`1`, `2`) and pinned to one node; GPU feature discovery's node labels `nvidia.com/gpu.count` and `nvidia.com/gpu.replicas` (`kube.GPU_COUNT_LABEL`, `kube.GPU_REPLICAS_LABEL`), which `check_plan` and the pods' node affinity read; per scored launch `server-2.log` and `router.log` in the runner's results folder |
| Comparison with docker, scale-out mode | second replica container `<first replica's name>-2` (a twin's: `rhvp-cmp-<id>-<NN>-a-same-2`); warm compile-cache volumes `<CACHE_VOLUME>-<last 8 characters of the run id>-<a or b>`; per scored launch `server-2.log` (artifact `server_log_2`) and `router.log`; env `RHVP_COMPARE_REPLICA_GPUS`, `RHVP_COMPARE_CPUSET`, `RHVP_COMPARE_CACHE_VOLUME` (from `REPLICA_GPUS`, `CPUSET`, `CACHE_VOLUME` in `scripts/compare-local.sh`); helper containers `rhvp-cmp-<last 8 characters of the run id>-gpu-probe-<1 or 2>` and `...-gpu-mig-<1 or 2>` (nvidia-smi, before the first launch); server-record fields `gpus` and `gpu_uuids` (in each launch's `servers`, `$defs/server`) |

## Repo map

```
AGENTS.md  CLAUDE.md (@AGENTS.md)  README.md  LICENSE (Apache-2.0)  LICENSES/BSD-3-Clause.txt
pattern-metadata.yaml  .gitignore
values-global.yaml            pattern name, global.gpuMachineSet, global.gpuOperators, global.model.*, main.variant
values-secret.yaml.template   secrets file format: hfmodel / hftoken, value "none"
variants/prod/values-prod.yaml   the cluster group: namespaces, operator subscriptions, Argo CD apps, imperative jobs
overrides/values-gpu-machineset-create-AWS.yaml   create-gpu-machineset imperative job and its RBAC
overrides/values-gpu-operators-install.yaml   global.gpuOperators install: the NFD and GPU Operator namespaces,
                                              Subscriptions and apps (nfd-config, nvidia-gpu-config)
overrides/values-scaleout-keda.yaml   keda mode: the Custom Metrics Autoscaler's namespace, Subscription and app,
                                      and the ray-serve-llm app's ignore rules restated with the pause annotation
                                      and the metrics token's data
ansible/rebuild-on-commit.yaml   imperative job: start a build when that build's source folder changed
ansible/rebuild_on_commit.py     the rebuild decision (which BuildConfigs to start), tested in tests/test_charts.py
charts/all/nfd-config/         NodeFeatureDiscovery instance
charts/all/nvidia-gpu-config/  NVIDIA ClusterPolicy (tolerates the nvidia.com/gpu taint; a 2 s DCGM interval
                               while the comparison is on); time-slicing-config.yaml (opt-in GPU time-slicing)
charts/all/ray-serve-llm/      BuildConfig, ImageStream, Deployment, Service, Route (off by default), PVC,
                               ConfigMap, ExternalSecret; compile-cache-pvc.yaml (compileCache node or shared);
                               scaledobject.yaml, triggerauthentication.yaml, servicemonitor.yaml,
                               metrics-token.yaml (keda mode only); _settings.tpl checks the scale-out settings
                               (settings-check.yaml renders nothing, it only fails a bad render)
charts/all/keda-config/        KedaController keda (an app only in keda mode)
charts/all/notebook/           BuildConfig, ImageStream, Deployment (JupyterLab + oauth-proxy), Service, Route,
                               ServiceAccount and RBAC; compare-job.yaml, compare-config.yaml (ConfigMap, PVC)
                               and compare-rbac.yaml, rendered only while global.compare.enabled is true
inference-server/              Containerfile, requirements.in, requirements-pip.lock (pip itself),
                               requirements-x86_64.lock, requirements-aarch64.lock, serve_app.py, entrypoint.sh,
                               rhvp_status.py, rhvp_scale.py (the rhvp-scale log line's shape)
notebooks/                     Containerfile, entrypoint.sh, .dockerignore, requirements.in, requirements.lock,
                               requirements-compare.in, requirements-compare.lock (pairbench, for the comparison),
                               make_notebook.py, rhvp_bench/ (words, draw, showing, measure, replay, prompts,
                               run), prompts.jsonl, recordings/synthetic.json,
                               inference_timings.ipynb (generated)
notebooks/rhvp_compare/        the image comparison: plan, runner, probe, guidellm, logs, energy, stats, analyze,
                               report, summary, kube (OpenShift backend), dockerapi (docker backend), example,
                               words, show, cells, servers (what each server start ran with), router (the
                               scale-out mode's request router); results.schema.json.
                               compare_images.ipynb and
                               compare-recordings/synthetic.json are generated by make_notebook.py
scripts/                       render-charts.sh, validate-values.sh (with pattern_settings.py, the scale-out
                               and GPU operator rules), local-smoke.sh, compare-local.sh
tests/                         test_serve_config.py, test_secrets_template.py, test_charts.py,
                               test_server_chart.py, test_scaleout_contract.py (conftest.py and
                               chart_helpers.py share one chart render),
                               test_notebook_rules.py, test_measure.py, test_compare_stats.py,
                               test_compare_parsers.py, test_compare_runner.py, test_compare_notebook.py,
                               test_compare_router.py, fixtures/ (trimmed GuideLLM reports, one with latency
                               goals; the mock configs for compare-local.sh, levels and scaleout; Ray Serve LLM
                               server logs rebuilt from real line shapes, ray-serve-start-*.log, for the
                               engine restart count)
.github/workflows/             lint.yml, jsonschema.yaml, notebook-check.yml
pattern.sh  Makefile  Makefile-common  ansible.cfg  .claude/skills/pattern-author/  .cursor/skills/pattern-author/
```

Patternizer manages `pattern.sh`, `Makefile-common`, `ansible.cfg` and both `pattern-author` skill folders. Do not
edit them by hand; `patternizer upgrade` rewrites them. You may add targets to `Makefile`.

### How the server starts

1. `entrypoint.sh` replaces the base image's NVIDIA entrypoint. It unsets `HF_TOKEN` unless the value starts
   with `hf_`, then starts `serve_app.py`.
2. `serve_app.py` reads `/config/llm.yaml` (the path in `RHVP_CONFIG`), then the environment overrides
   `RHVP_MODEL_ID`, `RHVP_MODEL_SOURCE`, `RHVP_MODEL_REVISION`, `RHVP_GPU_MEMORY_UTILIZATION`,
   `RHVP_MAX_MODEL_LEN` and `RHVP_STREAM_BATCHING_INTERVAL_MS` (a number of milliseconds, or `null`), and then any
   vLLM engine flags on its command line: every argument other than its own
   `--config`, `--print-config` and `--help` (`serve_app.engine_flags`; the chart passes none, the comparison
   passes a `ray-serve-llm` side's `args`). Each names a field of vLLM's `AsyncEngineArgs`, the class Ray Serve LLM
   builds the engine from, and becomes a key of `engine_kwargs`: `--attention-backend FLASHINFER` is
   `attention_backend: FLASHINFER`, `--max-num-seqs=64` is `max_num_seqs: 64`, `--enforce-eager` is `true` and
   `--no-enable-prefix-caching` is `enable_prefix_caching: false`. The syntax is its own, not `vllm serve`'s parser
   (`serve_app.ENGINE_FLAGS_HELP`): a list is one JSON list, `--no-` works for every boolean, and a value that is
   not text is read as JSON when it is JSON. When flags are given it imports that class and checks each name
   against its fields and each value against the field's declared type (`serve_app.value_kind`): a boolean takes no
   value; a whole number, a number and a list must be one; text stays text. Choices, enums, mappings, config
   objects and the items of a list go to the engine as given; they are not checked here, and vLLM's own config
   checks are expected to refuse most bad ones when the engine starts. Refused, each with exit 2 and a message that
   names the flag: a name that is not a field (vLLM's OpenAI server settings included) or not lower-case words, a
   flag that needs a value and has none, a value of the wrong kind, `None`, an empty value or `null` (all read as no
   value), a field that takes a boolean or a value, `--hf-token` (the token comes from `HF_TOKEN`), `--tokens-only`
   (`serve_app.RAY_FRONTEND_ONLY_KWARGS`: Ray Serve LLM gives that field to vLLM's OpenAI server settings only),
   `--model` and `--served-model-name`, a short flag, a dotted name, a flag given twice, a `--no-` flag with a value
   and a value with no flag before it. Of these, a value of the wrong kind and `None`, an empty value or `null` are
   quoted back; a short flag and a value with no flag before it are quoted whole; and a name that is not lower-case
   words or is dotted is quoted up to its first `=`, so a flag and its value given as one argument with a space
   between (`--api-key <key>`, `--hf-token <token>`) are quoted with the value. The other refusals never repeat the
   value, a `--no-` flag with a value included (so `--no-api-key=<key>` does not reach the log). `load_config` then
   refuses, from flags and `llm.yaml` alike, `model` and `served_model_name`, a true `disable_log_stats` and a
   `distributed_executor_backend` other than `ray` (`null` in `llm.yaml` included: Ray Serve LLM refuses the key
   whenever it is there and not `ray`), which Ray Serve LLM 2.60.0 would refuse only once Ray has started. The
   backend's message quotes the value given, as do the refusal of a `revision` that disagrees with `model_revision`
   and `load_config`'s range checks, such as a `max_model_len` or a `tensor_parallel_size` below 1, whichever of
   flags and `llm.yaml` gave it. No flag is dropped. Where vLLM cannot be imported (the unit tests) only the form is
   checked, with a warning. The rules were run side by side against `vllm serve`'s own parser for vLLM 0.33.0; for
   vLLM 0.29.0, which this image pins, only the fields' declared types at its tag were read.
3. It starts Ray with a 2 GiB object store and its Prometheus metrics on port 8080 (`metrics_port`), checks that
   Ray sees enough GPUs for `max_replicas` replicas (exit 2 when it does not and `max_replicas` is above 1), starts
   Serve on `0.0.0.0:8000`, and runs the Ray Serve LLM app `llm`. `serve.run` returns once the app is RUNNING,
   which means the model is loaded. `stream_batching_interval_ms` goes to `LLMConfig.experimental_configs`, from
   which Ray Serve LLM reads how long it collects a streamed answer's tokens before each write (0: every token as
   vLLM yields it); with `null` the key is left out, and Ray Serve LLM's `RAYLLM_MODEL_RESPONSE_BATCH_TIMEOUT_MS`
   (read when Ray Serve LLM is imported; 50 when unset) decides. The server logs the interval it passes, and a
   warning when that variable is set to a value other than the key's, which overrides it. It refuses to start when
   that variable is set but Python's `float()` does not read it as a number (empty, or text such as `50ms`), since
   Ray Serve LLM then fails to import. Checked on CPU only, in
   images with Ray 2.60.0 and a Ray 2.61 development build: the rendered `llm.yaml` gave Ray Serve LLM's own
   `LLMServer` an interval of 0 by default (also with the variable at 50), 12.5 when set to 12.5, and with `null`
   the variable's value (50 when unset); not with a running engine, and not with many streams at once.
4. It then writes the ready file `/tmp/r/rhvp-ready` and waits for SIGTERM. Until then it checks the app every 10
   seconds (every second while `max_replicas` is above `min_replicas`): it removes the ready file when the app stops
   being RUNNING, and logs one `rhvp-scale` line for each replica count that changed. Adding or removing a replica
   leaves the app RUNNING, so the ready file stays.
5. The startup and readiness probes check that ready file every 2 seconds. The kubelet runs the readiness probe
   as soon as the startup probe passes, so the pod turns Ready up to about 2 seconds after the file appears. That
   is the resolution of the notebook's start-to-Ready figure, which Kubernetes also records in whole seconds. The
   liveness probe calls `/-/healthz` every 30 seconds.

`llm.yaml` (the chart renders it from `values-global.yaml`):

```yaml
app_name: llm
model_id: qwen2.5-0.5b-instruct          # global.model.id; the name clients send as "model"
model_source: Qwen/Qwen2.5-0.5B-Instruct # global.model.source
model_revision: <global.model.revision, or empty>
engine_kwargs: {max_model_len: 8192, gpu_memory_utilization: 0.85, tensor_parallel_size: 1}
accelerator_type: null
min_replicas: 1
max_replicas: 1
object_store_memory_gb: 2
http_host: 0.0.0.0
http_port: 8000
# Written only when they differ from these defaults (so a default install renders the llm.yaml above):
# target_ongoing_requests: null  # server.autoscaling.targetOngoingRequests; required when max_replicas > 1
# gpu_per_replica: null          # server.gpuPerReplica; null: one GPU per tensor- and pipeline-parallel worker
# metrics_port: 8080             # server.metricsPort
# stream_batching_interval_ms: 0 # server.streamBatchingIntervalMs; 0: each streamed token as vLLM generates it,
#                                # 0.001 to 60000: milliseconds per write, null: Ray Serve LLM's own setting
```

`serve_app.py` refuses a config it cannot run, with a message that names the key: among others a `max_replicas`
above 1 without `target_ongoing_requests` (Ray Serve LLM's own target is 10^9), a `gpu_per_replica` that is not the
worker count of a tensor-parallel replica (or above 1 for a replica with one worker), and, for replicas that share a
GPU, a `gpu_memory_utilization` above `gpu_per_replica` minus 0.025 (`SHARED_GPU_RESERVE`, the same 0.025 in the
chart, `scripts/pattern_settings.py` and the comparison's `plan.SHARED_GPU_RESERVE`; tested together in
`tests/test_scaleout_contract.py`). The reserve is per replica, so the memory left free grows with the replicas on a
GPU, for each one's CUDA context; 0.025 of the GPU is an estimate of that context, not a measurement. It also refuses a
`stream_batching_interval_ms` that is not `null`, 0, or a number of milliseconds from 0.001 to 60000 (text such as
`"50"`, a boolean, a negative number and `1e-7` among them). The bounds keep the chart's value a plain number: Helm
writes a number below 0.000001 or from 10^21 up in exponent form, and PyYAML reads such a number as text when it has
no decimal point (`1e-7`, `1e+21`). The chart and `scripts/pattern_settings.py` refuse the same settings before a
sync; `scripts/validate-values.sh` reads a number in exponent form such as `5e1` in the values files as a number, as
Helm does (`pattern_settings.helm_loader`).

Server container environment:

- `RHVP_CONFIG=/config/llm.yaml`
- `HF_HOME=/models/hf`: model weights on the PVC, kept across restarts.
- Compile caches under `/cache`: `TRITON_CACHE_DIR=/cache/triton`, `VLLM_CACHE_ROOT=/cache/vllm`,
  `FLASHINFER_WORKSPACE_BASE=/cache/flashinfer`, `XDG_CACHE_HOME=/cache/xdg`. With `compileCache.mode: none` (the
  default) `/cache` is an emptyDir, emptied on every new pod. With `node` or `shared` it is PVC
  `ray-serve-llm-compile-cache`, kept for later pods; the pod then also gets `CUDA_CACHE_PATH=/cache/cuda`,
  `CUDA_CACHE_MAXSIZE=4294967296` (4 GiB) and `fsGroupChangePolicy: OnRootMismatch`. The comparison's docker backend
  gives a `compileCache: warm` side the same two CUDA variables (`dockerapi.WARM_CACHE_ENV`; a test keeps them equal
  to the chart's). There is no hostPath option: the restricted-v2 SCC the server runs under does not allow hostPath
  volumes.
- `RAY_TMPDIR=/tmp/r` on its own emptyDir. Keep it short: Ray puts Unix sockets under it, and socket paths are
  limited to 108 bytes.
- `/dev/shm`: an emptyDir with `medium: Memory`, 4Gi.
- `HF_TOKEN`: from `secretKeyRef` `huggingface-secret` / `hftoken`, `optional: true`.
- `RAY_AUTH_TOKEN_PATH=/etc/ray-auth/token`, in keda mode only: Ray's cluster token, from Secret
  `ray-serve-llm-metrics-token` key `token`, mounted read-only at that path with `subPath: token` (volume `ray-auth`,
  not optional), the token the ServiceMonitor scrapes the metrics port with. Every Ray process reads the file when that
  process starts: the head, each worker process added later, and a process that attaches (`rhvp_status.py`). Keep
  the `subPath`: Kubernetes rewrites a Secret volume's files in a running container when the Secret changes, but not a
  `subPath` file, and a Ray process that starts with a token other than the running cluster's is refused. In every
  other mode the chart leaves the image's own Ray token setup alone. The chart sets neither `RAY_AUTH_TOKEN` (Ray
  reads it before the path) nor `RAY_AUTH_MODE`.

Notebook environment (set by the chart, read by `rhvp_bench`):

- `RHVP_ENDPOINT=http://ray-serve-llm.vllm-inference.svc:8000`
- `RHVP_MODEL_ID=<global.model.id>`
- `RHVP_MODEL_REVISION=<global.model.revision>`, set only when the revision is not empty.
- `RHVP_SERVER_NAMESPACE=vllm-inference`
- `RHVP_SERVER_LABEL_SELECTOR=app.kubernetes.io/name=ray-serve-llm`
- Optional `RHVP_INTERNAL=1` adds an amber INTERNAL strip. It is off by default.
- Optional `RHVP_REPLAY=<path to a recorded run>` turns on REPLAY mode.

The notebook image bakes in the `notebooks/` folder at `/opt/app-root/notebooks`. Nothing is cloned at runtime.
When the pod starts, it copies that folder into a writable emptyDir at `/work`. It then runs JupyterLab on
`127.0.0.1:8888` with `root_dir` `/work`, behind an OpenShift oauth-proxy sidecar. The sidecar listens on port 8443
and uses a serving certificate; the Route uses `reencrypt` termination.

### How the image comparison runs

- Off by default. `global.compare` in `values-global.yaml` is the config; the notebook chart renders it, plus
  `global.model`, into ConfigMap `image-compare-config-<hash>`, and renders the Job and its RBAC.
- The Job's pod runs on the pinned GuideLLM image (`ghcr.io/vllm-project/guidellm:v0.8.1@sha256:015b...`); an init
  container copies `notebooks/rhvp_compare` into it from the notebook image, with pairbench as that image installed it
  (`/opt/app-root/lib/python3.12/site-packages/pairbench` and its `pairbench-<version>.dist-info`; pure Python, run
  by the GuideLLM image's own Python 3.13). A notebook image built before pairbench was added (or an `image.override`
  without it) fails that init container, and the Job with it. It runs on the notebook pod's node, because the results
  PVC is ReadWriteOnce and the notebook mounts it too.
- The runner (`python -m rhvp_compare run --backend kube`) first lists the runner pods in `vllm-notebook`
  (`app.kubernetes.io/name=image-compare`) and refuses to start (exit 5, `RUN_ACTIVE`, nothing changed) while an
  older one is Pending or Running. It then scales Deployment `ray-serve-llm` to 0, and for each launch creates a
  side pod in `vllm-inference` (every one after the first pinned to the first one's node; in mode `scaleout` a scored
  launch's second replica is a second pod on the same node), times its start, runs GuideLLM as its own child process
  (in mode `scaleout` with the router, also a child process of the runner), reads GPU power from the Thanos Querier,
  deletes the pod and saves `results.json`.
  A side's `args` follow its preset's own: `vllm serve` flags for `vllm-openai`, and vLLM engine
  arguments in the server's own syntax for `ray-serve-llm` (`serve_app.engine_flags`, "How the server starts").
  The `ray-serve-llm` preset also starts a side with `RAYLLM_MODEL_RESPONSE_BATCH_TIMEOUT_MS=0` and
  `RHVP_STREAM_BATCHING_INTERVAL_MS=null` (`plan.side_spec`), so the side sends each streamed token as it is
  generated, as the chart's server does by default. A side mounts the live server's ConfigMap `ray-serve-llm-config`,
  not `plan.llm_yaml` (the docker backend's file, with the key `null`), so without the second variable it would take
  the live server's interval; the variable makes it `null` and leaves Ray Serve LLM's variable to decide. A side on
  an older image, whose `serve_app.py` knows neither the key nor the second variable, ignores the key with a warning,
  with the same result. A side's own `env` replaces either variable's value.
  GuideLLM starts only after the side has listed the model on `GET /v1/models` and answered a streamed request
  (`probe`), and its backend carries `validate_backend=false` (`guidellm._backend`, both modes): GuideLLM v0.8.1's
  own check is `GET <target>/health` by default, and Ray Serve LLM's OpenAI app has no `/health` route, so with the
  check every load of the pattern's own server fails. In mode `scaleout` GuideLLM goes through the router, whose
  `/health` only says it has a replica. A round launches A, A~same, B and B~same once each (`plan.launch_order`,
  rotate-reverse); a twin (`A~same`) launches its side's image and spec under another lineup key, so a backend only
  names it differently (`plan.token`: `A-same` in pod names and labels). A side whose server exits at vLLM's
  start-up memory check (`Error in memory profiling` in its log) is started once more in the same launch
  (`runner.START_RETRIES`, the launch's `start_retries`, and the server record's own `start_retries`); in mode
  `scaleout` the second replica gets the same one retry, timed from its first ask. The pattern's own Ray Serve LLM
  server does not exit when its engine stops at that check: Ray Serve stops that replica and starts a new one inside
  the same start of the server, so the runner starts nothing again and `start_retries` stays 0. The runner counts
  these from the server's log once it is done with that server (`logs.memory_check_restarts`), in the launch's
  `engine_restarts` and the server record's own: a replica (one `(ServeReplica:<app>:<deployment> pid=N)` prefix)
  whose lines show `Error in memory profiling` counts once, however often Ray relays the error, when a new replica of
  the same deployment prints its first line after the first of those lines (Ray's late relayed copy of the error
  often comes after that new first line; the fixtures have both orders). Not counted: a replica stopped for any other
  reason, a last replica with none after it, and any `vllm serve` log (no such prefix). The server exits once Ray
  Serve gives up; in its first start, `runner._start_again` then starts it once more whenever
  `Error in memory profiling` is anywhere in its log, whatever its last replica stopped for (a KV cache too small,
  say), and `start_retries` and the launch's note count that exit as one at the memory check; the replicas counted
  before it carry over to the start again's record. A start-up figure includes the engine starts that stopped inside
  the start it is timed from (the side's last start; each start of the second replica), and `logs.breakdown` adds up
  every engine start's timed phases in a log, the stopped ones' included, while a moment (`engine_start` and the
  others in `logs.MOMENTS`) is its first line in the log, the stopped engine's. A GuideLLM time-out runs the launch
  once more with the same requests (`runner.LOAD_RETRIES`): in mode `levels` from a new start of the side, timed from
  that start; in mode `scaleout` from a new first replica, after GuideLLM, the router and the replicas that had
  started are taken down. The launch's `load_timeouts` is counted when the time-out happens and `load_retries` only
  once the second load begins; the first try's files are kept as `*.try-1.*`, its notes are marked `try 1:`, its
  server records are marked `discarded`, and its figures are cleared before the teardown. Time-outs count against
  their side in `analysis.load_timeouts` (totals) and per comparison. When the run ends it scales the server back to
  `compare.serverReplicas` (1), tries again for about a minute, reads the scale back, and records
  `run.server_restored`; a server that did not come back is exit 4.
- The ray-serve-llm app ignores the Deployment's `/spec/replicas` on every install, also with the comparison off: a
  sync never changes the server's replica count, so a server at 0 stays at 0 until someone scales it.
- SIGTERM (a deleted Job or `activeDeadlineSeconds`) cuts the current launch short and asks the server back at once,
  before the side is taken down (its pod waits for the GPU); every API call then gets a 10 s timeout. The pod's
  `terminationGracePeriodSeconds` (300) is above the worst case of that path (`kube.stop_budget_seconds()`, checked
  by `tests/test_charts.py`). In mode `scaleout` a stop also takes down the router and the second side pod
  (`kube.stop_budget_seconds(2)`), so the chart makes it at least 420 there (`notebook.compareGraceSeconds`).
- A Job's pod template cannot change, so the Job's name carries a hash of the whole rendered pod template
  (`notebook.comparePodTemplate`, hashed by `notebook.compareJobHash`). The run id carries a hash of the config
  only, so a new Job with the same config finds its run complete and exits at once, or refuses while the first
  still runs.
- The verdict rule (`stats.grade`, applied by `analyze._decide`): per case, the median over the rounds of B/A. Each
  side's noise floor is the largest `abs(median(side / twin) - 1)` over the cases of one metric and role
  (`case.group`); the bar is B's floor plus A's. The test is the exact two-sided sign test over the untied rounds,
  with Holm's correction over the primary cases only ("corrected for testing N cases at once"); a secondary case is
  graded on its uncorrected p-value. Outside the bar and p below alpha: `better` / `worse` (FASTER or SLOWER for
  times, BETTER or WORSE for rates and tokens per joule); inside and not significant: `tie`; otherwise `unresolved`
  (`inside_noise`, `not_significant`, or `order_unbalanced` when one side ran first more than 3 times as often).
  `no_verdict` comes first when a case cannot be graded, checked in this order: `more_than_one_node`, `side_failed`
  (a launch of A or B failed on its own account where the other did not; pattern-only, like the node check),
  `nonpositive_value` (a 0 on A or B in any round, paired or not, or in a side's pair with its twin, which takes the
  side's floor on that whole floor group), `single_measurement`, `too_few_rounds` (`stats.min_rounds` of the primary
  family, or of one case for a secondary), `no_floor`, `one_sided_floor`, `zero_floor`. There is no fixed tie band.
- What comes from pairbench (https://github.com/eugo-inc/pairbench): the exact sign test, Holm's correction and the
  fewest rounds (`stats.sign_test`, `stats.holm` and `stats.min_rounds` call `pairbench.stats.sign_test`, `holm` and
  `min_rounds`), pinned in `notebooks/requirements-compare.lock`. The adapters keep the pattern's arguments and
  results: `holm` hands pairbench the p-values sorted by case id, so its ties fall in the pattern's order, and
  `min_rounds` still counts fewer than 1 case as 1, which pairbench refuses. Everything else is the pattern's own: the
  per-round ratios and their median, the signs (a ratio of exactly 1 is a tie), the floors and the bar, `grade`, the
  order check, the reasons and their order, `max_cases` and `must_agree` (on top of the sign test). pairbench's own
  pairing, floor, rule and `analyse` are not used. There is no fallback: without pairbench, `rhvp_compare.stats` does
  not import. `tests/test_compare_stats.py` holds the three adapters to the pattern's earlier formulas on a grid of
  inputs and checks that they reach pairbench. Re-analyzing the repository's synthetic results and the results of 33
  earlier runs of the runner with real servers (39 attempts, 6 of them a run started again after it ended partial;
  kept outside the repository) with the earlier and the new code gave byte-identical `results.json` and `results.md`,
  on Python 3.12 and 3.13, and so did every analysis the unit tests make. The one difference: `analyze` takes alpha
  from the results file as it is, and a file edited by hand to an alpha of 1 or more, 0 or less, or NaN (the plan
  takes 0.0001 to 0.5; the schema allows any alpha above 0) now stops `python -m rhvp_compare analyze` with a
  ValueError from pairbench's `min_rounds`, where the earlier code gave an analysis for 1 or more and for NaN, and never
  finished for 0 or less.
- `notebooks/rhvp_compare/analyze.py` is the boundary to the statistics: `inputs(results)` turns `plan` and
  `launches` into plain rows (round, position, side, case, value), the twin map, the cases and the failed launches;
  `analyse(...)` turns those into `analysis` (`rule`, `comparisons`, `same_vs_same`, `nonpositive_twins`,
  `refusals`, `notes`, `outcomes`). Another tool that takes the same rows and writes the same `analysis` shape can
  replace `analyze.py` and `stats.py`. `results.schema.json` is version `rhvp-image-compare/2`; the notebook skips a
  file of another version with a notice.
- Run length with the defaults: 8 rounds of 4 launches plus 2 warm-ups, 34 server starts, an estimate of about 4 to
  6 hours (16 rounds: 66 starts, about 8 to 12 hours). Unmeasured on any cluster. The Job's
  `activeDeadlineSeconds` (48 h) and the per-launch side deadline (`kube.side_deadline`) and stop budget do not
  depend on the number of launches.
- Every sentence the comparison notebook shows is in `rhvp_compare/words.py`.

Checked on a plain Kubernetes cluster (kind) with GuideLLM mock servers: the Job and its RBAC, the server's scale to 0
and back, the status ConfigMap, the termination message, and the stop path (Job deleted mid-launch). That run came
before the refusal of a second run, the scale read-back, the early restore on a stop and the twins. On Kubernetes the
refusal of a second run, the early restore on a stop and the twins are covered by unit tests only (under docker, the
refusal and the stop path have run against the mock server); the scale read-back has since run on OpenShift (below).
Checked with real servers, under docker on one GPU (`scripts/compare-local.sh`), under the earlier rule (A and B only,
no twins): two 2-pair smoke runs of the example's two vLLM images with smaller settings. The second finished all 6
launches, measured by GuideLLM, with tokens per joule from nvidia-smi, one of them after a second start past vLLM's
start-up memory check (README, "On one machine"); the first, from before that second start, lost a launch to that check.
Checked with real servers under the twin rule, under docker on one machine with one GPU whose memory is the machine's
own: three comparisons whose sides were both `ray-serve-llm` (this pattern's own server), on two images with vLLM 0.33
(0.33.0 and a development build of it; not the vLLM 0.29.0 this image pins) built for the test with Ray Serve LLM and
this pattern's server files, each planned as 8 rounds with twins on both sides after one warm-up launch per side. Two
finished all 8 rounds (34 launches each). A memory guard around the runner stopped the third (SIGTERM) in round 7: that
launch is `STOPPED`, and the run was saved partial and analyzed on its 6 finished rounds. No other launch failed, and
`start_retries` was 0 in every launch (an engine that Ray Serve replaced inside a start does not count there; see the
restart count below). GuideLLM timed out in six launches across the three; each ran again from a new start
(`load_retries` 1) and finished. Their settings: the example's model, already on the model volume; one load level, 5 s
at 1 stream; `gpuMemoryUtilization` 0.12; `startup/first_answer_s` the only primary case; energy `none`; no
`SERVER_CONTAINER`; cold compile caches. Their code: `notebooks/rhvp_compare` and `scripts/compare-local.sh` of commit
`7ec0b75` (comments aside), and server files with an earlier, unmerged form of `serve_app.engine_flags` (each value read
as YAML, nothing checked against `AsyncEngineArgs`); one run's sides started through a wrapper `command` that then runs
the entrypoint, on the same images plus a test layer, which on one side also changed how the server's processes read the
model's weights (test code in them read every page of the larger weight tensors before vLLM copied them); that run's
wrapper also recorded the model files' page-cache state before each start, and a process outside the containers mapped
the model's file and touched its cached pages as each side started. The engine restart count
(`logs.memory_check_restarts`) was run afterward over the 66 server logs of two of those comparisons, the stopped one
and one that finished (`*.try-1.*` logs included): it gave 1 for each of the 7 logs in which one replica stopped at the
check and a new one followed (in one of them a replica before that one had stopped for another reason, a KV cache too
small for the model's length, and was not counted), and 0 for the other 59; over the 35 server logs of the other
finished comparison it gave 0 for each. Not among the three comparisons: an earlier try of one of the two that finished,
with the same settings and comparison code, finished 2 warm-up and 15 scored launches before a memory guard stopped it
in round 4 (saved partial), and other tries were stopped in their first warm-up launch. The `ray-serve-start-*.log`
fixtures are rebuilt from the line shapes of the 0.33.0 image's logs, with made-up process ids, replica ids, times,
durations, memory figures and memory setting, apart from the model's weight size (0.93 GiB, the same in every one of
those logs), the pattern's default model length (8192) and vLLM's own source line numbers; their only path and address
are the pattern's ready file (`/tmp/r/rhvp-ready`) and listen address (`http://0.0.0.0:8000/`), and no line is a copy of
a line of those logs.
Later runs with real servers under the twin rule, one of them with a warm compile cache, are not among the three.
Checked on OpenShift in the three runs described here (4.22.18, two throwaway clusters on AWS, each with the pattern's
g6.2xlarge node with one NVIDIA L4), without twins, each complete with every case NO VERDICT: mode `levels` with the
comparison code of an earlier commit of `main`, upstream `vllm/vllm-openai` v0.33.0 (A) against the image the cluster
builds from `inference-server/` (B, preset `ray-serve-llm`, vLLM 0.29.0, from the internal registry), one warm-up launch
per side and 2 rounds, 30 s at 1 and 8 streams (6 launches in 29 minutes; B's `engine_restarts`, which only a
`ray-serve-llm` side counts, 0 in each); mode `levels` with upstream vLLM v0.32.0 and v0.33.0, 1 round without a
warm-up, 30 s at each of 1, 4, 8, 16, 32 and 64 streams (21 minutes); and mode `scaleout` with the code as it
stood just before the statistics came from pairbench ("How scale out works"). In each, Argo CD created the Job, the
runner pod ran under the `restricted-v2` SCC, the server went
to 0 and back to one Ready pod, and no side pod was left. The `notebook` app showed Progressing while a Job ran, Healthy
once one had succeeded, and OutOfSync while an earlier run's Job and its config ConfigMap were still there (no pruning).
Both `levels` runs got tokens per joule from DCGM for every scored level (8 and 12), each from the energy counter with
15 power samples in its 30 s, so the 2 s interval held; which of the DCGM label names matched was not recorded. The
server records of OpenShift sides carry no server environment (`env_source: "none"`).
Not checked yet: the twin rule with real servers beyond those three runs: with a `vllm-openai` or `custom` side, on the
image `inference-server/` builds (vLLM 0.29.0), with the example's settings and default primary cases, with energy, with
`SERVER_CONTAINER`, with the current `serve_app.engine_flags` and the runner's later changes
(`engine_restarts` among them), and on more than one machine, GPU or model; on OpenShift, twins, a run that ends partial
or is stopped, the refusal of a second run, keda mode during a run, a side with `args` of its own, and the side pods'
SCC (not recorded); the DCGM reader against an exporter other than the one on those clusters, and which of its label
names (`exported_pod`, `exported_namespace`, `pod`, `Hostname`) a real exporter uses; the length of a run with the
defaults (an estimate of about 4 to 6 hours; none of the runs above times it); a launch whose GuideLLM timed out again
in its second try (`LOAD_FAILED`), which none of the runs above had; an engine restart count above 0 with vLLM 0.29.0
or on Kubernetes (the unit tests' logs have vLLM 0.33.0's line shapes and their backend is a fake); and the engine
restart count inside the second replica of mode `scaleout` and for a Ray Serve LLM server that exits once Ray Serve
gives up after replicas stopped at the check (the carry-over into the start again), which only the fake backend of the
unit tests has gone through.

### How scale out works

Three parts, each off by default. With every default, the local charts render exactly as before they existed
(`scripts/render-charts.sh`; the tests compare the renders).

- **Inside the pod** (`server.autoscaling`, `server.gpuPerReplica`): Ray Serve's own autoscaler, in the locked Ray
  release, adds and removes vLLM replicas in the one server pod. `serve_app.py` passes `target_ongoing_requests`
  into `autoscaling_config` and, when replicas share a GPU, a placement bundle of that share per worker
  (`placement_group_config`) and the same share as `VLLM_RAY_PER_WORKER_GPUS` in `LLMConfig.runtime_env`. vLLM
  0.29.0 reads that variable in its engine process, a child of the Serve replica, to size each worker's GPU request
  (1.0 when unset); Ray Serve LLM 2.60.0 sets it only for the workers, and LLMConfig.runtime_env is what reaches the
  replica (`LLMServer.get_deployment_options`). It does not depend on `global.scaleOut.mode`.
- **Across pods** (`global.scaleOut.mode: keda`): `variants/prod/values-prod.yaml` adds
  `/overrides/values-scaleout-<mode>.yaml` to `sharedValueFiles`; only the keda file exists. It installs Red Hat's
  Custom Metrics Autoscaler (operator, KedaController through the `keda-config` app), and the server chart renders the
  ScaledObject, TriggerAuthentication with its ServiceAccount and Roles, the ServiceMonitor and the `metrics` ports
  (see the names table). The trigger sums `ray_vllm_num_requests_waiting` over the server pods in `vllm-inference`,
  read from the Thanos Querier on port 9092 (namespace-scoped), and the HPA that KEDA creates divides it by the pod
  count. Between `minPods` and `maxPods` the HPA reads the metric on its own 15-second sync period:
  `pollingIntervalSeconds` only sets how often KEDA refreshes the ScaledObject's status. With Ray's metrics export
  (10 s) and the ServiceMonitor's scrape (15 s) before that read, a pod is asked for from a few seconds up to about 40
  seconds after requests start to wait. `cooldownSeconds` is the HPA's scale-down stabilization window. The
  ServiceMonitor scrapes with Ray's cluster token from Secret `ray-serve-llm-metrics-token`, which the server also
  starts Ray with (`RAY_AUTH_TOKEN_PATH`, "Server container environment"): a Ray build with token authentication on
  for its metrics endpoint answers 401 to a scrape without it, while Ray 2.60.0 checks no token on that port and takes
  the file as its cluster token. The chart generates the token with `randAlphaNum 64` on every render; the keda
  override's ignore rule on `/data/token` keeps the value Argo CD first created, as the notebook app's rule does for
  `notebook-proxy`. Nothing else in the chart may carry the value or a checksum of it, or the server would roll on
  every sync (`tests/test_charts.py` checks the render). When the value does change (an edit, the Secret deleted and
  created again, or the landing order under "Not checked yet"), the running server pods keep the old token through
  their `subPath` mount while the scrape sends the new one, so the docs say to delete the server pods after any
  change. Prerequisites: monitoring for user-defined projects enabled on the cluster, and a second GPU (another GPU
  node, or `timeSlicing.enabled` in `charts/all/nvidia-gpu-config/values.yaml`, or with `global.gpuOperators: reuse`
  time-slicing in the cluster's own ClusterPolicy, with `gpu_memory_utilization` lowered to leave room for the other
  pod; nothing checks that). Time-slicing covers every GPU node and namespace, while the
  comparison's `kube.gpu_free` looks only at `vllm-inference`: a GPU pod from another namespace can then share a timed
  GPU unseen, so the docs say to turn it off before a comparison in mode `levels`. The comparison's mode `scaleout`
  with `sharedGpu: true` needs it on with two shares; there a GPU pod from another namespace that holds the second
  share leaves the second replica with no node, and the launch fails (`kube.REPLICA_PENDING_S`) instead of being timed
  on a shared GPU, except for a pod that holds the second share before the forced step and is gone by then (the first
  replica's timed start and the surge before the step then share the GPU unseen). Server pods on more than one
  node need `modelStorage.accessMode: ReadWriteMany` and `compileCache.mode: shared`; a PVC's access mode cannot change
  after it is created.
- **The comparison's mode `scaleout`** (`global.compare.mode`): both backends run it. `scaleOut.sharedGpu` sets the
  memory default and the 0.475 refusal, and each backend checks where the replicas will run against it before
  anything changes (`check_plan`, `PLAN_INVALID`). On OpenShift (`KubeBackend.check_plan`) the runner lists the nodes
  of the sides' CPU type and needs a GPU node that fits (`kube.node_fits`): its labels (`kube.share_fits`:
  `nvidia.com/gpu.count` 1, `nvidia.com/gpu.replicas` 2 and `nvidia.com/gpu.sharing-strategy` `time-slicing` or absent
  for `true`; a count of 2 or more, replicas 1 or absent and the strategy `none` or absent for `false`; so MPS never
  fits, and a node without the strategy label is judged by the other two), and
  at least 2 `nvidia.com/gpu` in its `status.allocatable` (a device plugin with `renameByDefault: true` offers
  time-sliced shares as `nvidia.com/gpu.shared` instead, under the same labels). It refuses a
  `compare.side.gpuResourceName` other than `nvidia.com/gpu` (a MIG resource included). Each GPU side pod of the mode
  carries the matching required node affinity (`kube.share_affinity`, the labels only; the scheduler checks the
  allocatable); the second replica is pinned to the first replica's `kubernetes.io/hostname` (also where the
  ReadWriteOnce model volume is), that node is checked again with `node_fits`, and a second replica
  with no node after `kube.REPLICA_PENDING_S` (120 s) fails its launch. The router
  and GuideLLM run in the runner pod, in front of the side pods' IPs. A `ray-serve-llm` side mounts the live server's
  `llm.yaml`, so the docs say to keep `server.autoscaling` at one replica for this mode; nothing checks it. The
  OpenShift backend reads no GPU UUIDs (no `gpus` or `gpu_uuids` in its server records); `prepare()` adds a run note
  with the GPU share and raises the runner's own soft open-file limit to its hard limit (`kube.raise_open_files`,
  through `router.raise_open_file_limit`), so GuideLLM, its child process, inherits it; a note gives the limit, and
  another warns when it is below `kube.open_files_needed` (two for each request of the whole surge, plus
  `kube.ROUTER_SPARE_FILES`). The docker backend places
  the replicas from `RHVP_COMPARE_REPLICA_GPUS` (else `RHVP_COMPARE_GPUS`) and refuses, as `PLAN_INVALID` before
  anything changes, a plan whose `sharedGpu` does not match (`DockerBackend.check_plan`); `prepare()` reads each
  replica's GPUs with nvidia-smi into the server records (`gpus`, `gpu_uuids`, both shown in `results.md` and the
  notebook), and refuses two requests that reach one GPU while `sharedGpu` is false. nvidia-smi's GPU query in a
  container given a MIG device lists its parent GPU, so two MIG devices of one GPU share that UUID, and the records
  show the parent's. Two MIG requests (`0:0`, `MIG-...`; `dockerapi.is_mig_request`) are compared instead by the MIG
  device UUIDs `nvidia-smi -L` lists through each (`DockerBackend._compare_mig`, helper containers `gpu-mig-<n>`), so
  one device written two ways is refused; when it lists none, the run goes on with a note in `run.notes`. A MIG device
  and a whole GPU are compared by the parent's UUID. Each scored launch starts two servers (`plan.server_starts`: 66
  at the defaults, an estimate of about 5 to 7 hours, never timed at that length on a GPU or a cluster; the one run on
  OpenShift, 2 rounds without twins or a warm-up and 8 server starts, took about 38 minutes); each case is a floor
  group of its own, the two primaries need 7 rounds, and at 8 all 8 must agree.
  The router (`rhvp_compare/router.py`, a child of the runner) holds two connections for each request in flight:
  `scripts/compare-local.sh` starts the runner container with `--ulimit nofile=65536:65536`, soft and hard
  (`RUNNER_OPEN_FILES`, a whole number above 0; set but empty, no flag, and GuideLLM keeps docker's soft limit;
  anything else exits 2 before docker runs; if docker refuses to start the runner with "error setting rlimit", set it
  to a lower number, or set it but empty), since at docker's default soft limit of 1024 on the machine measured the
  router stopped taking connections at about 509 in flight; `router.serve` raises its own soft open-file limit to the
  hard limit when it starts and logs the limit before and after, and `router.OutOfFilesLog` logs running out of file
  handles at most once a second, with a count of the lines left out. A pod spec has no open-file setting, so on
  OpenShift the runner raises its own soft limit to the hard one, as the router does, and the run's notes record it;
  in the run on OpenShift they gave 1024 with a hard limit of 524288, raised to 524288.
  `scaleOut.requestRate`'s default (4) is a placeholder: the README's "Picking the surge rate" derives a rate from a
  levels run of the same sides, and `summary.surge_rate_fit` makes `results.md` and the notebook say when every scored
  surge of both sides met the goals, had at least half of its judged requests fail (`SURGE_FAILED_SHARE`), or had at
  most 5% within the goals (`SURGE_MET_SHARE_LOW`). In keda mode, the OpenShift runner pauses the ScaledObject before it
  scales the server to 0 (annotation `autoscaling.keda.sh/paused-replicas: "0"`; its Role allows get and patch on that
  one ScaledObject), and removes the pause when it brings the server back; the by-hand restore command it prints
  includes the `oc annotate` step, and so does the README's recovery for a runner killed outright (Argo CD ignores the
  annotation, so nothing else removes it). `kube.stop_budget_seconds()` is 290 s, under the runner pod's 300 s grace
  period; in mode `scaleout`, `kube.stop_budget_seconds(2)` is 370 s (the router's stop, `kube.ROUTER_STOP_S`, and the
  second side pod's teardown added), under the 420 s the chart gives the Job in that mode.

`global.scaleOut.mode` takes `none` or `keda`. `scripts/pattern_settings.py` (through `scripts/validate-values.sh`) and
`charts/all/ray-serve-llm/templates/_settings.tpl` refuse any other value with the same message; keep the two in step
(`tests/test_scaleout_contract.py`). Serving one model across several machines with Ray, as one Ray cluster, is out of
scope here; OpenShift AI offers KubeRay for that.

Switching keda mode off removes nothing: the apps sync without pruning, so the ScaledObject keeps scaling the
server until someone deletes it (README, "Scale out", has the commands), and the operator, `openshift-keda` and the
`keda-config` application stay.

The Subscription follows `stable` with automatic updates. Upstream KEDA 2.21 refuses a bound ServiceAccount token,
the kind `keda-trigger-auth-prometheus` uses, unless the operator maps that ServiceAccount to a token audience
(KEDA's 2.21 migration notes); the ScaledObject then never turns Ready and the server stops scaling while the
operator looks healthy. Red Hat's current operator is on the 2.19 line, which needs no mapping. If a Red Hat
release adds the rule, a mapping alone does not help here: KEDA then mints the token for a dedicated audience, and
the Thanos Querier's kube-rbac-proxy, which cluster monitoring manages, cannot be told to accept it (and mapping the
API server's own audience reopens what 2.21 closes). Move the TriggerAuthentication to a long-lived token Secret for
ServiceAccount `thanos`, or, as a stopgap, use KEDA's legacy token mode, which KEDA calls insecure.

Checked: unit tests and chart renders for all three parts. On CPU, in the server image, a stand-in Ray Serve app
with replicas that take 20 seconds to start scaled from 1 to 2 under `serve_app.serve_and_watch`: the `rhvp-scale`
lines showed the target and then the running count, the app stayed RUNNING, the ready file stayed, no request
failed, and `rhvp_status.py` printed the counts. On kind with upstream KEDA 2.19 (upstream 2.21 refused the bound
token, as above), the chart's own keda objects with stand-ins for the Thanos Querier and the server: 1 to 2 pods on
a waiting count above the threshold, held at 0 by the pause for a minute, back to 1 pod when the pause was removed;
every query carried the bearer token, the namespace and the rendered query. Through a Service, a new pod got none
of the requests of a client that kept 8 connections open; through `rhvp_compare/router.py`, it got a share. The
comparison's mode ran end to end with docker against GuideLLM mock servers
(`tests/fixtures/compare-scaleout-mock.json`: 7 rounds with twins, 28 launches, 56 server starts, about 46 minutes on
CPU): it completed with schema-valid results and the notebook rendered them. The faster-starting mock's second
replica gave `scale/first_answer_s` FASTER. The second replica got a share of the surge in 26 of the 28 launches. In
launch 20 (side A) and launch 15 (A~same, side A's twin) it joined the router 1.0 s and 1.9 s after the surge's last
request (sent 40 s after the first, while GuideLLM still ran) and got none; that run marked both as joined during the
surge, and `runner.joined_during_surge` now marks them as after it. The mock servers met the goals for every request, so
`surge/slo_attainment` was 1 in every launch and got NO VERDICT (`zero_floor`). With real vLLM servers under docker it
ran only in an earlier version, under the earlier rule (rounds of A and B, no twins), on one machine with one GPU: three
runs of one round each, one complete, two partial because nearly every request missed the goals at the rate they used.

Checked on OpenShift (4.22.18, throwaway clusters on AWS, the default g6.2xlarge node with one NVIDIA L4), one run of
each part:

- Inside the pod, with the server image the cluster built from an earlier commit of `main`: 1 to 2 replicas,
  `gpu_per_replica: 0.5`, `gpu_memory_utilization: 0.4`, `target_ongoing_requests: 4`, 32 clients at once. Both
  replicas started on the one L4 with `VLLM_RAY_PER_WORKER_GPUS` set, each engine in its replica's placement group (the
  GPU check at start passed); `rhvp_status.py` showed `LLMServer:qwen2_5-0_5b-instruct: running 2, target 2` 131 s
  after the load started; both replicas answered requests, 1,792 of 1,792 requests were answered, and the pod was still
  Ready at the end with no restart. The two `rhvp-scale` lines (`target`, then `running` 85 s later) reached the pod log
  inside Ray JSON log records, as each record's `message`; `logs.scale_events` reads that shape since then, in the unit
  tests only. The second replica was not seen removed (the run ended first).
- Keda mode, turned on in a running install, with `timeSlicing.enabled` (the node: allocatable `nvidia.com/gpu` 2,
  labels `nvidia.com/gpu.count` 1, `nvidia.com/gpu.replicas` 2, `nvidia.com/gpu.sharing-strategy` `time-slicing`,
  `nvidia.com/gpu.product` `NVIDIA-L4-SHARED`) and `gpu_memory_utilization` 0.4: Red Hat's operator
  (`custom-metrics-autoscaler.v2.19.0-4`) installed, app `keda-config` Synced and Healthy with KedaController `keda`
  reporting KEDA 2.19.0 installed, ScaledObject Ready. Under 32 clients the trigger's query read 24 waiting requests and
  the HPA had asked for a second pod by 35 s after the load started (checked about every 17 s); the second pod was Ready
  on the same node by 272 s and answered on its own address; 427 of 427 requests were answered; back to one pod by 285 s
  after the load stopped. During the load the second pod's metrics port exported `ray_vllm_num_requests_waiting` under
  that name. Before the first request Thanos had no series for it (for about 45 minutes, also after the server pod was
  deleted once), and the one scrape of the pod's port came back with no `ray_vllm_` lines; why is not proven. The server
  started with the `subPath` token mount.
- Mode `scaleout` as the OpenShift Job, with the code as it stood just before the statistics came from pairbench (the
  pattern's own sign test, Holm's correction and fewest rounds), `sharedGpu: true` on that node with time-slicing on
  (allocatable 2, `nvidia.com/gpu.count` 1, `nvidia.com/gpu.replicas` 2; the strategy label's value was not recorded
  for this node): upstream vLLM v0.32.0 (A) and v0.33.0 (B), preset `vllm-openai`, 2 rounds, no twins, no warm-up,
  53.8 requests/s for 420 s, the second replica asked for 30 s after the first request, goals of 1000 ms to the first
  token and 100 ms between tokens. `check_plan` passed; the notes gave the runner's open-file limit as 1024 (hard
  524288), raised to 524288; the Job's grace period was 420 s. All 4 scored launches completed with 0 failed requests
  and `start_retries`, `engine_restarts` and `load_retries` 0; the second replica's time to first answer was 74.3 to
  76.8 s (from its ask), it first got requests about 105 s after the surge's first request and got 42.7% to 43.1% of the
  requests; 27.8% (A) and 27.1% (B) of the requests sent were still running when the surge ended (left out, as GuideLLM
  does). Every case NO VERDICT (`too_few_rounds`). The rate was 1.2 times 44.8 requests/s, a lower bound from a `levels`
  run in which every level up to 64 streams met the goals on both sides; at 53.8 the share within the goals was 4.4% on
  both sides (medians), and `results.md` said a lower rate may suit these sides. The run took about 38 minutes.

The metrics token was checked on CPU only, without a cluster: one Ray head per container, started as `serve_app.py`
starts it, with no Serve app, a token file mounted at `/etc/ray-auth/token` and `RAY_AUTH_TOKEN_PATH` set. Ray 2.60.0
answered 200 on the metrics port with no token, a wrong one and the file's; it ran in token mode with the file's token
(without `RAY_AUTH_TOKEN_PATH` it also runs in token mode, with a token it writes to `~/.ray/auth_token`); and a
second process attached with `ray.init(address="auto")`, as `rhvp_status.py` does. A Ray 2.61 development build that
checks the token on that endpoint answered 401 with no token and 403 with a wrong one in each of four runs (one of
them with Ray's OpenTelemetry metrics path turned on), loaded the file's token, and answered 200 to 23 of the 24
scrapes with it; the other, the only such scrape of the first run, answered 500, and that run kept no detail of it.
With the path set and no file, both builds stopped in `ray.init` (`AuthenticationError`), which is why the chart's
mount is not optional. A review of the first version of the chart, which mounted the Secret as a directory, replaced
the token file in place under a running head (as the kubelet does to such a volume after the Secret changes), on CPU,
on Ray 2.60.0 and on the 2.61 development build. Before the swap, a second process attached with
`ray.init(address="auto")` and a new worker process came up in about 1 s. After it, an actor that was already running
still answered, but the attach failed (`ConnectionError` on 2.60.0, which `rhvp_status.py` reports as
`NO_CONTROLLER`; `AuthenticationError` on 2.61), new worker processes did not register within 40 to 60 s, and the
raylet logged `InvalidAuthToken`; the metrics port on 2.60.0 still answered 200. Putting the old file back made both
work again. That is why the chart now mounts the file with `subPath`.

Not checked yet:

- Scale events beyond the one run of each part above: a node with several GPUs, server pods on more than one node,
  the two layers together, more than 2 replicas or pods, Ray Serve removing a replica on a GPU, and the GPU check at
  start refusing too few GPUs.
- Replicas on a share of a GPU (`gpu_per_replica` below 1) have started on a real GPU only in the run above. An
  earlier try (one GPU, `gpu_per_replica: 0.5`) failed at the first replica's engine start with `Cannot schedule
  RayWorkerProc with the placement group ... {'CPU': 0, 'GPU': 1.0} cannot fit into ... [{'CPU': 1.0, 'GPU': 0.5}]`:
  the engine process did not see `VLLM_RAY_PER_WORKER_GPUS`. Before that run, the fix (the variable in
  `LLMConfig.runtime_env`) was checked on CPU only: Ray 2.60.0's Ray Serve LLM builder with a stand-in server class
  and a stand-in for vLLM's import surface, on a Ray node with one logical GPU; the replica's spawned child asked for
  one worker in the replica's placement group as `ray_executor_v2` does, and got the same error without the variable
  and a placed worker with it.
- In keda mode on OpenShift: the runner's own Role (the kind check paused as cluster admin, and no comparison ran in
  keda mode on OpenShift) and the Argo CD rule that keeps the pause annotation. Nor the metrics token there: the scrape
  worked, but Ray 2.60.0 checks no token on its metrics port, so that user-workload monitoring reads Secret
  `ray-serve-llm-metrics-token` and sends it is unchecked; nor that Argo CD keeps its first value, nor the order in
  which the ignore rule (in the clustergroup app) and the Secret (in the ray-serve-llm app) land when keda mode is
  turned on in a running install (it was turned on that way above, but whether the token changed was not checked). A
  second sync of the server's app before its rule arrives would replace the token; deleting the server pods then
  brings both sides back to one token. Nor that a running pod keeps its token file when the Secret changes: that a
  `subPath` file is not updated is from Kubernetes' documentation, not from a run (the `subPath` mount itself ran
  above, with no Secret change). Nor a Serve app running on a Ray build that checks the token.
- `ray_vllm_num_requests_waiting` has been seen under that name only under load (above); before the first request
  there was no series for it. KEDA reads an empty answer as 0, so a wrong name in another release means the server
  never scales, without an error.
- `compileCache.mode` `node` and `shared`, and `fsGroupChangePolicy: OnRootMismatch`, on OpenShift.
- The scale-out mode on OpenShift beyond the one run above: twins, a warm-up launch, a `ray-serve-llm` side,
  `sharedGpu: false` on a node with two GPUs or more, a second replica left with no node (`kube.REPLICA_PENDING_S`), a
  GuideLLM time-out, a stop (`kube.stop_budget_seconds(2)`), a run of the default length, the node check refusing a run
  (`kube.node_fits`, exit 2, `PLAN_INVALID`, naming what each GPU node offers), and the run note's warning of a low
  open-file limit (`kube.open_files_needed`); those last two have run only in the unit tests. The value of
  `nvidia.com/gpu.sharing-strategy` was read on the keda run's node, not on that run's. With twins, the current mode has
  run only against GuideLLM's mock servers: with docker, and on kind without a GPU (labels set by hand; one complete
  run, two stopped during the surge) in a version before the allocatable and sharing-strategy checks and the runner's
  open-file raise. The server records of OpenShift sides carry no server environment (`env_source: "none"`).
- An upscale replica that keeps failing: whether Ray then marks the app UNHEALTHY, which would remove the ready file
  and take the one serving replica out of the Service.
- The comparison's scale-out mode on MIG devices: that nvidia-smi's GPU query lists the parent GPU's UUID in a
  container given a MIG device, and that `nvidia-smi -L` lists the device's own UUID, is from NVIDIA's documented
  output, not from a run.

### Installing or reusing the GPU operators

`global.gpuOperators` (`values-global.yaml`) takes `install` or `reuse`. `variants/prod/values-prod.yaml` adds
`/overrides/values-gpu-operators-<value>.yaml` to `sharedValueFiles`, and only the install file exists: it holds
both operators' namespaces, Subscriptions and apps (names table, "GPU operators"), maps only, so it merges into the
variant's. With `reuse` the app of apps creates none of them. A Namespace in particular is never rendered for the
cluster's own operators: Argo CD would adopt it, and an uninstall could delete it with the operator in it.

The operators sit in the file that exists for `install` rather than behind a `disabled: true` file for `reuse` on
purpose: a misspelled or missing value then installs nothing, instead of a second copy of both operators next to the
cluster's (a second OperatorGroup in their namespace, and their Subscription and NodeFeatureDiscovery written over
where the names match the pattern's). The server chart, which always renders, refuses any value other than `install`
or `reuse`, and a missing one (`_settings.tpl`; no default in its `values.yaml`, which would hide a missing key);
`scripts/pattern_settings.py` (`check_gpu_config`, through `scripts/validate-values.sh`) refuses another value with
the same message. A missing one stops `validate-values.sh` before that, with its own message naming
`global.gpuOperators` (its `sharedValueFiles` entry needs the value). With `reuse` in `values-global.yaml`,
`check_gpu_config` also refuses `timeSlicing.enabled`, as does the GPU chart's
own render (`templates/time-slicing-config.yaml`); the two cases `validate-values.sh` adds with the other value
(`reuse` when `values-global.yaml` says `install`) leave time-slicing out, since with `install` it is the way to fit
two server pods on one GPU. The cluster renders no app for that chart with `reuse`, so on the cluster the
time-slicing setting would silently do nothing; only those two local checks catch it. With `reuse` every chart the
cluster still renders (all but `nfd-config` and `nvidia-gpu-config`) renders exactly as with `install`, and the app
of apps differs by exactly the eight objects in the names table's "GPU operators" row (`tests/test_charts.py`).

What the cluster's installs must provide is in the README, "Reuse the cluster's GPU operators": NFD labeling the GPU
nodes; a ready ClusterPolicy with a driver of 580 or newer, tolerating the GPU nodes' taints, labeling them
`nvidia.com/gpu.present=true` and offering `nvidia.com/gpu`; for the comparison's scale-out mode, GPU feature discovery
on (`spec.gfd.enabled`), whose node labels `KubeBackend.check_plan` reads; for the comparison's energy, the DCGM
exporter with its ServiceMonitor, the namespace label and the 2 s interval; for two pods on one GPU, its own
time-slicing with `renameByDefault: false`. Nothing on the cluster checks the energy part before a run: a too-slow DCGM
interval shows only as each level's note ("N power samples in the window; at least 10 are needed") once the run is under
way. The runner's scale-out node check (`KubeBackend.check_plan`) names the cluster's ClusterPolicy in its time-slicing
hints, and names GPU feature discovery instead when no GPU node of the sides' CPU type has its count label.

Run on a cluster once, on November 23, 2026 (the README's "Where this has run" says what was set up and checked): a
new OpenShift 4.22.18 cluster on AWS, NFD and the GPU Operator from OperatorHub, the pattern's own GPU node, no
time-slicing, then the install and an image comparison of 8 rounds. The cluster's ClusterPolicy, NodeFeatureDiscovery,
Subscriptions, OperatorGroups and namespaces kept their uid, generation, spec and owners throughout, and no Argo CD
application managed them. Not checked yet: a cluster that runs OpenShift AI, GPU nodes the pattern did not create,
time-slicing or MIG in the cluster's ClusterPolicy (so keda mode and the comparison's scale-out mode with `reuse`),
and uninstalling. Switching a running install is not supported and has not been tried. From `install` to `reuse`
nothing is removed, as the apps sync without pruning: the pattern's operator Namespaces, OperatorGroups and
Subscriptions stay, no longer updated, and its `nfd-config` and `nvidia-gpu-config` Applications stay with their
automated sync and keep applying the NodeFeatureDiscovery and ClusterPolicy from Git (the two charts still render them
when `values-global.yaml` says `reuse`). Both Applications carry Argo CD's foreground resources finalizer, so deleting
them deletes those two objects. From `reuse` to `install` adds a second copy of both operators.

## How to deploy

The pattern deploys through Red Hat OpenShift GitOps and the Validated Patterns Operator. Argo CD deploys what is
in **your fork on GitHub**, not the files on your disk: commit and push every change before you install or expect
it to sync.

### Prerequisites

- OpenShift 4.18 or newer, and cluster-admin access.
- Node Feature Discovery and the NVIDIA GPU Operator, one of:
  - a cluster that runs neither (the default, `global.gpuOperators: install`). The pattern installs both itself,
    with its own OperatorGroups and settings, and would clash with existing installs of them.
  - a cluster that already runs both, such as an OpenShift AI GPU cluster, with `global.gpuOperators: reuse` set in
    your fork before you install. Their ClusterPolicy must provide what the README, "Reuse the cluster's GPU
    operators", lists. This has run on one test cluster so far.

  The pattern has not been tried on a cluster that runs OpenShift AI.
- A StorageClass that provisions volumes dynamically, for the model PVC.
- The internal image registry set to `Managed`, and the `Build` and `ImageRegistry` cluster capabilities enabled.
  The cluster builds both images itself. To check:
  ```sh
  oc get configs.imageregistry.operator.openshift.io cluster -o jsonpath='{.spec.managementState}'
  oc get clusterversion version -o jsonpath='{.status.capabilities.enabledCapabilities}'
  ```
  On bare metal and vSphere the registry starts as `Removed`; see [Troubleshooting](#troubleshooting).
- A GPU. On AWS with `global.gpuMachineSet: create`, the pattern creates the node. Anywhere else, add a node with
  an NVIDIA GPU that the NVIDIA GPU Operator supports. One AWS exception: on a cluster installed into existing
  subnets (`platform.aws.subnets` or `vpc.subnets` in its install-config), the worker MachineSets name their subnet
  by ID, and the `create_machineset` job's template reads subnet filters, so it fails on every run. Set
  `gpuMachineSet: none` there and add the GPU MachineSet yourself. The server requests `nvidia.com/gpu: 1` and tolerates the
  `nvidia.com/gpu` taint.
- The CPU type of that GPU node. Both Deployments and both BuildConfigs select nodes with
  `kubernetes.io/arch: amd64`, from the `arch` value in `charts/all/ray-serve-llm/values.yaml` and
  `charts/all/notebook/values.yaml`. If your GPU node is arm64, set `arch: arm64` in
  `charts/all/ray-serve-llm/values.yaml` in your fork before you install; otherwise the server pod and its build
  never get a node. Leave the notebook chart on the CPU type of your ordinary workers: it needs no GPU and does not
  tolerate the GPU taint. The server's build also needs a node of that CPU type without the `nvidia.com/gpu`
  taint: a BuildConfig cannot carry tolerations (only a cluster-wide build override can add them), so a lone,
  tainted arm64 GPU node cannot run it.
- On your workstation:
  - podman 4.3 or newer. `pattern.sh` runs everything in a podman container; docker alone is not enough.
  - git.
  - `oc`, logged in to the cluster.
  - If `KUBECONFIG` is set, it must point inside `$HOME`, because only `$HOME` is mounted into the container.
- A fork of this repository, cloned locally, with your branch pushed and an upstream set. The installer reads the
  repo URL from the branch's upstream remote, and fails if the branch has no upstream.

### Path 1: command line (recommended)

1. Fork `https://github.com/eugo-inc/vllm-ray-inference-server` on GitHub, then clone your fork:
   ```sh
   git clone https://github.com/<you>/vllm-ray-inference-server.git
   cd vllm-ray-inference-server
   ```
2. Edit `values-global.yaml`:
   - `global.gpuMachineSet`: `create` (the default: one AWS g6.2xlarge with one L4) or `none`.
   - `global.gpuOperators`: `install` (the default: the pattern installs NFD and the NVIDIA GPU Operator) or
     `reuse` (the cluster already runs both; README, "Reuse the cluster's GPU operators"). If that cluster already
     has a GPU node for the server, set `global.gpuMachineSet: none` too, and `gpu.taintKey` in
     `charts/all/ray-serve-llm/values.yaml` if its GPU nodes are tainted with a key other than `nvidia.com/gpu`.
   - `global.model.id`: the name clients send as `model`.
   - `global.model.source`: the Hugging Face repo id.
   - `global.model.revision`: the Hugging Face commit to pin.

   Then commit and push:
   ```sh
   git commit -am "Configure my install"
   git push            # for a new branch: git push -u origin <branch>
   ```
3. Secrets. These are optional: the default model is not gated, and the installer falls back to the repo's
   template (token `none`). For a gated model, or to avoid picking up a leftover `~/values-secret.yaml` from
   another pattern, create the pattern's own secrets file:
   ```sh
   mkdir -p ~/.config/validated-patterns
   cp values-secret.yaml.template \
      ~/.config/validated-patterns/values-secret-vllm-ray-inference-server.yaml
   ```
   Put your `hf_...` token in the `hftoken` field's `value`. Never commit a filled-in secrets file;
   `.gitignore` ignores `values-secret*` except the template.
4. Log in and install:
   ```sh
   oc login --token=<token> --server=<api-url>
   ./pattern.sh make install
   ```
5. What `make install` does, step by step:
   1. `pattern.sh` starts `quay.io/validatedpatterns/utility-container` with podman, mounts the repo and
      `$HOME`, and runs `make` inside it.
   2. `make install` runs the Ansible playbook `rhvp.cluster_utils.install`. In order, it loads bootstrap
      secrets, deploys the operator, loads secrets, and then waits for every Argo CD application to be Synced and
      Healthy. The wait is at most 60 tries, 60 seconds apart.
   3. The operator step checks prerequisites first: cluster access, a StorageClass, the OpenShift version, and the
      name-length limit. It then renders the `pattern-install` Helm chart
      (`oci://quay.io/validatedpatterns/pattern-install`) with `values-global.yaml` plus your branch's upstream
      URL and branch name, and applies the result.
   4. That chart subscribes the cluster to the **Validated Patterns Operator** (`patterns-operator`, from
      `community-operators`). It also creates the **Pattern** custom resource: variant `prod`, Git target = your
      fork and branch.
   5. The operator installs **Red Hat OpenShift GitOps** and creates the Argo CD instance **`vp-gitops`** (in
      namespace `vp-gitops`). It then creates the app-of-apps: the **clustergroup** Helm chart (version `0.9.*`
      from charts.validatedpatterns.io), with `values-global.yaml` and `variants/prod/values-prod.yaml` from your
      fork.
   6. The clustergroup chart creates:
      - the namespaces and OperatorGroups;
      - the operator Subscriptions: External Secrets, and with `gpuOperators: install` NFD and the NVIDIA GPU
        Operator;
      - one Argo CD Application per app: `vault`, `openshift-external-secrets`, `ray-serve-llm` and `notebook`,
        and with `gpuOperators: install` also `nfd-config` and `nvidia-gpu-config`;
      - the imperative CronJob, which runs two jobs: `create-gpu-machineset` (the playbook
        `rhvp.cluster_utils.create_machineset`) on AWS when `gpuMachineSet: create`, and `rebuild-on-commit`.
   7. The apps then come up. The BuildConfigs build the server and notebook images inside the cluster. The server
      pod waits for a GPU node, downloads the model to its PVC, and turns Ready once the model is loaded.
6. `./pattern.sh make load-secrets` loads the secrets file again. Run it after you edit that file;
   `make install` already loaded it once. The secret travels from Vault (`secret/data/hub/hfmodel`, property
   `hftoken`) through ExternalSecret `huggingface-secret` to the Secret `huggingface-secret`. The server reads
   `HF_TOKEN` only when its container starts, so delete the pod to pick up a new token:
   `oc delete pod -l app.kubernetes.io/name=ray-serve-llm -n vllm-inference`.
7. `./pattern.sh make argo-healthcheck` checks again that every Argo CD application is Synced and Healthy. Use
   it after an install that timed out.

Secrets file search order. The first file found wins:

1. `$VALUES_SECRET`
2. `~/.config/hybrid-cloud-patterns/values-secret-vllm-ray-inference-server.yaml`
3. `~/.config/validated-patterns/values-secret-vllm-ray-inference-server.yaml`
4. `~/values-secret-vllm-ray-inference-server.yaml`
5. `~/values-secret.yaml`
6. the repo's `values-secret.yaml.template`

The template's `hftoken` value must stay a non-empty string (`"none"`). With `null`, the secrets loader aborts.

### Path 2: OpenShift web console

1. Go to **OperatorHub**, search for "Validated Patterns" and install the **Validated Patterns Operator**. It is a
   community operator, so accept the community-operator notice.
2. Go to **Installed Operators → Validated Patterns Operator → Pattern → Create instance**.
3. Fill in the form:
   - Name: `vllm-ray-inference-server`
   - **Cluster Group Name**: `prod`
   - Git Config → **Target Repo**: your fork's URL
   - Target Revision: your branch
4. Click **Create**.
5. Load secrets afterward. The console path loads none. Do this even for the default, ungated model: until
   `secret/data/hub/hfmodel` exists, ExternalSecret `huggingface-secret` is not Ready and Argo CD shows the
   `ray-serve-llm` app as Degraded. The server pod still starts, because the token is optional. Two ways:
   - From a clone of your fork, run `./pattern.sh make load-secrets`.
   - In the Vault UI, in the KV engine `secret`, create path `hub/hfmodel` with key `hftoken` (value `none`, or
     your token). The Vault root token is in Secret `vaultkeys` in namespace `imperative`.

### Path 3: private forks

Argo CD needs credentials to read a private fork:

1. Create an Argo CD repository Secret. It carries the label `argocd.argoproj.io/secret-type: repository` and
   your SSH key or token. You can create it through `bootstrap_secrets` in your secrets file.
2. Install with that Secret:
   ```sh
   ./pattern.sh make TOKEN_SECRET=<secret-name> TOKEN_NAMESPACE=<secret-namespace> install
   ```
3. See https://validatedpatterns.io/learn/private-repos/ for the Secret format.

That token covers Argo CD only. The in-cluster BuildConfigs (`ray-serve-llm`, `notebook`) also clone the
repo from `global.repoURL`, and they do not use that Secret. For a private fork, give each build its own git
credentials:

1. Create a source secret in each build's namespace, for example with a GitHub token:
   ```sh
   oc create secret generic pattern-git -n vllm-inference --type=kubernetes.io/basic-auth \
     --from-literal=username=<you> --from-literal=password=<token>
   oc create secret generic pattern-git -n vllm-notebook --type=kubernetes.io/basic-auth \
     --from-literal=username=<you> --from-literal=password=<token>
   ```
2. Set `build.sourceSecret: pattern-git` in `charts/all/ray-serve-llm/values.yaml` and
   `charts/all/notebook/values.yaml`, then push.

Or keep the fork public, or set `image.override` to a prebuilt image. On a running install, a new `image.override`
needs extra steps; see Troubleshooting.

### Updating a running install

- **Config and charts.** Push to your branch. Argo CD syncs automatically (automated sync, with retries).
- **Model or engine settings.** Changing `global.model.*` changes ConfigMap `ray-serve-llm-config`. Its checksum
  annotation then restarts the server pod, which downloads the new model. The Deployment uses `Recreate`, so the
  server is down until the new pod is Ready.
- **Images.** A BuildConfig's ConfigChange trigger fires only when the BuildConfig is created, so a push alone
  does not rebuild an image. There are two ways to rebuild:
  - **Automatic.** The `rebuild-on-commit` imperative job (`ansible/rebuild-on-commit.yaml`) runs every 10
    minutes, for both `ray-serve-llm` and `notebook`. It starts a build only when that build's folder
    (`inference-server/` or `notebooks/`) changed between the last successful build's commit and the head of the
    target branch. A failed build is not retried automatically. The full rule list is at the top of
    `ansible/rebuild_on_commit.py`. An error on one build (for example a Forbidden `start-build` when
    `rebuildOnCommit.enabled` is false in that chart) is printed as an `ERROR` line in the job's log, and the job
    still checks the other build and exits 0. The imperative CronJob runs its jobs one after another and stops at
    the first that fails, so `rebuild-on-commit` runs first and exits 0 on any `oc` or `git` error. Only its
    300-second job timeout can still fail the pod, and that also skips the GPU MachineSet job for that run.
  - **By hand:**
    ```sh
    oc start-build ray-serve-llm -n vllm-inference --follow
    oc start-build notebook -n vllm-notebook --follow
    ```

  A finished build updates the ImageStream. The image trigger annotation then rolls the Deployment.
- **Installs made under the old name.** Until November 2026 this pattern was named
  `rhvp-vllm-ray-inference-server`, and its namespaces were `rhvp-inference` and `rhvp-notebook`. An install made
  under the old name keeps working, and its Pattern resource and clustergroup app keep that name.
  - Merging the rename into your install's branch moves both apps to the new namespaces, `vllm-inference` and
    `vllm-notebook`, and nothing deletes the old ones. The `ray-serve-llm` and `notebook` apps sync without pruning
    (`automated: {}` in `variants/prod/values-prod.yaml`), and the Validated Patterns Operator syncs the
    clustergroup app without pruning too. So Argo CD creates a second server and a second notebook in the new
    namespaces, builds both images there (the server image takes about 18 minutes), and creates a new model PVC, so
    the new server downloads the model again. The old server keeps running in `rhvp-inference` and keeps its GPU.
    With one GPU, the new server pod stays Pending until you delete the old namespaces (see "Old namespaces" below).
    This is what the charts and the operator's code imply; it has not been tried on a cluster.
  - The installer takes the pattern name from `PATTERN_NAME` when it is set, and from `global.pattern` otherwise,
    and finds the Pattern resource and the secrets file by that name. So once your fork has the new
    `global.pattern`, set `PATTERN_NAME=rhvp-vllm-ray-inference-server` on every `./pattern.sh make` target you run
    on the old install, until you move it. Without it, nothing warns you:
    - `load-secrets` no longer finds a secrets file named for the old name. It loads `~/values-secret.yaml` if
      that exists, and the repo's template otherwise, and replaces a real token in Vault with that file's value
      (`none`, from the template).
    - `install` creates a second Pattern resource, `vllm-ray-inference-server`, next to the old one.
  - `display-secrets-info` is the exception: it always takes the name from `global.pattern` and ignores
    `PATTERN_NAME`. For it, set `VALUES_SECRET` to your old secrets file's path instead.

  To move the install to the new name, uninstall it first and merge the rename after:
  1. Download any notebook work you want to keep.
  2. Uninstall it under the old name: `PATTERN_NAME=rhvp-vllm-ray-inference-server ./pattern.sh make uninstall`
     (see [Uninstall](#uninstall)).
  3. Once the uninstall has finished, merge the rename into your branch, if you have not already, and push it. The
     uninstall has finished when the Pattern resource is gone and the operator has removed the apps:
     `oc get patterns -A` no longer lists `rhvp-vllm-ray-inference-server`, and `oc get applications.argoproj.io -A`
     lists none of its apps. Step 6 takes the pattern name from `global.pattern` in your clone, and Argo CD deploys
     from your fork, so the rename must be in both. Merging only after the uninstall keeps the old install from
     syncing the rename and starting a second server in the new namespaces. This order follows from the charts and
     the installer; it has not been tried on a cluster.
  4. Check for the old namespaces and delete them (see "Old namespaces" below).
  5. Rename your secrets file, if you have one:
     ```sh
     mv ~/.config/validated-patterns/values-secret-rhvp-vllm-ray-inference-server.yaml \
        ~/.config/validated-patterns/values-secret-vllm-ray-inference-server.yaml
     ```
  6. Install again: `./pattern.sh make install`. The new install creates `vllm-inference` and `vllm-notebook`,
     builds both images, and downloads the model again into a new PVC, `vllm-inference/ray-serve-llm-models`.

  Old namespaces. Neither path is sure to delete `rhvp-inference` and `rhvp-notebook`. A merge never does. The
  uninstall is experimental and has not been checked here; the framework's announcement says its last phase removes
  the app of apps, and with it the subscriptions and namespaces. Delete them only after the merge has synced or the uninstall has finished; before that, Argo CD can
  create them again. To check:
  ```sh
  oc get namespace rhvp-inference rhvp-notebook
  oc get pvc ray-serve-llm-models -n rhvp-inference
  ```
  To delete them with everything in them, including the old server, the old notebook and the downloaded model:
  ```sh
  oc delete namespace rhvp-inference rhvp-notebook
  ```
  Then check `oc get pv` for a volume whose claim was `rhvp-inference/ray-serve-llm-models`. A volume with the
  reclaim policy `Retain` stays after its claim is deleted; delete it with `oc delete pv <name>`. That deletes only
  the Kubernetes object: the backing disk (an EBS volume on AWS) stays until you delete it in your storage backend.

## Before changing structure, study how existing patterns do it

This pattern copies its structure from these sources. Read the matching one before you add an operator, an app, a
GPU knob, a build or a new serving mode:

- **The template.** https://github.com/validatedpatterns/multicloud-gitops
  - `values-global.yaml`
  - `variants/`: `variants/standalone/values-standalone.yaml` is the single-cluster shape.
  - `Makefile-common`
- **GPU enablement.** https://github.com/validatedpatterns-sandbox/ai-quickstart-rag
  - `overrides/values-gpu.yaml`: the NFD and GPU Operator namespaces, subscriptions and apps.
  - `overrides/values-gpu-AWS.yaml`: the `rhvp.cluster_utils.create_machineset` job and its `clusterRoleYaml`.
  - `charts/nvidia-config`: a ClusterPolicy that tolerates `nvidia.com/gpu`.
- **vLLM and the Hugging Face token.** https://github.com/validatedpatterns/rag-llm-gitops
  - `charts/all/vllm-inference-service`: vLLM on KServe, and the `hf_` token check.
  - `values-secret.yaml.template`: the `hfmodel` secret.
- **In-cluster builds under Argo CD.** https://github.com/validatedpatterns-sandbox/qna-chat-amd
  - `charts/all/llm/templates/llm-buildconfig.yaml`: a BuildConfig with `dockerfilePath`.
  - `values-hub.yaml`: `ignoreDifferences` for image-triggered Deployments and BuildConfig `.spec.paused`.
- **The framework itself.** The pattern-author skill in `.claude/skills/pattern-author/` (`SKILL.md`,
  `reference.md`) covers the clustergroup values schema, the values-file order, sharedValueFiles and imperative
  jobs. `.cursor/skills/pattern-author/` holds the same skill.

Framework facts that catch people out:

- Namespaces are maps, and maps merge across values files. Lists replace each other: a values file that sets
  `clusterGroup.imperative.jobs` replaces the whole job list from earlier files.
- A per-app `syncPolicy` replaces the chart's default (`automated: {}` plus `retry`) completely, so restate
  both.
- Missing `sharedValueFiles` are skipped. That is how `gpuMachineSet: none`, or any platform other than AWS,
  turns the MachineSet job off, and how `global.scaleOut.mode: none` adds no autoscaler: each app's valueFiles
  then lists `/overrides/values-scaleout-none.yaml`, which does not exist. It is also how `global.gpuOperators:
  reuse` installs neither GPU operator: they are in the install file, so a misspelled value installs nothing rather
  than a second copy ("Installing or reusing the GPU operators"). A value named in a `sharedValueFiles` entry must be
  set: a missing nested one (`global.scaleOut.mode`) fails the clustergroup chart's render, and a missing top-level
  one (`global.gpuOperators`) renders as an empty word there, so its file is skipped; `scripts/validate-values.sh`
  checks both.
- Keep app keys equal to chart folder names.

## Local development loop

Install the Python tools:

```sh
python3.12 -m pip install pytest pyyaml jsonschema nbformat nbclient nbconvert ipykernel httpx aiohttp==3.14.3
# pairbench, which the comparison's statistics need, pinned with its hashes (as CI installs it):
python3.12 -m pip install --no-deps --only-binary :all: --require-hashes -r notebooks/requirements-compare.lock
```

Then run these, all from the repo root:

| Command | What it does | Where it runs |
|---|---|---|
| `scripts/render-charts.sh` | renders every local chart and the clustergroup chart with the values the operator injects | needs docker or podman; uses the utility container |
| `scripts/validate-values.sh` | schema-checks the merged values (global, then `prod`, then each override) and the secrets template, and checks the scale-out settings | anywhere |
| `python3.12 -m pytest -q tests/` | runs the unit tests | anywhere; the chart tests (those that use the render in `tests/conftest.py`) are skipped without docker or podman and access to charts.validatedpatterns.io |
| `python3.12 notebooks/make_notebook.py --check` | regenerates both notebooks and their synthetic files and fails if one differs from the committed file | anywhere |
| `scripts/local-smoke.sh` | builds the server image and serves the model with docker | a machine with an NVIDIA GPU only |
| `scripts/compare-local.sh tests/fixtures/compare-local-mock.json <folder>` | the image comparison with docker, against two GuideLLM mock servers | needs docker and access to PyPI or a mirror (pairbench); GPU sides need an NVIDIA GPU |
| `scripts/compare-local.sh tests/fixtures/compare-scaleout-mock.json <folder>` | the comparison's scale-out mode with docker, against GuideLLM mock servers that take a while to start | needs docker and access to PyPI or a mirror (pairbench); no GPU |
| `python3.12 -m rhvp_compare validate --config <file>` (from `notebooks/`) | checks a comparison config and prints every round's order, the server starts and whether the rounds can give a verdict | anywhere |

## Bumping Ray and vLLM

Ray decides the vLLM version. The `llm` extra in Ray's `python/setup.py` pins exactly one vLLM: Ray 2.60.0 pins
`vllm[audio]==0.29.0`, which pins `torch==2.13.0`. Never bump vLLM alone.

1. Read `python/setup.py` at the new Ray tag (`https://github.com/ray-project/ray/blob/ray-<version>/python/setup.py`)
   for the vLLM pin. Then read that vLLM's requirements for its torch pin, and that torch's `cuda-toolkit` pin.
2. Edit `inference-server/requirements.in`: set `ray[llm]==<version>`. Set `cuda-toolkit[nvcc,cccl]` to torch's
   CUDA toolkit version.
3. If the CUDA version changed, move the `FROM` in `inference-server/Containerfile` to the matching
   `nvcr.io/nvidia/cuda:<cuda>-devel-ubi9` image, pinned by digest.
4. Regenerate both locks from `inference-server/` (the command is also in each lock file's header):
   ```sh
   uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_34 \
     --only-binary :all: --generate-hashes -o requirements-x86_64.lock
   uv pip compile requirements.in --python-version 3.12 --python-platform aarch64-manylinux_2_34 \
     --only-binary :all: --generate-hashes -o requirements-aarch64.lock
   ```
5. If the install step fails in hash mode, check `inference-server/requirements-pip.lock`: the image installs that
   pip first, and an older pip can reject a dependency that is asked for with an extra.
6. Update the `BASELINE` label in `notebooks/rhvp_bench/words.py`, then regenerate the notebook, and update the
   `version`, `description` and `io.k8s.description` image labels in `inference-server/Containerfile`. A test checks
   their Ray, vLLM, PyTorch and CUDA versions against the lock files. Update the versions in `README.md`, and the vLLM tag in the GPU support link below.
7. Run the whole local loop, then `scripts/local-smoke.sh` on a GPU machine.

## Bumping pairbench

pairbench (https://github.com/eugo-inc/pairbench) gives the comparison its sign test, Holm's correction and fewest
rounds ("How the image comparison runs"). For a new version:

1. `notebooks/requirements-compare.in`: set `pairbench==<version>`. Regenerate both locks from `notebooks/` with the
   commands in their headers (`requirements-compare.lock`, then `requirements.lock`, whose `requirements.in` takes in
   `requirements-compare.in`), and check that the only change in `requirements.lock` is pairbench's stanza. A test
   keeps the two locks' pins equal.
2. `charts/all/notebook/templates/compare-job.yaml`: the `pairbench-<version>.dist-info` folder the init container
   copies. `tests/test_charts.py` checks it against the lock.
3. Check that `pairbench.stats.sign_test`, `holm` and `min_rounds` still mean what `rhvp_compare/stats.py` says, run
   `tests/test_compare_stats.py`, and re-analyze recorded results with the old and the new version: the
   `results.json` and `results.md` must not change. Then run the whole local loop.

## Rules

1. **Stock packages only.** Allowed sources:
   - PyPI wheels;
   - public NVIDIA and Red Hat UBI base images;
   - Red Hat and certified operators;
   - Validated Patterns charts.

   Never add Eugo packages, Eugo images or Eugo registries, and never add a private package index.

   One exception, and only this one: pairbench (https://github.com/eugo-inc/pairbench), Eugo's open-source
   benchmarking tool (Apache-2.0). The image comparison takes its exact sign test, Holm's correction and fewest rounds
   from it ("How the image comparison runs"). It comes unmodified from PyPI, pinned with its hashes in
   `notebooks/requirements-compare.lock`, without extras, and goes only into the notebook image and the comparison's
   runner, never into the server image. Do not use more of pairbench, or add another Eugo package, without the
   maintainer's decision.

   The image comparison also runs GuideLLM's public release image, pinned by digest, and the two side images the
   user puts in `global.compare.sides`. Examples, defaults and tests use only public upstream images (two vLLM
   releases) or made-up `registry.example/...` names.
2. **Notebooks.** `notebooks/inference_timings.ipynb` and `notebooks/compare_images.ipynb` are generated by
   `notebooks/make_notebook.py` and are committed without outputs. Never hand-edit an `.ipynb`: change
   `make_notebook.py`, `rhvp_bench/` or `rhvp_compare/`, regenerate, and run `--check`.
   - `inference_timings.ipynb` times the stock server only. Its results are given in seconds and tokens per
     second, never as a ratio; `tests/test_notebook_rules.py` checks this.
   - `compare_images.ipynb` shows B/A ratios and the verdicts of the comparison's pre-declared rule, and nothing
     else that ranks the two images; `tests/test_compare_notebook.py` checks it.
   - In both, no sentence gives a cause for a result.
3. **Telemetry stays off.** Keep these settings in the server image: `RAY_USAGE_STATS_ENABLED=0`,
   `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `RAYLLM_ENABLE_REQUEST_PROMPT_LOGS=0`.
4. **Installer name limit.** `len(global.pattern) + 2 * len(variant)` must be under 47. Today it is
   25 + 2 * 4 = 33, so a variant name of up to 10 characters, such as `standalone` (25 + 2 * 10 = 45), still fits.
5. **Secrets.** Never commit real secret values.
6. **Keep the probes.** Readiness and startup check the ready file. Do not replace them with an HTTP check (see
   Troubleshooting).
7. **Keep `strategy: Recreate`** on the server Deployment.
8. **Imperative jobs must be idempotent.** They run every 10 minutes.
9. **License headers.** Server files that carry the Eugo BSD-3-Clause header keep it. Do not name private
   repositories in any file.

## Troubleshooting

- **Server crashes with `PermissionError` while writing a file, or the entrypoint exits with "... is not
  writable by uid ...".**
  - Why: OpenShift runs the container as an arbitrary UID in group 0. The image's virtualenv is read-only. Only
    `/opt/app-root/home`, `/cache`, `/models` and `/tmp/r` are writable by group 0. The chart mounts emptyDirs at
    `/cache` (or PVC `ray-serve-llm-compile-cache`, see `compileCache.mode`) and `/tmp/r`, and the PVC at `/models`.
  - Fix: point any new tool's cache at `/cache/<tool>` through its environment variable. For any new image folder
    that must be writable, run `chgrp 0` and `chmod g=u` in the same `RUN` that creates it.
- **Ray warns that it is using `/tmp` instead of `/dev/shm`, or the pod is OOMKilled.**
  - Why: Ray's object store lives in `/dev/shm`. The chart mounts a 4Gi memory-backed emptyDir there, and
    `object_store_memory_gb: 2` keeps the store inside it. A memory-backed emptyDir counts against the pod's
    memory limit.
  - Fix: if you raise either size, raise the `/dev/shm` `sizeLimit` and the memory limit with it.
- **Connections to the Service are refused while Ray reports the app as RUNNING.**
  - Why: Ray Serve's HTTP proxy listens on localhost by default. `serve_app.py` calls
    `serve.start(http_options={"host": "0.0.0.0", "port": 8000})` with `http_host` and `http_port` from
    `llm.yaml`.
  - Fix: keep `http_host: 0.0.0.0`.
- **The pod turned Ready before the model was loaded, and requests hang.**
  - Why: `/v1/models` and `/-/healthz` both answer before the model is loaded.
  - Fix: the readiness and startup probes must check the ready file `/tmp/r/rhvp-ready`. `serve_app.py` writes
    it only after `serve.run` returns with the app RUNNING. Use `/-/healthz` for liveness only.
- **FlashInfer errors about `nvcc`, or every pod start takes a long time.**
  - Why: FlashInfer compiles GPU kernels just in time with `nvcc`. The base is NVIDIA's CUDA *devel* UBI 9 image,
    which ships `nvcc`; `CUDA_HOME=/usr/local/cuda`; and the locks pin `cuda-toolkit[nvcc,cccl]` to the same CUDA
    version. With `compileCache.mode: none` (the default) the compile caches live in the `/cache` emptyDir, so
    every new pod compiles the kernels again.
  - Fix: do not switch to a `-runtime` or `-base` image. To keep the compiled kernels for later pods, set
    `compileCache.mode` to `node` (one GPU node) or `shared` (a ReadWriteMany StorageClass).
- **A request through a Route fails with 504 or times out.**
  - Why: the OpenShift router times out after 30 seconds by default. The optional server Route sets
    `haproxy.router.openshift.io/timeout: 10m`.
  - Note: the notebook talks to the Service directly, not through a Route.
- **After an update, the new server pod stays Pending, or the Deployment reports `ProgressDeadlineExceeded`.**
  - Why: a rolling update would need a second GPU and the same single-node (ReadWriteOnce) PVC. With one GPU it
    deadlocks.
  - Fix: keep `strategy: Recreate`, and expect the server to be down during each rollout.
- **The build fails to push, or no build ever runs.**
  - Why: the internal image registry must be `Managed`. On bare metal and vSphere it starts as `Removed`, because
    there is no shared object storage. The `Build` capability must also be enabled.
  - Fix: give the registry storage and set it to `Managed`. Or set `image.override` in both charts to prebuilt
    images that you host; the charts then render no BuildConfig and no image trigger (on a running install, see the
    `image.override` entry below for the extra steps). Without the registry, also
    set `oauthProxy.image` in `charts/all/notebook/values.yaml` to a Red Hat image such as
    `registry.redhat.io/openshift4/ose-oauth-proxy-rhel9:v4.19`, because the default proxy image is pulled
    through the internal registry.
- **You set or changed `image.override` on a running install, and the pod still runs the old image.**
  - Why: the Argo CD apps ignore the image of the image-triggered container (`server`, `notebook`), and with
    `RespectIgnoreDifferences=true` an ignored field keeps its live value even when Git changes it. The rule names
    only that container, so an `oauthProxy.image` change does sync; an `image.override` change does not.
  - Fix, in this order, when the install was building the image in the cluster (`image.override` was empty). To
    change one prebuilt image for another, only step 2 is needed: there is no trigger, and nothing to remove.
    1. Wait until Argo CD has applied the commit: this prints nothing once the image-trigger annotation is gone
       (before that, the trigger would put the old image back):
       `oc get deployment/ray-serve-llm -n vllm-inference -o jsonpath='{.metadata.annotations.image\.openshift\.io/triggers}'`
       (for the notebook, `deployment/notebook -n vllm-notebook`). The app still shows OutOfSync at this
       point, because of the objects step 3 removes.
    2. Set the image on the Deployment yourself. Argo CD ignores that one field, so the value stays:
       `oc set image deployment/ray-serve-llm server=<your image> -n vllm-inference`
       (for the notebook: `oc set image deployment/notebook notebook=<your image> -n vllm-notebook`).
       Wait until the new pod runs that image.
    3. Remove what the chart no longer renders. The apps sync automatically without pruning, so Argo CD leaves the
       BuildConfig, ImageStream and the rebuild Role and RoleBinding in place; the app stays OutOfSync and
       `rebuild-on-commit` keeps starting builds of that BuildConfig. Either sync once with Prune checked in the
       Argo CD UI, or delete them:
       `oc delete buildconfig/ray-serve-llm imagestream/ray-serve-llm role/rebuild-on-commit rolebinding/rebuild-on-commit -n vllm-inference`
       (for the notebook, `buildconfig/notebook imagestream/notebook` and the same Role and RoleBinding,
       in `vllm-notebook`). Do this only after step 2: the old pod's image is pulled through that ImageStream.
  - Do not delete the Deployment instead: these apps sync automatically without self-heal, so Argo CD recreates a
    deleted Deployment only on the next commit or a manual sync, and the server stays down until then.
  - A fresh install with `image.override` set from the start needs none of this: no BuildConfig, ImageStream,
    trigger or rebuild Role is ever created.
- **The image did not change after a push, or the first build failed.**
  - Why: ConfigChange triggers fire only when the BuildConfig is created. `rebuild-on-commit` starts a new build
    only after a push changes the build's folder, and it never retries a failed build on its own.
  - Fix: read the build log with `oc logs -f bc/ray-serve-llm -n vllm-inference`, fix the cause, then start the
    build by hand: `oc start-build ray-serve-llm -n vllm-inference --follow` (for the notebook:
    `oc start-build notebook -n vllm-notebook --follow`).
- **The `ray-serve-llm` Argo CD app shows OutOfSync.**
  - Why: the image trigger rewrites the Deployment's container image to a digest. The app entry in
    `variants/prod/values-prod.yaml` handles this with `ignoreDifferences` on the triggered container's image
    (`.spec.template.spec.containers[] | select(.name == "server") | .image`, and `"notebook"` for the notebook app)
    and `.spec.paused` (BuildConfig), plus `RespectIgnoreDifferences=true`.
  - Fix: if you add another image-triggered resource, add the same rules.
- **`make install` fails after about an hour, with apps not Healthy.**
  - Why: the installer waits at most 60 minutes, and a first install can take longer. Everything before the wait
    has already run: the pattern is installed and the secrets are loaded.
  - Fix: watch progress with `oc get builds -n vllm-inference` and `oc get pods -n vllm-inference`, then run
    `./pattern.sh make argo-healthcheck`. To skip the wait on a later install, run
    `VP_SKIP_ARGO_HEALTHCHECK=true ./pattern.sh make install`, then run `make argo-healthcheck` once things
    settle.
- **The server pod stays Pending with "Insufficient nvidia.com/gpu".**
  - Why: no node advertises `nvidia.com/gpu` yet. The chain is: the MachineSet job (AWS) → the node joins → NFD
    labels it → the GPU Operator installs the driver and device plugin. The ClusterPolicy must tolerate the
    `nvidia.com/gpu` taint. Argo CD sync waves do not order apps in this framework; apps retry instead. With
    `global.gpuOperators: reuse` that ClusterPolicy is the cluster's own: check that `oc get clusterpolicies` shows it
    `ready`, its `spec.daemonsets.tolerations`, and that it offers `nvidia.com/gpu` (not a MIG or `.shared` name),
    in the namespace its pods run in.
  - Check:
    ```sh
    oc get nodes -l nvidia.com/gpu.present=true
    oc get pods -n nvidia-gpu-operator
    oc describe node <gpu-node> | grep nvidia.com/gpu
    ```
- **Right after the install, ClusterSecretStore `vault-backend` reports "Vault is sealed" (code 503) or a 403,
  and ExternalSecret `huggingface-secret` reports `ClusterSecretStore "vault-backend" is not ready`.**
  - Why: the External Secrets Operator checks the store as soon as it exists, and Vault starts sealed. The
    framework's own `unsealvault-cronjob` (namespace `imperative`, every 5 minutes; separate from the pattern's
    imperative CronJob, which runs every 10) unseals Vault and sets up the Kubernetes login the store uses;
    `make install` waits for that, then loads the secrets. Until then the `openshift-external-secrets` and
    `ray-serve-llm` apps can show Degraded.
  - Fix: on the command-line install, none; it clears once the unseal job has run and `make install` has loaded the
    secrets. On the two live AWS installs the store turned Valid within 5 minutes of its first error, and the
    Secret was created within 7 minutes after that. If the store is still not Valid after 10 minutes, or the Secret
    is still missing after 15, check `oc get clustersecretstore vault-backend` and `oc get jobs -n imperative`. On
    the web-console install the store turns Valid by itself, but ExternalSecret `huggingface-secret` stays not Ready
    until you load the secrets ([Path 2](#path-2-openshift-web-console), step 5).
- **A gated model fails to download with 401 or 403.**
  - Why: the token is missing or not loaded. The entrypoint drops `HF_TOKEN` unless it starts with `hf_`.
  - Fix: put your `hf_` token in the secrets file, run `./pattern.sh make load-secrets`, wait for the Secret to
    refresh, then delete the server pod.

## GPU support

The server image is built for the GPU architectures in vLLM's release build list
(https://github.com/vllm-project/vllm/blob/v0.29.0/.buildkite/release-pipeline.yaml). On OpenShift we have tested only
the NVIDIA L4 on the default AWS node (README, "What this pattern was verified against").

## Uninstall

- `./pattern.sh make uninstall` is marked experimental by the framework
  (https://validatedpatterns.io/blog/2026-02-16-pattern-uninstall/). It deletes the Pattern resource, and the
  operator then removes the apps. Find the Pattern with `oc get patterns -A`.
- The Validated Patterns Operator, OpenShift GitOps and operator CRDs stay installed.
- Delete the GPU MachineSet yourself, after the pattern is gone or after `gpuMachineSet: none` has synced;
  otherwise the job may create it again:
  ```sh
  oc get machinesets -n openshift-machine-api \
    -o custom-columns=NAME:.metadata.name,TYPE:.spec.template.spec.providerSpec.value.instanceType
  oc delete machineset <name-of-the-g6.2xlarge-machineset> -n openshift-machine-api
  ```
- Check `oc get pv` for a leftover volume from claim `vllm-inference/ray-serve-llm-models`.
- With `global.gpuOperators: reuse` the pattern created no object of the cluster's NFD or GPU Operator, so its apps
  have none of them to remove (not tried on a cluster).
