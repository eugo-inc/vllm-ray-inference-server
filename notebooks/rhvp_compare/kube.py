# SPDX-License-Identifier: Apache-2.0
"""The OpenShift backend: each launch's side is a pod in the server namespace; the runner is a Job in the notebook
namespace (charts/all/notebook/templates/compare-*.yaml) and talks to the Kubernetes API with its service account
token through plain httpx (no Kubernetes client package).

A side pod runs like the pattern's own server: the restricted SCC's arbitrary UID in group 0, one GPU (request =
limit) with a toleration for the GPU taint, the CPU type's nodeSelector, the model PVC at /models, fresh emptyDirs
at /cache and /tmp/r, a memory-backed /dev/shm, and its own time limit (activeDeadlineSeconds), so a side outlives a
runner that was killed outright by one launch at most.

The live server. Before the first launch the runner scales the pattern's own server Deployment to 0, so the GPU and
the ReadWriteOnce model PVC are free, and when the run ends (complete, partial, stopped or failed) it scales it back
(to RHVP_COMPARE_SERVER_REPLICAS, default 1), trying again for about a minute and reading the scale back to check.
On a stop it does that first, before the side is taken down: the server's pod waits for the GPU. Argo CD ignores the
Deployment's replicas (variants/prod), so a sync during the run does not bring the server back onto the GPU. The
runner also waits until no other GPU pod is left in the namespace before each launch.

The server's autoscaler. With global.scaleOut.mode keda the chart names the server's ScaledObject in
RHVP_COMPARE_SERVER_SCALED_OBJECT. Before the scale to 0 the runner pauses it (annotation
autoscaling.keda.sh/paused-replicas: "0"), so the autoscaler cannot bring the server back mid-run, and when the run
ends it removes the pause before it scales the server back; the scale it reads back may then be above
RHVP_COMPARE_SERVER_REPLICAS, which the autoscaler is free to choose. A ScaledObject that does not exist is left
alone.

Mode scaleout (the scale round, see runner.py). The side's first replica is the launch's side pod; at the forced
step the runner asks for a second replica, a pod of its own (cmp-<run>-<NN>-<side>-2, label rhvp-compare/replica 2)
with the same image, arguments, environment and volumes, pinned to the node the first replica runs on. The router
(router.py) and GuideLLM run as child processes of the runner, in its pod, and reach the replicas by pod IP, as
GuideLLM does in mode levels. Each replica asks for one nvidia.com/gpu, so two replicas share one GPU only through
the GPU Operator's time-slicing. Before anything changes (`check_plan`) the runner lists the GPU nodes and refuses
the plan unless one fits scaleOut.sharedGpu, read from GPU feature discovery's labels: true needs a node with one GPU
(nvidia.com/gpu.count 1) offered as two time-sliced shares (nvidia.com/gpu.replicas 2, and
nvidia.com/gpu.sharing-strategy time-slicing or absent, not mps), so both replicas are on that GPU and, while both
run, nothing else has a share of it; false needs a node with two GPUs or more and no sharing (shares 1 or the label
absent, and the strategy none or absent). The node must also offer at least two nvidia.com/gpu to pods
(status.allocatable): a device plugin that renames time-sliced shares to nvidia.com/gpu.shared keeps the labels but
offers none (`node_fits`). Every GPU side pod of the mode carries a required node affinity for the labels, so the
scheduler picks such a node. A second replica that has no node REPLICA_PENDING_S after it was asked for fails the
launch with the scheduler's message (the GPU share it needs is held by something else, or the node is full). The
GPU UUIDs are not read here (no exec into the side pods), so the server records carry no `gpus`; the run's notes say
how the replicas share the GPU. A pod spec has no open-file setting, so before the first launch the runner raises its
own soft open-file limit to the hard one, which GuideLLM, its child process, inherits (the router does the same and
logs both); the notes give the limit, and warn when it is below what the surge could need (`open_files_needed`).

One run at a time. Before it changes anything the runner lists the runner pods in its own namespace
(app.kubernetes.io/name=image-compare): when another one is Pending or Running and is older (or as old, with a name
that sorts first), this run refuses to start. The oldest always wins, and nothing is left behind to go stale.

One node. Every side pod after the first is pinned to the node the first one ran on (its kubernetes.io/hostname
label), so all launches use the same GPU node; the results record every node a launch ran on. A twin's pod is its
side's pod under another name (cmp-<run>-<NN>-a-same).

Progress. After every launch the runner writes ConfigMap compare-<run id> in its own namespace: status.json (see
summary.py) and result.json, the whole results file while it fits.

The runner reaches a side by its pod IP (no Service). GuideLLM runs as a child process of the runner (see
guidellm.py), so its report lands straight on the results PVC.

The Hugging Face token goes into a side only by reference (secretKeyRef), and only when the Secret holds a real
token (it starts with hf_): the pattern's Secret holds "none" when no token was loaded, and some images would send
that to the Hub as a token. The runner reads the Secret to decide and never prints or stores it.

GPU power comes from DCGM through the cluster's monitoring (energy.py). Settings come from environment variables the
chart sets; the defaults match the pattern's names (AGENTS.md, "Names that must stay in sync").
"""

from __future__ import annotations

import base64
import json
import math
import os
import socket
import ssl
import time
from pathlib import Path
from collections.abc import Callable, Mapping

import httpx

from . import energy, guidellm, plan as plans, router, runner

SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")
SIDE_APP = "image-compare-side"
RUNNER_APP = "image-compare"
LABEL_RUN = "rhvp-compare/run"
PART_OF = "vllm-ray-inference-server"
#: Labels of the run's status ConfigMap.
STATUS_COMPONENT = "compare-run"
STATUS_RUN_LABEL = "compare-run-id"
#: Waiting reasons that never clear by themselves.
FATAL_WAITING = ("InvalidImageName", "CreateContainerConfigError", "CreateContainerError")
#: Waiting reasons that may clear (a slow or flaky registry), up to the pending limit.
PULL_WAITING = ("ErrImagePull", "ImagePullBackOff")


#: Timeouts, in seconds. Once a stop has begun (`hurry`) every call gets the short ones, so what is left after the
#: server is asked back fits the runner pod's grace period (see `stop_budget_seconds`).
API_TIMEOUT_S, LOG_TIMEOUT_S, DELETE_WAIT_S = 30.0, 60.0, 180.0
HURRY_TIMEOUT_S, HURRY_DELETE_WAIT_S = 10.0, 30.0
#: Waits between the tries to scale the server back, and how long the tries may go on (counted from the first);
#: a later call after that makes one try more.
RESTORE_BACKOFF_S = (2.0, 4.0, 8.0, 16.0)
RESTORE_WINDOW_S = 60.0
#: How long GuideLLM may take to stop (guidellm._stop: SIGTERM, 15 s, SIGKILL, 15 s).
GUIDELLM_STOP_S = 30.0
#: The annotation that pauses a KEDA ScaledObject at a fixed replica count.
KEDA_PAUSED = "autoscaling.keda.sh/paused-replicas"
#: Mode scaleout: GPU feature discovery's node labels for the GPUs a node has, the shares each is offered as, and how
#: they are shared (time-slicing, mps or none; a node without it is judged by the other two), the one GPU resource the
#: mode places replicas by, and the pod label of a replica's number.
GPU_COUNT_LABEL, GPU_REPLICAS_LABEL = "nvidia.com/gpu.count", "nvidia.com/gpu.replicas"
GPU_SHARING_LABEL = "nvidia.com/gpu.sharing-strategy"
#: The values of GPU_SHARING_LABEL each setting of scaleOut.sharedGpu accepts (None: the label absent).
GPU_SHARING_OK = {True: (None, "time-slicing"), False: (None, "none")}
WHOLE_GPU_RESOURCE = "nvidia.com/gpu"
LABEL_REPLICA = "rhvp-compare/replica"
#: Mode scaleout: how long the second replica may wait for a node. The first replica runs on that node with the
#: other share, so the scheduler's answer does not change by waiting longer.
REPLICA_PENDING_S = 120.0
#: How long the router may take to stop (router.Process.stop: SIGTERM, 10 s, SIGKILL, 10 s).
ROUTER_STOP_S = 20.0
#: Mode scaleout: the file handles the router may hold besides two sockets for each request in flight (its listening
#: socket, its log, the event loop's own), for `open_files_needed`.
ROUTER_SPARE_FILES = 64


def stop_budget_seconds(replicas: int = 1) -> float:
    """The longest a stop inside a launch can take with every API call at its short timeout: GuideLLM stopped, the
    server scaled back (tries in the window, the last one started at its end, then one more after the teardown;
    each try the autoscaler's unpause, a PATCH and a read), the side's teardown (its log, its pod and node, the delete
    and the wait), two saves with their status writes, and the cleanup's list and delete. A scale launch (mode
    scaleout, `replicas` 2) also stops the router and takes each added replica down (its log, the delete and the
    wait). terminationGracePeriodSeconds in the notebook chart must be above it, for each mode (tests/test_charts.py
    checks)."""
    t = HURRY_TIMEOUT_S
    restore = RESTORE_WINDOW_S + 2 * 3 * t
    teardown = t + 2 * t + t + HURRY_DELETE_WAIT_S + t
    added = (replicas - 1) * (t + t + HURRY_DELETE_WAIT_S + t)
    router = ROUTER_STOP_S if replicas > 1 else 0.0
    saves = 2 * 2 * t
    cleanup = 2 * t
    return GUIDELLM_STOP_S + router + restore + teardown + added + saves + cleanup


class Kube:
    """The few Kubernetes API calls the backend needs. Error texts carry the HTTP status, never a URL or a token."""

    def __init__(self, base_url: str | None = None, token: str | None = None, verify=None,
                 transport: httpx.BaseTransport | None = None, token_path: Path = SERVICE_ACCOUNT / "token") -> None:
        if base_url is None:
            host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
            port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
            base_url = f"https://[{host}]:{port}" if ":" in host else f"https://{host}:{port}"
        if verify is None:
            ca = SERVICE_ACCOUNT / "ca.crt"
            verify = ssl.create_default_context(cafile=str(ca)) if ca.is_file() else True
        self.timeout_s, self.log_timeout_s = API_TIMEOUT_S, LOG_TIMEOUT_S
        self.hurried = False
        self.http = httpx.Client(base_url=base_url, auth=energy.bearer(token_path, token), verify=verify,
                                 timeout=API_TIMEOUT_S, transport=transport)

    def hurry(self) -> None:
        """Short timeouts from now on: a stop has begun."""
        self.hurried = True
        self.timeout_s = self.log_timeout_s = HURRY_TIMEOUT_S

    def _call(self, method: str, path: str, what: str, ok_missing: bool = False, **kwargs) -> httpx.Response | None:
        kwargs["timeout"] = kwargs.get("timeout", self.timeout_s)
        if self.hurried:
            kwargs["timeout"] = min(kwargs["timeout"], HURRY_TIMEOUT_S)
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise runner.RoundError(f"the Kubernetes API did not answer while trying to {what} "
                                    f"({type(error).__name__})") from None
        if response.status_code == 404 and ok_missing:
            return None
        if response.status_code >= 400:
            try:
                reason = response.json().get("message", "")
            except ValueError:
                reason = ""
            raise runner.RoundError(f"could not {what}: HTTP {response.status_code} {reason}".strip())
        return response

    def pods(self, namespace: str, selector: str | None = None) -> list[dict]:
        params = {"labelSelector": selector} if selector else {}
        response = self._call("GET", f"/api/v1/namespaces/{namespace}/pods", f"list pods in {namespace}",
                              params=params)
        return list(response.json().get("items") or [])

    def pod(self, namespace: str, name: str) -> dict | None:
        response = self._call("GET", f"/api/v1/namespaces/{namespace}/pods/{name}", f"read pod {name}",
                              ok_missing=True)
        return None if response is None else response.json()

    def create_pod(self, namespace: str, body: Mapping) -> dict:
        return self._call("POST", f"/api/v1/namespaces/{namespace}/pods", f"create pod in {namespace}",
                          json=dict(body)).json()

    def delete_pod(self, namespace: str, name: str, grace_s: int = 30) -> None:
        self._call("DELETE", f"/api/v1/namespaces/{namespace}/pods/{name}", f"delete pod {name}", ok_missing=True,
                   params={"gracePeriodSeconds": str(grace_s)})

    def log(self, namespace: str, name: str, container: str) -> str:
        response = self._call("GET", f"/api/v1/namespaces/{namespace}/pods/{name}/log", f"read the log of {name}",
                              params={"container": container, "timestamps": "true"}, timeout=self.log_timeout_s)
        return response.text

    def node(self, name: str) -> dict | None:
        response = self._call("GET", f"/api/v1/nodes/{name}", f"read node {name}", ok_missing=True)
        return None if response is None else response.json()

    def nodes(self, selector: str) -> list[dict]:
        response = self._call("GET", "/api/v1/nodes", "list the nodes", params={"labelSelector": selector})
        return list(response.json().get("items") or [])

    def secret_value(self, namespace: str, name: str, key: str) -> str | None:
        response = self._call("GET", f"/api/v1/namespaces/{namespace}/secrets/{name}", f"read secret {name}",
                              ok_missing=True)
        if response is None:
            return None
        raw = (response.json().get("data") or {}).get(key)
        try:
            return base64.b64decode(raw).decode("utf-8").strip() if raw else None
        except (ValueError, UnicodeDecodeError):
            return None

    def replicas(self, namespace: str, deployment: str) -> int | None:
        response = self._call("GET", f"/apis/apps/v1/namespaces/{namespace}/deployments/{deployment}/scale",
                              f"read the scale of {deployment}", ok_missing=True)
        return None if response is None else int((response.json().get("spec") or {}).get("replicas") or 0)

    def scale(self, namespace: str, deployment: str, replicas: int) -> None:
        self._call("PATCH", f"/apis/apps/v1/namespaces/{namespace}/deployments/{deployment}/scale",
                   f"scale {deployment} to {replicas}", content=json.dumps({"spec": {"replicas": replicas}}),
                   headers={"Content-Type": "application/merge-patch+json"})

    def pause_scaled_object(self, namespace: str, name: str, paused: bool) -> bool:
        """Pause a KEDA ScaledObject at 0 replicas, or remove the pause; False when it does not exist."""
        value = "0" if paused else None
        response = self._call("PATCH", f"/apis/keda.sh/v1alpha1/namespaces/{namespace}/scaledobjects/{name}",
                              f"{'pause' if paused else 'resume'} the autoscaler {name}", ok_missing=True,
                              content=json.dumps({"metadata": {"annotations": {KEDA_PAUSED: value}}}),
                              headers={"Content-Type": "application/merge-patch+json"})
        return response is not None

    def put_configmap(self, namespace: str, name: str, labels: Mapping[str, str],
                      data: Mapping[str, str | None]) -> None:
        """Create the ConfigMap, or merge `data` into it (a None value removes that key)."""
        patched = self._call("PATCH", f"/api/v1/namespaces/{namespace}/configmaps/{name}", f"update ConfigMap {name}",
                             ok_missing=True, content=json.dumps({"data": dict(data)}),
                             headers={"Content-Type": "application/merge-patch+json"})
        if patched is None:
            self._call("POST", f"/api/v1/namespaces/{namespace}/configmaps", f"create ConfigMap {name}",
                       json={"apiVersion": "v1", "kind": "ConfigMap",
                             "metadata": {"name": name, "labels": dict(labels)},
                             "data": {k: v for k, v in data.items() if v is not None}})

    def cluster_version(self) -> str | None:
        response = self._call("GET", "/apis/config.openshift.io/v1/clusterversions/version",
                              "read the cluster version", ok_missing=True)
        if response is None:
            return None
        return ((response.json().get("status") or {}).get("desired") or {}).get("version")


def _condition(pod: Mapping, kind: str) -> Mapping:
    return next((c for c in (pod.get("status") or {}).get("conditions") or [] if c.get("type") == kind), {})


def _container_status(pod: Mapping, name: str = "server") -> Mapping:
    return next((c for c in (pod.get("status") or {}).get("containerStatuses") or [] if c.get("name") == name), {})


def _int(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def gpu_facts(labels: Mapping[str, str]) -> dict | None:
    """The GPU, from the NVIDIA GPU feature discovery labels on its node."""
    product = labels.get("nvidia.com/gpu.product")
    if not product:
        return None
    driver = labels.get("nvidia.com/cuda.driver-version.full")
    if not driver and labels.get("nvidia.com/cuda.driver.major"):
        driver = ".".join(labels.get(f"nvidia.com/cuda.driver.{p}", "") for p in ("major", "minor", "rev")).strip(".")
    cuda = labels.get("nvidia.com/cuda.runtime-version.full")
    if not cuda and labels.get("nvidia.com/cuda.runtime.major"):
        cuda = f"{labels['nvidia.com/cuda.runtime.major']}.{labels.get('nvidia.com/cuda.runtime.minor', '0')}"
    return {"product": product, "driver": driver or None, "cuda_driver": cuda or None,
            "memory_mib": _int(labels.get("nvidia.com/gpu.memory")), "count": _int(labels.get("nvidia.com/gpu.count")),
            "source": "node labels (NVIDIA GPU feature discovery)"}


def node_facts(node: Mapping) -> dict:
    labels = (node.get("metadata") or {}).get("labels") or {}
    info = (node.get("status") or {}).get("nodeInfo") or {}
    return {"name": (node.get("metadata") or {}).get("name"), "arch": labels.get("kubernetes.io/arch"),
            "instance_type": labels.get("node.kubernetes.io/instance-type"), "os_image": info.get("osImage"),
            "kernel": info.get("kernelVersion"), "container_runtime": info.get("containerRuntimeVersion")}


def image_digest(pod: Mapping) -> str | None:
    """repository@sha256:... of the side's container, as the node pulled it. When the runtime reports no repository
    digest (an imageID without "@sha256:"), the image the pod asked for, if that is pinned by digest; else None."""
    image_id = _container_status(pod).get("imageID") or ""
    for prefix in ("docker-pullable://", "docker://"):
        image_id = image_id.removeprefix(prefix)
    if "@sha256:" in image_id:
        return image_id
    asked = next((c.get("image") or "" for c in (pod.get("spec") or {}).get("containers") or []
                  if c.get("name") == "server"), "")
    return asked if "@sha256:" in asked else None


def _own_namespace() -> str | None:
    path = SERVICE_ACCOUNT / "namespace"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def side_deadline(plan: Mapping, launch: Mapping) -> int:
    """A side pod's own time limit: the start, twice GuideLLM's time limit, the energy settle time and ten minutes
    more. A pod serves one load: after a GuideLLM time-out the runner takes it down and the launch's second try starts
    a new pod, so the second load time is slack. In a scale launch (mode scaleout, a scored launch) the first replica
    also serves while the forced step passes and the second replica starts (up to its whole start time limit), so
    both replicas get those on top. The runner takes every side down itself; this only bounds a side whose runner
    was killed."""
    server, load = plan["server"], plan["load"]
    start = server["warmup_timeout_s"] if launch["kind"] == "warmup" else server["ready_timeout_s"]
    deadline = start + 2 * load["timeout_s"] + plan["energy"]["settle_s"] + 600
    if plan.get("mode") == plans.SCALEOUT and launch["kind"] != plans.WARMUP:
        deadline += plan["scale_out"]["step_after_s"] + server["ready_timeout_s"]
    return int(deadline)


def gpu_shares(labels: Mapping[str, str]) -> tuple[int | None, int]:
    """(GPUs, shares per GPU) of a node, from GPU feature discovery's labels: shares is 1 without time-slicing
    (label absent)."""
    count = _int(labels.get(GPU_COUNT_LABEL))
    shares = _int(labels.get(GPU_REPLICAS_LABEL, "1"))
    return count, shares if shares is not None else 0


def share_fits(labels: Mapping[str, str], shared: bool) -> bool:
    """Whether a node's labels let it hold a scale launch's two replicas as scaleOut.sharedGpu declares: shared, one
    GPU offered as exactly two time-sliced shares; not shared, two GPUs or more, not shared at all (no time-slicing,
    no MPS). A node without the sharing-strategy label passes on the other two."""
    count, shares = gpu_shares(labels)
    if count is None or labels.get(GPU_SHARING_LABEL) not in GPU_SHARING_OK[shared]:
        return False
    return count == 1 and shares == 2 if shared else count >= 2 and shares == 1


def gpu_allocatable(node: Mapping) -> int | None:
    """How many nvidia.com/gpu a node offers to pods (status.allocatable). None when it offers none, as when the
    device plugin renames time-sliced shares to nvidia.com/gpu.shared (renameByDefault: true)."""
    return _int(((node.get("status") or {}).get("allocatable") or {}).get(WHOLE_GPU_RESOURCE))


def node_fits(node: Mapping, shared: bool) -> bool:
    """`share_fits` on a node's labels, and the node offers the two replicas their nvidia.com/gpu (2 or more
    allocatable). The node affinity can ask only for the labels; the scheduler checks the rest when it places a pod."""
    labels = (node.get("metadata") or {}).get("labels") or {}
    return share_fits(labels, shared) and (gpu_allocatable(node) or 0) >= 2


def node_offer(node: Mapping) -> str:
    """What a GPU node offers a scale launch, in a refusal: its GPUs, shares, sharing strategy and allocatable."""
    meta = node.get("metadata") or {}
    labels = meta.get("labels") or {}
    count, shares = gpu_shares(labels)
    return (f"{meta.get('name')}: {count} GPU, {shares} share(s) each, sharing strategy "
            f"{labels.get(GPU_SHARING_LABEL) or 'not labeled'}, {gpu_allocatable(node) or 0} {WHOLE_GPU_RESOURCE} "
            "allocatable")


def share_affinity(shared: bool) -> dict:
    """The required node affinity of a scale launch's GPU side pods: a node whose labels fit `share_fits`."""
    if shared:
        gpus = [[{"key": GPU_COUNT_LABEL, "operator": "In", "values": ["1"]},
                 {"key": GPU_REPLICAS_LABEL, "operator": "In", "values": ["2"]}]]
    else:
        several = {"key": GPU_COUNT_LABEL, "operator": "Gt", "values": ["1"]}
        gpus = [[several, {"key": GPU_REPLICAS_LABEL, "operator": "DoesNotExist"}],
                [several, {"key": GPU_REPLICAS_LABEL, "operator": "In", "values": ["1"]}]]
    sharing = [{"key": GPU_SHARING_LABEL, "operator": "DoesNotExist"} if value is None else
               {"key": GPU_SHARING_LABEL, "operator": "In", "values": [value]} for value in GPU_SHARING_OK[shared]]
    terms = [{"matchExpressions": [*gpu, strategy]} for gpu in gpus for strategy in sharing]
    return {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": terms}}}


def share_text(shared: bool) -> str:
    return ("one GPU offered as 2 time-sliced shares (nvidia.com/gpu.count 1, nvidia.com/gpu.replicas 2, "
            "nvidia.com/gpu.sharing-strategy time-slicing or not set, 2 nvidia.com/gpu allocatable)" if shared
            else "2 GPUs or more, not shared (nvidia.com/gpu.count 2 or more, nvidia.com/gpu.replicas 1 or not set, "
            "nvidia.com/gpu.sharing-strategy none or not set, 2 nvidia.com/gpu allocatable or more)")


def raise_open_files() -> tuple[str, int | None]:
    """Mode scaleout: raise the runner's soft open-file limit to its hard limit, as the router raises its own
    (router.raise_open_file_limit), so GuideLLM, which the runner starts as a child process, inherits it. Returns the
    line for the run's notes and the soft limit read back after it (None when unlimited or not known). Never raises."""
    line = router.raise_open_file_limit("the runner")
    try:
        import resource
        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except (ImportError, OSError, ValueError):
        return line, None
    return line, None if soft == resource.RLIM_INFINITY else soft


def open_files_needed(scale: Mapping) -> int:
    """An open-file limit at which the router cannot run out of handles in a surge: two sockets for each of the
    surge's requests, were all of them in flight at once, and ROUTER_SPARE_FILES."""
    return 2 * math.ceil(scale["request_rate"] * scale["surge_s"]) + ROUTER_SPARE_FILES


class DcgmSampler:
    """GPU power from DCGM in the cluster's monitoring, read after the load (start and stop do nothing)."""

    source = "dcgm"

    def __init__(self, thanos: energy.Thanos, pod: str, namespace: str, node: str | None) -> None:
        self.thanos, self.pod, self.namespace, self.node = thanos, pod, namespace, node

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def samples(self, start: float, end: float):
        power, counter, matched, note = energy.dcgm_samples(self.thanos, self.pod, self.namespace, self.node, start,
                                                            end)
        return power, counter, note if note else (f"series matched by {matched}" if matched else None)


class KubeBackend:
    platform = "openshift"

    def __init__(self, kube: Kube | None = None, env: Mapping[str, str] | None = None,
                 results_root: Path = Path("/results"), clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, thanos_factory=energy.Thanos) -> None:
        env = os.environ if env is None else env
        self.kube = kube or Kube()
        self.namespace = env.get("RHVP_COMPARE_NAMESPACE", "vllm-inference")
        self.arch = env.get("RHVP_COMPARE_ARCH", "amd64")
        self.models_pvc = env.get("RHVP_COMPARE_MODELS_PVC", "ray-serve-llm-models")
        self.hf_secret = env.get("RHVP_COMPARE_HF_SECRET", "huggingface-secret")
        self.hf_key = env.get("RHVP_COMPARE_HF_KEY", "hftoken")
        self.llm_configmap = env.get("RHVP_COMPARE_LLM_CONFIGMAP", "ray-serve-llm-config")
        self.gpu_resource = env.get("RHVP_COMPARE_GPU_RESOURCE", "nvidia.com/gpu")
        self.gpu_taint = env.get("RHVP_COMPARE_GPU_TAINT_KEY", "nvidia.com/gpu")
        self.shm = env.get("RHVP_COMPARE_SHM", "4Gi")
        self.cache = env.get("RHVP_COMPARE_CACHE", "20Gi")
        self.pending_s = float(env.get("RHVP_COMPARE_PENDING_SECONDS", "900"))
        self.gpu_wait_s = float(env.get("RHVP_COMPARE_GPU_WAIT_SECONDS", "900"))
        self.server_deployment = env.get("RHVP_COMPARE_SERVER_DEPLOYMENT", "ray-serve-llm")
        self.server_replicas = int(env.get("RHVP_COMPARE_SERVER_REPLICAS", "1"))
        self.scaled_object = env.get("RHVP_COMPARE_SERVER_SCALED_OBJECT", "").strip()
        self._paused = False
        self.status_namespace = env.get("RHVP_COMPARE_STATUS_NAMESPACE") or _own_namespace()
        self.pod_name = env.get("RHVP_COMPARE_POD_NAME") or socket.gethostname()
        try:
            self.resources = json.loads(env.get("RHVP_COMPARE_SIDE_RESOURCES") or "{}")
        except ValueError:
            raise runner.RoundError("RHVP_COMPARE_SIDE_RESOURCES is not JSON") from None
        self.results_root = results_root
        self.clock, self.sleep = clock, sleep
        self.thanos_factory = thanos_factory
        self._thanos: energy.Thanos | None = None
        self.run_id = ""
        self.use_token = False
        self._token: str | None = None
        self._scaled_down = False
        self._restored = False
        self._restore_deadline: float | None = None
        self.pin_node: str | None = None  # the kubernetes.io/hostname of the first side's node
        self.delete_wait_s = DELETE_WAIT_S
        self._first: dict | None = None  # mode scaleout: the handle of the launch's first replica
        self.notes: list[str] = []  # read by the runner after prepare()

    restore_before_teardown = True  # the server's pod waits for the GPU while the side is taken down

    # ---- before and after the run ----

    def check_alone(self) -> None:
        """Refuse to start while another runner pod in this namespace is Pending or Running and is older."""
        if not self.status_namespace:
            return
        pods = self.kube.pods(self.status_namespace, f"app.kubernetes.io/name={RUNNER_APP}")

        def order(pod: Mapping) -> tuple[str, str]:
            meta = pod.get("metadata") or {}
            return meta.get("creationTimestamp") or "", meta.get("name") or ""

        mine = next((p for p in pods if (p.get("metadata") or {}).get("name") == self.pod_name), None)
        for pod in sorted(pods, key=order):
            meta, phase = pod.get("metadata") or {}, (pod.get("status") or {}).get("phase")
            if meta.get("name") == self.pod_name or phase not in ("Pending", "Running"):
                continue
            if mine is None or order(pod) < order(mine):
                job = (meta.get("labels") or {}).get("job-name")
                raise runner.RunActive(
                    f"another comparison is running: runner pod {meta.get('name')} ({phase}"
                    f"{', Job ' + job if job else ''}) in {self.status_namespace}. This run changed nothing; start "
                    f"it again (delete its Job and sync) once that one has ended.")

    def _scale(self, plan: Mapping) -> dict | None:
        """The scale-out settings when a scale launch would place GPU replicas, else None."""
        if plan.get("mode") != plans.SCALEOUT or not any(s.get("gpu", True) for s in plan["sides"]):
            return None
        return plan["scale_out"]

    def check_plan(self, plan: Mapping) -> None:
        """Mode scaleout with a GPU: refuse (PlanError, before anything changes) a plan no GPU node of the sides' CPU
        type can hold as scaleOut.sharedGpu declares (`node_fits`), or a GPU resource other than nvidia.com/gpu."""
        scale = self._scale(plan)
        if scale is None:
            return
        shared = scale["shared_gpu"]
        if self.gpu_resource != WHOLE_GPU_RESOURCE:
            raise plans.PlanError(
                f"mode scaleout on OpenShift places each replica on a whole GPU or a time-sliced share of one "
                f"({WHOLE_GPU_RESOURCE}); compare.side.gpuResourceName is {self.gpu_resource}")
        try:
            nodes = self.kube.nodes(f"kubernetes.io/arch={self.arch}")
        except runner.RoundError as error:
            raise plans.PlanError(
                f"mode scaleout reads the GPU nodes before it starts, and {error} (in mode scaleout the Job's "
                f"ClusterRole allows list on nodes)") from None
        offered, short = [], False
        for node in nodes:
            labels = (node.get("metadata") or {}).get("labels") or {}
            if gpu_shares(labels)[0] is None:
                continue
            if node_fits(node, shared):
                return
            short = short or share_fits(labels, shared)  # the labels fit, the allocatable GPUs do not
            offered.append(node_offer(node))
        what = "; ".join(offered) or f"no node with the label {GPU_COUNT_LABEL}"
        if short and shared:
            hint = (f"offer the shares to pods as {WHOLE_GPU_RESOURCE}: a node whose labels fit offers fewer than 2, "
                    f"as when the device plugin renames time-sliced shares to {WHOLE_GPU_RESOURCE}.shared "
                    "(renameByDefault: true; charts/all/nvidia-gpu-config sets it to false) or is still restarting "
                    "after a time-slicing change (wait for it and run again)")
        elif short:
            hint = (f"use a node that offers at least 2 {WHOLE_GPU_RESOURCE} to pods: a node whose labels fit offers "
                    "fewer than 2, as when one of its GPUs is unhealthy or the device plugin is still restarting "
                    "(wait for it and run again)")
        elif shared:
            hint = ("turn on GPU time-slicing (not MPS) with 2 shares (timeSlicing.enabled: true, replicas: 2, in "
                    "charts/all/nvidia-gpu-config/values.yaml) on a node with one GPU")
        else:
            hint = ("use a node with two GPUs or more and no GPU sharing (no time-slicing, no MPS), or set "
                    "scaleOut.sharedGpu: true")
        raise plans.PlanError(
            f"scaleOut.sharedGpu is {str(shared).lower()}: both replicas of a side run on one node, which needs "
            f"{share_text(shared)}; the {self.arch} GPU nodes offer {what}. To run it, {hint}")

    def prepare(self, plan: Mapping) -> None:
        self.run_id = plan["run_id"]
        scale = self._scale(plan)
        if scale is not None:
            self.notes.append(
                f"mode scaleout on OpenShift: the two replicas of each scored launch are pods on one node with "
                f"{share_text(scale['shared_gpu'])}; their GPU UUIDs are not read on OpenShift")
        if plan.get("mode") == plans.SCALEOUT:
            line, soft = raise_open_files()
            self.notes.append(f"{line}; GuideLLM, a child process of the runner, inherits it, and the router raises "
                              "its own soft limit the same way (router.log)")
            need, requests = open_files_needed(plan["scale_out"]), math.ceil(
                plan["scale_out"]["request_rate"] * plan["scale_out"]["surge_s"])
            if soft is not None and soft < need:
                self.notes.append(
                    f"warning: the runner's open-file limit is {soft}, below {need}: two for each of the surge's "
                    f"{requests} requests, were all of them in flight at once, and {ROUTER_SPARE_FILES} more. The "
                    "router holds two for each request in flight and fails new ones once it runs out (router.log "
                    "says so); raise the open-file limit the container runtime gives pods, or lower "
                    "scaleOut.requestRate or scaleOut.surgeSeconds")
        self._delete_run_pods()  # pods left by an earlier attempt of this run; also fails early without access
        token = self.kube.secret_value(self.namespace, self.hf_secret, self.hf_key) if self.hf_secret else None
        self.use_token = bool(token and token.startswith("hf_"))
        self._token = token if self.use_token else None
        if self.server_deployment and self.kube.replicas(self.namespace, self.server_deployment) is not None:
            self._scaled_down = True  # set first: a scale that fails half-way is still undone
            if self.scaled_object:  # first, or the autoscaler could bring the server back onto the GPU
                self._paused = True
                self._paused = self.kube.pause_scaled_object(self.namespace, self.scaled_object, True)
            self.kube.scale(self.namespace, self.server_deployment, 0)

    def describe(self) -> dict:
        version = self.kube.cluster_version()
        return {"cluster": {"openshift_version": version}} if version else {}

    def _delete_run_pods(self) -> None:
        for pod in self.kube.pods(self.namespace, f"{LABEL_RUN}={self.run_id}"):
            self.kube.delete_pod(self.namespace, pod["metadata"]["name"], grace_s=0)

    def restore_server(self) -> bool | None:
        """Scale the pattern's server back and read the scale to check; tries again with waits for up to
        RESTORE_WINDOW_S from the first try. True once it is back, None when it was never scaled down."""
        if not self._scaled_down:
            return True if self._restored else None
        if self._restore_deadline is None:
            self._restore_deadline = self.clock() + RESTORE_WINDOW_S
        waits = iter(RESTORE_BACKOFF_S)
        while True:
            try:
                if self._paused:  # the autoscaler first, so it does not hold the server at 0
                    self.kube.pause_scaled_object(self.namespace, self.scaled_object, False)
                    self._paused = False
                self.kube.scale(self.namespace, self.server_deployment, self.server_replicas)
                now = self.kube.replicas(self.namespace, self.server_deployment)
                if now == self.server_replicas or (self.scaled_object and now is not None
                                                   and now >= self.server_replicas):
                    self._scaled_down, self._restored = False, True
                    return True
                last = f"its scale reads {now} replicas after the change"
            except runner.RoundError as error:
                last = str(error)
            wait = next(waits, None)
            if wait is None or self.clock() + wait > self._restore_deadline:
                break
            self.sleep(wait)
        resume = (f"oc annotate scaledobject/{self.scaled_object} -n {self.namespace} {KEDA_PAUSED}- && "
                  if self._paused else "")
        raise runner.RoundError(
            f"could not scale {self.server_deployment} back to {self.server_replicas} ({last}). Bring it back by hand: "
            f"oc delete pod -n {self.namespace} -l app.kubernetes.io/name={SIDE_APP} && {resume}oc scale "
            f"deployment/{self.server_deployment} --replicas={self.server_replicas} -n {self.namespace}")

    def on_stop(self) -> None:
        """A stop has begun: short timeouts and waits from now on. Sets flags only (it may run in the signal
        handler)."""
        self.kube.hurry()
        self.delete_wait_s = HURRY_DELETE_WAIT_S

    def cleanup(self) -> None:
        """Delete the run's side pods (the runner has asked the server back first)."""
        if self.run_id:
            self._delete_run_pods()

    def publish(self, run_id: str, status_text: str, result_text: str | None) -> None:
        if not self.status_namespace:
            return
        self.kube.put_configmap(self.status_namespace, f"compare-{run_id}",
                                {"app.kubernetes.io/component": STATUS_COMPONENT, STATUS_RUN_LABEL: run_id,
                                 "app.kubernetes.io/part-of": PART_OF},
                                {"status.json": status_text, "result.json": result_text})

    # ---- one launch ----

    def _requests_gpu(self, pod: Mapping) -> bool:
        for container in (pod.get("spec") or {}).get("containers") or []:
            resources = container.get("resources") or {}
            for part in ("limits", "requests"):
                if str((resources.get(part) or {}).get(self.gpu_resource, "0")) not in ("0", ""):
                    return True
        return False

    def gpu_free(self, plan: Mapping, side: Mapping) -> None:
        """Wait until no other pod in the namespace holds or waits for a GPU (for example the pattern's own server
        while its scale-down to 0 goes through)."""
        if not side.get("gpu", True):
            return
        deadline = self.clock() + self.gpu_wait_s
        while True:
            holders = [p["metadata"]["name"] for p in self.kube.pods(self.namespace)
                       if (p.get("status") or {}).get("phase") in ("Pending", "Running") and self._requests_gpu(p)]
            if not holders:
                return
            if self.clock() > deadline:
                raise runner.RoundError(f"other pods in {self.namespace} still hold the GPU: {', '.join(holders)}",
                                        runner.GPU_NOT_FREE)
            self.sleep(10)

    def pod_body(self, plan: Mapping, launch: Mapping, side: Mapping, spec: Mapping, replica: int = 1,
                 node: str | None = None) -> dict:
        """The side's pod; in mode scaleout replica 2 is the one the forced step adds, on `node` (the
        kubernetes.io/hostname of the first replica's node)."""
        key = launch["side"].replace("~", "-")  # plan.token: a pod name and a label value take no "~"
        name = f"cmp-{self.run_id[-8:]}-{launch['index']:02d}-{key.lower()}" + (f"-{replica}" if replica > 1 else "")
        scale = self._scale(plan) if spec.get("gpu") else None
        env = [{"name": k, "value": v} for k, v in spec["env"].items()]
        if self.use_token:
            env.append({"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": self.hf_secret, "key": self.hf_key,
                                                                          "optional": True}}})
        resources = json.loads(json.dumps(self.resources or {}))
        tolerations = []
        if spec.get("gpu"):
            for part in ("requests", "limits"):
                resources.setdefault(part, {})[self.gpu_resource] = 1
            tolerations = [{"key": self.gpu_taint, "operator": "Exists", "effect": "NoSchedule"}]
        mounts = [{"name": "models", "mountPath": "/models"}, {"name": "cache", "mountPath": "/cache"},
                  {"name": "ray-tmp", "mountPath": "/tmp/r"}, {"name": "dshm", "mountPath": "/dev/shm"}]
        volumes = [{"name": "models", "persistentVolumeClaim": {"claimName": self.models_pvc}},
                   {"name": "cache", "emptyDir": {"sizeLimit": self.cache}},
                   {"name": "ray-tmp", "emptyDir": {}},
                   {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": self.shm}}]
        if spec.get("llm_config"):
            mounts.append({"name": "config", "mountPath": "/config", "readOnly": True})
            volumes.append({"name": "config", "configMap": {"name": self.llm_configmap}})
        container: dict = {
            "name": "server", "image": side["image"], "imagePullPolicy": "IfNotPresent", "args": list(spec["args"]),
            "env": env, "ports": [{"name": "http", "containerPort": plan["server"]["port"]}],
            "resources": resources, "volumeMounts": mounts,
            "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
        }
        if spec.get("command"):
            container["command"] = list(spec["command"])
        pin = node or self.pin_node
        body = {
            "apiVersion": "v1", "kind": "Pod",
            "metadata": {"name": name, "labels": {
                "app.kubernetes.io/name": SIDE_APP, "app.kubernetes.io/part-of": PART_OF, LABEL_RUN: self.run_id,
                "rhvp-compare/side": key, "rhvp-compare/launch": str(launch["index"]),
                "rhvp-compare/round": str(launch["round"]) if launch.get("round") else "warmup"}},
            "spec": {
                "restartPolicy": "Never", "terminationGracePeriodSeconds": 30, "automountServiceAccountToken": False,
                "activeDeadlineSeconds": side_deadline(plan, launch),
                "enableServiceLinks": False,
                "securityContext": {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}},
                "nodeSelector": {"kubernetes.io/arch": self.arch,
                                 **({"kubernetes.io/hostname": pin} if pin else {})},
                "tolerations": tolerations,
                "containers": [container], "volumes": volumes,
            },
        }
        if plan.get("mode") == plans.SCALEOUT:
            body["metadata"]["labels"][LABEL_REPLICA] = str(replica)
        if scale is not None:
            body["spec"]["affinity"] = share_affinity(scale["shared_gpu"])
        return body

    def start(self, plan: Mapping, launch: Mapping, side: Mapping, spec: Mapping, replica: int = 1) -> dict:
        """Create the side's pod (replica 1), or in mode scaleout the replica the forced step adds (replica 2), on
        the node the launch's first replica runs on."""
        node = self._first_replica_node(plan, spec) if replica > 1 else None
        body = self.pod_body(plan, launch, side, spec, replica, node)
        name = body["metadata"]["name"]
        if self.kube.pod(self.namespace, name) is not None:  # left from an earlier attempt
            self._delete_and_wait(name)
        self.kube.create_pod(self.namespace, body)
        handle = {"name": name, "image": side["image"], "port": plan["server"]["port"], "gpu": bool(spec.get("gpu")),
                  "created": self.clock(), "pull_since": None, "last": None, "last_at": -1e9}
        if replica > 1:
            handle.update(replica=replica, pending_s=REPLICA_PENDING_S)
        else:
            self._first = handle
        return handle

    def _first_replica_node(self, plan: Mapping, spec: Mapping) -> str:
        """The kubernetes.io/hostname of the node the launch's first replica runs on, where the second one must run
        too (its GPU share, and the ReadWriteOnce model volume). For a GPU side, that node is checked against
        scaleOut.sharedGpu (`node_fits`; the node affinity already asked the scheduler for its labels)."""
        pod = self._pod(self._first, fresh=True) if self._first is not None else None
        node_name = ((pod or {}).get("spec") or {}).get("nodeName")
        node = self.kube.node(node_name) if node_name else None
        labels = ((node or {}).get("metadata") or {}).get("labels") or {}
        if not labels.get("kubernetes.io/hostname"):
            raise runner.RoundError("the second replica cannot be placed: the node of the first replica is not known",
                                    runner.SIDE_FAILED)
        scale = self._scale(plan) if spec.get("gpu") else None
        if scale is not None and not node_fits(node, scale["shared_gpu"]):
            raise runner.RoundError(
                f"the first replica runs on node {node_offer(node)}; scaleOut.sharedGpu "
                f"{str(scale['shared_gpu']).lower()} needs {share_text(scale['shared_gpu'])}", runner.SIDE_FAILED)
        return labels["kubernetes.io/hostname"]

    def _pod(self, handle: dict, fresh: bool = False) -> dict | None:
        """The side's pod, read at most every 2 seconds (the probes ask twice a second)."""
        if fresh or self.clock() - handle["last_at"] >= 2.0:
            handle["last"] = self.kube.pod(self.namespace, handle["name"])
            handle["last_at"] = self.clock()
        return handle["last"]

    def _problem(self, handle: dict, pod: Mapping | None) -> str | None:
        if pod is None:
            return "the side's pod is gone"
        status = pod.get("status") or {}
        phase = status.get("phase")
        state = _container_status(pod).get("state") or {}
        if "terminated" in state:
            ended = state["terminated"]
            return f"the server exited: {ended.get('reason') or 'Error'}, exit code {ended.get('exitCode')}"
        if phase in ("Failed", "Succeeded"):
            return f"the side's pod ended ({phase}: {status.get('reason') or status.get('message') or 'no reason'})"
        waiting = state.get("waiting") or {}
        reason = waiting.get("reason")
        if reason in FATAL_WAITING:
            return f"the server could not start: {reason}: {waiting.get('message', '')}".strip()
        if reason in PULL_WAITING:
            handle["pull_since"] = handle["pull_since"] or self.clock()
            if self.clock() - handle["pull_since"] > self.pending_s:
                return f"the image could not be pulled: {waiting.get('message') or reason}"
        else:
            handle["pull_since"] = None
        scheduled = _condition(pod, "PodScheduled")
        pending_s = handle.get("pending_s", self.pending_s)
        if scheduled.get("status") == "False" and self.clock() - handle["created"] > pending_s:
            who = f"replica {handle['replica']}" if handle.get("replica", 1) > 1 else "the side"
            return f"{who} never got a node in {pending_s:g} s: {scheduled.get('message') or 'unschedulable'}"
        return None

    def endpoint(self, handle: dict, deadline: float) -> str:
        while True:
            pod = self._pod(handle, fresh=True)
            problem = self._problem(handle, pod)
            if problem:
                raise runner.RoundError(problem, runner.SIDE_FAILED)
            ip = (pod.get("status") or {}).get("podIP")
            if ip:
                return f"http://[{ip}]:{handle['port']}" if ":" in ip else f"http://{ip}:{handle['port']}"
            if self.clock() > deadline:
                raise runner.RoundError("the side's pod got no IP address before the time limit",
                                        runner.SIDE_NEVER_READY)
            self.sleep(1.0)

    def alive(self, handle: dict) -> str | None:
        return self._problem(handle, self._pod(handle))

    def run_load(self, plan: Mapping, launch: Mapping, target: str, out_dir: Path) -> Path:
        # GuideLLM downloads the model's tokenizer, which needs the token for a gated model (never "none")
        return guidellm.run(plan, target, out_dir, env={"HF_TOKEN": self._token} if self._token else None)

    def start_load(self, plan: Mapping, launch: Mapping, target: str, out_dir: Path) -> guidellm.Running:
        """Mode scaleout: the surge, GuideLLM in the background against the router (the token as in run_load)."""
        return guidellm.start(plan, target, out_dir, env={"HF_TOKEN": self._token} if self._token else None)

    def router(self, plan: Mapping, targets: list[str], out_dir: Path):
        """Mode scaleout: the router (router.py) in this pod, in front of the replicas' pod IPs."""
        from . import router as routers  # its server needs aiohttp, which the runner's image has

        try:
            return routers.spawn(targets, out_dir / "router.log")
        except routers.RouterError as error:
            raise runner.RoundError(str(error)) from None

    def sampler(self, plan: Mapping, handle: dict):
        if plan["energy"]["source"] != "dcgm" or not handle.get("gpu"):
            return None
        if self._thanos is None:
            self._thanos = self.thanos_factory(plan["energy"]["thanos_url"])
        pod = self._pod(handle, fresh=True) or {}
        return DcgmSampler(self._thanos, handle["name"], self.namespace, (pod.get("spec") or {}).get("nodeName"))

    def logs(self, handle: dict) -> str:
        return self.kube.log(self.namespace, handle["name"], "server")

    def facts(self, handle: dict) -> dict:
        pod = self._pod(handle, fresh=True) or {}
        node_name = (pod.get("spec") or {}).get("nodeName")
        out: dict = {"node_name": node_name, "image_digest": image_digest(pod)}
        if node_name:
            node = self.kube.node(node_name)
            if node:
                labels = (node.get("metadata") or {}).get("labels") or {}
                out["node"] = node_facts(node)
                if handle.get("gpu"):
                    out["gpu"] = gpu_facts(labels)
                if self.pin_node is None and labels.get("kubernetes.io/hostname"):
                    self.pin_node = labels["kubernetes.io/hostname"]  # every later side runs on this node
        return out

    def _delete_and_wait(self, name: str) -> None:
        self.kube.delete_pod(self.namespace, name)
        start = self.clock()
        while self.kube.pod(self.namespace, name) is not None:
            waited = self.clock() - start
            if waited > self.delete_wait_s:  # shortened when a stop begins during the wait
                raise runner.RoundError(f"pod {name} was not gone {waited:.0f} s after it was deleted")
            self.sleep(2.0)

    def stop(self, handle: dict) -> None:
        """Delete the side and wait until it is gone, so the GPU and the model PVC are free for the next launch."""
        self._delete_and_wait(handle["name"])
