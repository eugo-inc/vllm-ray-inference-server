#!/usr/bin/env bash
# Compare two inference-server images on one machine with docker, no cluster: the same runner, launches, GuideLLM
# load and analysis as the OpenShift Job (see README, "Compare two images"), with GPU power from nvidia-smi instead
# of DCGM. Needs docker, access to PyPI (or a mirror) for pairbench (step 2 below) and, for GPU sides, one NVIDIA
# GPU with the NVIDIA Container Toolkit; nothing else on the host.
#
#   scripts/compare-local.sh <config.json> <results folder>
#
# The config file is JSON with the keys of global.compare in values-global.yaml, plus "model" (the keys of
# global.model). tests/fixtures/compare-local-mock.json is a small one that needs no GPU: two GuideLLM mock
# servers with different speeds. tests/fixtures/compare-scaleout-mock.json is the same in mode scaleout: mock
# servers that take a while to start, one surge per scored launch, and a second replica at the forced step.
#
# What it does:
#   1. Creates the docker network $NETWORK if it is missing.
#   2. Installs pairbench (the comparison's sign test, Holm's correction and fewest rounds), which the GuideLLM
#      image does not have, from notebooks/requirements-compare.lock: pip in a throwaway container of the GuideLLM
#      image downloads it from PyPI (or the mirror or proxy that PIP_INDEX_URL, HTTPS_PROXY and the like name, passed
#      in when set), checks its hash, and installs it into a new temporary folder, removed when the script ends. No
#      other package is installed. A failed install stops the script with exit 3 before the runner starts; pip's own
#      message above it says why.
#   3. Runs the runner (notebooks/rhvp_compare, and that folder, both mounted read-only) in a container of the
#      GuideLLM image on that network, with the docker socket mounted. The runner starts each side and the nvidia-smi
#      sampler as sibling containers on the same network, one launch at a time, and runs GuideLLM as its own child
#      process, as the OpenShift Job does.
#   4. Prints where results.json and results.md are. The exit code is the runner's: 0 when every launch finished,
#      1 for a partial run, 2 for a config that cannot run, 3 for a run that failed outside a launch (or pairbench
#      that could not be installed, step 2), 4 when the server container (SERVER_CONTAINER) could not be started
#      again, 5 when another comparison was running.
#
# One comparison at a time: the runner container carries the label rhvp-compare.runner, and a run refuses to start
# while an older one is running. Stopping the runner (docker stop, or Ctrl-C on this script) cuts the current launch
# short, takes its side down and saves the results as partial, within STOP_TIMEOUT.
#
# Each side runs as an arbitrary UID in group 0 (as on OpenShift), with its caches on a fresh tmpfs and the model
# weights on the docker volume $MODELS_VOLUME, kept between runs (remove it with docker volume rm). A side with
# compileCache: warm gets its own cache volume instead (named after $CACHE_VOLUME and the run), filled by its warm-up
# launch and removed when the run ends.
#
# Mode scaleout (mode: scaleout in the config): each scored launch starts a second container of the side, named
# like the first plus -2, at the forced step; the runner puts a small router in front of the two (router.py, a child
# process of the runner, like GuideLLM). Both replicas get GPUS unless REPLICA_GPUS gives each its own; the run is
# refused when that does not match scaleOut.sharedGpu (true: one GPU for both, false: a GPU of its own for each).
#
# The runner container's open-file limit is RUNNER_OPEN_FILES, soft and hard (65536 by default; docker's default soft
# limit was 1024 on the machine measured, and depends on the docker daemon's settings), for GuideLLM and the router:
# the router holds two connections for each request in flight, so at 1024 it stopped accepting at about 509 in flight.
# Set but empty, it leaves docker's default: the router still raises its own soft limit to the hard one, but GuideLLM
# keeps the soft limit, so at 1024 it can hold at most about 1,000 connections. A value that is not a whole number
# above 0 stops the script before docker runs (exit 2). If docker refuses to start the runner with "error setting
# rlimit", set RUNNER_OPEN_FILES to a lower number, or set it but empty.
#
# Settings (environment variables):
#   LOAD_IMAGE      the GuideLLM image the runner runs in (default: the pinned release); set it to the same
#                   image when the config sets load.image
#   NETWORK         docker network (default rhvp-compare)
#   DOCKER_SOCKET   the docker daemon's socket, mounted into the runner at /var/run/docker.sock (default
#                   /var/run/docker.sock); it must be the daemon the docker command here talks to
#   MODELS_VOLUME   docker volume for /models (default rhvp-compare-models)
#   GPUS            "all" (default) or a device id such as 0; give one id on a machine with several GPUs
#   REPLICA_GPUS    mode scaleout: one device id per replica, comma-separated (for example 0,1, or two MIG
#                   devices such as 0:0,0:1), for scaleOut.sharedGpu: false; default empty, every replica gets
#                   GPUS (one GPU shared, as scaleOut.sharedGpu: true expects)
#   CPUSET          CPUs the runner and every container it starts may use, as docker writes them (for example
#                   0-3; default all)
#   CACHE_VOLUME    name prefix of the warm compile-cache volumes (default rhvp-compare-cache)
#   HF_TOKEN        passed on when it starts with hf_ (gated models)
#   RUNNER_MEMORY   memory limit of the runner container, GuideLLM included (default 8g)
#   RUNNER_OPEN_FILES  open-file limit of the runner container, soft and hard: a whole number above 0 (default
#                   65536), or empty for docker's default, which GuideLLM then keeps (see above)
#   SIDE_MEMORY     memory limit of each side's container, as docker writes it (default none, for example 32g)
#   SERVER_CONTAINER  a container that normally holds the GPU: stopped for the run, started again after it (one
#                   that stopped by itself before the run took it is left as it is; results.md says so)
#   STOP_TIMEOUT    seconds the runner has to stop cleanly after docker stop (default 120)
set -euo pipefail

if [ "$#" -ne 2 ]; then
  sed -n '2,13p' "$0" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
mkdir -p "$2"
RESULTS="$(cd "$2" && pwd)"
NETWORK="${NETWORK:-rhvp-compare}"
MODELS_VOLUME="${MODELS_VOLUME:-rhvp-compare-models}"
GPUS="${GPUS:-all}"
REPLICA_GPUS="${REPLICA_GPUS:-}"
CPUSET="${CPUSET:-}"
CACHE_VOLUME="${CACHE_VOLUME:-rhvp-compare-cache}"
RUNNER_MEMORY="${RUNNER_MEMORY:-8g}"
RUNNER_OPEN_FILES="${RUNNER_OPEN_FILES-65536}"  # no colon: set but empty stays empty (docker's default)
SIDE_MEMORY="${SIDE_MEMORY:-}"
SERVER_CONTAINER="${SERVER_CONTAINER:-}"
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"
RUNNER_NAME="rhvp-compare-runner-$$"
SOCKET="${DOCKER_SOCKET:-/var/run/docker.sock}"

[ -f "$CONFIG" ] || { echo "no config file at $CONFIG" >&2; exit 2; }
[ -S "$SOCKET" ] || { echo "no docker socket at $SOCKET" >&2; exit 2; }
case "$RUNNER_OPEN_FILES" in
  '') ;;
  0* | *[!0-9]*)
    echo "RUNNER_OPEN_FILES must be a whole number above 0, or empty for docker's default; got: $RUNNER_OPEN_FILES" >&2
    exit 2 ;;
esac

# The pinned GuideLLM release (GUIDELLM_IMAGE in notebooks/rhvp_compare/plan.py; a test keeps the two equal).
LOAD_IMAGE="${LOAD_IMAGE:-ghcr.io/vllm-project/guidellm:v0.8.1@sha256:015b22441020e6014b4c06d89882921eb29e9b9a6f2e274b253afef4029aa081}"
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null

token_env=()
case "${HF_TOKEN:-}" in hf_*) token_env=(-e HF_TOKEN) ;; esac
cpuset_flag=()
[ -z "$CPUSET" ] || cpuset_flag=(--cpuset-cpus "$CPUSET")
ulimit_flag=()
[ -z "$RUNNER_OPEN_FILES" ] || ulimit_flag=(--ulimit "nofile=$RUNNER_OPEN_FILES:$RUNNER_OPEN_FILES")

# pairbench for the runner (step 2 above): hash-checked, wheels only, nothing else, into a folder of its own.
# NVIDIA_VISIBLE_DEVICES=void keeps a GPU out of this container where docker's default runtime is NVIDIA's.
# pip's index and proxy settings, where set, go into it, for a machine that reaches PyPI through a mirror or a proxy.
PYTHON_DEPS="$(mktemp -d "${TMPDIR:-/tmp}/rhvp-compare-pairbench.XXXXXX")"
trap 'rm -rf "$PYTHON_DEPS"' EXIT
pip_env=()
for name in PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST HTTPS_PROXY https_proxy HTTP_PROXY http_proxy \
  NO_PROXY no_proxy; do
  [ -z "${!name:-}" ] || pip_env+=(-e "$name")
done
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e NVIDIA_VISIBLE_DEVICES=void "${pip_env[@]}" \
  -v "$REPO/notebooks/requirements-compare.lock:/requirements-compare.lock:ro" \
  -v "$PYTHON_DEPS:/deps" \
  --entrypoint python "$LOAD_IMAGE" \
  -m pip install --disable-pip-version-check --no-cache-dir --no-deps --only-binary :all: --require-hashes \
  --target /deps -r /requirements-compare.lock >&2 \
  || { echo "could not install pairbench from notebooks/requirements-compare.lock (pip says why above)" >&2; exit 3; }

docker run --rm --name "$RUNNER_NAME" --network "$NETWORK" \
  --label rhvp-compare.runner=1 --memory "$RUNNER_MEMORY" --stop-timeout "$STOP_TIMEOUT" --init \
  "${ulimit_flag[@]}" \
  "${cpuset_flag[@]}" \
  --user "$(id -u):$(id -g)" --group-add "$(stat -c %g "$SOCKET")" \
  -v "$SOCKET:/var/run/docker.sock" \
  -v "$RESULTS:/results" \
  -v "$CONFIG:/config/compare.json:ro" \
  -v "$REPO/notebooks/rhvp_compare:/runner/rhvp_compare:ro" \
  -v "$PYTHON_DEPS:/runner-deps:ro" \
  -e PYTHONPATH=/runner:/runner-deps \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp/home \
  -e HF_HOME=/tmp/hf \
  -e RHVP_COMPARE_NETWORK="$NETWORK" \
  -e RHVP_COMPARE_HOST_RESULTS="$RESULTS" \
  -e RHVP_COMPARE_MODELS_VOLUME="$MODELS_VOLUME" \
  -e RHVP_COMPARE_GPUS="$GPUS" \
  -e RHVP_COMPARE_REPLICA_GPUS="$REPLICA_GPUS" \
  -e RHVP_COMPARE_CPUSET="$CPUSET" \
  -e RHVP_COMPARE_CACHE_VOLUME="$CACHE_VOLUME" \
  -e RHVP_COMPARE_MEMORY="$SIDE_MEMORY" \
  -e RHVP_COMPARE_SERVER_CONTAINER="$SERVER_CONTAINER" \
  -e RHVP_COMPARE_RUNNER_NAME="$RUNNER_NAME" \
  "${token_env[@]}" \
  -w /runner --entrypoint python "$LOAD_IMAGE" \
  -m rhvp_compare run --backend docker --config /config/compare.json --results /results
