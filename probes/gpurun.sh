#!/bin/bash
# usage: gpurun.sh LABEL /probes/PROBE.py [probe args...]   (laptop GPU, max-power with trap)
set -u
label=$1; shift
trap 'echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null' EXIT
echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
# The probes run on gpu-lab's image and its training/ mount, a separate repo, so
# record which one this run started with; "dirty" = uncommitted changes under training/.
g=$HOME/gpu-lab
stamp="gpu-lab $(git -C $g rev-parse --short HEAD)$(git -C $g diff --quiet HEAD -- training || echo ' dirty'), image $(docker image inspect -f '{{.Id}}' gpu-lab:training | cut -c8-19)"
repo=$(cd "$(dirname "$0")/.." && pwd)  # this checkout, so a worktree runs its own probes
docker run --rm --init --gpus all --ipc=host -v /srv/model-cache:/hf:ro \
  -v $repo/probes:/probes:ro -v $repo/results:/out \
  -v $HOME/gpu-lab/training:/gpulab/training:ro \
  -e PLATFORM_PROFILE=$(cat /sys/firmware/acpi/platform_profile) \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTHONDONTWRITEBYTECODE=1 -e LEAN_LORA -e EXPERT_LORA -e EXPERT_R -e LM_HEAD_LORA -e CE_CHUNK -e DRY_RUN -e GUARD -e RENDER --user $(id -u):$(id -g) \
  --entrypoint python3 gpu-lab:training "$@" > $repo/results/$label.log 2>&1
rc=$?
echo "$stamp" >> $repo/results/$label.log
echo "exit $rc" >> $repo/results/$label.log
