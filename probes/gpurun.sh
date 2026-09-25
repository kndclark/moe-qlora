#!/bin/bash
# usage: gpurun.sh LABEL MOUNT_EXTRA -- probe args...   (laptop GPU, max-power with trap)
set -u
label=$1; shift
trap 'echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null' EXIT
echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
docker run --rm --init --gpus all --ipc=host -v /srv/model-cache:/hf:ro \
  -v $HOME/moe-qlora/probes:/probes:ro -v $HOME/moe-qlora/results:/out \
  -v $HOME/gpu-lab/training:/gpulab/training:ro \
  -e PLATFORM_PROFILE=$(cat /sys/firmware/acpi/platform_profile) \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTHONDONTWRITEBYTECODE=1 -e LEAN_LORA -e EXPERT_LORA -e EXPERT_R -e LM_HEAD_LORA -e CE_CHUNK --user $(id -u):$(id -g) \
  --entrypoint python3 gpu-lab:training "$@" > $HOME/moe-qlora/results/$label.log 2>&1
echo "exit $?" >> $HOME/moe-qlora/results/$label.log
