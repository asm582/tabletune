#!/usr/bin/env bash
# Render the llm-d optimized-baseline model server for one configuration (config-only changes to the guide):
#   ./render.sh <config> [table-json]   -> overlays/<config>/  (apply with: kubectl apply -n vllm-dsd -k overlays/<config>)
# The guide as shipped: Qwen3-32B, TP=2, 2 replicas (4 GPUs). Configuration changes only:
# --max-num-seqs 256, --max-model-len 4096, the vLLM nightly image (V2 runner keeps full CUDA graphs per draft length), the shared volume
# for the model cache and the unmodified TableTune package (PYTHONPATH), all replicas on one node.
set -euo pipefail
CFG=$1; TABLE=${2:-}
NIGHTLY=nightly-43b4aaea3e40e20ef53ce2bfecfa1b72060bf2b5
TUNE=/work/tabletune/experiments/results/tune-h100-32b-tp2-llmd-nightly
D=overlays/$CFG; mkdir -p $D
# LLMD_GUIDES: the guides/ folder of an llm-d clone (main at 4dc1573); kustomize needs it relative to the overlay
LLMD_GUIDES=${LLMD_GUIDES:?set LLMD_GUIDES to <llm-d clone>/guides}
GUIDE=$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$LLMD_GUIDES" "$D")
# NODE: pin all replicas to one node (as in the committed results); unset, any H100 node
if [[ -n "${NODE:-}" ]]; then NODESEL="{kubernetes.io/hostname: $NODE}"; else NODESEL="{nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3}"; fi
IMG_TAG=$NIGHTLY; V2=1; SPEC=""; CTRL=""
eagle() { echo "{\"method\":\"eagle3\",\"model\":\"AngelSlim/Qwen3-32B_eagle3\",\"num_speculative_tokens\":$1$2}"; }
case $CFG in
  guide)      IMG_TAG=v0.30.0; V2=0 ;;                      # the guide's own image, no speculation
  nospec)     ;;
  k1|k2|k3)   SPEC=$(eagle ${CFG#k} "") ;;
  docs)       SPEC=$(eagle 3 ",\"num_speculative_tokens_per_batch_size\":[[1,64,3],[65,128,1],[129,512,0]]") ;;
  tuned)      SPEC=$(eagle 3 ",\"num_speculative_tokens_per_batch_size\":$TABLE") ;;
  tuned-ctrl) SPEC=$(eagle 3 ",\"num_speculative_tokens_per_batch_size\":$TABLE"); CTRL=1 ;;
  tuned13)    SPEC=$(eagle 3 ",\"num_speculative_tokens_per_batch_size\":$TABLE") ;;   # table from the length-0/1/3 profiles
  tuned13-ctrl) SPEC=$(eagle 3 ",\"num_speculative_tokens_per_batch_size\":$TABLE"); CTRL=1; TUNE=${TUNE}-k013 ;;
  docs-ctrl)  SPEC=$(eagle 3 ",\"num_speculative_tokens_per_batch_size\":[[1,64,3],[65,128,1],[129,512,0]]"); CTRL=1 ;;
  *) echo "unknown config $CFG"; exit 1 ;;
esac
cat > $D/kustomization.yaml <<K
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - $GUIDE/recipes/modelserver/base/single-host/default
namePrefix: optimized-baseline-nvidia-gpu-vllm-
images:
  - name: REPLACE_MODEL_SERVER_IMAGE
    newName: docker.io/vllm/vllm-openai
    newTag: $IMG_TAG
labels:
  - pairs:
      llm-d.ai/model: Qwen3-32B
      llm-d.ai/guide: optimized-baseline
      llm-d.ai/accelerator-variant: gpu
      llm-d.ai/accelerator-vendor: nvidia
    includeSelectors: true
    includeTemplates: true
patches:
  - path: patch-vllm.yaml
K
{
cat <<P
apiVersion: apps/v1
kind: Deployment
metadata:
  name: decode
  annotations: {tabletune.config: "$CFG"}
spec:
  replicas: 2
  template:
    metadata:
      annotations: {tabletune.config: "$CFG"}
    spec:
      nodeSelector: $NODESEL
      tolerations:
        - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
      containers:
        - name: modelserver
          command: ["vllm", "serve"]
          args:
            - "Qwen/Qwen3-32B"
            - "--disable-access-log-for-endpoints=/health,/metrics,/v1/models"
            - "--tensor-parallel-size=2"
            - "--max-num-seqs=256"
            - "--max-model-len=4096"
            - "--seed=0"
            - "--enable-auto-tool-choice"
            - "--tool-call-parser=hermes"
            - "--reasoning-parser=qwen3"
P
[[ -n $SPEC ]] && echo "            - \"--speculative-config=$(echo $SPEC | sed 's/"/\\"/g')\""
cat <<P
          env:
            - {name: HF_HOME, value: /work/hf}
            - {name: HF_HUB_OFFLINE, value: "1"}
            - {name: VLLM_USE_V2_MODEL_RUNNER, value: "$V2"}
P
if [[ -n $CTRL ]]; then cat <<P
            - {name: PYTHONPATH, value: /work/dsd-pkg}   # the unmodified TableTune package (controller plugin)
            - {name: VLLM_DSD_CONTROLLER, value: "1"}
            - {name: VLLM_DSD_PRIOR, value: "$TUNE/acceptance.json"}
            - {name: VLLM_DSD_COSTS, value: "$TUNE/costs_k*.json"}
            - {name: VLLM_DSD_HYSTERESIS, value: "8"}
            - {name: VLLM_DSD_REFIT_STEPS, value: "200"}
            - {name: VLLM_DSD_LOG_STEPS, value: "1000"}
P
fi
cat <<P
          resources:
            limits: {cpu: '16', memory: 128Gi, nvidia.com/gpu: 2}
            requests: {cpu: '8', memory: 96Gi, nvidia.com/gpu: 2}
          volumeMounts:
            - {mountPath: /dev/shm, name: shm}
            - {mountPath: /.cache, name: torch-compile-cache}
            - {mountPath: /.triton, name: triton-cache}
            - {mountPath: /.config, name: vllm-config}
            - {mountPath: /work, name: work}
          startupProbe: {initialDelaySeconds: 15, periodSeconds: 30, timeoutSeconds: 5, failureThreshold: 120}
          livenessProbe: {periodSeconds: 10, timeoutSeconds: 5, failureThreshold: 3}
          readinessProbe: {periodSeconds: 5, timeoutSeconds: 2, failureThreshold: 3}
      volumes:
        - {name: shm, emptyDir: {medium: Memory, sizeLimit: 20Gi}}
        - {name: torch-compile-cache, emptyDir: {}}
        - {name: triton-cache, emptyDir: {}}
        - {name: vllm-config, emptyDir: {}}
        - {name: work, persistentVolumeClaim: {claimName: dsd-work}}
P
} > $D/patch-vllm.yaml
echo "rendered $D"
