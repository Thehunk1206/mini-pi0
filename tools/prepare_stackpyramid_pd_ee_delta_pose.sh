#!/usr/bin/env bash
set -euo pipefail

# Prepare StackPyramid-v1 motion-planning data for mini-pi0 training:
# 1. replay ManiSkill motion-planning demos with rgbd observations,
# 2. keep the source controller by default, or optionally convert actions,
# 3. convert the replayed trajectories to robomimic HDF5.
#
# Useful overrides:
#   COUNT=5 NUM_ENVS=4 bash tools/prepare_stackpyramid_pd_ee_delta_pose.sh
#   TARGET_CONTROL_MODE=pd_ee_delta_pose bash tools/prepare_stackpyramid_pd_ee_delta_pose.sh
#   SOURCE_H5=... ROBO_OUT=... bash tools/prepare_stackpyramid_pd_ee_delta_pose.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
PYTHON="${PYTHON:-.venv/bin/python}"
NUM_ENVS="${NUM_ENVS:-16}"
COUNT="${COUNT:-}"
TARGET_CONTROL_MODE="${TARGET_CONTROL_MODE:-pd_joint_pos}"

SOURCE_H5="${SOURCE_H5:-demos/maniskill/StackPyramid-v1/motionplanning/trajectory.h5}"
WORK_DIR="${WORK_DIR:-demos/maniskill/StackPyramid-v1/motionplanning_${TARGET_CONTROL_MODE}}"
ROBO_OUT="${ROBO_OUT:-data/robomimic/maniskill/stackpyramid/mp/rgbd_${TARGET_CONTROL_MODE}.hdf5}"

REPLAY_PREFIX="${WORK_DIR}/trajectory.rgbd.${TARGET_CONTROL_MODE}.physx_cpu"
REPLAY_H5="${REPLAY_PREFIX}.h5"
REPLAY_JSON="${REPLAY_PREFIX}.json"

mkdir -p "${WORK_DIR}"
mkdir -p "$(dirname "${ROBO_OUT}")"

COUNT_ARGS=()
CONVERT_LIMIT_ARGS=()
if [[ -n "${COUNT}" ]]; then
  COUNT_ARGS=(--count "${COUNT}")
  CONVERT_LIMIT_ARGS=(--limit "${COUNT}")
fi

echo "[1/2] Replaying StackPyramid-v1 with rgbd + ${TARGET_CONTROL_MODE}, physx_cpu, num-envs=${NUM_ENVS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" tools/replay_maniskill_local_env.py \
  --env-id StackPyramid-v1 \
  --work-dir "${WORK_DIR}" \
  --traj-path "${SOURCE_H5}" \
  --obs-mode rgbd \
  --target-control-mode "${TARGET_CONTROL_MODE}" \
  --save-traj \
  --sim-backend physx_cpu \
  --num-envs "${NUM_ENVS}" \
  "${COUNT_ARGS[@]}"

mapfile -t REPLAY_H5S < <(find "${WORK_DIR}" -maxdepth 1 -type f -name "trajectory.rgbd.${TARGET_CONTROL_MODE}.physx_cpu*.h5" | sort -V)
if [[ "${#REPLAY_H5S[@]}" -eq 0 && -f "${REPLAY_H5}" ]]; then
  REPLAY_H5S=("${REPLAY_H5}")
fi
if [[ "${#REPLAY_H5S[@]}" -eq 0 ]]; then
  echo "No replay HDF5 files found under ${WORK_DIR}" >&2
  exit 1
fi

REPLAY_JSONS=()
for h5_path in "${REPLAY_H5S[@]}"; do
  json_path="${h5_path%.h5}.json"
  if [[ ! -f "${json_path}" ]]; then
    echo "Missing replay JSON for ${h5_path}: ${json_path}" >&2
    exit 1
  fi
  REPLAY_JSONS+=("${json_path}")
done

echo "[2/2] Converting ${#REPLAY_H5S[@]} replay file(s) to robomimic HDF5"
"${PYTHON}" -m mini_pi0.cli.main convert-maniskill-trajectory \
  --input_hdf5 "${REPLAY_H5S[@]}" \
  --input_json "${REPLAY_JSONS[@]}" \
  --output_hdf5 "${ROBO_OUT}" \
  --image_camera_map agentview_image=base_camera,robot0_eye_in_hand_image=hand_camera \
  --state_keys robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos \
  --overwrite \
  "${CONVERT_LIMIT_ARGS[@]}"

echo "Done."
echo "Replay HDF5 files:"
printf '  %s\n' "${REPLAY_H5S[@]}"
echo "Training HDF5: ${ROBO_OUT}"
