#!/usr/bin/env bash
set -euo pipefail

# Prepare PegInsertionSide data for mini-pi0 training:
# 1. replay ManiSkill motion-planning demos through the local 4-camera env,
# 2. augment the replay with compact contact/force features,
# 3. convert the contact-augmented replay to robomimic HDF5.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
NUM_ENVS="${NUM_ENVS:-16}"
COUNT="${COUNT:-}"

SOURCE_H5="${SOURCE_H5:-demos/maniskill/PegInsertionSide-v1/motionplanning/trajectory.h5}"
WORK_DIR="${WORK_DIR:-demos/maniskill/PegInsertionSide-v1/motionplanning_holecam}"
ROBO_OUT="${ROBO_OUT:-data/robomimic/maniskill/peginsertionside/mp/rgbd_pd_ee_delta_pose_holecam_contacts.hdf5}"

REPLAY_H5="${WORK_DIR}/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5"
REPLAY_JSON="${WORK_DIR}/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.json"
CONTACT_H5="${WORK_DIR}/trajectory.rgbd.pd_ee_delta_pose.contacts.physx_cpu.h5"
CONTACT_JSON="${WORK_DIR}/trajectory.rgbd.pd_ee_delta_pose.contacts.physx_cpu.json"

mkdir -p "${WORK_DIR}"
mkdir -p "$(dirname "${ROBO_OUT}")"

COUNT_ARGS=()
LIMIT_ARGS=()
if [[ -n "${COUNT}" ]]; then
  COUNT_ARGS=(--count "${COUNT}")
  LIMIT_ARGS=(--limit "${COUNT}")
fi

echo "[1/4] Replaying with MiniPi0PegInsertionSide-v1, physx_cpu, num-envs=${NUM_ENVS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" tools/replay_maniskill_local_env.py \
  --env-id MiniPi0PegInsertionSide-v1 \
  --work-dir "${WORK_DIR}" \
  --traj-path "${SOURCE_H5}" \
  --obs-mode rgbd \
  --target-control-mode pd_ee_delta_pose \
  --save-traj \
  --reward-mode dense \
  --sim-backend physx_cpu \
  --num-envs "${NUM_ENVS}" \
  "${COUNT_ARGS[@]}"

echo "[2/4] Recording contact/force features into replay copy"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" tools/replay_maniskill_contacts.py \
  --traj-path "${REPLAY_H5}" \
  --base-hdf5 "${REPLAY_H5}" \
  --out-hdf5 "${CONTACT_H5}" \
  --obs-mode rgbd \
  --control-mode pd_ee_delta_pose \
  --sim-backend physx_cpu \
  --reward-mode dense \
  --use-first-env-state \
  --overwrite \
  "${LIMIT_ARGS[@]}"

echo "[3/4] Copying replay metadata for contact-augmented file"
cp "${REPLAY_JSON}" "${CONTACT_JSON}"

echo "[4/4] Converting to robomimic HDF5 with four cameras and contact obs"
"${PYTHON}" -m mini_pi0.cli.main convert-maniskill-trajectory \
  --input_hdf5 "${CONTACT_H5}" \
  --input_json "${CONTACT_JSON}" \
  --output_hdf5 "${ROBO_OUT}" \
  --image_camera_map agentview_image=base_camera,robot0_eye_in_hand_image=hand_camera,hole_left_image=hole_left_camera,hole_right_image=hole_right_camera \
  --overwrite

echo "Done."
echo "Replay HDF5: ${REPLAY_H5}"
echo "Contact HDF5: ${CONTACT_H5}"
echo "Training HDF5: ${ROBO_OUT}"
