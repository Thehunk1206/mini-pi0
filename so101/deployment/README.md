# Experimental SO-arm deployment prototype

`deploy_so100.py` is an older policy-deployment sketch that reads and writes
servo registers directly. Its kinematics, register addresses, joint mapping,
and 0–180 degree clipping are not validated for the calibrated SO-101 follower.

It is retained as reference code only. Do not run it on the physical SO-101.
A supported deployment path should use LeRobot's calibrated `SO101Follower`,
enforce saved joint ranges and rate limits, monitor following error and
temperature, and provide reliable torque-disable and incident handling before
it is connected to hardware.
