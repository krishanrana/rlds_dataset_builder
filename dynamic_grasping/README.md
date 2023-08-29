## Dynamic Grasping Dataset: 

This dataset consists of 812 successful trajectories pertaining to top-down dynamic grasping of a moving object using a Franka Panda robot manipulator arm. The object is placed on a XY motion platform that can move objects through arbitrary trajectories at various speeds. The system uses a CoreXY motion platform design that is described here. All parts in the design can be 3D printed or easily sourced. A detailed description of how to replicate the dynamic motion platform is provided at [here](https://github.com/BenBurgessLimerick/DGBench). The controller used for dataset collection is based on this [paper](https://arxiv.org/pdf/2204.13879v1.pdf) which utilises an eye-in-hand camera placed between the grippers of the robot.

A single trajectory consists of the following information:

* Robot State: 7x Joint Angles, 2x Gripper Position, 7x Joint Velocities 

* Observation: 1x Hand Camera Image, 1x External Camera Image

* Action: 3x XYZ Cartesian Position of the End Effector, 4x Orientation Quaternion of the End Effector. All actions are with respect to the base frame of the robot.

* Reward: A reward of 1 is provided when the object is between the gripper fingers and the gripper position is less than a fixed threshold.

* Language Instruction: "Pick up the red block"


![sample_traj](https://drive.google.com/uc?export=view&id=1EUdqJgB2m_AOqog9FVYi40qrpgRMWY8A)


