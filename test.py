import pybullet as p
import pybullet_data
import numpy as np
import csv
import math

# Initialize PyBullet simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to additional PyBullet data

# Load Panda robot model
panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Let the simulation settle
p.setGravity(0, 0, -9.8)

num_joints = p.getNumJoints(panda)
print("Number of joints in the robot:", num_joints)

# Joint indices for the Panda robot
joint_indices = [0, 1, 2, 3, 4, 5, 6]

# Define target joint positions for forward kinematics
target_joint_positions = [0.0, -0.3, 0.0, -2.0, 0.0, 2.0, 0.8]

# Set the joint positions
for i in range(len(joint_indices)):
    p.resetJointState(panda, joint_indices[i], target_joint_positions[i])

# Perform forward kinematics to get the end-effector state
end_effector_state = p.getLinkState(panda, 11)  # Link index 11 is the end-effector

print("End-effector position (forward kinematics):", end_effector_state[4])
print("End-effector orientation (forward kinematics):", end_effector_state[5])

# Define target position and orientation for inverse kinematics
target_position = end_effector_state[4]
target_orientation = end_effector_state[5]  # Orientation in quaternion

# Perform inverse kinematics to get the joint positions
ik_joint_positions = p.calculateInverseKinematics(panda, 11, target_position, targetOrientation=target_orientation)

print("Joint positions (inverse kinematics):", ik_joint_positions)

# Disconnect from PyBullet simulation
p.disconnect()