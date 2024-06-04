import pybullet as p
import pybullet_data
import numpy as np
import csv
import math

def collect_data(num_samples):
    
    # Initialize PyBullet simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to additional PyBullet data

    # Load Panda robot model
    panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # Let the simulation settle
    p.setGravity(0, 0, -9.81)   

    # Joint indices for the Panda robot
    joint_indices = [0, 1, 2, 3, 4, 5, 6]

    for i in range(num_samples):
        
        # Define target joint positions for forward kinematics
        target_joint_positions = [0.0661,-0.252,-2.8296,-2.474,0.8625,-1.7355,1.3257]

        # Set the joint positions
        for i in range(len(joint_indices)):
            p.resetJointState(panda, joint_indices[i], target_joint_positions[i])
        
        # Perform forward kinematics to get the end-effector state
        end_effector_state = p.getLinkState(panda, 11)  # Link index 11 is the end-effector

        x,y,z = end_effector_state[4]
        alpha, beta, gamma = p.getEulerFromQuaternion(end_effector_state[5])

        # Define target position and orientation for inverse kinematics
        target_position = end_effector_state[4]
        target_orientation = end_effector_state[5]  # Orientation in quaternion

        # Perform inverse kinematics to get the joint positions
        #ik_joint_positions = p.calculateInverseKinematics(panda, 11, target_position, targetOrientation=target_orientation)

        print("x,y,z:",x,y,z)
        print(alpha,beta,gamma)
        #print("joints:",ik_joint_positions)

    # Disconnect from PyBullet simulation
    p.disconnect()

collect_data(1)
#collect_data(823553)
