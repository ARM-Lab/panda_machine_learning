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

    # Open data.csv file for writing
    with open('panda_random_data.csv', 'w', newline='') as csvfile:
        # Create a writer object for writing to CSV file
        writer = csv.writer(csvfile)
        # Write header to the file
        writer.writerow(['x', 'y', 'z', 'alpha', 'beta', 'gamma', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'])
        
        for i in range(num_samples):
            q1 = round(np.random.uniform(low=-166*math.pi/180, high=166*math.pi/180),4)
            q2 = round(np.random.uniform(low=-101*math.pi/180, high=101*math.pi/180),4)
            q3 = round(np.random.uniform(low=-166*math.pi/180, high=166*math.pi/180),4)
            q4 = round(np.random.uniform(low=-174*math.pi/180, high=-4*math.pi/180),4)
            q5 = round(np.random.uniform(low=  -1*math.pi/180, high=215*math.pi/180),4)
            q6 = round(np.random.uniform(low=-166*math.pi/180, high=166*math.pi/180),4)
            q7 = round(np.random.uniform(low=-166*math.pi/180, high=166*math.pi/180),4)

            # Define target joint positions for forward kinematics
            target_joint_positions = [q1,q2,q3,q4,q5,q6,q7]

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
            ik_joint_positions = p.calculateInverseKinematics(panda, 11, target_position, targetOrientation=target_orientation)
            
            # Write values to the file
            writer.writerow([round(x,4), round(y,4), round(z,4), round(alpha,4), round(beta,4), round(gamma,4), round(q1,4), round(q2,4), round(q3,4), round(q4,4), round(q5,4), round(q6,4), round(q7,4)])
        #    print("x,y,z:",x,y,z)
        #    print(alpha,beta,gamma)
        #    print("joints:",ik_joint_positions)
        #    print("q:",q1,q2,q3,q4,q5,q6,q7)

    # Disconnect from PyBullet simulation
    p.disconnect()

collect_data(3000000)
#collect_data(823553)
