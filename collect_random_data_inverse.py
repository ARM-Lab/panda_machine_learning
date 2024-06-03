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

    # Open data.csv file for writing
    with open('panda_random_data.csv', 'w', newline='') as csvfile:
        # Create a writer object for writing to CSV file
        writer = csv.writer(csvfile)
        # Write header to the file
        writer.writerow(['x', 'y', 'z', 'alfa', 'beta', 'gamma', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'])
        
        for i in range(num_samples):
            x = round(np.random.uniform(low=-0.5, high=0.5),2)
            y = round(np.random.uniform(low=-0.5, high=0.5),2)
            z = round(np.random.uniform(low= 0, high=0.8),2)
            alfa = round(np.random.uniform(low=-3.14, high=3.14),2)
            beta = round(np.random.uniform(low=-3.14, high=3.14),2)
            gamma = round(np.random.uniform(low=-3.14, high=3.14),2)


            desired_position = [x, y, z]
            desired_orientation = p.getQuaternionFromEuler([alfa, beta, gamma])

            # Index of the panda_hand in the URDF
            end_effector_index = 11  

            joint_poses = p.calculateInverseKinematics(panda, end_effector_index, desired_position, desired_orientation)
            q1, q2, q3, q4, q5, q6, q7,_,_ = joint_poses
            print(joint_poses)

            # Write values to the file
            writer.writerow([round(x,2), round(y,2), round(z,2), round(alfa,2), round(beta,2), round(gamma,2), round(q1,2), round(q2,2), round(q3,2), round(q4,2), round(q5,2), round(q6,2), round(q7,2)])

    # Disconnect from PyBullet simulation
    p.disconnect()

collect_data(2000000)
#collect_data(823553)
