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
        writer.writerow(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'x', 'y', 'z', 'alfa', 'beta', 'gamma'])
        
        for i in range(num_samples):
            q1 = round(np.random.uniform(low=-166, high=166),2)
            q2 = round(np.random.uniform(low=-101, high=101),2)
            q3 = round(np.random.uniform(low=-166, high=166),2)
            q4 = round(np.random.uniform(low=-174, high=-4),2)
            q5 = round(np.random.uniform(low=  -1, high=215),2)
            q6 = round(np.random.uniform(low=-166, high=166),2)
            q7 = round(np.random.uniform(low=-166, high=166),2)

            # Set the position of robot joints
            p.setJointMotorControlArray(panda, [0, 1, 2, 3, 4, 5, 6], p.POSITION_CONTROL, 
                                        targetPositions=[round(q1*math.pi/180,2), round(q2*math.pi/180,2), round(q3*math.pi/180,2), round(q4*math.pi/180,2), round(q5*math.pi/180,2), round(q6*math.pi/180,2), round(q7*math.pi/180,2)])
            
            # Simulate one time step
            p.stepSimulation()

            # Get position and orientation of end effector
            link_state = p.getLinkState(panda, 7, computeForwardKinematics=True)
            position = link_state[0]
            orientation = link_state[1]
            x, y, z = position
            alfa, beta, gamma = p.getEulerFromQuaternion(orientation)
            
            # Write values to the file
            writer.writerow([round(x,2), round(y,2), round(z,2), round(alfa,2), round(beta,2), round(gamma,2), round(q1*math.pi/180,2), round(q2*math.pi/180,2), round(q3*math.pi/180,2), round(q4*math.pi/180,2), round(q5*math.pi/180,2), round(q6*math.pi/180,2), round(q7*math.pi/180,2)])

    # Disconnect from PyBullet simulation
    p.disconnect()

#collect_data(1000000)
collect_data(823553)
