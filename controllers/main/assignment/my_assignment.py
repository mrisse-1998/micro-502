import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors. 
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

def get_command(sensor_data, camera_data, dt):

    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    # Initialization : if get_command has no "mode" value.
    if not hasattr(get_command, "mode"):
        get_command.mode = "Initialization"
        get_command.wake_up = False
        get_command.capture_step = 1
        get_command.directions = []
        get_command.positions = []
        get_command.gates_pos = []
        get_command.gate_index = 0 
        

        

    # Get the positions from the motion planner.
    # TODO : implement this after the general algorithm is done.

    # mp_obj = MotionPlanner3D()
    # setpoints = mp_obj.trajectory_setpoints
    # timepoints = mp_obj.time_setpoints
    # assert setpoints is not None, "No valid trajectory reference setpoints found"
    # tol_goal = 0.25


    
    if get_command.mode == "Initialization":

        print("Mode : Initialization")

        # First capture
        if get_command.capture_step == 1:
            get_command.img_flux, _, p_cam, _ = next_gate_detection(camera_data)
            get_command.directions.append(get_world_position(p_cam, sensor_data))
            get_command.positions.append(np.array([sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]]))
            get_command.last_yaw = sensor_data["yaw"]
            get_command.t_start = time.time()
            get_command.capture_step = 2
            print("Captured first image")
        
        # Move and wait
        elif get_command.capture_step == 2 and time.time() - get_command.t_start < 5:
            print("Moving to second position")
            P = get_command.positions[0]
            control_command = [P[0], P[1], 2, get_command.last_yaw + 0.3]

        # Second capture
        elif get_command.capture_step == 2:
            get_command.img_flux, _, p_cam, _ = next_gate_detection(camera_data)
            get_command.directions.append(get_world_position(p_cam, sensor_data))
            get_command.positions.append(np.array([sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]]))
            get_command.last_yaw = sensor_data["yaw"]
            get_command.t_start = time.time()
            get_command.capture_step = 3
            print("Captured second image")

        # Move and wait
        elif get_command.capture_step == 3 and time.time() - get_command.t_start < 5:
            print("Moving to third position")
            P = get_command.positions[1]
            control_command = [P[0] - 0.2, P[1] - 1, 1, get_command.last_yaw + 0.3]
        
        elif get_command.capture_step == 3:
            get_command.img_flux, _, p_cam, _ = next_gate_detection(camera_data)
            get_command.directions.append(get_world_position(p_cam, sensor_data))
            get_command.positions.append(np.array([sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]]))

            # Triangulate
            H1 = triangulate(get_command.directions[0], get_command.directions[1],
                             get_command.positions[0], get_command.positions[1])
            H2 = triangulate(get_command.directions[1], get_command.directions[2],
                             get_command.positions[1], get_command.positions[2])
    
            get_command.H = (H1 + H2) / 2
            get_command.gates_pos.append(get_command.H)
            get_command.mode = "Orientation"


            # Clean all positions, directions from the first orientation
            get_command.capture_step = 1
            get_command.t_start = time.time()
            get_command.directions, get_command.positions = [], []
            


    if get_command.mode == "Orientation":
        i = get_command.gate_index
        print(f"Mode : Orientation to identify gate number {i + 2}")

        if get_command.capture_step == 1 and time.time() - get_command.t_start < 10:
            print("Move to gate pos")
            # Move to known position
            H = get_command.gates_pos[i]
            control_command = [H[0], H[1], H[2], sensor_data['yaw']]
            get_command.last_yaw = sensor_data['yaw']
            get_command.capture_step = 1


        elif get_command.capture_step == 1:
            get_command.img_flux, _, p_cam, _ = next_gate_detection(camera_data)
            if p_cam is None:
                # Slowly scan
                H = get_command.gates_pos[i]
                control_command = [H[0], H[1], H[2], sensor_data['yaw'] + 0.2]
            else:
                get_command.directions.append(get_world_position(p_cam, sensor_data))
                get_command.positions.append(np.array([sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]]))
                get_command.capture_step = 2
                get_command.t_start = time.time()
                print(f"Captured first image of gate {i}")
        
        elif get_command.capture_step == 2 and time.time() - get_command.t_start < 10:
            P = get_command.positions[0]
            r = get_command.directions[0]  # direction to gate in world frame

            # Compute yaw to point in direction of r (just use x and y)
            yaw_to_gate = np.arctan2(r[1], r[0])
            
            # Create command
            control_command = [P[0], P[1], P[2]+0.4, yaw_to_gate]
        
        # Second capture
        elif get_command.capture_step == 2:
            get_command.img_flux, _, p_cam, _ = next_gate_detection(camera_data)
            get_command.directions.append(get_world_position(p_cam, sensor_data))
            get_command.positions.append(np.array([sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]]))
            get_command.capture_step = 3
            get_command.t_start = time.time()
            print(f"Captured second image of gate {i}")

        # Move to 3rd capture
        elif get_command.capture_step == 3 and time.time() - get_command.t_start < 10:
            P1 = get_command.positions[0]
            P2 = get_command.positions[1]
            r1 = get_command.directions[0]
            r2 = get_command.directions[1]

            # # Triangulate the gate position
            H = triangulate(r1, r2, P1, P2)

            # # Compute halfway point from current position (P2)
            mid_pos = (P2 + H) / 2
            mid_pos[2] = H[2]  # keep same height as drone

            # # Compute yaw to face the gate
            direction_to_gate = H - P2
            yaw_to_gate = np.arctan2(direction_to_gate[1], direction_to_gate[0])

            # # Command
            control_command = [mid_pos[0], mid_pos[1], mid_pos[2], yaw_to_gate]




        elif get_command.capture_step == 3:
            get_command.img_flux, _, p_cam, normale = next_gate_detection(camera_data)
            get_command.directions.append(get_world_position(p_cam, sensor_data))
            get_command.positions.append(np.array([sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]]))
            print(f"Captured third image of gate {i}")

            # Triangulate
            H = triangulate(get_command.directions[1], get_command.directions[2],
                             get_command.positions[1], get_command.positions[2])
            get_command.gates_pos.append(H)

            # Reset for next gate
            get_command.capture_step = 1
            get_command.directions.clear()
            get_command.positions.clear()
            get_command.gate_index += 1
            get_command.t_start = time.time()

            if get_command.gate_index >= 4:  # All gates found
                get_command.mode = "Preparation"
            


    if get_command.mode == "Recognition":

        print("MODE : Recognition lap")
        
    if get_command.mode == "Preparation":
        print("MODE : Preparation")

    if get_command.mode == "Run":
        print("MODE : Run")


    # Print sensor values
    #print("\n--- Sensor Data ---")
    #print(f"Position   : x={sensor_data['x_global']:.2f}, y={sensor_data['y_global']:.2f}, z={sensor_data['z_global']:.2f}")
    #print(f"Velocity   : vx={sensor_data['v_x']:.2f}, vy={sensor_data['v_y']:.2f}, vz={sensor_data['v_z']:.2f}")
    #print(f"Acceleration: ax={sensor_data['ax_global']:.2f}, ay={sensor_data['ay_global']:.2f}, az={sensor_data['az_global']:.2f}")
    #print(f"Yaw        : {sensor_data['yaw']:.2f} rad")
 
    
    # ---- YOUR CODE HERE ----
    if 'control_command' not in locals():
        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
    
    cv2.imshow("Image", get_command.img_flux)
    cv2.waitKey(1)

    
    return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians

def triangulate(r, s, P, Q):
    """
    r, s: Direction vectors in world frame from camera detections
    P, Q: Positions of drone when each image was taken
    Returns: Estimated 3D position of the object (gate center)
    """
    A = np.array([[np.dot(r, r), -np.dot(s, r)],
                  [np.dot(r, s), -np.dot(s, s)]])
    
    b = np.array([[-np.dot((P - Q), r)],
                  [-np.dot((P - Q), s)]])
    
    lambda_sol, mu_sol = np.linalg.solve(A, b)

    F = P + lambda_sol * r
    G = Q + mu_sol * s
    H = (F + G) / 2

    return H

def get_world_position(p_camera_frame, sensor_data):
    R_C2D = ([0, 0, 1], [-1, 0, 0], [0, -1, 0])
    R_D2W = euler2rotmat([sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw']])
    R_C2W = R_D2W @ R_C2D

    p_world_frame = R_C2W @ p_camera_frame

    return p_world_frame


def next_gate_detection(camera_data):

    b = camera_data[:, :, 0]
    g = camera_data[:, :, 1]
    r = camera_data[:, :, 2]

    threshold_r = 180
    threshold_g = 170
    threshold_b = 170

    img_th = np.where(
    (r > threshold_r) &
    (g < threshold_g) &
    (b > threshold_b),
    255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cam_copy = camera_data.copy()

    gate_pixels = None

    if contours:

        # Compute the largest area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > 500:
            cv2.drawContours(cam_copy, [largest_contour], -1, (255, 0, 0), 2)
            gate_pixels = largest_contour.reshape(-1, 2)
        else:
            print("Contour too small, ignoring.")
            return cam_copy, None, None, None
    else:
        return cam_copy, None, None, None
        


    # Using moments to get a better gate centroïd
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
    else:
        centroid_x, centroid_y = 0, 0  # fallback

    centroid = np.array([centroid_x, centroid_y])
    cv2.circle(cam_copy, (int(centroid[0]), int(centroid[1])), radius=5, color=(0, 0, 255), thickness=-1)


    # Normale
    # Get orientation using minAreaRect
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(int)

    # Direction vector along width of rectangle (for normal computation)
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]

    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        gate_direction_image = edge1
    else:
        gate_direction_image = edge2

    gate_direction_image = gate_direction_image / np.linalg.norm(gate_direction_image)

    # Normal = perpendicular in 2D
    normal_image = np.array([-gate_direction_image[1], gate_direction_image[0]])

    # For visualization (draw the normal)
    normal_tip = (int(centroid[0] + 50 * normal_image[0]), int(centroid[1] + 50 * normal_image[1]))
    cv2.arrowedLine(cam_copy, (centroid_x, centroid_y), normal_tip, (0, 255, 0), 1)

    # 3D vector in camera frame
    normal_camera_frame = np.append(normal_image, 0.0)  # Z = 0 in image plane

    img_center_x = img_th.shape[1] / 2  # width / 2
    img_center_y = img_th.shape[0] / 2  # height / 2
    p_camera_frame = centroid - np.array([img_center_x, img_center_y])
    p_camera_frame = np.append(p_camera_frame, 161.013922282)

    
    return cam_copy, gate_pixels, p_camera_frame, normal_camera_frame

def euler2rotmat(euler_angles):
    
    R = np.eye(3)
    
    # Here you need to implement the rotation matrix
    # First calculate the rotation matrix for each angle (roll, pitch, yaw)
    # Then multiply the matrices together to get the total rotation matrix

    # Inputs:
    #           euler_angles: A list of 3 Euler angles [roll, pitch, yaw] in radians
    # Outputs:
    #           R: A 3x3 numpy array that represents the rotation matrix of the euler angles
    
    # --- YOUR CODE HERE ---

    # --- SAMPLE SOLUTION ---

    R_roll = np.array([ [1, 0, 0], 
                        [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
                        [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])]])
    
    R_pitch = np.array([[np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
                        [0, 1, 0],
                        [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])]])
    
    R_yaw = np.array([ [np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
                          [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
                          [0, 0, 1]])
    
    R = R_yaw @ R_pitch @ R_roll

    return R

class MotionPlanner3D():
    
    def __init__(self):
        # Inputs:
        # - start: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar 
        # - obstacles: 2D array with obstacle locations and obstacle widths [x, y, z, dx, dy, dz]*n_obs
        # - bounds: The bounds of the environment [x_min, x_max, y_min, y_max, z_min, z_max]
        # - grid_size: The grid size of the environment (scalar)
        # - goal: The final goal position of the drone (tuple of 3) 
        
        ## DO NOT MODIFY --------------------------------------------------------------------------------------- ##

        # We don't need an AStart. We already have the

        # FIRST PHASE : hardcoding of the gate positions.
        g1 = np.array([2.12417, 1.83874, 1.23681])
        g2 = np.array([5.12138, 2.30441, 0.780412])
        g3 = np.array([7.20097, 3.26934, 1.28575])
        g4 = np.array([5.29501, 6.74127, 1.18919])
        g5 = np.array([2.52189, 5.50474, 1.04415])

        self.path = [g1, g2, g3, g4, g5, g1, g2, g3, g4, g5, g1, g2, g3, g4, g5]

        self.trajectory_setpoints = None

        self.init_params(self.path)

        self.run_planner(self.path)

        # ---------------------------------------------------------------------------------------------------- ##

    def run_planner(self, path_waypoints):    
        # Run the subsequent functions to compute the polynomial coefficients and extract and visualize the trajectory setpoints
         ## DO NOT MODIFY --------------------------------------------------------------------------------------- ##
    
        poly_coeffs = self.compute_poly_coefficients(path_waypoints)
        self.trajectory_setpoints, self.time_setpoints = self.poly_setpoint_extraction(poly_coeffs, path_waypoints)

        ## ---------------------------------------------------------------------------------------------------- ##

    def init_params(self, path_waypoints):

        # Inputs:
        # - path_waypoints: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar

        # TUNE THE FOLLOWING PARAMETERS (PART 2) ----------------------------------------------------------------- ##
        self.disc_steps = 20 #Integer number steps to divide every path segment into to provide the reference positions for PID control # IDEAL: Between 10 and 20
        self.vel_lim = 7.0 #Velocity limit of the drone (m/s)
        self.acc_lim = 50.0 #Acceleration limit of the drone (m/s²)
        t_f = 2.8  # Final time at the end of the path (s)

        # Determine the number of segments of the path
        self.times = np.linspace(0, t_f, len(path_waypoints)) # The time vector at each path waypoint to traverse (Vector of size m) (must be 0 at start)

    def compute_poly_matrix(self, t):
        # Inputs:
        # - t: The time of evaluation of the A matrix (t=0 at the start of a path segment, else t >= 0) [Scalar]
        # Outputs: 
        # - The constraint matrix "A_m(t)" [5 x 6]
        # The "A_m" matrix is used to represent the system of equations [x, \dot{x}, \ddot{x}, \dddot{x}, \ddddot{x}]^T  = A_m(t) * poly_coeffs (where poly_coeffs = [c_0, c_1, c_2, c_3, c_4, c_5]^T and represents the unknown polynomial coefficients for one segment)
        A_m = np.zeros((5,6))
        
        # TASK: Fill in the constraint factor matrix values where each row corresponds to the positions, velocities, accelerations, snap and jerk here
        # SOLUTION ---------------------------------------------------------------------------------- ## 
        
        A_m = np.array([
            [t**5, t**4, t**3, t**2, t, 1], #pos
            [5*(t**4), 4*(t**3), 3*(t**2), 2*t, 1, 0], #vel
            [20*(t**3), 12*(t**2), 6*t, 2, 0, 0], #acc  
            [60*(t**2), 24*t, 6, 0, 0, 0], #jerk
            [120*t, 24, 0, 0, 0, 0] #snap
        ])

        return A_m

    def compute_poly_coefficients(self, path_waypoints):
        
        # Computes a minimum jerk trajectory given time and position waypoints.
        # Inputs:
        # - path_waypoints: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar
        # Outputs:
        # - poly_coeffs: The polynomial coefficients for each segment of the path [6(m-1) x 3]

        # Use the following variables and the class function self.compute_poly_matrix(t) to solve for the polynomial coefficients
        
        seg_times = np.diff(self.times) #The time taken to complete each path segment
        m = len(path_waypoints) #Number of path waypoints (including start and end)
        poly_coeffs = np.zeros((6*(m-1),3))

        # YOUR SOLUTION HERE ---------------------------------------------------------------------------------- ## 

        # 1. Fill the entries of the constraint matrix A and equality vector b for x,y and z dimensions in the system A * poly_coeffs = b. Consider the constraints according to the lecture: We should have a total of 6*(m-1) constraints for each dimension.
        # 2. Solve for poly_coeffs given the defined system

        for dim in range(3):  # Compute for x, y, and z separately
            A = np.zeros((6*(m-1), 6*(m-1)))
            b = np.zeros(6*(m-1))
            pos = np.array([p[dim] for p in path_waypoints])
            A_0 = self.compute_poly_matrix(0) # A_0 gives the constraint factor matrix A_m for any segment at t=0, this is valid for the starting conditions at every path segment

            # SOLUTION
            row = 0
            for i in range(m-1):
                pos_0 = pos[i] #Starting position of the segment
                pos_f = pos[i+1] #Final position of the segment
                # The prescribed zero velocity (v) and acceleration (a) values at the start and goal position of the entire path
                v_0, a_0 = 0, 0
                v_f, a_f = 0, 0
                A_f = self.compute_poly_matrix(seg_times[i]) # A_f gives the constraint factor matrix A_m for a segment i at its relative end time t=seg_times[i]
                if i == 0: # First path segment
                #     # 1. Implement the initial constraints here for the first segment using A_0
                #     # 2. Implement the final position and the continuity constraints for velocity, acceleration, snap and jerk at the end of the first segment here using A_0 and A_f (check hints in the exercise description)
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[1] #Initial velocity constraint
                    b[row] = v_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[2] #Initial acceleration constraint
                    b[row] = a_0
                    row += 1
                    #Continuity of velocity, acceleration, jerk, snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i < m-2: # Intermediate path segments
                #     # 1. Similarly, implement the initial and final position constraints here for each intermediate path segment
                #     # 2. Similarly, implement the end of the continuity constraints for velocity, acceleration, snap and jerk at the end of each intermediate segment here using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    #Continuity of velocity, acceleration, jerk and snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i == m-2: #Final path segment
                #     # 1. Implement the initial and final position, velocity and accelerations constraints here for the final path segment using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[1] #Final velocity constraint
                    b[row] = v_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[2] #Final acceleration constraint
                    b[row] = a_f
                    row += 1
            # Solve for the polynomial coefficients for the dimension dim

            poly_coeffs[:,dim] = np.linalg.solve(A, b)   

        return poly_coeffs

    def poly_setpoint_extraction(self, poly_coeffs, path_waypoints):

        # DO NOT MODIFY --------------------------------------------------------------------------------------- ##

        # Uses the class features: self.disc_steps, self.times, self.poly_coeffs, self.vel_lim, self.acc_lim
        x_vals, y_vals, z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))
        v_x_vals, v_y_vals, v_z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))
        a_x_vals, a_y_vals, a_z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))

        # Define the time reference in self.disc_steps number of segements
        time_setpoints = np.linspace(self.times[0], self.times[-1], self.disc_steps*len(self.times))  # Fine time intervals

        # Extract the x,y and z direction polynomial coefficient vectors
        coeff_x = poly_coeffs[:,0]
        coeff_y = poly_coeffs[:,1]
        coeff_z = poly_coeffs[:,2]

        for i,t in enumerate(time_setpoints):
            seg_idx = min(max(np.searchsorted(self.times, t)-1,0), len(coeff_x) - 1)
            # Determine the x,y and z position reference points at every refernce time
            x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_x[seg_idx*6:(seg_idx+1)*6])
            y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_y[seg_idx*6:(seg_idx+1)*6])
            z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z velocities at every reference time
            v_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_x[seg_idx*6:(seg_idx+1)*6])
            v_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_y[seg_idx*6:(seg_idx+1)*6])
            v_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z accelerations at every reference time
            a_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_x[seg_idx*6:(seg_idx+1)*6])
            a_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_y[seg_idx*6:(seg_idx+1)*6])
            a_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_z[seg_idx*6:(seg_idx+1)*6])

        yaw_vals = np.zeros((self.disc_steps*len(self.times),1))
        trajectory_setpoints = np.hstack((x_vals, y_vals, z_vals, yaw_vals))

        self.plot(path_waypoints, trajectory_setpoints)
            
        # Find the maximum absolute velocity during the segment
        vel_max = np.max(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        vel_mean = np.mean(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        acc_max = np.max(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))
        acc_mean = np.mean(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))

        print("Maximum flight speed: " + str(vel_max))
        print("Average flight speed: " + str(vel_mean))
        print("Average flight acceleration: " + str(acc_mean))
        print("Maximum flight acceleration: " + str(acc_max))
        
        # Check that it is less than an upper limit velocity v_lim
        assert vel_max <= self.vel_lim, "The drone velocity exceeds the limit velocity : " + str(vel_max) + " m/s"
        assert acc_max <= self.acc_lim, "The drone acceleration exceeds the limit acceleration : " + str(acc_max) + " m/s²"

        # ---------------------------------------------------------------------------------------------------- ##

        return trajectory_setpoints, time_setpoints
    
    
    def plot(self, path_waypoints, trajectory_setpoints):

        # Plot 3D trajectory
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        

        ax.plot(trajectory_setpoints[:,0], trajectory_setpoints[:,1], trajectory_setpoints[:,2], label="Minimum-Jerk Trajectory", linewidth=2)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 3)
        ax.set_zlim(0, 1.5)

        # Plot waypoints
        waypoints_x = [p[0] for p in path_waypoints]
        waypoints_y = [p[1] for p in path_waypoints]
        waypoints_z = [p[2] for p in path_waypoints]
        ax.scatter(waypoints_x, waypoints_y, waypoints_z, color='red', marker='o', label="Waypoints")

        # Labels and legend
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Motion planning trajectories")
        ax.legend()
        plt.show()
