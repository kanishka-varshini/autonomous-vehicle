import numpy as np
import matplotlib.pyplot as plt

# Function to calculate curvature based on three waypoints
def calculate_curvature(wp1, wp2, wp3):
    v1 = np.array([wp2[0] - wp1[0], wp2[1] - wp1[1]])
    v2 = np.array([wp3[0] - wp2[0], wp3[1] - wp2[1]])
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))
    curvature = angle / np.linalg.norm(v2)
    return curvature

# Function to decide velocity profile based on curvature
def decide_velocity(curvature, v_max, v_turn):
    if curvature > 0.1:
        return v_turn
    else:
        return v_max

# Function to smoothly transition between velocities
def smooth_transition(v_start, v_end, time, t_step):
    """
    Create a smooth velocity transition using a 3rd order polynomial (minimum jerk).

    Parameters:
    - v_start: Starting velocity
    - v_end: Target velocity
    - time: Total time for the transition
    - t_step: Time step for simulation

    Returns:
    - v_profile: Smoothed velocity profile
    """
    t_profile = np.arange(0, time, t_step)
    v_profile = []

    # 3rd order polynomial coefficients for smooth transition
    a0 = v_start
    a1 = 0  # Starting with 0 acceleration
    a2 = 3 * (v_end - v_start) / time**2
    a3 = -2 * (v_end - v_start) / time**3

    for t in t_profile:
        v_t = a0 + a1 * t + a2 * t**2 + a3 * t**3
        v_profile.append(v_t)

    return v_profile

# Function to simulate robot following trajectory with adaptive velocity and braking
def follow_trajectory(waypoints, v_max, v_turn, a_max, brake_signal, t_brake):
    n_points = len(waypoints)
    v_profile = []
    d_profile = [0]
    time_step = 0.1  # Small time step for each segment

    # Initial velocity
    v_current = 0.0
    v_target = v_max

    for i in range(1, n_points - 2):
        wp1, wp2, wp3 = waypoints[i-1], waypoints[i], waypoints[i+1]
        curvature = calculate_curvature(wp1, wp2, wp3)
        
        # Decide target velocity based on curvature
        v_target = decide_velocity(curvature, v_max, v_turn)

        # If brake signal is active, set velocity to 0 with smooth deceleration
        if brake_signal[i] == 1:
            v_target = 0.0

        # Smooth velocity transition (both acceleration and deceleration)
        smooth_v = smooth_transition(v_current, v_target, t_brake, time_step)
        v_profile.extend(smooth_v)

        # Update distance traveled
        for v_t in smooth_v:
            d_profile.append(d_profile[-1] + v_t * time_step)

        # Update current velocity for next step
        v_current = v_target

    # Time array for plotting
    t_profile = np.arange(0, len(v_profile) * time_step, time_step)
    return t_profile, v_profile, d_profile

# Define global trajectory as a list of waypoints (x, y)
waypoints = [
    (0, 0),   # Starting point
    (10, 5),  # Curve 1
    (20, 15), # Curve 2
    (30, 20), # Straight path
    (35, 20), # Turn left (90 degrees)
    (35, 30), # Go straight after 90 degree turn
    (35, 40), # Turn right (90 degrees)
    (45, 40), # Go straight after 90 degree turn
    (55, 50), # Curve 3
    (65, 55), # Curve 4
    (75, 55), # Straight path
    (80, 50), # Turn left (90 degrees)
    (80, 40), # Straight down
    (80, 30), # Turn left (90 degrees)
    (70, 30), # Straight after left turn
    (60, 30), # Turn right (90 degrees)
    (60, 40) # Finish point
]

# Parameters for velocity control
v_max = 5.0  # Maximum velocity (m/s)
v_turn = 2.0  # Reduced velocity for turns (m/s)
a_max = 0.5  # Maximum acceleration (m/s^2)
t_brake = 2.0  # Time for smooth braking or transition (seconds)

# Brake signal (0 for no braking, 1 for braking)
brake_signal = [0,0,0,0, 0, 0, 0, 1, 1, 1, 0, 0,0,0,1,1]  # Example: Apply brakes for certain segments

# Simulate robot following the trajectory with brake signal
t, v, d = follow_trajectory(waypoints, v_max, v_turn, a_max, brake_signal, t_brake)


# Plot velocity vs time
plt.figure(figsize=(10, 5))
plt.plot(t, v, label='Smoothed Velocity (m/s)')
plt.title('Adaptive Smoothed Velocity Profile with Braking Signal')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.show()


# Extract x and y coordinates from waypoints
x_coords = [wp[0] for wp in waypoints]
y_coords = [wp[1] for wp in waypoints]

# Plot the trajectory curve
plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords, marker='o', color='b', label='Trajectory')
plt.title('Example Trajectory Path')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)

# Mark the waypoints
for i, wp in enumerate(waypoints):
    plt.text(wp[0], wp[1], f'WP{i}', fontsize=9, ha='right')

plt.legend()
plt.show()
