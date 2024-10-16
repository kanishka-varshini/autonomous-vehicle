import numpy as np
import matplotlib.pyplot as plt

# Function to generate smooth minimum-jerk (third-order) velocity profile
def minimum_jerk_profile(v0, v_max, vf, a_max, d_total):
    """
    Generates a minimum jerk (3rd order) velocity profile.

    Parameters:
    - v0: initial velocity (m/s)
    - v_max: maximum velocity (m/s)
    - vf: final velocity (m/s)
    - a_max: maximum acceleration (m/s^2)
    - d_total: total distance to travel (m)

    Returns:
    - t: time array (s)
    - v: velocity array (m/s)
    - d: distance array (m)
    """
    # Acceleration phase duration (from v0 to v_max)
    t_acc = (v_max - v0) / a_max
    d_acc = 0.5 * (v0 + v_max) * t_acc

    # Deceleration phase duration (from v_max to vf)
    t_dec = (v_max - vf) / a_max
    d_dec = 0.5 * (v_max + vf) * t_dec

    # Constant velocity phase duration
    if d_total > d_acc + d_dec:
        d_const = d_total - (d_acc + d_dec)
        t_const = d_const / v_max
    else:
        # No constant velocity phase, adjust t_acc and t_dec for triangular profile
        t_acc = np.sqrt(d_total / (a_max * (1 + v0/v_max + vf/v_max)))
        t_dec = t_acc * (vf / v_max)
        t_const = 0

    t_total = t_acc + t_const + t_dec

    # Time arrays for each phase
    t1 = np.linspace(0, t_acc, num=100)
    t2 = np.linspace(t_acc, t_acc + t_const, num=100)
    t3 = np.linspace(t_acc + t_const, t_total, num=100)

    # Define cubic polynomial for acceleration phase (3rd-order polynomial)
    def cubic_acceleration(t, v0, v_max, t_acc):
        a_0 = v0
        a_1 = 0
        a_2 = 3 * (v_max - v0) / (t_acc ** 2)
        a_3 = -2 * (v_max - v0) / (t_acc ** 3)
        return a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3

    # Define cubic polynomial for deceleration phase (3rd-order polynomial)
    def cubic_deceleration(t, v_max, vf, t_dec):
        a_0 = v_max
        a_1 = 0
        a_2 = -3 * (v_max - vf) / (t_dec ** 2)
        a_3 = 2 * (v_max - vf) / (t_dec ** 3)
        return a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3

    # Compute velocities for each phase
    v1 = cubic_acceleration(t1, v0, v_max, t_acc)
    v2 = np.full_like(t2, v_max)  # Constant velocity phase
    v3 = cubic_deceleration(t3 - (t_acc + t_const), v_max, vf, t_dec)

    # Compute distances for each phase by integrating the velocity
    d1 = np.cumsum(v1) * (t1[1] - t1[0])
    d2 = d1[-1] + v_max * (t2 - t_acc)
    d3 = d2[-1] + np.cumsum(v3) * (t3[1] - t3[0])

    # Combine time, velocity, and distance arrays
    t = np.concatenate([t1, t2, t3])
    v = np.concatenate([v1, v2, v3])
    d = np.concatenate([d1, d2, d3])

    return t, v, d

# Parameters
v0 = 0.0     # Initial velocity (m/s)
v_max = 5.0  # Maximum velocity (m/s)
vf = 0.0     # Final velocity (m/s), set to a smaller speed if needed
a_max = 1.0  # Maximum acceleration (m/s^2)
d_total = 50.0 # Total distance (m)

# Generate the smooth minimum-jerk velocity profile
t, v, d = minimum_jerk_profile(v0, v_max, vf, a_max, d_total)

# Plot velocity vs. time
plt.figure(figsize=(10, 5))
plt.plot(t, v, label='Velocity (m/s)')
plt.title('Minimum Jerk Velocity Profile')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.show()

# Plot distance vs. time
plt.figure(figsize=(10, 5))
plt.plot(t, d, label='Distance (m)')
plt.title('Distance Traveled Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.grid(True)
plt.legend()
plt.show()
