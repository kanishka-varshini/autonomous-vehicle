import numpy as np
import matplotlib.pyplot as plt

# Function to generate trapezoidal velocity profile
def trapezoidal_velocity_profile(v0, v_max, vf, a_max, d_total):
    """
    Generates a trapezoidal velocity profile.

    Parameters:
    - v0: initial velocity (m/s)
    - v_max: maximum velocity (m/s)
    - vf: final velocity (m/s), can be zero or a smaller speed
    - a_max: maximum acceleration (m/s^2)
    - d_total: total distance to travel (m)

    Returns:
    - t: time array (s)
    - v: velocity array (m/s)
    - d: distance array (m)
    """
    # Acceleration phase
    t_acc = (v_max - v0) / a_max
    d_acc = 0.5 * (v0 + v_max) * t_acc

    # Deceleration phase
    t_dec = (v_max - vf) / a_max
    d_dec = 0.5 * (v_max + vf) * t_dec

    # Check if there's enough distance for a constant velocity phase
    if d_total <= d_acc + d_dec:
        # No constant velocity phase, triangular profile
        d_acc = d_total / 2
        d_dec = d_total / 2
        t_acc = np.sqrt(2 * d_acc / a_max)
        t_dec = np.sqrt(2 * d_dec / a_max)
        t_const = 0
    else:
        # Constant velocity phase
        d_const = d_total - (d_acc + d_dec)
        t_const = d_const / v_max

    # Total time
    t_total = t_acc + t_const + t_dec

    # Time arrays for each phase
    t1 = np.linspace(0, t_acc, num=100)
    t2 = np.linspace(t_acc, t_acc + t_const, num=100)
    t3 = np.linspace(t_acc + t_const, t_total, num=100)

    # Velocity for each phase
    v1 = v0 + a_max * t1
    v2 = np.full_like(t2, v_max)
    v3 = v_max - a_max * (t3 - (t_acc + t_const))

    # Distance for each phase
    d1 = v0 * t1 + 0.5 * a_max * t1**2
    d2 = d_acc + v_max * (t2 - t_acc)
    d3 = d_acc + d_const + v_max * (t3 - (t_acc + t_const)) - 0.5 * a_max * (t3 - (t_acc + t_const))**2

    # Combine time, velocity, and distance
    t = np.concatenate([t1, t2, t3])
    v = np.concatenate([v1, v2, v3])
    d = np.concatenate([d1, d2, d3])

    return t, v, d

# Parameters
v0 = 0.0     # Initial velocity (m/s)
v_max = 5.0  # Maximum velocity (m/s)
vf = 2.0     # Final velocity (m/s), change to 0 for full stop
a_max = 1.0  # Maximum acceleration (m/s^2)
d_total = 50.0 # Total distance (m)

# Generate the velocity profile
t, v, d = trapezoidal_velocity_profile(v0, v_max, vf, a_max, d_total)

# Plotting velocity vs. time
plt.figure(figsize=(10, 5))
plt.plot(t, v, label='Velocity (m/s)')
plt.title('Trapezoidal Velocity Profile')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.show()

# Plotting distance vs. time
plt.figure(figsize=(10, 5))
plt.plot(t, d, label='Distance (m)')
plt.title('Distance Traveled Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.grid(True)
plt.legend()
plt.show()
