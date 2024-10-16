import numpy as np
import matplotlib.pyplot as plt

# Define constants
v_max = 8.0      # Max velocity (m/s)
a_max = 1.0      # Max acceleration (m/s^2)
d_total = 10.0   # Total distance (m)

# Function to generate a trapezoidal velocity profiles
def trapezoidal_velocity_profile(d_total, v_max, a_max):
    # Calculate the time for acceleration and deceleration
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2

    if d_total <= 2 * d_acc:
        # No constant velocity phase
        t_acc = np.sqrt(d_total / a_max)
        t_const = 0
    else:
        # Constant velocity phase
        d_const = d_total - 2 * d_acc
        t_const = d_const / v_max
    
    return t_acc, t_const

# Function to compute the S-curve profile using cubic polynomials
def s_curve_velocity_profile(t_acc, t_const, t_total):
    # Using cubic splines or polynomials for smoother acceleration
    t_s = np.linspace(0, t_total, 1000)
    vel_s = np.zeros_like(t_s)
    
    for i, t in enumerate(t_s):
        if t < t_acc:
            vel_s[i] = a_max * t * (3 - 2 * (t / t_acc))
        elif t_acc <= t < t_acc + t_const:
            vel_s[i] = v_max
        elif t >= t_acc + t_const:
            t_decel = t - t_acc - t_const
            vel_s[i] = v_max * (1 - (t_decel / t_acc) ** 2)
    
    return t_s, vel_s

# Using the Newton-Raphson method to refine time estimates
def newton_raphson_time_estimation(v_max, d_total, a_max):
    def f(t): 
        return v_max * t - 0.5 * a_max * t**2 - d_total
    
    def df(t): 
        return v_max - a_max * t
    
    t_guess = d_total / v_max
    for _ in range(10):
        t_guess = t_guess - f(t_guess) / df(t_guess)
    
    return t_guess

# Main program
t_acc, t_const = trapezoidal_velocity_profile(d_total, v_max, a_max)
t_total = 2 * t_acc + t_const

t_s, vel_s = s_curve_velocity_profile(t_acc, t_const, t_total)

# Plotting the results
plt.plot(t_s, vel_s)
plt.title('S-curve Velocity Profile')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.show()
