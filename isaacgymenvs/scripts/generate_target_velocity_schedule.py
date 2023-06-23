import numpy as np
from scipy.interpolate import CubicSpline

def generate_fixed_speed_schedule(n_timesteps):
    return np.ones((n_timesteps, 1)) * 1.0

def generate_random_speed_schedule(n_timesteps):
    speed_schedule = np.zeros((n_timesteps, 1)) 
    # Generate a random sine wave combination 
    ts = np.linspace(0, 1, n_timesteps)
    for i in range(5):
        a = np.random.uniform() 
        phi = np.random.uniform() * 2 * np.pi 
        omega = np.random.uniform(low = 0.1, high = 20)
        speed_schedule[:,0] += a * np.sin(omega * ts + phi)
    return speed_schedule

def generate_sigmoid_speed_schedile(n_timesteps):

    vel1 = [np.rando.uniform(), ] * n_timesteps

    # vel2 = np.array([np.round(random.random(), decimals =1),] * self.data_size)
    vel2 = [np.rando.uniform(),] * n_timesteps

    if vel1 is vel2:
        vel2 =[np.random.uniform(), ] * n_timesteps

    w = 0.15
    D = np.linspace(0, 2, n_timesteps)
    sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
    vels = vel1 + (vel2 - vel1) * (1 - sigmaD)

    return vels

def generate_random_polar_direction_schedule(n_timesteps):
    direction_schedule = np.zeros((n_timesteps, 3))
    ts = np.linspace(0, 1, n_timesteps)

    theta = np.zeros(n_timesteps)
    for i in range(5):
        phi = np.random.uniform() * 2 * np.pi 
        omega = np.random.uniform(low = 0.1, high = 20)
        theta += np.sin(omega * ts + phi)

    direction_schedule[:, 0] += np.cos(theta)
    direction_schedule[:, 1] += np.sin(theta)
    return direction_schedule

def generate_cubic_spline_direction_schedule(n_timesteps):
    direction_schedule = np.zeros((n_timesteps, 2))
    
    xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # Select random direction
    ys = np.random.normal(loc=np.zeros((xs.shape[0], 2)), scale=np.ones((xs.shape[0], 2)))
    ys = ys / np.linalg.norm(ys, axis=-1, keepdims=True)
    
    # Interpolate to fill in schedule
    cs = CubicSpline(xs, ys)
    direction_schedule += cs(ts, 1)
    return direction_schedule

if __name__ == "__main__":
    
    n_timesteps = 2000
    ts = np.linspace(0, 1, n_timesteps)

    speed_schedule = generate_random_speed_schedule(n_timesteps)
    direction_schedule = generate_random_polar_direction_schedule(n_timesteps)
    velocity_schedule = speed_schedule * direction_schedule

    # Visualize the schedule
    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots()
    combined_schedule = np.concatenate([direction_schedule, speed_schedule], axis=-1)

    inp = input("[S]ave, or [E]xit\n")
    if inp.lower() == "s":
        np.save('schedule.npy', combined_schedule)