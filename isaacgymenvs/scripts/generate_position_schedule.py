import numpy as np

if __name__ == "__main__":
    
    n_timesteps = 2000
    ts = np.linspace(0, 1, n_timesteps)

    waypoints = [
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    position_schedule = [w for w in waypoints for _ in range(n_timesteps // len(waypoints))]
    position_schedule = np.array(position_schedule)

    # Visualize the schedule
    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots()
    ax.plot(ts, position_schedule)
    fig.show() 
    
    inp = input("[S]ave, or [E]xit\n")
    if inp.lower() == "s":
        np.save('pos_schedule.npy', position_schedule)