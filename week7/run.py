# ===== IMPORTING LIBRARIES =====

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from functools import partial

# ===== GENERAL OPTIMIZER FUNCTION =====

def optimize(cost, random_jump, init_guess, T, decay,
             k, iterations, save_list=False):
    """
    Optimize a function through annealing, and return
    a list of intermediate guesses.

    -----

    `cost` - calculate cost of guess
    `random_jump` - take a guess and a distance, and move
    randomly
    `init_guess` - initial (random) guess
    `T` - starting temperature
    `decay` - decay rate for temperature
    `k` - boltzmann constant
    `iterations` - number of iterations to make
    `save_list` - if `True`, this returns the of intermediate
    guesses made -- otherwise, returns just the final guess
    """

    p_arr = None

    if save_list:
        p_arr = [init_guess]

    x = init_guess
    c = cost(x)

    for i in range(iterations):
        print(f"Progress: {100 * i / iterations:<8.4}%", end="\r")
        # create a new guess
        new_x = random_jump(
            x,
            (np.random.random_sample() - 0.5) * T
        )

        new_c = cost(new_x)

        if new_c < c: # if it's better, jump
            x = new_x
            c = new_c
            if save_list:
                p_arr.append(x)
        else:
            # otherwise, jump with some probability
            toss = np.random.random_sample()
            if toss < np.exp(-(new_c - c) / (k * T)):
                x = new_x
                c = new_c
                if save_list:
                    p_arr.append(x)

        T *= decay

    if save_list:
        return p_arr
    else:
        return x

# ===== TSP FUNCTIONS =====

def tsp_cost(cities, guess):
    """
    Gives the cost of travelling between cities.

    -----

    `cities` is the list of co-ordinates of the cities
    `guess` is the order in which you visit the cities
    """
    sum_dist = 0

    for i in range(-1, len(guess) - 1):
        sum_dist += np.linalg.norm(cities[guess[i]] - cities[guess[i + 1]])

    return sum_dist

def tsp_jump(guess, distance):
    """
    Swap up to `distance` + 1 values in `guess` to make a
    new guess
    """
    swaps = np.random.randint(int(np.abs(distance)) + 1) + 1
    # print(swaps)
    new_guess = guess.copy()

    for _ in range(swaps):
        swap0, swap1 = np.random.randint(len(guess), size=2)
        tmp = new_guess[swap0]
        new_guess[swap0] = new_guess[swap1]
        new_guess[swap1] = tmp

    return new_guess

def tsp_greedy_solve(cities):
    """
    Get an initial guess for the TSP greedily
    """
    guess = [0]
    unused = list(range(1, len(cities)))
    prev = cities[0]

    for i in range(1, len(cities)):
        min_city = unused[0]
        min_dist = np.linalg.norm(cities[i] - prev)

        for j in range(1, len(unused)):
            dist = np.linalg.norm(cities[unused[j]] -  prev)
            if dist < min_dist:
                min_city = unused[j]
                min_dist = dist

        unused.remove(min_city)
        prev = cities[min_city]
        guess.append(min_city)

    print(guess)
    return guess


# ===== CHOOSING FILE =====

choice = input(
"""Which file to work on?
1 = tsp_10.txt
2 = tsp_100.txt (random initial guess)
3 = tsp_100.txt (greedy initial guess)
"""
)

input_filename = None
iterations = None
init_guess = None
start_temp_scale = None
interval = None

if choice == "1":
    input_filename = "tsp_10.txt"
    start_temp_scale = 1/5
    iterations = 500
    interval = 60
elif choice == "2":
    input_filename = "tsp_100.txt"
    start_temp_scale = 1/5
    iterations = 30000
    interval = 1
elif choice == "3":
    input_filename = "tsp_100.txt"
    start_temp_scale = 1/10
    iterations = 5000
    interval = 6
else:
    print("choosing tsp_10.txt")
    start_temp_scale = 1/5
    input_filename = "tsp_10.txt"
    iterations = 500
    interval = 60

# ===== READING FILE =====

lines = open(input_filename, "r").readlines()

# list of co-ordinates
cities = []

for line in lines[1:]:
    s = line.split()
    cities.append(np.array([float(s[0]), float(s[1])]))

if choice == "3":
    init_guess = tsp_greedy_solve(cities)
else:
    init_guess = list(range(len(cities)))

# ===== PLOT CITIES =====

fig, ax = plt.subplots()

ln, = ax.plot([], [])
lnpoints, = ax.plot(
    [cities[i][0] for i in range(len(cities))],
    [cities[i][1] for i in range(len(cities))],
    'go'
)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# ===== GET OPTIMUM PATHS =====

visited_paths = optimize(
    partial(tsp_cost, cities),
    tsp_jump,
    init_guess,
    len(cities) * start_temp_scale,
    0.99,
    0.1,
    iterations,
    save_list=True
)


print(f"\nBest cost found: {tsp_cost(cities, visited_paths[-1])}")

# ===== ANIMATE =====

def update(frame):
    f = int(np.floor(frame))
    coords = []
    for c in visited_paths[f]:
        coords.append(cities[c])
    coords.append(cities[visited_paths[f][0]])
    coords = np.array(coords)
    ln.set_data(coords[:, 0], coords[:, 1])
    return ln,

ani = FuncAnimation(fig, update, frames=range(len(visited_paths)), interval=1, repeat=False)
plt.show()
