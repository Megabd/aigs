# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
from pickletools import read_stringnl_noescape_pair
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
from pcgym import PcgrlEnv
from pcgym.envs.helper import get_int_prob, get_string_map
from PIL import Image



# Evolutionary algorithms 1

# %% n-dimensional function with a strange topology
@partial(np.vectorize, signature="(d)->()")
def griewank_function(pop):  # this is kind of our fitness function (except we a minimizing)
    return 1 + np.sum(pop**2) / 4000 - np.prod(np.cos(pop / np.sqrt(np.arange(1, pop.size + 1))))


@partial(np.vectorize, signature="(d)->(d)", excluded=[0])
def mutate(sigma, pop):  
    # Add Gaussian noise to each element of `pop`, mean = 0, standard deviation = sigma, Shape of noise matches pop.shape
    return pop + np.random.normal(0, sigma, pop.shape)


@partial(np.vectorize, signature="(d),(d)->(d)")
def crossover(x1, x2):  # We interlop between the two parents, this time with a shared scalar 
    a = np.random.rand()
    return x1 * a + x2 * (1 - a)


def step(pop, cfg):
    loss = griewank_function(pop)
    idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)]  # select best
    best = np.tile(pop[idxs], (int(cfg.population * cfg.proportion), 1))  # copy the best
    pop = crossover(best, best[np.random.permutation(best.shape[0])])  # cross over
    return mutate(cfg.sigma, pop), loss  # return new generation and loss and mutate


def main(cfg):
    pop = np.random.uniform(-50.0, 50.0, size=(cfg.population, 2)) # Making a random starting population with values between -50 and 50

    # main loop
    for gen in range(cfg.generation):
        pop, loss = step(pop, cfg)

        # Save the best of each generation and put them in a plot
        best_idx = np.argmin(loss)
        best_vec, best_val = pop[best_idx], loss[best_idx]

        plt.clf()
        plt.scatter(pop[:, 0], pop[:, 1], s=10, alpha=0.5, label="population")
        plt.scatter(*best_vec, c="red", s=40, marker="*", label="best")
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        plt.title(f"gen {gen:03d} | best={best_val:.6f}")
        # Pause to make progress observeable
        plt.pause(1)  


"""     
#  Evolutionary algorithms 2
def fitness(stats):
    # Helper functions, takes a value and a cap, returns 0 to 1 based on how close to cap
    def low_good_cap(x, cap):
        # score = 1 when x <= cap, else decreases ~ inversely
        x = float(x)
        return 1.0 if x <= cap else cap / x

    def high_good_cap(x, cap):
        # score = 0..1, saturates at cap
        x = float(x)
        return min(x / cap, 1.0)

    # hard code some hard coded weights
    s_distfloor = low_good_cap(stats["dist-floor"], 10.0) * 10
    s_disjtube  = low_good_cap(stats["disjoint-tubes"], 10.0) * 10
    s_noise     = low_good_cap(stats["noise"], 100.0) * 10
    s_empty     = high_good_cap(stats["empty"], 1000.0) * 10

    secondary = (s_distfloor + s_disjtube + s_noise + s_empty)

    primary = -stats["dist-win"] # 113 to 0, reversed so big is good

    return primary + secondary

# get key from utils.
def get_key(b, resolution):
    # suppose that b is in [0, 1]*
    return tuple(
        [int(x * resolution) if x < 1 else (resolution - 1) for x in b]
    )  # edge case when the behavior is exactly the bound you put it with the previous cell

# Makes a random map
def RandomMap(env):
    return np.random.randint(0, env.get_num_tiles(), env._rep._map.shape)

# Randomly swap a few tiles to random tiles
def mutate_map(level_map, num_mutations=1, num_tiles=None):

    new_map = level_map.copy()
    h, w = new_map.shape
    if num_tiles is None:
        num_tiles = int(new_map.max()) + 1

    # For each mutation, we choose a random tile (between 0 and height and width), and we make that tile into a random tile
    for _ in range(num_mutations):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        new_map[y, x] = np.random.randint(0, num_tiles)

    return new_map
resolution = 10
maxJumps = 100 
maxEnemies = 200

def MAP_Elite(cfg, env):
    archive = {}
    G = int(0.1 * cfg.budget)  # number of random solutions to start filling the archive
    for i in range(cfg.budget):
        if (i < G):
            x = RandomMap(env) # Fill the archive with random maps in the start
        else:
            # Choose a random map and mutate it 
            rec = np.random.choice(list(archive.values()))
            x = mutate_map(rec["solution"], num_mutations=200, num_tiles=env.get_num_tiles()) 
        # Get all the stats
        stats = env._prob.get_stats(get_string_map(x, env._prob.get_tile_types()))
        jumps = stats["jumps"]
        enemies = stats["enemies"]
        b = [jumps/maxJumps, enemies/maxEnemies]
        p = fitness(stats)
        key = get_key(b, resolution) 
        # if the map is not in the archive, or the mutated map is better than the old one, we store the new map
        if key not in archive or archive[key]["fitness"] < p:
            archive[key] = {"fitness": p, "behavior": (jumps, enemies), "solution": x}
    return archive

            

    
# %% Setup
def main(cfg):
    env, pop = init_pcgym(cfg)
    mapElite = MAP_Elite(cfg, env)
    target_key = get_key([10/maxJumps, 5/maxEnemies], resolution)  # e.g., ~10 jumps, ~5 enemies
    if target_key in mapElite:
        env._rep._map = mapElite[target_key]["solution"]
    # Else get map with highest fitness
    else:
        best_key = max(mapElite, key=lambda k: mapElite[k]["fitness"])
        env._rep._map = mapElite[best_key]["solution"]
    # Save the map, print the stats and fitness
    Image.fromarray(env.render()).save("map.png")
    map = get_string_map(env._rep._map, env._prob.get_tile_types())
    behavior = env._prob.get_stats(map)
    print(behavior)
    print(fitness(behavior))
    exit()


# %% Init population (maps)
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.budget, *env._rep._map.shape)) 
    return env, pop
"""