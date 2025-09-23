# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
from pcgym import PcgrlEnv
from pcgym.envs.helper import get_int_prob, get_string_map


# %% n-dimensional function with a strange topology
@partial(np.vectorize, signature="(d)->()")
def griewank_function(pop):  # this is kind of our fitness function (except we a minimizing)
    return 1 + np.sum(pop**2) / 4000 - np.prod(np.cos(pop / np.sqrt(np.arange(1, pop.size + 1))))


@partial(np.vectorize, signature="(d)->(d)", excluded=[0])
def mutate(sigma, pop):  # What are we doing here?
    return pop + np.random.normal(0, sigma, pop.shape)


@partial(np.vectorize, signature="(d),(d)->(d)")
def crossover(x1, x2):  # TODO: think about what we are doing here. Is it smart?
    return x1 * np.random.rand() + x2 * (1 - np.random.rand())


def step(pop, cfg):
    loss = griewank_function(pop)
    idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)]  # select best
    best = np.tile(pop[idxs], (int(cfg.population * cfg.proportion), 1))  # cross over
    pop = crossover(best, best[np.random.permutation(best.shape[0])])  # mutate
    return mutate(cfg.sigma, pop), loss  # return new generation and loss


# %% Setup
def main(cfg):
    env, pop = init_pcgym(cfg)
    map = get_string_map(env._rep._map, env._prob.get_tile_types())
    behavior = env._prob.get_stats(map)
    print(behavior)
    exit()


# %% Init population (maps)
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.n, *env._rep._map.shape))  # type: ignore
    return env, pop
