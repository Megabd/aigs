# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
from pcgym import PcgrlEnv
from PIL import Image


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
    env._rep._map = pop[0]
    Image.fromarray(env.render()).save("map.png")


# %% Init population (maps)
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.n, *env._rep._map.shape))  # type: ignore
    return env, pop


#########################################################################
##  BELOW HERE THERE BE DRAGONS (left over stuff i played around with) ##
#########################################################################

# %% Plotting function that I think we should put in utils.py
# def our_plot_function(fn):
#     x1 = np.linspace(-10, 10, 100)
#     x2 = np.linspace(-10, 10, 100)
#     xs = np.stack(np.meshgrid(x1, x2), axis=-1)
#     ys = fn(xs)
#     plt.imshow(ys, cmap="viridis")
#     plt.colorbar()
#     plt.show()


# env, pop = init(ctx.config)
# from pcgym.envs import PcgrlEnv
# from typing import Tuple
# from pcgym.envs.helper import get_string_map

# import qdax
# from qdax.core.map_elites import MAPElites
# from PIL import Image
# from qdax.core.emitters.mutation_operators import polynomial_mutation

# fitness, behavior = map(np.array, zip(*list(map(lambda p: eval(env, p), pop))))
# print(behavior.shape, fitness.shape)
# # print(fitness)
# # print(behavior)
# # print(list(map(lambda p: mutate(ctx.config, env, p), pop)))


# # %% using gym-pcg to evaluate map quality
# def eval(env, p) -> Tuple[int, np.ndarray]:
#     env._rep._map = p
#     string_map = get_string_map(env._rep._map, env._prob.get_tile_types())
#     stats = env._prob.get_stats(string_map)
#     behavior = np.array([stats["disjoint-tubes"], stats["empty"]])
#     return env._prob.get_reward(stats, {k: 0 for k in stats.keys()}), behavior


# def mutate(cfg, env, p) -> np.ndarray:
#     mask = np.random.random(p.shape) < cfg.qd.p
#     p[mask] = np.random.randint(0, env.get_num_tiles(), p[mask].shape)
#     return p


# def archive(state, p):
#     raise NotImplementedError
