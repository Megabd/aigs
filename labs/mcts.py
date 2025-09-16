# imports
from __future__ import annotations
import numpy as np
import aigs
from aigs import State, Env
from dataclasses import dataclass, field


# %% Setup
env: Env
amountOfVisted = 0


# %%
def minimax(state: State, maxim: bool) -> int:
    if state.ended:
        return state.point
    else:
        temp: int = -10 if maxim else 10
        for action in np.where(state.legal)[0]:  # for all legal actions
            value = minimax(env.step(state, action), not maxim)
            temp = max(temp, value) if maxim else min(temp, value)
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int) -> int:
    if state.ended:
        return state.point
    else:
        temp: int = -10 if maxim else 10
        for action in np.where(state.legal)[0]:  # for all legal actions
            value = alpha_beta(env.step(state, action), not maxim, alpha, beta)
            temp = max(temp, value) if maxim else min(temp, value)
            if maxim:
                alpha = max (alpha, temp)
                if beta <= alpha:
                    break
            else:
                beta = min(beta, temp)
                if beta <= alpha:
                    break
        return temp



@dataclass
class Node:
    def __init__(self, state : State, parent : Node, action : int):
        self.state = state
        self.parent = parent
        self.visted = 0
        self.reward = 0
        self.action = action
        self.children = []


# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:
    v0 = Node(state, None, None)
    for _ in range(cfg.compute):
        v1 = tree_policy(v0, cfg)
        delta = default_policy(v1.state)
        backup(v1,delta)
    return best_child(v0, cfg.c).action


def tree_policy(node: Node, cfg) -> Node:
    while not node.state.ended:
        a = untriedAction(node)
        if a is not None:
            return expand(node, a)
        else:
            node = best_child(node, cfg.c)
    return node

def untriedAction (v : Node) -> int:
    legal = np.where(v.state.legal)[0]
    tried = {child.action for child in v.children}
    untried = [a for a in legal]
    for a in legal:
        if a in tried:
            untried.remove(a)
    if len(untried) <= 0:
        return None
    return untried[0]

def expand(v: Node, a : int) -> Node:
    newChild = Node(env.step(v.state, a), v, a)
    v.children.append(newChild)
    return newChild


def best_child(root: Node, c) -> Node:
    bestScore = 0
    best = root.children[0]                    
    for child in root.children:
        score = UCB1(root, child)
        if score > bestScore:
            bestScore = score
            best = child
    return best

def UCB1(root : Node, node : Node) -> float:
    return node.reward + np.sqrt(np.log2(root.visted) / node.visted)

def default_policy(state: State) -> int:
    while not state.ended:
        actions = np.where(state.legal)[0]
        if len(actions) == 0:
            break
        action = np.random.choice(actions)
        state = env.step(state, action)
    return state.point


def backup(node, delta) -> None:
    while node is not None:
        node.visted += 1
        node.reward += delta
        delta = -delta
        node = node.parent


# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    state = env.init()

    while not state.ended:
        actions = np.where(state.legal)[0]  # the actions to choose from

        match getattr(cfg, state.player):
            case "random":
                a = np.random.choice(actions).item()

            case "human":
                print(state, end="\n\n")
                a = int(input(f"Place your piece ({'x' if state.minim else 'o'}): "))

            case "minimax":
                values = [minimax(env.step(state, a), not state.maxim) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "alpha_beta":
                values = [alpha_beta(env.step(state, a), not state.maxim, -1, 1) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "monte_carlo":
                a = monte_carlo(state, cfg)

            case _:
                raise ValueError(f"Unknown player {state.player}")

        state = env.step(state, a)

    print(f"{['nobody', 'o', 'x'][state.point]} won", state, sep="\n")
