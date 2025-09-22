#import "conf.typ": lab
#import "@preview/lovelace:0.3.0": *
#import "@preview/wrap-it:0.1.1": wrap-content

#set text(size: 12pt)
#set par(justify: true, leading: 1.2em)

#show: doc => lab(title: "Quality Divertsty", date: "Sep", doc)

#align(center, text(1.5em, tracking: 0.1em, upper[Quality Diversity]))
#v(2em)

Quality diversity algorithms, rather than finding a _single_ good solution to a
problem, constructs an archive of _different_ high performing solutions.
"Different" can be defined in various ways, an obvious—and only recently
possible—example being solutions that _look_ different to a vision model
@kumar2024. A popular#footnote[At least amongst the researches teaching you]
quality diversity algorithm is the pedagogically named "Multi-dimensional
Archive of Phenotypic Elites" (MAP-Elites) @mouret2015. Think of it as having a
fitness dimension, on which we maximize, and a set of behavioral dimensions we
want to cover/explore.

= Evolutionary algorithms



Today (September 16th, 2025) we will be playing with evolutionary algorithms, as
a way to get ready for quality diversity algorithms. Evolutionary algorithms are
at their core, extremely intuitive: randomly mutate, see what works, and then
further mutate on that. Inspired by the frequent bifurcation of species into two
sexes\*, we can further mix parts of one good solution with another, combining
them to get a new (perhaps even better solution).

#figure(pseudocode-list(stroke: none, line-numbering: none)[
  _Evolutionary optimization pseudocode_
  + *for* generation in range(N)
    + fitness = eval_function(population)
    + idxs = argsort(population)
    + population = population[idxs]
    + population = cross_over(population, n)
    + population = mutate(population)
  + *end*
])

You will now:
1. Select a test function for optimization #footnote[#link(
    "https://en.wikipedia.org/wiki/Test_functions_for_optimization",
  )]
2. Implement it in python (and visualize it)
3. Find its optimal solution using an evolutionary algorithm from the lecture
4. Make some cool plots of the results (get creative)




= PCGYM

Recall that Gym @towers2024 is a framework for reinforcement learning
environments, consisting of an `init` and a `step` method (the same as those in
our `aigs/games.py` file). Note further that (as the term suggests) procedural
content generation (PCG) focuses on the _procedure_ that generates a given piece
of content, rather than the content itself. To that effect we have made `pcgym`
#box[(itself derived from `pcgrl` @khalifa2020)] that enables quick ideation of
levels, supporting the kind of methods this lab is meant to have you play with.
You will now:
1. Explore `pcgym` #footnote[My fork of `gym-pcgrl` modified to work with our
    course. It is located at #link(
      "https://github.com/syrkis/pcgym",
    ) but is already included in our environment] and have a random agent play a
  level of any game.
2. Replace the random agent with a _randomly initialized_ agent, that maps game
  states to action.
3. Improve the randomly initialized agent using the basic evolutionary algorithm
  used in the previous task.
4. Bonus: think about whether the cross-over operator makes sense for your
  agent, and why / why not this is the case.

= A\*

A\* (A-star) is a pathfinding algorithm combining cost-so-far and estimated
cost-to-goal, balancing shortest-path (like Dijkstra) with goal-directed search
(like greedy best-first). In games like Mario, A\* is used to guide agents
efficiently through levels by evaluating possible moves via a cost function
(e.g., distance, obstacles) and a heuristic (e.g., estimated steps to the flag),
generating optimal or near-optimal action sequences toward the goal. You will
now:
1. Think a bit about A\* (what it is and how you would implement it)
2. Implement A\* or find an implementation online and have it play a level from
  `pcgym`

= picbreeder

Play around with pic breeder by:
1. Drawing something
2. Trying to then find it

= Novelty search

In `pcgym` make a setup that finds _new_ levels.

= Content generation

We can optimize levels, just like we can optimize players. Thinking about what
it means for a level to be "fit", you must now:
1. Define a fitness function for a level in `pcgym` (A\* can play a role in
  evaluating a level)
2. Define two _behavioral_ dimensions, meaning ways in which levels can be
  different, that are _not_ fitness related (e.g., number of jumps).
3. Generate an archive of good levels that are different from one another with
  (Timothée's beloved) MAP-Elite algorithm.

#figure(
  image("me_pseudocode.png"),
  caption: [MAP-Elite algorithm is per the original paper @mouret2015],
)


#[
  #show heading.where(level: 1): set heading(numbering: none)
  = Index of Sources <touying:unoutlined>
  #bibliography("zotero.bib", title: none, style: "ieee")
]
