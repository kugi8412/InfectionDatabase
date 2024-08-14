# InfectionDatabase
Class containing information on infections, which are transmitted between persons according to a net of mutual relations G = (V, E). 
The vertices V represent persons, and the edges E represent the length of the relations between people.
Each value is less than a fixed parameter DMAX ∈ N. Furthermore, we distinguish between two
types of contingencies:
1) direct contingencies between persons connected by an edge in a graph G
2) indirect infections are infections requiring an intermediary (even if the intermediary is not infected).
Infections spread through the graph G according to the following rules:
- if a person V became infected at time T, then a person U can become infected at any point in the
time interval (T + dU, V, T + DMAX), where dV, dU is the length of the shortest path
between V and U ∈ G;
- each case is assigned a source of infection (the chronologically last case of
infection from which it may have become infected) or it may be the beginning of a new outbreak of
infections;
- subsequent cases are added to the database in a chronological order.

## Requirements
- python version 3.12
