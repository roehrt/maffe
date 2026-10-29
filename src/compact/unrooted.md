# Experimental unrooted compact formulation

`unrooted_solver.cpp` is a separate entry point: its trees never pass through
`Tree`, `AnnotatedInstance`, or the rooted reduction orchestrator. Newick roots
of degree two are suppressed before indexing. The remaining internal vertices
have degree three. Single-leaf and two-leaf inputs are supported as degenerate
cases; at least two input trees are required.

For every triple of leaves, record its median in each input tree and the three
vectors of incident neighbours toward the witness leaves. Sort the three vectors
together to canonically name the ports. Merge states only when both their median
tuples and matched port vectors agree. Sorting directions independently in each
tree would lose the cross-tree correspondence and is incorrect.

Connections have fixed endpoint vertices and fixed path footprints in every
tree. Generate leaf pairs, the three spokes of each triple, and one connection
for each quartet with a common split in all input trees. For a common split
ab|cd, its endpoints are the states witnessed by abc and acd. All connections
are deduplicated by endpoint states. Every edge of a suppressed agreement
component has witnesses of one of these forms, so the construction is complete.

For a state s, activation y_s, and its port p, impose

    sum(x_e : e incident to port p of s) = y_s.

For every vertex of every tree, total state activation at that vertex plus
connection usage of that vertex is at most one. Connections use their path
interiors and labelled endpoints; internal endpoints are charged through y_s.
All variables are binary; their continuous relaxation uses bounds [0,1]. The objective is

    n + sum(y_s) - sum(x_e).

A component with L >= 2 leaves has L-2 branching states and 2L-3 connections,
so it saves L-1 components relative to all singletons. Leaves not incident to
a selected connection remain singleton components.

## LP equivalence

The reference relaxation has a variable lambda_C for every non-singleton
agreement component, capacities sum(lambda_C : v in T_i[C]) <= 1 for every
input vertex, and objective n - sum((|C|-1) lambda_C).

Every such component maps to states and connections with the same resource
usage and objective. Conversely, start at any positive connection in a feasible
fractional compact solution. At each internal state reached, the port equalities
provide a positive connection through each other port. This expansion cannot
return to an earlier vertex: its projection in each input tree never reverses
direction at a branch or along a path, and a tree has no closed walk without a
reversal. It thus terminates at distinct labelled leaves, giving a tree embedded
in all input trees with the same topology. Subtract the minimum variable value
on that component and repeat. Port equalities are preserved, at least one
positive variable disappears per iteration, and the resulting decomposition
preserves all capacities and the objective. This establishes equivalence to the
agreement-component packing relaxation.

## Implementation limits

The initial implementation eagerly builds O(n^3) states, O(n^4) candidate
connections, and all vertex-capacity rows. Path coefficients can require
O(t n^5) space in the worst case. The same model builder feeds SCIP or, when
compiled with Gurobi enabled, Gurobi. There is no rectangle folding or reduction
pass. Both backends seed the all-singleton forest and send solver logs to stderr.
