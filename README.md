# maffe

`maffe` is an exact and heuristic solver for maximum agreement forest for two or more rooted bifurcating phylogenetic trees.
It won the first place in all three tracks (Exact, Heuristic, Lower Bound) of the [PACE challenge 2026](https://pacechallenge.org/2026/).

**Acknowledgements** The implementation was assisted by OpenAI’s Codex with GPT 5.4 and GPT
5.5. We extensively tested the solver with fuzzing and [STRIDE instances](https://pace2026.imada.sdu.dk/) and take responsibility for
its correctness.

## Dependencies

- C++23 compiler
- Meson >= 1.3
- Ninja
- CMake
- Git
- pkg-config
- patch

The solver depends on SCIP Optimization Suite 10.0.2 and HiGHS 1.15.1.
The competition build also uses ConicBundle for the heuristic binary.
The Meson build script will automatically download and build all required solver dependencies.
We optionally support Gurobi as (I)LP backend and expose Python bindings.

## Build

On Debian/Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y build-essential meson ninja-build cmake git pkg-config patch
```

Configure and build:

```sh
meson setup build --buildtype=release
meson compile -C build
```

Run the solver with

```sh
./build/maffe < instance.trees
```

where `instances.trees` contains a list of bifurcating phylogenetic trees. 

## Competition Build

To build the competition binaries, which were submitted to optil.io and to the PACE Challenge 2026, use the Docker-based build script:

```sh
./scripts/build_competition.sh
```

This writes static binaries `maffe-exact`, `maffe-heuristic`, and `maffe-lowerbound` to `dist-competition/`.
However, for non-competition use, we recommend the default build.
