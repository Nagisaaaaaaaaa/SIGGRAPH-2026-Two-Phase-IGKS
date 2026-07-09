# SIGGRAPH-2026-Two-Phase-IGKS

The official *reference* implementation for the SIGGRAPH 2026 paper "[An Extended Full GKS Formulation for High-Efficiency and Low-Memory Two-Phase Flow Simulation](https://dl.acm.org/doi/10.1145/3811290)".

## Author's Note

Unfortunately, we ultimately had to settle for a C++ & CUDA reference implementation. Our original plan was actually to write it in Python. However, even with [Taichi](https://github.com/taichi-dev/taichi)—arguably the best graphics library out there right now—support for template metaprogramming remains quite primitive. As a result, the Python code ended up being significantly longer than the C++ version, and the compilation times were agonizingly slow (sometimes even taking longer than the simulation itself!). Since we wanted this reference codebase to read more like "executable pseudocode", we decided to stick with C++. The trade-off, of course, is that getting it to compile will (very likely) be a bit of a headache.

To prioritize readability, we stripped out all performance optimizations. However, making it run faster is quite straightforward: simply move the boundary condition handling into a separate kernel and eliminate as many redundant floating-point operations as possible. Doing just this is enough to hit the performance benchmarks reported in our paper. Naturally, there's plenty of room for further optimization, which we encourage you to explore on your own.

You can find the core kernel for the two-phase IGKS in `SimulateOneStep.cu`. It maps exactly to Appendix A in the paper and is remarkably concise—just over 300 lines long. While we only provide a 2D implementation here, extending it to 3D is a breeze. Just do a global search for `Your code here` and fill in the blanks; you'll only need to write about 150 lines of code in total.
