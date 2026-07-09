# SIGGRAPH-2026-Two-Phase-IGKS

The official *reference* implementation for the SIGGRAPH 2026 paper "[An Extended Full GKS Formulation for High-Efficiency and Low-Memory Two-Phase Flow Simulation](https://dl.acm.org/doi/10.1145/3811290)".

## Author's Note

Unfortunately, we ultimately had to settle for a C++ & CUDA reference implementation. Our original plan was actually to write it in Python. However, even with [Taichi](https://github.com/taichi-dev/taichi)—arguably the best graphics library out there right now—support for template metaprogramming remains quite primitive. As a result, the Python code ended up being significantly longer than the C++ version, and the compilation times were agonizingly slow (sometimes even taking longer than the simulation itself!). Since we wanted this reference codebase to read more like "executable pseudocode", we decided to stick with C++. The trade-off, of course, is that getting it to compile will (very likely) be a bit of a headache.

To prioritize readability, we stripped out all performance optimizations. However, making it run faster is quite straightforward: simply move the boundary condition handling into a separate kernel and eliminate as many redundant floating-point operations as possible. Doing just this is enough to hit the performance benchmarks reported in our paper. Naturally, there's plenty of room for further optimization, which we encourage you to explore on your own.

You can find the core kernel for the two-phase IGKS in `SimulateOneStep.cu`. It maps exactly to Appendix A in the paper and is remarkably concise—just over 300 lines long. While we only provide a 2D implementation here, extending it to 3D is a breeze. Just do a global search for `Your code here` and fill in the blanks; you'll only need to write about 150 lines of code in total.

## Getting Started

While the core algorithm of this reference implementation is written in C++ and CUDA, parameter configuration and post-processing visualization are handled entirely in Python. This setup is incredibly convenient, as it allows you to tweak parameters on the fly without needing to recompile the C++ code. Additionally, since we heavily rely on some bleeding-edge C++ features, a very recent compiler is an absolute must. Older versions stand practically no chance of compiling this codebase successfully (T^T). Below is the complete guide for setting up your environment and compiling the code:

- **Install a sufficiently new C++ compiler.** We have tested the codebase on the following versions of the "Big Three" compilers:
  | Compiler | Version |
  |----------|---------|
  | gcc      | 15.2    |
  | clang    | 22.1    |
  | msvc     | 18.6    |

- **Install a sufficiently new CUDA Toolkit.** Currently, the oldest NVCC version that fully supports all three compilers mentioned above is 13.3.

- **Install Python and the required dependencies:**

  ```bash
  pip install matplotlib nanobind numpy pyvista taichi
  ```

- **Set up the `CPM_SOURCE_CACHE` environment variable** (temporarily or permanently) to a designated directory. Using `~/CPM` as an example:

  ```bash
  mkdir -p ~/CPM
  export CPM_SOURCE_CACHE="$HOME/CPM"
  ```

- **Generate the CMake project:**

  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  ```

- **Build the project:**

  ```bash
  cmake --build build -j --target igks_tp
  ```

- **Run the simulation:**

  ```bash
  python IGKS_TP/Main.py
  ```
