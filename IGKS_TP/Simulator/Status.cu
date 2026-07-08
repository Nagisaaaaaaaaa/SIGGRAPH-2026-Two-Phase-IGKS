#include "../Simulator.hpp"

namespace igks::tp {

[[nodiscard]] SimulatorStatus Simulator::Status() {
  auto check = [](cudaError_t error) {
    if (error != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(error));
  };

  std::array<usize, d> phiShape{};
  std::array<isize, d> phiStrides{};
  std::array<usize, d + 1> uShape{};
  std::array<isize, d + 1> uStrides{};
  std::array<usize, d> pShape{};
  std::array<isize, d> pStrides{};

  [&]<usize... id>(std::index_sequence<id...>) {
    ((phiShape[id] = static_cast<usize>(descriptor().resolution()(C<id>{})), uShape[id] = phiShape[id],
      pShape[id] = phiShape[id]),
     ...);
  }(std::make_index_sequence<d>{});
  uShape[d] = d;

  usize size = 1;
  ForEach<d>([&](auto id) { size *= phiShape[id]; });

  phiStrides[0] = 1;
  ForEach<d>([&](auto id) {
    if constexpr (id == 0)
      return;
    phiStrides[id] = phiStrides[id - 1] * static_cast<isize>(phiShape[id - 1]);
  });

  ForEach<d>([&](auto id) { uStrides[id] = phiStrides[id], pStrides[id] = phiStrides[id]; });
  uStrides[d] = static_cast<isize>(size);

  {
    nb::gil_scoped_release release;

    check(cudaMemcpyAsync(
        thrust::raw_pointer_cast(storage().phiHost.data()), thrust::raw_pointer_cast(storage().phiDevice.data()),
        size * sizeof(Real), cudaMemcpyDeviceToHost
    ));

    ForEach<d>([&](auto id) {
      check(cudaMemcpyAsync(
          thrust::raw_pointer_cast(storage().uHost.data()) + id * size,
          thrust::raw_pointer_cast(storage().uDevice[id].data()), size * sizeof(Real), cudaMemcpyDeviceToHost
      ));
    });

    check(cudaMemcpyAsync(
        thrust::raw_pointer_cast(storage().pHost.data()), thrust::raw_pointer_cast(storage().pDevice.data()),
        size * sizeof(Real), cudaMemcpyDeviceToHost
    ));

    check(cudaDeviceSynchronize());
  }

  auto owner = nb::find(*this);
  return {
      .time = static_cast<Real>(time()),
      .phi = {thrust::raw_pointer_cast(storage().phiHost.data()), d, phiShape.data(), owner, phiStrides.data()},
      .u = {thrust::raw_pointer_cast(storage().uHost.data()), d + 1, uShape.data(), owner, uStrides.data()},
      .p = {thrust::raw_pointer_cast(storage().pHost.data()), d, pShape.data(), owner, pStrides.data()},
  };
}

} // namespace igks::tp
