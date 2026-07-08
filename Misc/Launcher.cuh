#pragma once

#include "Layout.hpp"

#include <cuda_runtime_api.h>
#include <stdexcept>

namespace igks::tp {

namespace detail::launcher {

template <is::Integral TIdx, typename F>
GLOBAL static void KernelLaunchIdx(TIdx size, F f) {
  usize i = static_cast<usize>(threadIdx.x) + static_cast<usize>(blockIdx.x) * static_cast<usize>(blockDim.x);
  if (i < static_cast<usize>(size))
    f(static_cast<TIdx>(i));
}

template <Layout TLayout, typename F>
GLOBAL static void KernelLaunchLayout(TLayout layout, usize size, F f) {
  usize i = static_cast<usize>(threadIdx.x) + static_cast<usize>(blockIdx.x) * static_cast<usize>(blockDim.x);
  if (i < size) {
    if constexpr (requires { f(value_type::Cast<u64>(layout(i))); })
      f(value_type::Cast<u64>(layout(i)));
    else if constexpr (requires { f(value_type::Cast<i64>(layout(i))); })
      f(value_type::Cast<i64>(layout(i)));
    else if constexpr (requires { f(value_type::Cast<u32>(layout(i))); })
      f(value_type::Cast<u32>(layout(i)));
    else if constexpr (requires { f(value_type::Cast<i32>(layout(i))); })
      f(value_type::Cast<i32>(layout(i)));
  }
}

//
//
//
template <typename Derived>
class LauncherBase {
private:
  [[nodiscard]] Derived &derived() { return *static_cast<Derived *>(this); }

  [[nodiscard]] Derived const &derived() const { return *static_cast<Derived const *>(this); }

protected:
  LauncherBase() = default;

private:
  static constexpr usize defaultBlockSize = 256;

  usize blockSize_ = defaultBlockSize;
  usize overallSize_ = 0;

protected:
  [[nodiscard]] usize blockSize() const { return blockSize_; }

  decltype(auto) blockSize(usize v) {
    blockSize_ = v;
    return derived();
  }

  [[nodiscard]] usize overallSize() const { return overallSize_; }

  decltype(auto) overallSize(usize v) {
    overallSize_ = v;
    return derived();
  }

protected:
  template <typename Kernel, typename... Args>
  void Launch(Kernel &&kernel, Args &&...args) {
    if (overallSize_ == 0 || blockSize_ == 0)
      return;

    auto check = [](cudaError_t error) {
      if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    };

    usize gridSize = (overallSize_ + blockSize_ - 1) / blockSize_;
    if (gridSize > static_cast<usize>(std::numeric_limits<unsigned int>::max()))
      check(cudaErrorInvalidConfiguration);

    kernel<<<static_cast<unsigned int>(gridSize), static_cast<unsigned int>(blockSize_)>>>(std::forward<Args>(args)...);
    check(cudaGetLastError());
  }
};

} // namespace detail::launcher

//
//
//
//
//
template <typename... Ts>
class Launcher;

template <typename Kernel>
class Launcher<Kernel> : public detail::launcher::LauncherBase<Launcher<Kernel>> {
private:
  using Base = detail::launcher::LauncherBase<Launcher<Kernel>>;

public:
  explicit Launcher(Kernel const &kernel) : kernel_(kernel) {}

public:
  using Base::blockSize;
  using Base::overallSize;

  template <typename... Args>
  void Launch(Args &&...args) {
    Base::Launch(kernel_, std::forward<Args>(args)...);
  }

private:
  Kernel const &kernel_;
};

template <typename Kernel>
Launcher(Kernel const &kernel) -> Launcher<Kernel>;

//
//
//
template <is::Integral TIdx, typename F>
class Launcher<TIdx, F> : public detail::launcher::LauncherBase<Launcher<TIdx, F>> {
private:
  using Base = detail::launcher::LauncherBase<Launcher<TIdx, F>>;

public:
  Launcher(TIdx const &size, F const &f) : f_(f) { Base::overallSize(size); }

public:
  using Base::blockSize;

  void Launch() {
    Base::Launch(detail::launcher::KernelLaunchIdx<TIdx, F>, static_cast<TIdx>(Base::overallSize()), f_);
  }

private:
  F f_;
};

template <is::Integral TIdx, typename F>
Launcher(TIdx const &size, F const &f) -> Launcher<TIdx, F>;

//
//
//
template <Layout TLayout, typename F>
class Launcher<TLayout, F> : public detail::launcher::LauncherBase<Launcher<TLayout, F>> {
private:
  using Base = detail::launcher::LauncherBase<Launcher<TLayout, F>>;

public:
  Launcher(TLayout const &layout, F const &f) : layout_(layout), f_(f) {
    usize size = [&]<usize... row>(std::index_sequence<row...>) constexpr {
      return (static_cast<usize>(1) * ... * layout_.size()(C<row>{}));
    }(std::make_index_sequence<decltype(layout_.size())::rows()>{});
    Base::overallSize(size);
  }

public:
  using Base::blockSize;

  void Launch() { Base::Launch(detail::launcher::KernelLaunchLayout<TLayout, F>, layout_, Base::overallSize(), f_); }

private:
  TLayout layout_;
  F f_;
};

template <Layout TLayout, typename F>
Launcher(TLayout const &layout, F const &f) -> Launcher<TLayout, F>;

} // namespace igks::tp
