#pragma once

#include "Misc/Config.hpp"
#include "Misc/MDView.hpp"

#include <nanobind/ndarray.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace igks::tp {

/// \brief Speed of sound.
CONSTANT static constexpr Real cs = std::numbers::inv_sqrt3_v<Real>;

/// \brief Squared speed of sound.
CONSTANT static constexpr Real cs2 = 1.0 / 3.0;

/// \brief Inverse of (squared speed of sound).
CONSTANT static constexpr Real cs2Inv = 3.0;

/// \brief Inverse of (2 times squared speed of sound).
CONSTANT static constexpr Real lambda = 1.5;

namespace nb = nanobind;

using Crd = VecD<isize>::AsStorage;

enum class BC : u8 { none = 0, noSlip, periodic };

//
//
//
//
//
class GridDescriptor {
public:
  [[nodiscard]] HOST_DEVICE Crd resolution() const { return value_type::Cast<isize>(layout_.size()); }

  [[nodiscard]] HOST_DEVICE auto const &layout() const { return layout_; }

  [[nodiscard]] HOST_DEVICE auto &layout() { return layout_; }

  [[nodiscard]] HOST_DEVICE auto const &rho() const { return rho_; }

  [[nodiscard]] HOST_DEVICE auto &rho() { return rho_; }

  [[nodiscard]] HOST_DEVICE auto const &tau() const { return tau_; }

  [[nodiscard]] HOST_DEVICE auto &tau() { return tau_; }

  [[nodiscard]] HOST_DEVICE auto const &gravity() const { return gravity_; }

  [[nodiscard]] HOST_DEVICE auto &gravity() { return gravity_; }

  [[nodiscard]] HOST_DEVICE auto const &interfaceWidth() const { return interfaceWidth_; }

  [[nodiscard]] HOST_DEVICE auto &interfaceWidth() { return interfaceWidth_; }

  [[nodiscard]] HOST_DEVICE auto const &mobility() const { return mobility_; }

  [[nodiscard]] HOST_DEVICE auto &mobility() { return mobility_; }

  [[nodiscard]] HOST_DEVICE auto const &sigma() const { return sigma_; }

  [[nodiscard]] HOST_DEVICE auto &sigma() { return sigma_; }

  [[nodiscard]] HOST_DEVICE auto const &dt() const { return dt_; }

  [[nodiscard]] HOST_DEVICE auto &dt() { return dt_; }

public:
  [[nodiscard]] HOST_DEVICE auto phi() const { return Mat{0_R, 1_R}; }

  [[nodiscard]] HOST_DEVICE Real rho(Real const &phiClamped) const {
    return rho().Dot(Mat{phi()(1) - phiClamped, phiClamped - phi()(0)});
  }

  [[nodiscard]] HOST_DEVICE Real tau(Real const &phiClamped) const {
    auto nu = tau() * cs2;
    Vec2r<>::AsStorage mu;
    ForEach<2>([&](auto id) { mu(id) = rho()(id) * nu(id); });
    auto res_mu = mu.Dot(Mat{phi()(1) - phiClamped, phiClamped - phi()(0)});
    auto res_nu = res_mu / rho(phiClamped);
    return res_nu * cs2Inv;
  }

private:
  LeftLayout<Vec<usize, d>::AsStorage> layout_{};
  Vec2r<>::AsStorage rho_{};
  Vec2r<>::AsStorage tau_{};
  VecDr<>::AsStorage gravity_{};
  Real interfaceWidth_{};
  Real mobility_{};
  Real sigma_{};

  Real dt_{};
};

//
//
//
struct GridStorage {
  // TODO: Should never use thrust.
  thrust::device_vector<Real> phiDevice;
  thrust::device_vector<Real> phiTempDevice;
  std::array<thrust::device_vector<Real>, d> uDevice;
  std::array<thrust::device_vector<Real>, d> uTempDevice;
  thrust::device_vector<Real> pDevice;
  thrust::device_vector<Real> pTempDevice;

  thrust::host_vector<Real> phiHost;
  thrust::host_vector<Real> uHost; // TODO: Better to use `VecDr<>::AsStorage` here.
  thrust::host_vector<Real> pHost;

  GridStorage() = default;

  explicit GridStorage(GridDescriptor const &descriptor) {
    usize nCells = 1;
    ForEach<d>([&](auto id) { nCells *= descriptor.resolution()(id); });

    phiDevice.resize(nCells);
    phiTempDevice.resize(nCells);
    ForEach<d>([&](auto id) { uDevice[id].resize(nCells); });
    ForEach<d>([&](auto id) { uTempDevice[id].resize(nCells); });
    pDevice.resize(nCells);
    pTempDevice.resize(nCells);

    phiHost.resize(nCells);
    uHost.resize(d * nCells);
    pHost.resize(nCells);
  }
};

//
//
//
class GridAccessor : public GridDescriptor {
public:
  using GridDescriptor::phi;

  GridAccessor() = default;

  GridAccessor(GridDescriptor const &descriptor, GridStorage &storage) : GridDescriptor(descriptor) {
    phi_ = storage.phiDevice.data().get();
    phiTemp_ = storage.phiTempDevice.data().get();
    ForEach<d>([&](auto id) { u_[id] = storage.uDevice[id].data().get(); });
    ForEach<d>([&](auto id) { uTemp_[id] = storage.uTempDevice[id].data().get(); });
    p_ = storage.pDevice.data().get();
    pTemp_ = storage.pTempDevice.data().get();
  }

public:
#if 1
  // Rayleigh-Taylor instability.
  [[nodiscard]] HOST_DEVICE auto bc(Crd const &crd) const {
    bool isOutOfDomainAtAxis1 = crd(1) < 0 || crd(1) >= resolution()(1);
    bool isOutOfDomainExceptAxis1 = false;
    ForEach<d>([&](auto id) {
      if constexpr (id == 1)
        return;
      isOutOfDomainExceptAxis1 |= crd(id) < 0 || crd(id) >= resolution()(id);
    });

    BC res;
    if (isOutOfDomainAtAxis1) [[unlikely]]
      res = BC::noSlip;
    else if (isOutOfDomainExceptAxis1) [[unlikely]]
      res = BC::periodic;
    else
      res = BC::none;
    return res;
  }

  [[nodiscard]] HOST_DEVICE auto phi(Crd const &crd) const {
    Crd crdMapped{crd};
    ForEach<d>([&](auto id) {
      if constexpr (id == 1) {
        if (crdMapped(id) < 0)
          crdMapped(id) = -crdMapped(id) - 1;
        else if (crdMapped(id) >= resolution()(id))
          crdMapped(id) = 2 * resolution()(id) - crdMapped(id) - 1;
      } else {
        if (crdMapped(id) < 0)
          crdMapped(id) += resolution()(id);
        else if (crdMapped(id) >= resolution()(id))
          crdMapped(id) -= resolution()(id);
      }
    });

    return phi_[layout()(crdMapped)];
  }

  [[nodiscard]] HOST_DEVICE auto u(Crd const &crd) const {
    Crd crdMapped{crd};
    int flip = 1;
    ForEach<d>([&](auto id) {
      if constexpr (id == 1) {
        if (crdMapped(id) < 0) {
          crdMapped(id) = -crdMapped(id) - 1;
          flip *= -1;
        } else if (crdMapped(id) >= resolution()(id)) {
          crdMapped(id) = 2 * resolution()(id) - crdMapped(id) - 1;
          flip *= -1;
        }
      } else {
        if (crdMapped(id) < 0)
          crdMapped(id) += resolution()(id);
        else if (crdMapped(id) >= resolution()(id))
          crdMapped(id) -= resolution()(id);
      }
    });

    auto uMapped = [&]<usize... id>(std::index_sequence<id...>) {
      return Mat{u_[id][layout()(crdMapped)]...};
    }(std::make_index_sequence<d>{});

    return flip * uMapped;
  }

  [[nodiscard]] HOST_DEVICE auto p(Crd const &crd) const {
    Crd crdMapped{crd};
    ForEach<d>([&](auto id) {
      if constexpr (id == 1) {
        if (crdMapped(id) < 0)
          crdMapped(id) = -crdMapped(id) - 1;
        else if (crdMapped(id) >= resolution()(id))
          crdMapped(id) = 2 * resolution()(id) - crdMapped(id) - 1;
      } else {
        if (crdMapped(id) < 0)
          crdMapped(id) += resolution()(id);
        else if (crdMapped(id) >= resolution()(id))
          crdMapped(id) -= resolution()(id);
      }
    });

    return p_[layout()(crdMapped)];
  }
#elif 0
  // Dam break.
  // TODO: Your code here.
#endif

  HOST_DEVICE void phi(Crd const &crd, Real v) { phi_[layout()(crd)] = v; }

  HOST_DEVICE void u(Crd const &crd, is::Vec auto const &v) const {
    auto const i = layout()(crd);
    ForEach<d>([&](auto id) { u_[id][i] = v(id); });
  }

  HOST_DEVICE void p(Crd const &crd, Real v) { p_[layout()(crd)] = v; }

  HOST_DEVICE void phiTemp(Crd const &crd, Real v) { phiTemp_[layout()(crd)] = v; }

  HOST_DEVICE void uTemp(Crd const &crd, is::Vec auto const &v) const {
    auto const i = layout()(crd);
    ForEach<d>([&](auto id) { uTemp_[id][i] = v(id); });
  }

  HOST_DEVICE void pTemp(Crd const &crd, Real v) { pTemp_[layout()(crd)] = v; }

private:
  Real *phi_{};
  Real *phiTemp_{};
  std::array<Real *, d> u_{};
  std::array<Real *, d> uTemp_{};
  Real *p_{};
  Real *pTemp_{};
};

//
//
//
//
//
struct SimulatorConfig {
  std::array<isize, d> resolution{};
  std::array<Real, 2> rho{};
  std::array<Real, 2> nu{};
  std::array<Real, d> gravity{};
  Real interfaceWidth{};
  Real mobility{};
  Real sigma{};
  Real cfl{};
};

struct SimulatorStatus {
  Real time{};
  nb::ndarray<nb::numpy, Real, nb::ndim<d>> phi;
  nb::ndarray<nb::numpy, Real, nb::ndim<d + 1>> u;
  nb::ndarray<nb::numpy, Real, nb::ndim<d>> p;
};

class Simulator {
public:
  [[nodiscard]] auto const &cfl() const { return cfl_; }

  [[nodiscard]] auto &cfl() { return cfl_; }

  [[nodiscard]] auto const &time() const { return time_; }

  [[nodiscard]] auto &time() { return time_; }

  [[nodiscard]] auto const &descriptor() const { return descriptor_; }

  [[nodiscard]] auto &descriptor() { return descriptor_; }

  [[nodiscard]] auto const &storage() const { return storage_; }

  [[nodiscard]] auto &storage() { return storage_; }

  [[nodiscard]] GridAccessor accessor() { return {descriptor(), storage()}; }

public:
  void Config(SimulatorConfig const &config);
  void AllocAndInit();
  void ApplySource();
  void SimulateOneStep();
  [[nodiscard]] SimulatorStatus Status();

private:
  Real cfl_{};

  double time_{};

  GridDescriptor descriptor_{};
  GridStorage storage_{};
};

//
//
//
//
//
// Shortcuts.
using $R0 = std::conditional_t<
    d == 1,
    std::decay_t<decltype(_0)>,
    std::conditional_t<
        d == 2,
        std::decay_t<decltype(_00)>,
        std::conditional_t<d == 3, std::decay_t<decltype(_000)>, void>>>;
using $R1 = std::conditional_t<
    d == 1,
    std::decay_t<decltype(_1)>,
    std::conditional_t<
        d == 2,
        std::decay_t<decltype(_10)>,
        std::conditional_t<d == 3, std::decay_t<decltype(_100)>, void>>>;
using $R2 = std::conditional_t<
    d == 1,
    C<static_cast<usize>(0)>,
    std::conditional_t<
        d == 2,
        std::decay_t<decltype(_01)>,
        std::conditional_t<d == 3, std::decay_t<decltype(_010)>, void>>>;
using $R3 = std::conditional_t<
    d == 1,
    C<static_cast<usize>(0)>,
    std::conditional_t<
        d == 2,
        C<static_cast<usize>(0)>,
        std::conditional_t<d == 3, std::decay_t<decltype(_001)>, void>>>;

CONSTANT static constexpr $R0 $r0;
CONSTANT static constexpr $R1 $r1;
CONSTANT static constexpr $R2 $r2;
CONSTANT static constexpr $R3 $r3;

[[nodiscard]] constexpr auto $r(is::C auto id) {
  if constexpr (id == 0)
    return $r0;
  else if constexpr (id == 1)
    return $r1;
  else if constexpr (id == 2)
    return $r2;
  else if constexpr (id == 3)
    return $r3;
}

} // namespace igks::tp
