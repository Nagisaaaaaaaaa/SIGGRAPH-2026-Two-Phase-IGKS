#include "../Simulator.hpp"
#include "Misc/Launcher.cuh"

namespace igks::tp {

void Simulator::AllocAndInit() {
  storage() = GridStorage(descriptor());

  Launcher(accessor().layout(), [g = accessor()] DEVICE(Crd const &crd) mutable {
    Real phi = g.phi()(0);
    auto u = VecDr<>::Constant<0.0_R>();
    Real p = 0_R;

    auto pos = value_type::Cast<Real>(crd) + VecDr<>::Constant<0.5_R>();
    auto res = value_type::Cast<Real>(g.resolution());

#if 1
    // Rayleigh-Taylor instability.
    {
      pos(1) -= g.resolution()(1) / 2_R;

      if constexpr (d == 2) {
        auto y0 = 0.1_R * res(1) / 4_R * std::cos(2_R * pi<Real> * pos(0) / res(0));
        phi = (g.phi()(1) + g.phi()(0)) / 2_R +
              (g.phi()(1) - g.phi()(0)) / 2_R * std::tanh(2_R * (pos(1) - y0) / g.interfaceWidth());
      } else if constexpr (d == 3) {
        auto y0 = 0.05_R * res(1) / 4_R *
                  (std::cos(2_R * pi<Real> * pos(0) / res(0)) + std::cos(2_R * pi<Real> * pos(2) / res(2)));
        phi = (g.phi()(1) + g.phi()(0)) / 2_R +
              (g.phi()(1) - g.phi()(0)) / 2_R * std::tanh(2_R * (pos(1) - y0) / g.interfaceWidth());
      }
    }
#elif 0
    // Dam break.
    { phi = (pos(0) < res(0) / 4_R && pos(1) < res(1) * (2_R / 3_R)) ? g.phi()(1) : g.phi()(0); }
#endif

    g.phi(crd, phi);
    g.phiTemp(crd, phi);
    g.u(crd, u);
    g.uTemp(crd, u);
    g.p(crd, p);
    g.pTemp(crd, p);
  }).Launch();
}

} // namespace igks::tp
