#include "../Simulator.hpp"
#include "Misc/CartesianInterface.hpp"
#include "Misc/Launcher.cuh"
#include "Misc/MaxwellBoltzmannDistribution.hpp"

#include <thrust/transform_reduce.h>

namespace igks::tp {

void Simulator::SimulateOneStep() {
#if 0
  descriptor().dt() = 0.5_R;
#elif 1
  auto first = thrust::make_counting_iterator<usize>(0);
  auto last = first + storage().uDevice[0].size();
  auto uNormMax = thrust::transform_reduce(first, last, [g = accessor()] DEVICE(usize i) -> Real {
    auto norm = [](auto const &v) { return std::sqrt(v.Dot(v)); };
    return norm(g.u(g.layout()(static_cast<isize>(i))));
  }, 0_R, cuda::maximum<Real>());
  descriptor().dt() = cfl() / (cs + uNormMax);
#elif 0
  auto first = thrust::make_counting_iterator<usize>(0);
  auto last = first + storage().uDevice[0].size();
  auto spectralRadiusSumMax = thrust::transform_reduce(first, last, [g = accessor()] DEVICE(usize i) -> Real {
    auto u = g.u(g.layout()(static_cast<isize>(i)));
    auto sum = static_cast<Real>(d) * cs;
    ForEach<d>([&](auto id) { sum += std::abs(u(id)); });
    return sum;
  }, 0_R, cuda::maximum<Real>());
  descriptor().dt() = cfl() / spectralRadiusSumMax;
#endif

  Launcher(accessor().layout(), [g = accessor()] DEVICE(Crd const &crd) mutable {
    auto bcsWorld = [&](auto s) { return g.bc(crd + s); };
    auto phisWorld = [&](auto s) { return g.phi(crd + s); };
    auto usWorld = [&](auto s) { return g.u(crd + s); };
    auto psWorld = [&](auto s) { return g.p(crd + s); };

    auto grad = [](auto const &vs) {
      if constexpr (d == 2) {
        using T = std::decay_t<decltype(vs(_oo))>;

        T pvpx = (1_R / 12_R) * (((vs(_pp) - vs(_nn)) + (vs(_pn) - vs(_np)))) + (1_R / 3_R) * (vs(_po) - vs(_no));
        T pvpy = (1_R / 12_R) * (((vs(_pp) - vs(_nn)) + (vs(_np) - vs(_pn)))) + (1_R / 3_R) * (vs(_op) - vs(_on));

        if constexpr (is::Vec<T>) {
          MatDr<>::AsStorage res;
          res(0, 0) = pvpx(0), res(1, 0) = pvpx(1), res(0, 1) = pvpy(0), res(1, 1) = pvpy(1);
          return res;
        } else {
          return Mat{pvpx, pvpy};
        }
      } else if constexpr (d == 3) {
        // TODO: Your code here.
      }
    };

    auto laplace = [](auto const &vs) {
      if constexpr (d == 2) {
        using T = std::decay_t<decltype(vs(_oo))>;

        T voo = vs(_oo);
        T res = (1_R / 6_R) * ((((vs(_pp) - voo) + (vs(_nn) - voo)) + ((vs(_pn) - voo) + (vs(_np) - voo)))) +
                (2_R / 3_R) * ((((vs(_po) - voo) + (vs(_no) - voo)) + ((vs(_op) - voo) + (vs(_on) - voo))));
        return res;
      } else if constexpr (d == 3) {
        // TODO: Your code here.
      }
    };

    Real totalFlux_phi{0_R};
    Real totalFlux_phi_regularized{0_R};
    VecDr<>::AsStorage totalFlux_rhou{VecDr<>::Zero()};
    Real totalFlux_p_div_cs2{0_R};

    ForEach<CartesianInterfaces<d>>([&]<typename Interface>() -> void {
      auto bcs = [&](auto s) { return Interface::World2Local(bcsWorld(Interface::Local2World(s))); };
      auto phis = [&](auto s) { return Interface::World2Local(phisWorld(Interface::Local2World(s))); };
      auto us = [&](auto s) { return Interface::World2Local(usWorld(Interface::Local2World(s))); };
      auto ps = [&](auto s) { return Interface::World2Local(psWorld(Interface::Local2World(s))); };

      auto grad = [](auto const &vs) {
        if constexpr (d == 2) {
          using T = std::decay_t<decltype(vs(_oo))>;

          T pvpx = vs(_po) - vs(_oo);
          T pvpy = ((vs(_op) + vs(_pp)) - (vs(_on) + vs(_pn))) / 4_R;

          if constexpr (is::Vec<T>) {
            MatDr<>::AsStorage res;
            res(0, 0) = pvpx(0), res(1, 0) = pvpx(1), res(0, 1) = pvpy(0), res(1, 1) = pvpy(1);
            return res;
          } else {
            return Mat{pvpx, pvpy};
          }
        } else if constexpr (d == 3) {
          // TODO: Your code here.
        }
      };

      auto laplace = [](auto const &vs) {
        if constexpr (d == 2) {
          using T = std::decay_t<decltype(vs(_oo))>;

          T res = /*0_R +*/ (((vs(_op) + vs(_on)) - 2_R * vs(_oo)) + ((vs(_pp) + vs(_pn)) - 2_R * vs(_po))) / 2_R;
          return res;
        } else if constexpr (d == 3) {
          // TODO: Your code here.
        }
      };

      // TODO: You can also try interpolating with conservative variables, as in classical FVM methods.
      auto phi = (phis($o) + phis($p)) / 2_R;
      auto u = (us($o) + us($p)) / 2_R;
      auto p = (ps($o) + ps($p)) / 2_R;

      auto rho = g.rho(std::clamp(phi, g.phi()(0), g.phi()(1)));
      auto tau = g.tau(std::clamp(phi, g.phi()(0), g.phi()(1)));

      auto grad_phi = grad(phis);
      auto grad_rho = (g.rho()(1) - g.rho()(0)) * grad_phi;
      auto grad_p = grad(ps);
      auto grad_psi_div_cs2 = grad_p * cs2Inv - grad_rho;
      auto grad_u = grad(us);
      auto laplace_phi = laplace(phis);

      auto force_rhou = (48_R * (g.sigma() / g.interfaceWidth()) *
                             (((phi - g.phi()(0)) * (phi - g.phi()(1))) * (phi - (g.phi()(0) + g.phi()(1)) / 2_R)) -
                         1.5_R * (g.sigma() * g.interfaceWidth()) * laplace_phi) *
                            grad_phi +
                        rho * Interface::World2Local(g.gravity());
      auto fb_div_cs2 = -grad_psi_div_cs2 + force_rhou * cs2Inv;

      auto m0 = [](auto order, auto r012) {
        return MaxwellBoltzmannDistribution<d, lambda>::Moment<decltype(order + r012), $O>(VecDr<>::Zero());
      };
      auto m = [&u](auto order, auto r012) {
        return MaxwellBoltzmannDistribution<d, lambda>::Moment<decltype(order + r012), $O>(u);
      };
      auto geq = [&](auto order, auto r012) { return rho * m(order, r012) + (p * cs2Inv - rho) * m0(order, r012); };

      // Compute "a", "b", ..., and "A".
      auto solve = [&](auto const &h, auto &a) {
        // Compute a1, a2, ...
        ForEach<d>([&](auto id) { a(id + i1) = cs2Inv * (h(id + i1) - u(id) * h(0)); });

        // Compute a0.
        a(i0) = 0;
        ForEach<d>([&](auto id) { a(i0) += u(id) * a(id + i1); });
        a(i0) = h(0) - a(i0);
      };

      Vec<Real, d + 1>::AsStorage a;
      Real ap;
      if constexpr (d >= 1) {
        a(0) = 0;
        ForEach<d>([&](auto id) { a(0) += u(id) * grad_u(id, 0); });
        a(0) = grad_rho(0) - rho * a(0) * cs2Inv;
        ForEach<d>([&](auto id) { a(id + 1) = rho * grad_u(id, 0) * cs2Inv; });
        ap = grad_p(0) * cs2Inv - grad_rho(0);
      }

      [[maybe_unused]] Vec<Real, d + 1>::AsStorage b;
      [[maybe_unused]] Real bp;
      if constexpr (d >= 2) {
        b(0) = 0;
        ForEach<d>([&](auto id) { b(0) += u(id) * grad_u(id, 1); });
        b(0) = grad_rho(1) - rho * b(0) * cs2Inv;
        ForEach<d>([&](auto id) { b(id + 1) = rho * grad_u(id, 1) * cs2Inv; });
        bp = grad_p(1) * cs2Inv - grad_rho(1);
      }

      [[maybe_unused]] Vec<Real, d + 1>::AsStorage c;
      [[maybe_unused]] Real cp;
      // TODO: Your code here.

      Vec<Real, d + 1>::AsStorage A;
      Real Ap = -u.Dot(grad_psi_div_cs2);
      {
        Vec<Real, d + 1>::AsStorage h;
        auto hi = [&](auto r012) {
          if constexpr (d == 2)
            return -(
                ((a(0) * m(_10, r012) + b(0) * m(_01, r012)) + (a(1) * m(_20, r012) + b(2) * m(_02, r012)) +
                 (a(2) + b(1)) * m(_11, r012)) +
                (ap * m0(_10, r012) + bp * m0(_01, r012))
            );
          // else if constexpr (d == 3)
          // TODO: Your code here.
        };
        ForEach<d + 1>([&](auto id) { h(id) = hi($r(id)); });
        h(0) += -u.Dot(grad_psi_div_cs2) - Ap;
        ForEach<d>([&](auto id) { h(id + 1) += force_rhou(id); });
        solve(h, A);
      }

      // Compute flux.
      auto flux = [&](auto r012) {
        if constexpr (d == 2)
          return geq(_10, r012) -
                 tau * (((a(0) * m(_20, r012) + b(0) * m(_11, r012)) + (a(1) * m(_30, r012) + b(2) * m(_12, r012)) +
                         (a(2) + b(1)) * m(_21, r012)) +
                        (ap * m0(_20, r012) + bp * m0(_11, r012))) +
                 (g.dt() / 2_R - tau) *
                     ((A(0) * m(_10, r012) + (A(1) * m(_20, r012) + A(2) * m(_11, r012))) + Ap * m0(_10, r012)) +
                 tau * (-(m(_10, r012) * u.Dot(fb_div_cs2) + m0(_10, r012) * u.Dot(grad_psi_div_cs2)) +
                        (Mat{m(_20, r012), m(_11, r012)}.Dot(fb_div_cs2) +
                         Mat{m0(_20, r012), m0(_11, r012)}.Dot(grad_psi_div_cs2)));
        // else if constexpr (d == 3)
        // TODO: Your code here.
      };

      auto flux_p_div_cs2 = flux($r0);
      VecDr<>::AsStorage flux_rhou;
      ForEach<d>([&](auto id) { flux_rhou(id) = flux($r(id + i1)); });

      auto dir_phi = grad_phi;
      dir_phi(0) = ((phis($p) + phis($q)) - (phis($n) + phis($o))) / 4_R;
      auto norm = [](auto const &v) { return std::sqrt(v.Dot(v)); };

      auto flux_phi = flux_p_div_cs2 / (g.rho()(1) - g.rho()(0));
      auto flux_phi_regularized =
          flux_phi -
          (bcs($p) != BC::noSlip
               ? g.mobility() * (grad_phi(0) - (4_R / g.interfaceWidth()) * ((phi - g.phi()(0)) * (g.phi()(1) - phi)) *
                                                   (dir_phi(0) / (norm(dir_phi) + 1e-12_R))) // TODO: Magic number here.
               : 0_R);

      totalFlux_phi += Interface::Local2World(flux_phi);
      totalFlux_phi_regularized += Interface::Local2World(flux_phi_regularized);
      totalFlux_rhou += Interface::Local2World(flux_rhou);
      totalFlux_p_div_cs2 += Interface::Local2World(flux_p_div_cs2);
    });

    auto phi = phisWorld($o);
    auto u = usWorld($o);
    auto p = psWorld($o);

    auto rho = g.rho(std::clamp(phi, g.phi()(0), g.phi()(1)));

    auto grad_phi = grad(phisWorld);
    auto grad_p = grad(psWorld);
    auto grad_rho = (g.rho()(1) - g.rho()(0)) * grad_phi;
    auto grad_psi = grad_p - grad_rho * cs2;
    auto laplace_phi = laplace(phisWorld);

    auto force_rhou_Fs = (48_R * (g.sigma() / g.interfaceWidth()) *
                              (((phi - g.phi()(0)) * (phi - g.phi()(1))) * (phi - (g.phi()(0) + g.phi()(1)) / 2_R)) -
                          1.5_R * (g.sigma() * g.interfaceWidth()) * laplace_phi) *
                         grad_phi;
    VecDr<>::AsStorage force_rhou_Fa{VecDr<>::Zero()};
    // Apply velocity filter.
#if 0
    ForEach<d>([&](auto id) {
      constexpr auto sP = value_type::Cast<isize>($r(id + i1));
      constexpr auto sN = -sP;

      auto uPH = (u + usWorld(sP)) / 2_R;
      auto uNH = (u + usWorld(sN)) / 2_R;
      auto phiPH = (phi + phisWorld(sP)) / 2_R;
      auto phiNH = (phi + phisWorld(sN)) / 2_R;
      auto nuPH = g.tau(std::clamp(phiPH, g.phi()(0), g.phi()(1))) * cs2;
      auto nuNH = g.tau(std::clamp(phiNH, g.phi()(0), g.phi()(1))) * cs2;

      if (std::abs(uPH(id) / nuPH) >= 2_R)
        force_rhou_Fa += std::abs(uPH(id)) * (usWorld(sP) - u);
      if (std::abs(uNH(id) / nuNH) >= 2_R)
        force_rhou_Fa += std::abs(uNH(id)) * (usWorld(sN) - u);
    });
    force_rhou_Fa *= rho / 2_R;
#endif
    auto force_u_G = g.gravity();
    auto force_p = -u.Dot(grad_psi);

    auto phiPost = phi - totalFlux_phi * g.dt();
    auto rhoPost = g.rho(std::clamp(phiPost, g.phi()(0), g.phi()(1)));

    auto phiPost_regularized = phi - totalFlux_phi_regularized * g.dt();
    auto uPost =
        (rho * u - totalFlux_rhou * g.dt() + (force_rhou_Fs + force_rhou_Fa) * g.dt()) / rhoPost + force_u_G * g.dt();
    auto pPost = p - totalFlux_p_div_cs2 * cs2 * g.dt() + force_p * g.dt();

    g.phiTemp(crd, phiPost_regularized);
    g.uTemp(crd, uPost);
    g.pTemp(crd, pPost);
  }).Launch();

  // Swap storages.
  {
    using std::swap;
    swap(storage().phiDevice, storage().phiTempDevice);
    swap(storage().uDevice, storage().uTempDevice);
    swap(storage().pDevice, storage().pTempDevice);
  }

  // Apply pressure normalization.
#if 1
  auto pSum = thrust::reduce(storage().pDevice.begin(), storage().pDevice.end(), 0_R);
  auto pAvg = pSum / static_cast<Real>(storage().pDevice.size());
  Launcher(accessor().layout(), [g = accessor(), pAvg] DEVICE(Crd const &crd) mutable {
    g.p(crd, g.p(crd) - pAvg);
  }).Launch();
#endif

  // Apply pressure filter.
#if 1
  Launcher(accessor().layout(), [g = accessor()] DEVICE(Crd const &crd) mutable {
    auto ps = [&](auto s) { return g.p(crd + s); };

    auto filter = [](auto const &vs) {
      if constexpr (d == 2)
        return (1_R / 36_R) * ((vs(_pp) + vs(_nn)) + (vs(_pn) + vs(_np))) +
               (1_R / 9_R) * ((vs(_po) + vs(_no)) + (vs(_op) + vs(_on))) + (4_R / 9_R) * vs(_oo);
      // else if constexpr (d == 3)
      // TODO: Your code here.
    };
    auto pFiltered = filter(ps);

    g.pTemp(crd, pFiltered);
  }).Launch();

  // Swap storages.
  {
    using std::swap;
    swap(storage().pDevice, storage().pTempDevice);
  }
#endif

  time() += descriptor().dt();
}

} // namespace igks::tp
