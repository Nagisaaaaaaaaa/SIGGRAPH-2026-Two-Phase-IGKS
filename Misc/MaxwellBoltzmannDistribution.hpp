#pragma once

#include "Base.hpp"

namespace igks::tp {

namespace detail::maxwell_boltzmann_distribution {

// Check template parameters for `Moment` at compile time.
template <typename Real, usize dim, typename TOrder, typename TDomain, typename TU>
static consteval void StaticTestMoment() {
  static_assert(is::Vec<TOrder> && is::Vec<TDomain> && is::Vec<TU>, "Template parameters of `Moment` must be `Vec`s");

  static_assert(is::Empty<TOrder>, "`TOrder` must be static");
  static_assert(is::Empty<TDomain>, "`TDomain` must be static");

  static_assert(
      is::Same<typename TOrder::value_type, usize> && TOrder::rows() == dim,
      "`TOrder` must be a `dim`-dimensional `usize` vector"
  );
  static_assert(
      is::Same<typename TDomain::value_type, isize> && TDomain::rows() == dim,
      "`TDomain` must be a `dim`-dimensional `isize` vector"
  );
  static_assert(
      is::Same<typename TU::value_type, Real> && TU::rows() == dim, "`TU` must be a `dim`-dimensional `Real` vector"
  );

  ForEach<dim>([](auto i) {
    constexpr isize domainI = TDomain{}(i);
    static_assert(domainI == -1 || domainI == 0 || domainI == 1, "Domain entries must be -1, 0, or 1");
  });
}

//
//
//
template <typename Constants>
struct except_tail_constants;

template <typename... ConstantsPerCol>
struct except_tail_constants<Types<ConstantsPerCol...>> {
  using type = Types<typename ConstantsPerCol::template PopBack<>...>;
};

template <typename Constants>
using except_tail_constants_t = typename except_tail_constants<Constants>::type;

template <typename Constants>
struct tail_constants;

template <typename... ConstantsPerCol>
struct tail_constants<Types<ConstantsPerCol...>> {
  using type = Types<Types<typename ConstantsPerCol::template Back<>>...>;
};

template <typename Constants>
using tail_constants_t = typename tail_constants<Constants>::type;

// Get the type of `TMat` without its last row.
template <typename TMat>
struct except_tail {};

template <typename T, usize rows, usize cols, typename Constants, typename Token>
struct except_tail<Mat<T, rows, cols, Constants, Token>> {
  static_assert(rows > 0, "The last row can only be removed from a non-empty matrix");

  using type = Mat<T, rows - 1, cols, except_tail_constants_t<Constants>, Token>;
};

template <typename TMat>
using except_tail_t = typename except_tail<TMat>::type;

// Get the type of `TMat` containing only its last row.
template <typename TMat>
struct tail {};

template <typename T, usize rows, usize cols, typename Constants, typename Token>
struct tail<Mat<T, rows, cols, Constants, Token>> {
  static_assert(rows > 0, "The last row is only available for non-empty matrices");

  using type = Mat<T, 1, cols, tail_constants_t<Constants>, Token>;
};

template <typename TMat>
using tail_t = typename tail<TMat>::type;

} // namespace detail::maxwell_boltzmann_distribution

//
//
//
//
//
template <usize dim, auto lambda>
class MaxwellBoltzmannDistribution;

//! The n-D Maxwell-Boltzmann distribution can be reduced to the 1D case,
//! especially when computing moments.
//! The specializations below exploit this reduction.
template <auto lambda>
class MaxwellBoltzmannDistribution<1, lambda> {
private:
  using Real = decltype(lambda);
  static_assert(std::floating_point<Real>, "`lambda` must be a floating point value");

public:
  template <typename TOrder, typename TDomain, typename TU>
  [[nodiscard]] HOST_DEVICE static constexpr Real Moment(const TU &u) {
    detail::maxwell_boltzmann_distribution::StaticTestMoment<Real, 1, TOrder, TDomain, TU>();

    constexpr Real rt = 1.0 / (2.0 * lambda);

    constexpr usize order = TOrder{}(C<0>{});
    constexpr isize domain = TDomain{}(C<0>{});

    Real u0 = u(C<0>{});

    if constexpr (order == 0) {
      if constexpr (domain == 0)
        return static_cast<Real>(1);
      else if constexpr (domain == 1)
        return std::erfc(-std::sqrt(lambda) * u0) / static_cast<Real>(2);
      else if constexpr (domain == -1)
        return std::erfc(std::sqrt(lambda) * u0) / static_cast<Real>(2);
    } else if constexpr (order == 1) {
      if constexpr (domain == 0)
        return u0;
      else if constexpr (domain == 1)
        return u0 * Moment<decltype(Mat{C<static_cast<usize>(0)>{}}), TDomain>(u) +
               std::exp(-lambda * (u0 * u0)) / (std::sqrt(pi<Real> * lambda) * static_cast<Real>(2));
      else if constexpr (domain == -1)
        return u0 * Moment<decltype(Mat{C<static_cast<usize>(0)>{}}), TDomain>(u) -
               std::exp(-lambda * (u0 * u0)) / (std::sqrt(pi<Real> * lambda) * static_cast<Real>(2));
    } else {
      return u0 * Moment<decltype(TOrder{} - Mat{C<static_cast<usize>(1)>{}}), TDomain>(u) +
             ((order - 1) * rt) * Moment<decltype(TOrder{} - Mat{C<static_cast<usize>(2)>{}}), TDomain>(u);
    }
  }
};

//
//
//
template <usize dim, auto lambda>
  requires(dim > 1)
class MaxwellBoltzmannDistribution<dim, lambda> {
private:
  using Real = decltype(lambda);

public:
  template <typename TOrder, typename TDomain, typename TU>
  [[nodiscard]] HOST_DEVICE static constexpr Real Moment(const TU &u) {
    detail::maxwell_boltzmann_distribution::StaticTestMoment<Real, dim, TOrder, TDomain, TU>();

    using TOrderL = detail::maxwell_boltzmann_distribution::except_tail_t<TOrder>;
    using TOrderR = detail::maxwell_boltzmann_distribution::tail_t<TOrder>;
    using TDomainL = detail::maxwell_boltzmann_distribution::except_tail_t<TDomain>;
    using TDomainR = detail::maxwell_boltzmann_distribution::tail_t<TDomain>;
    using TUL = detail::maxwell_boltzmann_distribution::except_tail_t<TU>;
    using TUR = detail::maxwell_boltzmann_distribution::tail_t<TU>;

    TUL uL;
    ForEach<dim - 1>([&](auto i) { uL(i) = u(i); });
    TUR uR;
    uR(C<0>{}) = u(C<dim - 1>{});

    return MaxwellBoltzmannDistribution<dim - 1, lambda>::template Moment<TOrderL, TDomainL, TUL>(uL) *
           MaxwellBoltzmannDistribution<1, lambda>::template Moment<TOrderR, TDomainR, TUR>(uR);
  }
};

} // namespace igks::tp
