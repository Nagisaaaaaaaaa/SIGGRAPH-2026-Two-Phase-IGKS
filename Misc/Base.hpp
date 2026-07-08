#pragma once

#include <Arietta/Mat.hpp>
#include <algorithm>
#include <numbers>

namespace igks::tp {

using namespace arietta;

#ifdef __CUDACC__
  #define GLOBAL      __global__
  #define HOST        __host__
  #define DEVICE      __device__
  #define HOST_DEVICE __host__ __device__

  #define CONSTANT __constant__
#else
  #define GLOBAL
  #define HOST
  #define DEVICE
  #define HOST_DEVICE

  #define CONSTANT
#endif

//
//
//
namespace value_type {

namespace detail::base {

template <typename T>
struct Of {
  using type = T;
};

template <typename T>
  requires requires { typename T::value_type; }
struct Of<T> {
  using type = T::value_type;
};

} // namespace detail::base

template <typename T>
using Of = detail::base::Of<T>::type;

template <typename T>
[[nodiscard]] HOST_DEVICE constexpr auto Cast(is::Integral auto const &v) {
  return static_cast<T>(v);
}

template <typename T>
[[nodiscard]] HOST_DEVICE constexpr auto Cast(is::C auto v) {
  return C<static_cast<T>(decltype(v)::value)>{};
}

template <typename T, is::Mat M>
[[nodiscard]] HOST_DEVICE constexpr auto Cast(M const &m) {
  return [&]<usize... col>(std::index_sequence<col...>) constexpr {
    return Mat{[&]<usize c, usize... row>(C<c>, std::index_sequence<row...>) constexpr {
      return Mat{Cast<T>(m(C<row>{}, C<c>{}))...};
    }(C<col>{}, std::make_index_sequence<M::rows()>{})...};
  }(std::make_index_sequence<M::cols()>{});
}

} // namespace value_type

//
//
//
template <typename T>
static constexpr T pi = std::numbers::pi_v<T>;

/// \brief 1 / 3.1415926...
template <typename T>
static constexpr T piInv = std::numbers::inv_pi_v<T>;

/// \brief 2.7182818284...
template <typename T>
static constexpr T e = std::numbers::e_v<T>;

[[nodiscard]] HOST_DEVICE constexpr auto Lerp(auto const &x, auto const &y, auto const &t) { return x + t * (y - x); }

//
//
//
//
//
// Shortcuts.
CONSTANT static constexpr C<static_cast<usize>(0)> i0{};
CONSTANT static constexpr C<static_cast<usize>(1)> i1{};
CONSTANT static constexpr C<static_cast<usize>(2)> i2{};
CONSTANT static constexpr C<static_cast<usize>(3)> i3{};

//
//
//
#define IGKSTP_M                                                                                                       \
  C<static_cast<isize>(-2)> {}
#define IGKSTP_N                                                                                                       \
  C<static_cast<isize>(-1)> {}
#define IGKSTP_O                                                                                                       \
  C<static_cast<isize>(0)> {}
#define IGKSTP_P                                                                                                       \
  C<static_cast<isize>(1)> {}
#define IGKSTP_Q                                                                                                       \
  C<static_cast<isize>(2)> {}

CONSTANT static constexpr auto _n = Mat{IGKSTP_N};
CONSTANT static constexpr auto _o = Mat{IGKSTP_O};
CONSTANT static constexpr auto _p = Mat{IGKSTP_P};

CONSTANT static constexpr auto _m = Mat{IGKSTP_M};
CONSTANT static constexpr auto _q = Mat{IGKSTP_Q};

CONSTANT static constexpr auto _nn = Mat{IGKSTP_N, IGKSTP_N};
CONSTANT static constexpr auto _on = Mat{IGKSTP_O, IGKSTP_N};
CONSTANT static constexpr auto _pn = Mat{IGKSTP_P, IGKSTP_N};

CONSTANT static constexpr auto _no = Mat{IGKSTP_N, IGKSTP_O};
CONSTANT static constexpr auto _oo = Mat{IGKSTP_O, IGKSTP_O};
CONSTANT static constexpr auto _po = Mat{IGKSTP_P, IGKSTP_O};

CONSTANT static constexpr auto _np = Mat{IGKSTP_N, IGKSTP_P};
CONSTANT static constexpr auto _op = Mat{IGKSTP_O, IGKSTP_P};
CONSTANT static constexpr auto _pp = Mat{IGKSTP_P, IGKSTP_P};

CONSTANT static constexpr auto _om = Mat{IGKSTP_O, IGKSTP_M};
CONSTANT static constexpr auto _mo = Mat{IGKSTP_M, IGKSTP_O};
CONSTANT static constexpr auto _qo = Mat{IGKSTP_Q, IGKSTP_O};
CONSTANT static constexpr auto _oq = Mat{IGKSTP_O, IGKSTP_Q};

CONSTANT static constexpr auto _nnn = Mat{IGKSTP_N, IGKSTP_N, IGKSTP_N};
CONSTANT static constexpr auto _onn = Mat{IGKSTP_O, IGKSTP_N, IGKSTP_N};
CONSTANT static constexpr auto _pnn = Mat{IGKSTP_P, IGKSTP_N, IGKSTP_N};

CONSTANT static constexpr auto _non = Mat{IGKSTP_N, IGKSTP_O, IGKSTP_N};
CONSTANT static constexpr auto _oon = Mat{IGKSTP_O, IGKSTP_O, IGKSTP_N};
CONSTANT static constexpr auto _pon = Mat{IGKSTP_P, IGKSTP_O, IGKSTP_N};

CONSTANT static constexpr auto _npn = Mat{IGKSTP_N, IGKSTP_P, IGKSTP_N};
CONSTANT static constexpr auto _opn = Mat{IGKSTP_O, IGKSTP_P, IGKSTP_N};
CONSTANT static constexpr auto _ppn = Mat{IGKSTP_P, IGKSTP_P, IGKSTP_N};

CONSTANT static constexpr auto _nno = Mat{IGKSTP_N, IGKSTP_N, IGKSTP_O};
CONSTANT static constexpr auto _ono = Mat{IGKSTP_O, IGKSTP_N, IGKSTP_O};
CONSTANT static constexpr auto _pno = Mat{IGKSTP_P, IGKSTP_N, IGKSTP_O};

CONSTANT static constexpr auto _noo = Mat{IGKSTP_N, IGKSTP_O, IGKSTP_O};
CONSTANT static constexpr auto _ooo = Mat{IGKSTP_O, IGKSTP_O, IGKSTP_O};
CONSTANT static constexpr auto _poo = Mat{IGKSTP_P, IGKSTP_O, IGKSTP_O};

CONSTANT static constexpr auto _npo = Mat{IGKSTP_N, IGKSTP_P, IGKSTP_O};
CONSTANT static constexpr auto _opo = Mat{IGKSTP_O, IGKSTP_P, IGKSTP_O};
CONSTANT static constexpr auto _ppo = Mat{IGKSTP_P, IGKSTP_P, IGKSTP_O};

CONSTANT static constexpr auto _nnp = Mat{IGKSTP_N, IGKSTP_N, IGKSTP_P};
CONSTANT static constexpr auto _onp = Mat{IGKSTP_O, IGKSTP_N, IGKSTP_P};
CONSTANT static constexpr auto _pnp = Mat{IGKSTP_P, IGKSTP_N, IGKSTP_P};

CONSTANT static constexpr auto _nop = Mat{IGKSTP_N, IGKSTP_O, IGKSTP_P};
CONSTANT static constexpr auto _oop = Mat{IGKSTP_O, IGKSTP_O, IGKSTP_P};
CONSTANT static constexpr auto _pop = Mat{IGKSTP_P, IGKSTP_O, IGKSTP_P};

CONSTANT static constexpr auto _npp = Mat{IGKSTP_N, IGKSTP_P, IGKSTP_P};
CONSTANT static constexpr auto _opp = Mat{IGKSTP_O, IGKSTP_P, IGKSTP_P};
CONSTANT static constexpr auto _ppp = Mat{IGKSTP_P, IGKSTP_P, IGKSTP_P};

CONSTANT static constexpr auto _oom = Mat{IGKSTP_O, IGKSTP_O, IGKSTP_M};
CONSTANT static constexpr auto _omo = Mat{IGKSTP_O, IGKSTP_M, IGKSTP_O};
CONSTANT static constexpr auto _moo = Mat{IGKSTP_M, IGKSTP_O, IGKSTP_O};
CONSTANT static constexpr auto _qoo = Mat{IGKSTP_Q, IGKSTP_O, IGKSTP_O};
CONSTANT static constexpr auto _oqo = Mat{IGKSTP_O, IGKSTP_Q, IGKSTP_O};
CONSTANT static constexpr auto _ooq = Mat{IGKSTP_O, IGKSTP_O, IGKSTP_Q};

#undef IGKSTP_M
#undef IGKSTP_N
#undef IGKSTP_O
#undef IGKSTP_P
#undef IGKSTP_Q

using N = std::decay_t<decltype(_n)>;
using O = std::decay_t<decltype(_o)>;
using P = std::decay_t<decltype(_p)>;

using M = std::decay_t<decltype(_m)>;
using Q = std::decay_t<decltype(_q)>;

using NN = std::decay_t<decltype(_nn)>;
using ON = std::decay_t<decltype(_on)>;
using PN = std::decay_t<decltype(_pn)>;

using NO = std::decay_t<decltype(_no)>;
using OO = std::decay_t<decltype(_oo)>;
using PO = std::decay_t<decltype(_po)>;

using NP = std::decay_t<decltype(_np)>;
using OP = std::decay_t<decltype(_op)>;
using PP = std::decay_t<decltype(_pp)>;

using OM = std::decay_t<decltype(_om)>;
using MO = std::decay_t<decltype(_mo)>;
using QO = std::decay_t<decltype(_qo)>;
using OQ = std::decay_t<decltype(_oq)>;

using NNN = std::decay_t<decltype(_nnn)>;
using ONN = std::decay_t<decltype(_onn)>;
using PNN = std::decay_t<decltype(_pnn)>;

using NON = std::decay_t<decltype(_non)>;
using OON = std::decay_t<decltype(_oon)>;
using PON = std::decay_t<decltype(_pon)>;

using NPN = std::decay_t<decltype(_npn)>;
using OPN = std::decay_t<decltype(_opn)>;
using PPN = std::decay_t<decltype(_ppn)>;

using NNO = std::decay_t<decltype(_nno)>;
using ONO = std::decay_t<decltype(_ono)>;
using PNO = std::decay_t<decltype(_pno)>;

using NOO = std::decay_t<decltype(_noo)>;
using OOO = std::decay_t<decltype(_ooo)>;
using POO = std::decay_t<decltype(_poo)>;

using NPO = std::decay_t<decltype(_npo)>;
using OPO = std::decay_t<decltype(_opo)>;
using PPO = std::decay_t<decltype(_ppo)>;

using NNP = std::decay_t<decltype(_nnp)>;
using ONP = std::decay_t<decltype(_onp)>;
using PNP = std::decay_t<decltype(_pnp)>;

using NOP = std::decay_t<decltype(_nop)>;
using OOP = std::decay_t<decltype(_oop)>;
using POP = std::decay_t<decltype(_pop)>;

using NPP = std::decay_t<decltype(_npp)>;
using OPP = std::decay_t<decltype(_opp)>;
using PPP = std::decay_t<decltype(_ppp)>;

using OOM = std::decay_t<decltype(_oom)>;
using OMO = std::decay_t<decltype(_omo)>;
using MOO = std::decay_t<decltype(_moo)>;
using QOO = std::decay_t<decltype(_qoo)>;
using OQO = std::decay_t<decltype(_oqo)>;
using OOQ = std::decay_t<decltype(_ooq)>;

//
//
//
#define IGKSTP_0                                                                                                       \
  C<static_cast<usize>(0)> {}
#define IGKSTP_1                                                                                                       \
  C<static_cast<usize>(1)> {}
#define IGKSTP_2                                                                                                       \
  C<static_cast<usize>(2)> {}
#define IGKSTP_3                                                                                                       \
  C<static_cast<usize>(3)> {}

CONSTANT static constexpr auto _0 = Mat{IGKSTP_0};
CONSTANT static constexpr auto _1 = Mat{IGKSTP_1};
CONSTANT static constexpr auto _2 = Mat{IGKSTP_2};
CONSTANT static constexpr auto _3 = Mat{IGKSTP_3};

CONSTANT static constexpr auto _00 = Mat{IGKSTP_0, IGKSTP_0};

CONSTANT static constexpr auto _10 = Mat{IGKSTP_1, IGKSTP_0};
CONSTANT static constexpr auto _01 = Mat{IGKSTP_0, IGKSTP_1};

CONSTANT static constexpr auto _20 = Mat{IGKSTP_2, IGKSTP_0};
CONSTANT static constexpr auto _11 = Mat{IGKSTP_1, IGKSTP_1};
CONSTANT static constexpr auto _02 = Mat{IGKSTP_0, IGKSTP_2};

CONSTANT static constexpr auto _30 = Mat{IGKSTP_3, IGKSTP_0};
CONSTANT static constexpr auto _21 = Mat{IGKSTP_2, IGKSTP_1};
CONSTANT static constexpr auto _12 = Mat{IGKSTP_1, IGKSTP_2};
CONSTANT static constexpr auto _03 = Mat{IGKSTP_0, IGKSTP_3};

CONSTANT static constexpr auto _000 = Mat{IGKSTP_0, IGKSTP_0, IGKSTP_0};

CONSTANT static constexpr auto _100 = Mat{IGKSTP_1, IGKSTP_0, IGKSTP_0};
CONSTANT static constexpr auto _010 = Mat{IGKSTP_0, IGKSTP_1, IGKSTP_0};
CONSTANT static constexpr auto _001 = Mat{IGKSTP_0, IGKSTP_0, IGKSTP_1};

CONSTANT static constexpr auto _200 = Mat{IGKSTP_2, IGKSTP_0, IGKSTP_0};
CONSTANT static constexpr auto _110 = Mat{IGKSTP_1, IGKSTP_1, IGKSTP_0};
CONSTANT static constexpr auto _020 = Mat{IGKSTP_0, IGKSTP_2, IGKSTP_0};
CONSTANT static constexpr auto _101 = Mat{IGKSTP_1, IGKSTP_0, IGKSTP_1};
CONSTANT static constexpr auto _011 = Mat{IGKSTP_0, IGKSTP_1, IGKSTP_1};
CONSTANT static constexpr auto _002 = Mat{IGKSTP_0, IGKSTP_0, IGKSTP_2};

CONSTANT static constexpr auto _300 = Mat{IGKSTP_3, IGKSTP_0, IGKSTP_0};
CONSTANT static constexpr auto _210 = Mat{IGKSTP_2, IGKSTP_1, IGKSTP_0};
CONSTANT static constexpr auto _120 = Mat{IGKSTP_1, IGKSTP_2, IGKSTP_0};
CONSTANT static constexpr auto _030 = Mat{IGKSTP_0, IGKSTP_3, IGKSTP_0};
CONSTANT static constexpr auto _201 = Mat{IGKSTP_2, IGKSTP_0, IGKSTP_1};
CONSTANT static constexpr auto _111 = Mat{IGKSTP_1, IGKSTP_1, IGKSTP_1};
CONSTANT static constexpr auto _021 = Mat{IGKSTP_0, IGKSTP_2, IGKSTP_1};
CONSTANT static constexpr auto _102 = Mat{IGKSTP_1, IGKSTP_0, IGKSTP_2};
CONSTANT static constexpr auto _012 = Mat{IGKSTP_0, IGKSTP_1, IGKSTP_2};
CONSTANT static constexpr auto _003 = Mat{IGKSTP_0, IGKSTP_0, IGKSTP_3};

#undef IGKSTP_0
#undef IGKSTP_1
#undef IGKSTP_2
#undef IGKSTP_3

} // namespace igks::tp
