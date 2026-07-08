#pragma once

#include "Base.hpp"

namespace igks::tp {

// Configurations.

/// \brief Number of dimensions of the simulation system.
CONSTANT static constexpr usize d = 2;

using Real = float;

//
//
//
//
//
// Aliases.
template <typename T, typename... Ts>
using MatD = Mat<T, d, d, Ts...>;
template <typename T, typename... Ts>
using VecD = Vec<T, d, Ts...>;

template <typename... Ts>
using Mat1r = Mat1<Real, Ts...>;
template <typename... Ts>
using Mat2r = Mat2<Real, Ts...>;
template <typename... Ts>
using Mat3r = Mat3<Real, Ts...>;
template <typename... Ts>
using Mat4r = Mat4<Real, Ts...>;
template <typename... Ts>
using MatDr = MatD<Real, Ts...>;

template <typename... Ts>
using Vec1r = Vec1<Real, Ts...>;
template <typename... Ts>
using Vec2r = Vec2<Real, Ts...>;
template <typename... Ts>
using Vec3r = Vec3<Real, Ts...>;
template <typename... Ts>
using Vec4r = Vec4<Real, Ts...>;
template <typename... Ts>
using VecDr = VecD<Real, Ts...>;

//
//
//
// Literal operators.
HOST_DEVICE constexpr Real operator"" _R(long double v) { return static_cast<Real>(v); }

HOST_DEVICE constexpr Real operator"" _R(unsigned long long v) { return static_cast<Real>(v); }

//
//
//
// Shortcuts.
using $M = std::conditional_t<d == 1, M, std::conditional_t<d == 2, MO, std::conditional_t<d == 3, MOO, void>>>;
using $N = std::conditional_t<d == 1, N, std::conditional_t<d == 2, NO, std::conditional_t<d == 3, NOO, void>>>;
using $O = std::conditional_t<d == 1, O, std::conditional_t<d == 2, OO, std::conditional_t<d == 3, OOO, void>>>;
using $P = std::conditional_t<d == 1, P, std::conditional_t<d == 2, PO, std::conditional_t<d == 3, POO, void>>>;
using $Q = std::conditional_t<d == 1, Q, std::conditional_t<d == 2, QO, std::conditional_t<d == 3, QOO, void>>>;

CONSTANT static constexpr $M $m;
CONSTANT static constexpr $N $n;
CONSTANT static constexpr $O $o;
CONSTANT static constexpr $P $p;
CONSTANT static constexpr $Q $q;

} // namespace igks::tp
