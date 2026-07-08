#pragma once

#include "Base.hpp"

namespace igks::tp {

/// \brief Control volume interface for Cartesian grids of `dim` dimensions.
template <usize dim, usize... optionalInterface>
struct CartesianInterface;

//
//
//
template <>
struct CartesianInterface<1> {
  static constexpr usize l = 0; // Left.
  static constexpr usize r = 1; // Right.

  static constexpr usize size = 2;

  template <usize i, typename T>
  [[nodiscard]] HOST_DEVICE static constexpr auto World2Local(T const &vWorld) {
    // TODO: Implement this.
  }

  template <usize i, typename T>
  [[nodiscard]] HOST_DEVICE static constexpr auto Local2World(T const &vLocal) {
    // TODO: Implement this.
  }
};

//
//
//
template <>
struct CartesianInterface<2> : CartesianInterface<1> {
  static constexpr usize d = 2; // Down.
  static constexpr usize u = 3; // Up.

  static constexpr usize size = 4;

  template <usize i, typename T>
  [[nodiscard]] HOST_DEVICE static constexpr auto World2Local(T const &vWorld) {
    if constexpr (is::Vec<T>) {
      if constexpr (T::rows() == 2) {
        if constexpr (i == l)
          return Mat{-vWorld(C<0>{}), -vWorld(C<1>{})};
        else if constexpr (i == r)
          return Mat{vWorld(C<0>{}), vWorld(C<1>{})};
        else if constexpr (i == d)
          return Mat{-vWorld(C<1>{}), vWorld(C<0>{})};
        else if constexpr (i == u)
          return Mat{vWorld(C<1>{}), -vWorld(C<0>{})};
      }
    } else if constexpr (is::Mat<T>) {
      if constexpr (T::rows() == 2 && T::cols() == 2) {
        auto v00 = vWorld(C<0>{}, C<0>{});
        auto v10 = vWorld(C<1>{}, C<0>{});
        auto v01 = vWorld(C<0>{}, C<1>{});
        auto v11 = vWorld(C<1>{}, C<1>{});

        if constexpr (i == l)
          return Mat{Mat{v00, v10}, Mat{v01, v11}};
        else if constexpr (i == r)
          return Mat{Mat{v00, v10}, Mat{v01, v11}};
        else if constexpr (i == d)
          return Mat{Mat{v11, -v01}, Mat{-v10, v00}};
        else if constexpr (i == u)
          return Mat{Mat{v11, -v01}, Mat{-v10, v00}};
      }
    } else {
      return vWorld;
    }
  }

  template <usize i, typename T>
  [[nodiscard]] HOST_DEVICE static constexpr auto Local2World(T const &vLocal) {
    if constexpr (is::Vec<T>) {
      if constexpr (T::rows() == 2) {
        if constexpr (i == l)
          return Mat{-vLocal(C<0>{}), -vLocal(C<1>{})};
        else if constexpr (i == r)
          return Mat{vLocal(C<0>{}), vLocal(C<1>{})};
        else if constexpr (i == d)
          return Mat{vLocal(C<1>{}), -vLocal(C<0>{})};
        else if constexpr (i == u)
          return Mat{-vLocal(C<1>{}), vLocal(C<0>{})};
      }
    } else if constexpr (is::Mat<T>) {
      if constexpr (T::rows() == 2 && T::cols() == 2) {
        auto v00 = vLocal(C<0>{}, C<0>{});
        auto v10 = vLocal(C<1>{}, C<0>{});
        auto v01 = vLocal(C<0>{}, C<1>{});
        auto v11 = vLocal(C<1>{}, C<1>{});

        if constexpr (i == l)
          return Mat{Mat{v00, v10}, Mat{v01, v11}};
        else if constexpr (i == r)
          return Mat{Mat{v00, v10}, Mat{v01, v11}};
        else if constexpr (i == d)
          return Mat{Mat{v11, -v01}, Mat{-v10, v00}};
        else if constexpr (i == u)
          return Mat{Mat{v11, -v01}, Mat{-v10, v00}};
      }
    } else {
      return vLocal;
    }
  }
};

//
//
//
template <>
struct CartesianInterface<3> : CartesianInterface<2> {
  static constexpr usize b = 4; // Back.
  static constexpr usize f = 5; // Front.

  static constexpr usize size = 6;

  template <usize i, typename T>
  [[nodiscard]] HOST_DEVICE static constexpr auto World2Local(T const &vWorld) {
    if constexpr (is::Vec<T>) {
      if constexpr (T::rows() == 3) {
        if constexpr (i == l)
          return Mat{-vWorld(C<0>{}), -vWorld(C<1>{}), vWorld(C<2>{})};
        else if constexpr (i == r)
          return Mat{vWorld(C<0>{}), vWorld(C<1>{}), vWorld(C<2>{})};
        else if constexpr (i == d)
          return Mat{-vWorld(C<1>{}), vWorld(C<0>{}), vWorld(C<2>{})};
        else if constexpr (i == u)
          return Mat{vWorld(C<1>{}), -vWorld(C<0>{}), vWorld(C<2>{})};
        else if constexpr (i == b)
          return Mat{-vWorld(C<2>{}), vWorld(C<1>{}), vWorld(C<0>{})};
        else if constexpr (i == f)
          return Mat{vWorld(C<2>{}), vWorld(C<1>{}), -vWorld(C<0>{})};
      }
    } else if constexpr (is::Mat<T>) {
      if constexpr (T::rows() == 3 && T::cols() == 3) {
        auto v00 = vWorld(C<0>{}, C<0>{});
        auto v10 = vWorld(C<1>{}, C<0>{});
        auto v20 = vWorld(C<2>{}, C<0>{});
        auto v01 = vWorld(C<0>{}, C<1>{});
        auto v11 = vWorld(C<1>{}, C<1>{});
        auto v21 = vWorld(C<2>{}, C<1>{});
        auto v02 = vWorld(C<0>{}, C<2>{});
        auto v12 = vWorld(C<1>{}, C<2>{});
        auto v22 = vWorld(C<2>{}, C<2>{});

        if constexpr (i == l)
          return Mat{Mat{v00, v10, -v20}, Mat{v01, v11, -v21}, Mat{-v02, -v12, v22}};
        else if constexpr (i == r)
          return Mat{Mat{v00, v10, v20}, Mat{v01, v11, v21}, Mat{v02, v12, v22}};
        else if constexpr (i == d)
          return Mat{Mat{v11, -v01, -v21}, Mat{-v10, v00, v20}, Mat{-v12, v02, v22}};
        else if constexpr (i == u)
          return Mat{Mat{v11, -v01, v21}, Mat{-v10, v00, -v20}, Mat{v12, -v02, v22}};
        else if constexpr (i == b)
          return Mat{Mat{v22, -v12, -v02}, Mat{-v21, v11, v01}, Mat{-v20, v10, v00}};
        else if constexpr (i == f)
          return Mat{Mat{v22, v12, -v02}, Mat{v21, v11, -v01}, Mat{-v20, -v10, v00}};
      }
    } else {
      return vWorld;
    }
  }

  template <usize i, typename T>
  [[nodiscard]] HOST_DEVICE static constexpr auto Local2World(T const &vLocal) {
    if constexpr (is::Vec<T>) {
      if constexpr (T::rows() == 3) {
        if constexpr (i == l)
          return Mat{-vLocal(C<0>{}), -vLocal(C<1>{}), vLocal(C<2>{})};
        else if constexpr (i == r)
          return Mat{vLocal(C<0>{}), vLocal(C<1>{}), vLocal(C<2>{})};
        else if constexpr (i == d)
          return Mat{vLocal(C<1>{}), -vLocal(C<0>{}), vLocal(C<2>{})};
        else if constexpr (i == u)
          return Mat{-vLocal(C<1>{}), vLocal(C<0>{}), vLocal(C<2>{})};
        else if constexpr (i == b)
          return Mat{vLocal(C<2>{}), vLocal(C<1>{}), -vLocal(C<0>{})};
        else if constexpr (i == f)
          return Mat{-vLocal(C<2>{}), vLocal(C<1>{}), vLocal(C<0>{})};
      }
    } else if constexpr (is::Mat<T>) {
      if constexpr (T::rows() == 3 && T::cols() == 3) {
        auto v00 = vLocal(C<0>{}, C<0>{});
        auto v10 = vLocal(C<1>{}, C<0>{});
        auto v20 = vLocal(C<2>{}, C<0>{});
        auto v01 = vLocal(C<0>{}, C<1>{});
        auto v11 = vLocal(C<1>{}, C<1>{});
        auto v21 = vLocal(C<2>{}, C<1>{});
        auto v02 = vLocal(C<0>{}, C<2>{});
        auto v12 = vLocal(C<1>{}, C<2>{});
        auto v22 = vLocal(C<2>{}, C<2>{});

        if constexpr (i == l)
          return Mat{Mat{v00, v10, -v20}, Mat{v01, v11, -v21}, Mat{-v02, -v12, v22}};
        else if constexpr (i == r)
          return Mat{Mat{v00, v10, v20}, Mat{v01, v11, v21}, Mat{v02, v12, v22}};
        else if constexpr (i == d)
          return Mat{Mat{v11, -v01, v21}, Mat{-v10, v00, -v20}, Mat{v12, -v02, v22}};
        else if constexpr (i == u)
          return Mat{Mat{v11, -v01, -v21}, Mat{-v10, v00, v20}, Mat{-v12, v02, v22}};
        else if constexpr (i == b)
          return Mat{Mat{v22, v12, -v02}, Mat{v21, v11, -v01}, Mat{-v20, -v10, v00}};
        else if constexpr (i == f)
          return Mat{Mat{v22, -v12, -v02}, Mat{-v21, v11, v01}, Mat{-v20, v10, v00}};
      }
    } else {
      return vLocal;
    }
  }
};

//
//
//
template <usize dim, usize i>
struct CartesianInterface<dim, i> {
  template <typename T>
  [[nodiscard]] HOST_DEVICE static constexpr auto World2Local(T const &vWorld) {
    return CartesianInterface<dim>::template World2Local<i>(vWorld);
  }

  template <typename T>
  [[nodiscard]] HOST_DEVICE static constexpr auto Local2World(T const &vLocal) {
    return CartesianInterface<dim>::template Local2World<i>(vLocal);
  }
};

//
//
//
//
//
namespace detail::cartesian_interface {

template <usize dim, typename Interfaces>
struct CartesianInterfacesImpl;

template <usize dim, usize... iInterfaces>
struct CartesianInterfacesImpl<dim, std::index_sequence<iInterfaces...>> {
  using type = Types<CartesianInterface<dim, iInterfaces>...>;
};

} // namespace detail::cartesian_interface

//
//
//
/// \brief Types of control volume interfaces for Cartesian grids of `dim` dimensions.
template <usize dim>
using CartesianInterfaces = detail::cartesian_interface::
    CartesianInterfacesImpl<dim, std::make_index_sequence<CartesianInterface<dim>::size>>::type;

} // namespace igks::tp
