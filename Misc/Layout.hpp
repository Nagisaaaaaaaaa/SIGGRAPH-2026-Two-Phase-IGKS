#pragma once

#include "Base.hpp"

namespace igks::tp {

namespace detail::layout {

template <typename T>
concept Crd = is::Vec<T> && is::Integral<typename T::value_type>;

template <typename T>
concept Crduz = is::Vec<T> && is::Same<typename T::value_type, usize>;

template <Crduz Size>
struct LayoutBase : Size {
  LayoutBase() = default;

  HOST_DEVICE constexpr explicit LayoutBase(Crd auto const &size) : Size(value_type::Cast<usize>(size)) {}

  [[nodiscard]] HOST_DEVICE constexpr Size size() const { return *this; }
};

//
//
//
template <usize row, typename Size>
[[nodiscard]] HOST_DEVICE constexpr auto LeftStride(Size const &size) {
  if constexpr (row == 0)
    return C<static_cast<Size::value_type>(1)>{};
  else
    return LeftStride<row - 1>(size) * size(C<row - 1>{});
}

template <usize row, typename Size, typename Crd>
[[nodiscard]] HOST_DEVICE constexpr auto LeftIndex(Size const &size, Crd const &crd) {
  if constexpr (row == 0)
    return crd(C<static_cast<usize>(0)>{});
  else
    return LeftIndex<row - 1>(size, crd) + crd(C<row>{}) * LeftStride<row>(size);
}

} // namespace detail::layout

//
//
//
//
//
template <typename T>
concept Layout = requires(T const &layout, usize i) {
  { layout.size() } -> detail::layout::Crduz;
  { layout(i) } -> detail::layout::Crduz;
  { layout(layout(i)) } -> is::Same<usize>;
};

//
//
//
template <detail::layout::Crduz Size>
struct LeftLayout : detail::layout::LayoutBase<Size> {
  using Base = detail::layout::LayoutBase<Size>;
  using Base::Base;
  using Base::size;

  template <typename I, typename V = value_type::Of<I>>
    requires((is::Integral<I> || is::C<I>) && is::Integral<V>)
  [[nodiscard]] HOST_DEVICE constexpr auto operator()(I i) const {
    return [&]<usize... row>(std::index_sequence<row...>) constexpr {
      return Mat{value_type::Cast<V>((i / detail::layout::LeftStride<row>(size())) % size()(C<row>{}))...};
    }(std::make_index_sequence<Size::rows()>{});
  }

  template <detail::layout::Crd Crd, typename V = Crd::value_type>
  [[nodiscard]] HOST_DEVICE constexpr auto operator()(Crd const &crd) const {
    static_assert(Size::rows() == Crd::rows(), "Layout coordinate must have the same rank as size");

    return value_type::Cast<V>(detail::layout::LeftIndex<Crd::rows() - 1>(size(), crd));
  }
};

template <detail::layout::Crd Size>
LeftLayout(Size const &) -> LeftLayout<decltype(value_type::Cast<usize>(std::declval<Size>()))>;

} // namespace igks::tp
