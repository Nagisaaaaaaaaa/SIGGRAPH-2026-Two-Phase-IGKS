#pragma once

#include "Layout.hpp"

namespace igks::tp {

template <Layout Layout, typename View>
class MDView : Layout {
public:
  using Base = Layout;
  using Base::size;

  using value_type = decltype([]() {
    if constexpr (requires { typename View::value_type; })
      return std::type_identity<typename View::value_type>{};
    else if constexpr (requires { typename std::remove_cvref_t<decltype(std::declval<View>()[C<0>{}])>::value_type; })
      return std::type_identity<typename std::remove_cvref_t<decltype(std::declval<View>()[C<0>{}])>::value_type>{};
    else
      return std::type_identity<std::remove_cvref_t<decltype(std::declval<View>()[C<0>{}])>>{};
  }())::type;

  MDView() = default;

  HOST_DEVICE explicit MDView(Layout const &layout, View const &view) : Base(layout), view_(view) {}

  HOST_DEVICE explicit MDView(Layout const &layout, View &&view) : Base(layout), view_(std::move(view)) {}

  //
  //
  //
public:
  [[nodiscard]] HOST_DEVICE constexpr Layout layout() const { return *this; }

  template <typename tag = void>
  [[nodiscard]] HOST_DEVICE constexpr auto data() const & {
    return igks::tp::MDView{layout(), view_.data()};
  }

  template <typename tag = void>
  HOST_DEVICE constexpr void data() const && = delete;

  template <typename tag = void>
  [[nodiscard]] HOST_DEVICE constexpr auto data() & {
    return igks::tp::MDView{layout(), view_.data()};
  }

  template <typename tag = void>
  HOST_DEVICE constexpr void data() && = delete;

public:
  [[nodiscard]] HOST_DEVICE constexpr decltype(auto) operator()(is::Vec auto const &crd) const {
    return view_[Base::operator()(crd)];
  }

  [[nodiscard]] HOST_DEVICE constexpr decltype(auto) operator()(is::Vec auto const &crd) {
    return view_[Base::operator()(crd)];
  }

private:
  View view_;
};

} // namespace igks::tp
