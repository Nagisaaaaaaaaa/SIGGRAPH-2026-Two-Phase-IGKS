#include "../Simulator.hpp"

namespace igks::tp {

void Simulator::Config(SimulatorConfig const &config) {
  auto toMat = []<typename T>(T const &v) {
    return [&]<usize... id>(std::index_sequence<id...>) {
      return typename Vec<typename T::value_type, std::tuple_size_v<T>>::AsStorage{v[id]...};
    }(std::make_index_sequence<std::tuple_size_v<T>>{});
  };

  descriptor().layout() =
      std::decay_t<decltype(descriptor().layout())>(value_type::Cast<usize>(toMat(config.resolution)));
  descriptor().rho() = toMat(config.rho);
  descriptor().tau() = toMat(config.nu) * cs2Inv;
  descriptor().gravity() = toMat(config.gravity);
  descriptor().interfaceWidth() = config.interfaceWidth;
  descriptor().mobility() = config.mobility;
  descriptor().sigma() = config.sigma;
  cfl() = config.cfl;
  time() = 0;
}

} // namespace igks::tp
