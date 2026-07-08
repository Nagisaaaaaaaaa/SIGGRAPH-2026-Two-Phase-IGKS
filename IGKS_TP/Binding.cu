#include "Simulator.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

namespace nb = nanobind;

NB_MODULE(igks_tp, m) {
  m.doc() = "Python bindings for IGKS TP.";

  using namespace igks::tp;

  nb::class_<SimulatorConfig>(m, "SimulatorConfig")
      .def(nb::init<>())
      .def_rw("resolution", &SimulatorConfig::resolution)
      .def_rw("rho", &SimulatorConfig::rho)
      .def_rw("nu", &SimulatorConfig::nu)
      .def_rw("gravity", &SimulatorConfig::gravity)
      .def_rw("interfaceWidth", &SimulatorConfig::interfaceWidth)
      .def_rw("mobility", &SimulatorConfig::mobility)
      .def_rw("sigma", &SimulatorConfig::sigma)
      .def_rw("cfl", &SimulatorConfig::cfl);

  nb::class_<SimulatorStatus>(m, "SimulatorStatus")
      .def_prop_ro("time", [](SimulatorStatus const &status) { return status.time; })
      .def_prop_ro("phi", [](SimulatorStatus const &status) { return status.phi; }, nb::rv_policy::reference)
      .def_prop_ro("u", [](SimulatorStatus const &status) { return status.u; }, nb::rv_policy::reference)
      .def_prop_ro("p", [](SimulatorStatus const &status) { return status.p; }, nb::rv_policy::reference);

  nb::class_<Simulator>(m, "Simulator")
      .def(nb::init<>())
      .def("Config", &Simulator::Config)
      .def("AllocAndInit", &Simulator::AllocAndInit)
      .def("ApplySource", &Simulator::ApplySource, nb::call_guard<nb::gil_scoped_release>())
      .def("SimulateOneStep", &Simulator::SimulateOneStep, nb::call_guard<nb::gil_scoped_release>())
      .def("Status", &Simulator::Status);
}
