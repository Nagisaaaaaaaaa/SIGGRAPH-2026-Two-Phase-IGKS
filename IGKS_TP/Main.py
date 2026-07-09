from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Misc"))
import Binding

Binding.find_igks_tp()

import igks_tp
from concurrent.futures import ThreadPoolExecutor
import math
from Misc.Colormap import colormap
import numpy as np
import pyvista as pv
import taichi as ti
from time import sleep

d = Binding.d
real = Binding.real
ti.real = Binding.ti.real

ti.init(
  advanced_optimization=False,
  arch=ti.gpu,
  cfg_optimization=False,
  debug=False,
  default_fp=ti.real,
  fast_math=False,
  log_level=ti.INFO,
  offline_cache=True,
  verbose=True,
)


def main() -> None:
  config = igks_tp.SimulatorConfig()

  if 1:
    # Rayleigh-Taylor instability.
    L = 200
    rho_ratio = 3  # 1000
    At = (1 - 1 / rho_ratio) / (1 + 1 / rho_ratio)
    Re = 3000
    mu_ratio = 1  # 100
    g = 5e-7
    Pe = 1000  # 200
    # Ca = 0.26  # 0.44
    # We = 50000

    u_ref = math.sqrt(g * L)
    t_ref = math.sqrt(L / (g * At))
    nu1 = u_ref * L / Re
    nu0 = rho_ratio * nu1 / mu_ratio

    config.resolution = [L, 4 * L] if d == 2 else [L, 4 * L, L]
    config.rho = [1 / rho_ratio, 1]
    config.nu = [nu0, nu1]
    config.gravity = [0, -g] if d == 2 else [0, -g, 0]
    config.interfaceWidth = 5
    config.mobility = u_ref * L / Pe
    config.sigma = 0
    # config.sigma = 1 * nu1 * u_ref / Ca
    # config.sigma = 1 * L * math.sqrt(g * L) / We
  elif 0:
    # Dam break.
    a = 0.2
    L = 100
    scale = [4, 3] if d == 2 else [4, 3, 1]
    n2 = 2
    rho_ratio = 1000
    u02 = 9.8 * n2 * a
    Re1 = rho_ratio * math.sqrt(u02) * n2 * a / 1e-3
    Re0 = 1 * math.sqrt(u02) * n2 * a / 1e-5
    u_ref = math.sqrt(2.5e-6 / 9.8 * (L * u02) / a)

    g = 9.8 * ((a * u_ref**2) / (L * u02))
    t_ref = 1 * ((L * math.sqrt(u02)) / (a * u_ref))
    nu1 = u_ref * n2 * L / Re1
    nu0 = u_ref * n2 * L / Re0

    config.resolution = [s * L for s in scale]
    config.rho = [1 / rho_ratio, 1]
    config.nu = [nu0, nu1]
    config.gravity = [0, -g] if d == 2 else [0, -g, 0]
    config.interfaceWidth = 5  # 3
    config.mobility = 0.1  # 1 / 12
    config.sigma = 0.072 * ((1 * L * u_ref**2) / (rho_ratio * a * u02))

  config.cfl = 0.6

  simulator = igks_tp.Simulator()
  simulator.Config(config)
  simulator.AllocAndInit()

  if d == 2:
    window = ti.ui.Window("IGKS TP", tuple(config.resolution))
    while window.running:
      for _ in range(1000):
        simulator.ApplySource()
        simulator.SimulateOneStep()
      status = simulator.Status()
      phi = status.phi
      phi_range = [0.0, 1.0]
      phi = (phi - phi_range[0]) / (phi_range[1] - phi_range[0])
      rgb = colormap("turbo", phi)
      image = np.ascontiguousarray(np.rot90(rgb.transpose(1, 0, 2)[::-1, :, :], k=-1).astype(real))
      window.get_canvas().set_image(image)
      window.show()
  elif d == 3:
    plotter = pv.Plotter()
    grid = pv.ImageData(dimensions=tuple(config.resolution), spacing=(1.0, 1.0, 1.0), origin=(0.5, 0.5, 0.5))
    grid.point_data["phi"] = np.empty(grid.n_points, dtype=real)
    actor = None

    def update_grid(status) -> None:
      nonlocal actor

      grid.point_data["phi"][:] = np.ravel(status.phi, order="F")
      grid.point_data["phi"].Modified()
      grid.Modified()

      iso_surface = grid.contour([0.5], scalars="phi")
      if actor is None:
        actor = plotter.add_mesh(iso_surface, scalars="phi", clim=[0.0, 1.0], cmap="turbo")
      else:
        actor.mapper.dataset = iso_surface

    update_grid(simulator.Status())

    plotter.add_axes()
    plotter.view_xy()
    plotter.show(title="IGKS TP", interactive_update=True, auto_close=False)

    def simulate_batch():
      for _ in range(500):
        simulator.ApplySource()
        simulator.SimulateOneStep()
      return simulator.Status()

    executor = ThreadPoolExecutor(max_workers=1)
    simulation = executor.submit(simulate_batch)
    try:
      while not plotter._closed and not plotter.iren.interactor.GetDone():
        if simulation.done():
          status = simulation.result()
          update_grid(status)
          simulation = executor.submit(simulate_batch)
        plotter.update()
        sleep(1 / 240)
    finally:
      executor.shutdown(wait=True)
      plotter.close()


if __name__ == "__main__":
  main()
