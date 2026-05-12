# Implementation Notes

## 1. Main file

`src/three_body_simulation.py` is the original project script, renamed to avoid spaces and accents in the filename. The file has intentionally been kept as a single script so that the original work is preserved and can be traced easily.

The public-facing documentation is written in English. Some function names inside the script remain in Spanish because renaming them would require a larger refactor and could accidentally break internal calls.

## 2. `System` class

The `System` class stores the physical state of the simulation:

- number of bodies,
- positions `x` with shape `(N, 3)`,
- velocities `v` with shape `(N, 3)`,
- masses `m`,
- gravitational constant `G`.

It also applies a centre-of-mass correction so that the plotted system does not drift across the screen as a whole.

## 3. Initial conditions

`Cond_iniciales_definidas(cond_inicial)` acts as a database of predefined systems. It returns the physical system, labels, colours and plotting options.

The available configurations include Lagrange, Broucke, figure-eight, butterfly, dragonfly, yarn, Skinny Pineapple, Li-Liao, free fall, a chaotic three-body case, Earth-Moon-Sun, the simplified Solar System and several larger or three-dimensional extensions.

## 4. Acceleration calculation

`calcula_a(a, system)` implements the Newtonian acceleration directly:

$$
\mathbf a_i=G\sum_{j\neq i}m_j\frac{\mathbf r_j-\mathbf r_i}{\|\mathbf r_j-\mathbf r_i\|^3}.
$$

The implementation uses NumPy broadcasting to compute pairwise separations efficiently. Self-interactions are removed by setting the diagonal contribution to zero.

## 5. Integrators

The main integration routines are:

- `paso_Euler`: explicit Euler.
- `backward_euler`: approximate implicit Euler using fixed-point iteration.
- `paso_verlet_velocidades`: velocity Verlet.
- `paso_rk5`: fifth-order Runge-Kutta-Fehlberg.

These names are preserved from the original code. The README and documentation explain their English meaning.

## 6. Diagnostics

The code includes several diagnostics:

- `compute_rel_energy_error`: relative total-energy error.
- `compute_rel_angular_momentum_error`: relative angular-momentum error.
- `calcular_energias_individuales`: individual and total energies.
- `comparar_errores_metodos`: comparison of error behaviour across methods.
- `error_vs_dt_sistemas`: global error as a function of time step.
- `seccion_poincare`: Poincaré-section crossings.

These diagnostics are essential because visual inspection alone is not enough to determine whether an orbital simulation is physically reliable.

## 7. Visualization

The script contains routines for:

- 2D trajectory plots,
- 3D trajectory plots,
- GIF generation,
- energy-error plots,
- angular-momentum-error plots,
- method-comparison plots,
- Poincaré sections.

The most useful outputs have been copied into `assets/` so that they can be displayed directly in the GitHub README and documentation.

## 8. Recommended future refactor

A more professional package-style version could be organized as:

```text
src/
  initial_conditions.py
  integrators.py
  diagnostics.py
  plotting.py
  main.py
examples/
tests/
```

The refactor should be conservative. It should not change formulas, units, masses or initial conditions unless the change is explicitly justified and verified against the current outputs.

## 9. Suggested validation after refactoring

After any code reorganization, the following checks should be run:

1. Generate the same representative GIFs.
2. Recompute the energy-error and angular-momentum-error plots.
3. Compare RKF5 and Velocity Verlet on the chaotic benchmark.
4. Verify that the Poincaré-section figure is still generated correctly.
5. Confirm that all Markdown image links still resolve on GitHub.


## 10. External inspiration and license hygiene

This project acknowledges Ching-Yin Ng's `grav_sim` repository and the **“5 steps to N-body simulation”** tutorial as an educational inspiration for the $N$-body simulation workflow.

- Tutorial: <https://alvinng4.github.io/grav_sim/5_steps_to_n_body_simulation/>
- Repository: <https://github.com/alvinng4/grav_sim>

At the time of packaging, no external `grav_sim` source files are intentionally included in this repository. If future work copies or adapts implementation details from `grav_sim`, the copied/adapted files should preserve the original MIT license notice and this repository should keep the corresponding attribution in `THIRD_PARTY_NOTICES.md`.
