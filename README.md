# Numerical Simulation of the Three-Body Problem

A computational physics project devoted to the **gravitational three-body problem** and its extension to larger $N$-body systems. The repository compares four numerical integrators — explicit Euler, an approximate Backward Euler scheme, velocity Verlet and fifth-order Runge-Kutta-Fehlberg — through trajectories, GIF animations, energy conservation, angular-momentum conservation, Poincaré sections and simplified Solar System experiments.

![Method error summary](assets/figures/analysis/method_errors_summary.png)

## Repository contents

- `src/three_body_simulation.py`: main Python script from the project, renamed for GitHub-friendly use.
- `docs/theory_and_history.md`: full theoretical background, historical context and mathematical formulation.
- `docs/results_and_discussion.md`: detailed discussion of the numerical results and method comparison.
- `docs/implementation_notes.md`: mapping between the equations and the implementation.
- `assets/gifs/`: selected GIFs for the most representative simulations.
- `assets/figures/`: static plots, screenshots and numerical-analysis figures.
- `docs/source_material/`: original supporting material used as the basis for the project.

## Quick visual overview

### Chaotic three-body benchmark

| Explicit Euler | RKF5 | Velocity Verlet |
|---|---|---|
| ![Chaotic Euler](assets/gifs/method_comparison/chaotic_euler.gif) | ![Chaotic RKF5](assets/gifs/method_comparison/chaotic_rkf5.gif) | ![Chaotic velocity Verlet](assets/gifs/method_comparison/chaotic_velocity_verlet.gif) |

### Periodic orbits and larger systems

| Broucke orbit | Skinny Pineapple orbit | Simplified Solar System |
|---|---|---|
| ![Broucke RKF5](assets/gifs/periodic_orbits/broucke_rkf5.gif) | ![Skinny Pineapple RKF5](assets/gifs/periodic_orbits/skinny_pineapple_rkf5.gif) | ![Solar System RKF5](assets/gifs/solar_system/solar_system_rkf5.gif) |

## Physical formulation

For $N$ bodies with masses $m_i$, positions $\mathbf r_i$ and velocities $\mathbf v_i$, the gravitational acceleration of body $i$ is

$$
\ddot{\mathbf r}_i = G\sum_{j\neq i}m_j\frac{\mathbf r_j-\mathbf r_i}{\|\mathbf r_j-\mathbf r_i\|^3}.
$$

The problem is integrated as a first-order dynamical system:

$$
\dot{\mathbf r}_i=\mathbf v_i,\qquad \dot{\mathbf v}_i=\mathbf a_i.
$$

Because the system is isolated and the gravitational forces are internal and central, the exact dynamics conserve total mechanical energy and total angular momentum:

$$
E=\sum_i \frac12 m_i\|\mathbf v_i\|^2-G\sum_{i<j}\frac{m_i m_j}{\|\mathbf r_i-\mathbf r_j\|},
\qquad
\mathbf L=\sum_i m_i\mathbf r_i\times\mathbf v_i.
$$

For that reason, the project does not only ask whether a trajectory looks visually attractive. It also asks whether the integrator respects the physics.

## Numerical methods compared

| Method | Global order | Approximate cost per step | Role in the project |
|---|---:|---:|---|
| Explicit Euler | $O(\Delta t)$ | 1 acceleration evaluation | Pedagogical baseline; fast but strongly drifting. |
| Approximate Backward Euler | $O(\Delta t)$ | Iterative | Implicit idea, but not competitive in this implementation. |
| Velocity Verlet | $O(\Delta t^2)$ | 2 acceleration evaluations | Best balance between stability, cost and physical conservation. |
| RKF5 | $O(\Delta t^5)$ | 6 acceleration evaluations | Best global accuracy among the implemented methods. |

## Main conclusion

- **RKF5** is the most accurate method for comparing sensitive and chaotic trajectories.
- **Velocity Verlet** provides the best practical compromise: it preserves the mechanical structure very well at moderate cost.
- **Explicit Euler** is useful as a didactic reference because it clearly shows energy drift and orbital deformation.
- **Backward Euler** is not competitive here because the implemented fixed-point approximation remains first order and may introduce error spikes when forces vary rapidly.

## Running the project

```bash
pip install -r requirements.txt
python src/three_body_simulation.py
```

The script keeps commented blocks for activating different systems, plots and diagnostics. This preserves the original project workflow while keeping the repository easy to inspect.

## Documentation

- [Theory and history](docs/theory_and_history.md)
- [Results and discussion](docs/results_and_discussion.md)
- [Implementation notes](docs/implementation_notes.md)
- [Asset index](assets/ASSETS_INDEX.md)
- [Acknowledgements](ACKNOWLEDGEMENTS.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [References](REFERENCES.md)


## Acknowledgements and inspiration

This project was inspired in part by Ching-Yin Ng's `grav_sim` project and, in particular, by the tutorial **“5 steps to N-body simulation”**, which presents a clear step-by-step introduction to building gravitational $N$-body simulations in Python.

- Tutorial: <https://alvinng4.github.io/grav_sim/5_steps_to_n_body_simulation/>
- Repository: <https://github.com/alvinng4/grav_sim>

The present repository is an independent computational-physics practice project. The citation above is included to acknowledge the educational resource that helped motivate the implementation style and simulation workflow. No external `grav_sim` source code is intentionally vendored in this repository. If any future refactor directly copies or adapts code from `grav_sim`, the original MIT license notice must be preserved.

## Suggested future improvements

The current repository intentionally preserves the original single-script structure. A more software-engineering-oriented version could split the code into modules such as `initial_conditions.py`, `integrators.py`, `diagnostics.py`, `plotting.py`, `examples/` and `tests/`. That refactor should be conservative: formulas, units, masses and initial conditions should not be changed unless the change is explicitly justified.
