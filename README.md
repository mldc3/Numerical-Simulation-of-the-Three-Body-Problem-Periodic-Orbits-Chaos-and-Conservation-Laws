# Numerical Simulation of the Three-Body Problem
### Periodic Orbits, Chaos, and Conservation Laws

This repository presents a computational-physics study of the gravitational **three-body problem** (and extensions to larger N-body systems), with emphasis on:

- periodic and quasi-periodic trajectories,
- chaotic sensitivity to initial conditions,
- numerical stability,
- conservation of total energy and angular momentum.

The project compares multiple time-integration strategies (Euler, Backward Euler, velocity Verlet, RKF5) through trajectory analysis, diagnostics, and visual outputs.

---

## Why this project matters

The three-body problem is a classic nonlinear system in celestial mechanics. Unlike the two-body case, closed-form solutions are generally unavailable, so simulation quality depends strongly on numerical method choice. This project uses that setting to study how integrator behavior affects:

- physical fidelity,
- long-time drift in invariants,
- and qualitative orbit classification (regular vs chaotic).

---

## Physical model

The simulations follow Newtonian gravity for point masses:

$$
\mathbf{F}_{ij} = G\,\frac{m_i m_j}{\|\mathbf{r}_j - \mathbf{r}_i\|^3}(\mathbf{r}_j - \mathbf{r}_i)
$$

Core diagnostics include:

- **Total energy** $E = T + V$
- **Total angular momentum** $\mathbf{L}$
- **Phase-space / Poincaré-style behavior** for regularity vs chaos

> The scientific intent is to preserve physical logic while comparing numerical schemes, not to alter governing equations or initial conditions arbitrarily.

---

## Historical and scientific context

The three-body problem has shaped modern dynamics from Newton and Euler to Poincaré and contemporary chaos theory. This repository positions that classical problem in a modern numerical workflow:

1. define physically meaningful initial states,
2. integrate with multiple solvers,
3. evaluate invariants and trajectory structure,
4. interpret numerical artifacts versus real dynamics.

---

## Methods compared

- **Forward Euler** – simple baseline, typically dissipative/unstable for long runs.
- **Backward Euler** – implicit and more stable, but can overdamp motion.
- **Velocity Verlet** – symplectic-style behavior, often better long-term conservation.
- **RKF5** – adaptive high-order method for accuracy-focused integration.

---

## Results and visual outputs

Typical outputs used in this project:

- trajectory plots in configuration space,
- animations (GIFs) of orbital evolution,
- energy and angular-momentum error curves,
- Poincaré-style sections for dynamical classification,
- simplified Solar System experiments.

If your branch includes generated media, present them under a section like:

```markdown
## Gallery
![Three-body orbit](assets/gifs/three_body_orbit.gif)
![Energy drift comparison](assets/images/energy_drift.png)
```

---

## Running the project

Because this repository may be used in different local setups, a typical Python workflow is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python <main_simulation_script>.py
```

If your local version includes notebooks or separate scripts, run those entry points directly and keep analysis artifacts (plots/GIFs) organized under clear folders (for example `assets/`, `results/`, or `figures/`).

---

## Suggested repository organization

For clear GitHub presentation, keep scientific code and communication assets separated:

- `src/` or script root for simulation code
- `docs/` for explanatory markdown notes
- `assets/gifs/` and `assets/images/` for visual outputs used in this README
- `results/` for reproducible experiment outputs

---

## Attribution

This work is **inspired in part by Ching-Yin Ng's `grav_sim` tutorial/workflow**, adapted here for comparative numerical experiments in the three-body and N-body context.

Please retain this attribution when reusing or extending the workflow.

---

## License and reuse

If you reuse this repository for research or teaching, cite both:

1. this project, and
2. the referenced `grav_sim` inspiration where relevant.
