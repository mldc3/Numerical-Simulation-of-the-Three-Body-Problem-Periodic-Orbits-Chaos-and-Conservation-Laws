# Numerical Simulation of the Three-Body Problem: Periodic Orbits, Chaos and Conservation Laws

Computational-physics project exploring gravitational three-body and small $N$-body systems with multiple numerical integrators, trajectory visualizations, and conservation-law diagnostics.

This is an independent practice project inspired in part by Ching-Yin Ng's educational `grav_sim` materials; it is **not** the original `grav_sim` repository.

## Visual preview

![Chaotic RKF5](assets/gifs/chaotic_cases/chaotic_rkf5.gif)
![Broucke RKF5](assets/gifs/periodic_orbits/broucke_rkf5.gif)
![Solar System RKF5](assets/gifs/solar_system/solar_system_rkf5.gif)

![Method error summary](assets/figures/diagnostics/method_errors_summary.png)

## Theory summary

The project integrates Newtonian $N$-body equations in 3D:

$$
\ddot{\mathbf r}_i=G\sum_{j\neq i}m_j\frac{\mathbf r_j-\mathbf r_i}{\|\mathbf r_j-\mathbf r_i\|^3}
$$

and evaluates how numerical methods affect physical reliability through:
- trajectory quality,
- total energy conservation,
- total angular momentum conservation,
- long-time stability in periodic and chaotic regimes.

## Numerical methods

| Method | Type | Global order | Project role |
|---|---|---:|---|
| Euler | Explicit | 1 | Pedagogical baseline; fastest per step but largest drift |
| Backward Euler (fixed-point approximation) | Implicit approximation | 1 | Stability comparison baseline in this implementation |
| Velocity Verlet | Symplectic-style mechanical integrator | 2 | Best cost/conservation compromise |
| RKF5 | Runge-Kutta-Fehlberg | 5 | Highest accuracy in tested benchmarks |

## Results summary

- **RKF5** provides the best accuracy and strongest conservation diagnostics.
- **Velocity Verlet** is the most practical compromise between cost and physical fidelity.
- **Euler** is useful for comparison but not reliable for long-term orbital behavior.
- **Backward Euler** is not competitive in this specific fixed-point implementation.

See detailed analysis in:
- [`docs/theory_and_history.md`](docs/theory_and_history.md)
- [`docs/results_and_discussion.md`](docs/results_and_discussion.md)
- [`docs/implementation_notes.md`](docs/implementation_notes.md)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Execution

```bash
python src/three_body_simulation.py
```

## Repository structure

```text
.
├── README.md
├── PROJECT_INFO.md
├── COPILOT_PROMPT.md
├── ACKNOWLEDGEMENTS.md
├── REFERENCES.md
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── three_body_simulation.py
│   └── README.md
├── docs/
│   ├── theory_and_history.md
│   ├── results_and_discussion.md
│   └── implementation_notes.md
└── assets/
    ├── ASSETS_INDEX.md
    ├── gifs/
    │   ├── periodic_orbits/
    │   ├── chaotic_cases/
    │   ├── solar_system/
    │   └── basic_cases/
    └── figures/
        ├── periodic_orbits/
        ├── chaotic_cases/
        ├── solar_system/
        ├── diagnostics/
        └── basic_cases/
```

## Acknowledgements

This project was inspired in part by Ching-Yin Ng's `grav_sim` project and tutorial:
- Tutorial: <https://alvinng4.github.io/grav_sim/5_steps_to_n_body_simulation/>
- Repository: <https://github.com/alvinng4/grav_sim>

This repository remains an independent computational-physics practice project.

## References

See [`REFERENCES.md`](REFERENCES.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
