# Source Code Overview

`three_body_simulation.py` contains the full simulation logic for the three-body and small $N$-body experiments, including:
- predefined initial-condition systems,
- Newtonian acceleration computation,
- Euler / Backward Euler / Velocity Verlet / RKF5 integrators,
- diagnostics (energy, angular momentum, error trends, Poincaré sections),
- plotting and GIF generation helpers.

## Run

From repository root:

```bash
python src/three_body_simulation.py
```

## Required packages

Install with:

```bash
pip install -r requirements.txt
```

Dependencies:
- `numpy`
- `matplotlib`
- `pillow`
