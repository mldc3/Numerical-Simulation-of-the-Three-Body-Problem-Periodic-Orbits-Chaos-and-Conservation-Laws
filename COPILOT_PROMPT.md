# Prompt for GitHub Copilot

@copilot Reorganize this repository so it becomes a clean, professional and easy-to-understand GitHub project, without changing the physics or the numerical results.

Context:
- Computational physics project about the three-body problem and extensions to $N$-body systems.
- Main code file: `src/three_body_simulation.py`.
- Long-form documentation: `docs/theory_and_history.md`, `docs/results_and_discussion.md`, `docs/implementation_notes.md`.
- Selected visual assets: `assets/gifs/` and `assets/figures/`.
- Original supporting material has been cleaned from this repository.

Goal:
When someone opens the repository, it should look like a complete and polished project: theory, history, equations, explained results, GIFs placed in the right sections, and code that is easy to locate and run.


Attribution requirement:
- Preserve the acknowledgement to Ching-Yin Ng's `grav_sim` project and the tutorial **“5 steps to N-body simulation”**.
- Keep the links in `README.md`, `docs/theory_and_history.md`, `docs/results_and_discussion.md`, `docs/implementation_notes.md`, `ACKNOWLEDGEMENTS.md` and `THIRD_PARTY_NOTICES.md`.
- Do not remove or weaken the attribution while reorganizing the repository.
- Do not claim that this repository is the original `grav_sim` project. It is an independent practice project inspired in part by that resource.

Tasks:
1. Keep a clear structure: `src/`, `docs/`, `assets/gifs/`, `assets/figures/`.
2. Improve `README.md` as the main landing page: title, description, GIFs, method table, main results, installation, execution, links to docs and conclusions.
3. Check that all relative links to images and GIFs work correctly on GitHub.
4. Use equations compatible with GitHub Markdown.
5. Keep `requirements.txt` minimal with `numpy`, `matplotlib` and `pillow`.
6. Add a clear results section: RKF5 as the most accurate method, velocity Verlet as the best compromise, Euler as a pedagogical baseline, and Backward Euler as not competitive in this implementation.
7. Add limitations and future work: refactoring, CLI, tests, adaptive integrators and optimization for large $N$.
8. Do not include the original large ZIP file or empty/duplicate GIFs.
9. Do not change formulas, initial conditions, masses or units unless the change is explicitly justified.
10. Preserve all acknowledgements, references and third-party notices.
11. Leave the repository ready for a clean commit with the suggested message: `Add three-body simulation project with theory, results, visual assets and attribution`.
