# Theory and History of the Three-Body Problem

## 1. Motivation

The three-body problem studies the motion of three masses interacting through their mutual gravitational attraction. At first sight it looks like a direct extension of the two-body problem, but the qualitative behaviour changes completely. The two-body problem admits a closed-form solution in terms of conic sections, whereas the general three-body problem does not have a general analytical solution. Instead, it exhibits periodic orbits, close encounters, escape events, binary formation, long-term instability and deterministic chaos.

This makes the three-body problem an ideal computational-physics project. It is simple enough to formulate from Newton's law of gravitation, but rich enough to test whether a numerical method is physically trustworthy. A good simulation should not merely draw attractive trajectories. It should also preserve the invariants of the exact dynamics as well as possible, especially the total mechanical energy and total angular momentum.

## 2. Historical context

### 2.1 Newton and the two-body problem

Newton's law of universal gravitation explains Kepler's laws by reducing the two-body problem to the motion of a single reduced mass in a central inverse-square potential. The resulting trajectories are conic sections: ellipses, parabolas or hyperbolas depending on the total energy. This case is integrable because the motion has enough conserved quantities to determine the orbit analytically.

The difficulty appears when a third gravitationally interacting mass is added. The Sun-Earth-Moon system is the classical motivating example: the Moon is mainly bound to the Earth, but the solar perturbation cannot be ignored. This turns celestial mechanics from a closed-form problem into one of the central long-term challenges of mathematical physics.

### 2.2 Euler, Lagrange and special solutions

Although the general problem is not analytically solvable, some special solutions are known. Euler found collinear configurations in which the three bodies remain aligned while rotating or scaling in a coordinated way. Lagrange found equilateral-triangle configurations in which the shape of the system is preserved in a rotating frame. These solutions are important because they show that the absence of a general solution does not mean the absence of structure.

In numerical work, such regular configurations are useful test cases. If a method cannot reproduce a simple symmetric orbit for a sufficiently long time, it will not be reliable in more sensitive cases.

### 2.3 Poincaré and deterministic chaos

Poincaré's work on the three-body problem revealed that the issue was not only technical difficulty. The phase-space structure of the problem can be extremely complicated. Trajectories with nearly identical initial conditions can separate dramatically over time. This is deterministic chaos: the equations are deterministic, but long-term prediction becomes practically limited because small physical or numerical errors are amplified.

This idea is central to the project. In a chaotic system, numerical error is not a minor detail. An inaccurate integrator can generate behaviour that looks complex but is not a faithful representation of the intended physical system.

### 2.4 Modern computational discovery of periodic orbits

Modern computing has revealed many families of periodic three-body orbits, including figure-eight, Broucke, butterfly, dragonfly, yarn, Li-Liao and Skinny Pineapple-type trajectories. Many of these are found numerically by searching for initial conditions that close after a period. They are excellent benchmark cases because they combine visual clarity with strong sensitivity to integration error.

The project uses these orbits, together with chaotic and Solar-System-like cases, to compare the implemented integrators under different physical conditions.

## 3. Newtonian equations of motion

Consider $N$ point masses $m_i$ with positions $\mathbf r_i(t)$ and velocities $\mathbf v_i(t)$. The gravitational force exerted by body $j$ on body $i$ is

$$
\mathbf F_{ij}=Gm_i m_j\frac{\mathbf r_j-\mathbf r_i}{\|\mathbf r_j-\mathbf r_i\|^3}.
$$

Summing all pairwise contributions gives

$$
m_i\ddot{\mathbf r}_i=G\sum_{j\neq i}m_i m_j\frac{\mathbf r_j-\mathbf r_i}{\|\mathbf r_j-\mathbf r_i\|^3}.
$$

After cancelling $m_i$, the acceleration of body $i$ is

$$
\boxed{\ddot{\mathbf r}_i=G\sum_{j\neq i}m_j\frac{\mathbf r_j-\mathbf r_i}{\|\mathbf r_j-\mathbf r_i\|^3}}.
$$

This is the core equation implemented in the code. Each body is accelerated by every other body, and the direct summation of all pairwise interactions scales as $O(N^2)$ per time step.

## 4. First-order formulation

Numerical integrators usually operate on first-order systems. Defining

$$
\mathbf v_i=\dot{\mathbf r}_i,
$$

the second-order Newtonian equation becomes

$$
\dot{\mathbf r}_i=\mathbf v_i,
\qquad
\dot{\mathbf v}_i=\mathbf a_i(\mathbf r_1,\ldots,\mathbf r_N).
$$

For three bodies in three spatial dimensions, there are 18 dynamical variables: 9 position components and 9 velocity components. This is why a full three-body initial condition requires the positions and velocities of all three bodies.

## 5. Centre-of-mass frame

For visualization and numerical clarity, the system is shifted to its centre-of-mass frame. The centre-of-mass position and velocity are

$$
\mathbf R_{CM}=\frac{1}{M}\sum_i m_i\mathbf r_i,
\qquad
\mathbf V_{CM}=\frac{1}{M}\sum_i m_i\mathbf v_i,
\qquad
M=\sum_i m_i.
$$

The transformed variables are

$$
\mathbf r_i\leftarrow\mathbf r_i-\mathbf R_{CM},
\qquad
\mathbf v_i\leftarrow\mathbf v_i-\mathbf V_{CM}.
$$

This transformation does not change the relative gravitational dynamics. It simply removes a uniform translation of the whole system, so the plotted trajectories are easier to interpret.

## 6. Conserved quantities

### 6.1 Total mechanical energy

The exact Newtonian $N$-body system conserves the total mechanical energy

$$
E=\sum_i \frac12 m_i\|\mathbf v_i\|^2-G\sum_{i<j}\frac{m_i m_j}{\|\mathbf r_i-\mathbf r_j\|}.
$$

The first term is kinetic energy. The second term is gravitational potential energy. Because the potential is negative, strongly bound systems have lower total energy.

In the project, the relative energy error is measured as

$$
\varepsilon_E(n)=\frac{|E_n-E_0|}{|E_0|}.
$$

A systematic growth of $\varepsilon_E$ means that the integrator is injecting or removing artificial energy.

### 6.2 Total angular momentum

The angular momentum is

$$
\mathbf L=\sum_i m_i\mathbf r_i\times\mathbf v_i.
$$

Gravitational forces act along the line joining pairs of particles, so the internal torques cancel and the total angular momentum is conserved. The code measures the relative angular-momentum error as

$$
\varepsilon_L(n)=\frac{\|\mathbf L_n-\mathbf L_0\|}{\|\mathbf L_0\|}.
$$

Angular momentum is often better preserved than energy because it depends linearly on positions and velocities, whereas energy includes quadratic velocity terms and singular potential terms during close encounters.

## 7. Numerical integration methods

### 7.1 Explicit Euler

Explicit Euler updates positions and velocities using only the current slope:

$$
\mathbf r_{n+1}=\mathbf r_n+\mathbf v_n\Delta t,
\qquad
\mathbf v_{n+1}=\mathbf v_n+\mathbf a_n\Delta t.
$$

It has first-order global accuracy. Its main advantage is simplicity and low cost. Its main weakness is systematic drift: it does not preserve the geometry of Hamiltonian motion and therefore performs poorly in long gravitational simulations.

### 7.2 Approximate Backward Euler

Backward Euler uses the future state in the update:

$$
\mathbf v_{n+1}=\mathbf v_n+\mathbf a(\mathbf r_{n+1})\Delta t,
\qquad
\mathbf r_{n+1}=\mathbf r_n+\mathbf v_{n+1}\Delta t.
$$

Because the acceleration depends on the unknown future position, the method is implicit. In this project it is approximated by fixed-point iteration. Although implicit Euler can be valuable for stiff dissipative systems, this implementation remains first order and is not especially well suited to conservative orbital dynamics.

### 7.3 Velocity Verlet

Velocity Verlet is designed for second-order mechanical systems:

$$
\mathbf r_{n+1}=\mathbf r_n+\mathbf v_n\Delta t+\frac12\mathbf a_n\Delta t^2,
$$

$$
\mathbf a_{n+1}=\mathbf a(\mathbf r_{n+1}),
$$

$$
\mathbf v_{n+1}=\mathbf v_n+\frac12(\mathbf a_n+\mathbf a_{n+1})\Delta t.
$$

It is second-order accurate, time-reversible for constant time step, and much better adapted to conservative mechanics than explicit Euler. In practice, it often keeps the energy bounded around the correct value rather than producing monotonic drift.

### 7.4 Fifth-order Runge-Kutta-Fehlberg

The RKF5 method estimates the state by combining several intermediate slopes:

$$
y_{n+1}=y_n+\Delta t\sum_k b_k k_k.
$$

Here $y$ contains both positions and velocities. The method has high global accuracy, but it requires more acceleration evaluations per step. It is therefore more expensive than Euler or Verlet, but it is the most accurate method included in the project.

## 8. Poincaré sections

A Poincaré section records intersections of a trajectory with a chosen surface in phase space. In the code, an upward crossing through $y=0$ is used and the pair $(x,v_x)$ is stored.

Regular or quasi-periodic motion tends to produce repeated curves or structured patterns. Chaotic motion tends to produce a more scattered cloud of points. This is why Poincaré sections are useful: they reveal phase-space structure that is not always obvious from a trajectory plot alone.

The Poincaré-Bendixson theorem applies only to autonomous two-dimensional continuous systems under specific hypotheses. The full three-body problem has a much higher-dimensional phase space, so it is not restricted by that theorem and can display chaotic dynamics.

## 9. Extension to larger $N$-body systems

The same Newtonian formulation applies to Earth-Moon-Sun simulations, simplified Solar System models and idealized galactic configurations. The difficulty becomes computational: direct pairwise summation scales as $O(N^2)$ per step. For very large systems, one would normally need softening, adaptive time stepping, tree methods, parallelization and physically consistent initial-condition generation.

## 10. Final idea

This project shows that the physical formulation and the numerical method cannot be separated. The same equations can produce a reliable simulation or an unphysical one depending on the integrator. That is why the analysis combines trajectories, animations, energy error, angular-momentum error, computational cost and sensitivity to initial conditions.


## 11. References and further reading

- Ching-Yin Ng, `grav_sim`: $N$-body gravity simulation library with C and Python API. Repository: <https://github.com/alvinng4/grav_sim>.
- Ching-Yin Ng, **“5 steps to N-body simulation”**. Tutorial: <https://alvinng4.github.io/grav_sim/5_steps_to_n_body_simulation/>.
- Javier Roa, Adrian S. Hamers, Maxwell X. Cai and Nathan W. C. Leigh, *Moving Planets Around: An Introduction to N-Body Simulations Applied to Exoplanetary Systems*, MIT Press, 2020.
- Richard L. Burden and J. Douglas Faires, *Numerical Analysis*, Cengage Learning, 9th edition, 2011.

The `grav_sim` tutorial is cited as an inspiration and further-reading resource for the numerical $N$-body simulation workflow. The mathematical and physical discussion in this repository remains focused on the project implementation and the supplied practice material.
