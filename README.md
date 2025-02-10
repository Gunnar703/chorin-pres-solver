# chorin-pressure-solver
Implements Chorin's pressure projection method to simulate transient, incompressible fluid flow in a rectangular domain.

## Smoke Visualization
This code implements smoke flow visualization by introducing a virtual 'smoke density' field (virtual because it does not affect the simulation). Smoke is added at a user-specified location and propagated according to the continuity equation, where the density terms are replaced by 'smoke density' and the velocity terms are the same as in the simulation. An example visualization is shown below.

<p align="center">
  <img width="256" height="256" src="smoke_anim.gif" alt="animated" />
  <br/>
  Smoke visualization. All BCs are homogeneous Dirichlet except for the two high-pressure 'holes' on either end of the domain. Those are set to $p=200.0$
</p>

## Other Animations
From left to right: Pressure, X-Velocity, and Y-Velocity.
<p align="center">
  <img width="256" height="256" src="pres_anim.gif" alt="animated"/>
  <img width="256" height="256" src="u_anim.gif" alt="animated"/>
  <img width="256" height="256" src="v_anim.gif" alt="animated"/>
</p>

# Chorin's Method
Chorin's pressure-projection method is an explicit method of simulating transient, incompressible fluid flows. It involves splitting the computation into three steps:
1. Computation of an estimated acceleration field
2. Solution of a Poisson equation to find a pressure field which satisfies the continuity equation (0 divergence for incompressible flows)
3. Computation of a corrected acceleration field based on the outputs of steps 1 and 2.

## Derivation
Start with the incompressible Navier Stokes equations (omitting energy since it is completely decoupled from continuity and momentum for incompressible flows).
```math
\frac{\partial u}{\partial t} = \nu\nabla^2u - \left(u\cdot\nabla\right)u - \frac{1}{\rho}\nabla p
```
```math
\nabla\cdot u =0
```
Introduce an estimated acceleration that satisfies the momentum equations without the pressure term
```math
\frac{\partial u^*}{\partial t} = \nu\nabla^2u - \left(u\cdot\nabla\right)u \tag{A}
```
Rearrange the momentum equation and the estimator momentum equation (above).
```math
\frac{\partial u}{\partial t} + \frac{1}{\rho}\nabla p = \nu\nabla^2u - \left(u\cdot\nabla\right)u \tag{1}
```
```math
\frac{\partial u^*}{\partial t} = \nu\nabla^2u - \left(u\cdot\nabla\right)u \tag{2}
```
Subtract (1) - (2):
```math
\frac{\partial u}{\partial t} - \frac{\partial u^*}{\partial t} + \frac{1}{\rho}\nabla p = 0\tag{C}
```
Take the divergence of both sides and note that the divergence operator and the time derivative operator commute with one another.
```math
\nabla\cdot\frac{\partial u}{\partial t} - \nabla\cdot\frac{\partial u^*}{\partial t} = - \frac{1}{\rho}\nabla\cdot\nabla p
```
```math
\frac{\partial}{\partial t}\left( \nabla\cdot u \right) - \nabla\cdot\frac{\partial u^*}{\partial t} = - \frac{1}{\rho}\nabla^2 p
```
The first term is zero from the continuity equation, therefore
```math
 \nabla^2 p = \rho\left(\nabla\cdot\frac{\partial u^*}{\partial t}\right) \tag{B}
```

Equations A, B, and C (repeated below) represent the three steps to applying Chorin's pressure-projection method.
1. Estimate the flow acceleration
```math
\frac{\partial u^*}{\partial t} = \nu\nabla^2u - \left(u\cdot\nabla\right)u \tag{A}
```
2. Compute a pressure field that satisfies the continuity equation
```math
 \nabla^2 p = \rho\left(\nabla\cdot\frac{\partial u^*}{\partial t}\right) \tag{B}
```
3. Correct the flow acceleration estimate
```math
\frac{\partial u}{\partial t} = \frac{\partial u^*}{\partial t} - \frac{1}{\rho}\nabla p \tag{C*}
```

## Discretization
In this code, time derivatives are approximated using a first order forward difference (Euler) scheme and spatial derivatives are approximated using a second-order central difference scheme. The laplacian operator is approximated using a five-point stencil.

In summary, this code uses at FTCS (Forward in Time, Central in Space) numerical scheme to discretize the differential operators.

## Implementation
The code is implemented in C++/CUDA. Equation B is solved using the Jacobi method without over/under-relaxation.
