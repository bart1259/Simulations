# Finite Volume Method (FVM) for 2D Heat Diffusion Equation on irregular grid

### Problem Setup

Consider the heat diffusion equation:

$$ \frac{\partial u}{\partial t} = \alpha \nabla^2 u $$

where:
- $ u $ is the temperature field as a function of space and time.
- $ \alpha $ is the thermal diffusivity of the material.
- $ \nabla^2 $ denotes the Laplacian operator.

### Finite Volume Method Approach

1. **Domain Discretization:**
   - Divide the spatial domain into a finite number of small control volumes (CVs or "cells").
   - Each control volume is centered around a node where you want to approximate the temperature $ u $.

2. **Integral Formulation:**
   - Integrate the governing equation over each control volume. For a control volume $ V_i $:

   $$ \int_{V_i} \frac{\partial u}{\partial t} \, dV = \alpha \int_{V_i} \nabla^2 u \, dV $$

3. **Discretization in Time and Space:**
   - Use the divergence theorem to convert the volume integral of the Laplacian to a surface integral:

   $$ \frac{d}{dt} \int_{V_i} u \, dV = \alpha \int_{S_i} \nabla u \cdot \hat{n} \, dS $$

   where $ S_i $ is the surface of the control volume and $ \hat{n} $ is the outward normal.

4. **Approximation of Terms:**
   - **Time Derivative:** Using a simple forward difference for time, for a small time step $ \Delta t $,

   $$ \frac{1}{\Delta t} (u_i^{n+1} - u_i^{n}) V_i = \alpha \sum_{\text{faces } j} (u_j - u_i) \frac{A_j}{d_j} $$

   where:
   - $ u_i^n $ is the temperature at node $ i $ at time step $ n $.
   - $ V_i $ is the volume of the control volume.
   - $ A_j $ is the area of face $ j $ of the control volume.
   - $ d_j $ is the distance between the centroid of volume $ i $ and the centroid of the neighboring volume sharing face $ j $.
   - The above equation assumes a simple linear approximation for the temperature gradient between neighboring centroid points.

5. **Assembly of Linear System:**
   - Organize the equations from all control volumes into a system of linear equations, which can be solved using standard numerical methods.

6. **Boundary and Initial Conditions:**
   - Apply appropriate boundary conditions (e.g., Dirichlet or Neumann) and initial conditions to ensure a well-posed problem.