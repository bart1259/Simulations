import numpy as np
import jax.numpy as jnp
from jax import jit
import jax

from functools import partial
from scipy.spatial import KDTree

from src.config import FluidSimConfig2D, FluidSimConfig3D
from src.kernel_functions import (
    kernel_function_poly6,
    kernel_function_gradient_spiky,
    kernel_function_viscosity_laplacian,
)

import time

MAX_NEIGHBORS = 32

class FluidSimulationEngine:
    def __init__(self, config: FluidSimConfig2D):
        if type(config) == FluidSimConfig2D:
            self.dimensions = 2
        elif type(config) == FluidSimConfig3D:
            self.dimensions = 3
        else:
            raise ValueError("Invalid configuration type. Expected FluidSimConfig2D or FluidSimConfig3D")

        self.config = config
        self.time = 0.0
        self.is_running = True

        self.spawns_realized = [False for _ in range(len(config.spawn_groups))]

        self.positions = jnp.zeros((0, self.dimensions), dtype=jnp.float32) # TODO: Consider making particles closer to eachother close in index space
        self.velocities = jnp.zeros((0, self.dimensions), dtype=jnp.float32)

        # Check if there are any spawn groups at time 0 defined
        for i, spawn_group in enumerate(config.spawn_groups):
            if spawn_group.spawn_time == 0.0:
                self._spawn_particles(i)

        self.neighborhood_calculation_time = 0.0
        self.step_time = 0.0

    def _get_neighborhood_matrix(self, positions, h):
        """
        positions: [n, d] array of particle positions
        h: float, smoothing length
        returns: [n, n] array of booleans indicating if a particle is in the neighborhood of another particle
        """
        # IDK if this can be written efficiently in jax so we'll use sklearn
        assert type(positions) == np.ndarray, "positions must be a numpy array"
        tree = KDTree(positions)
        neighbor_indexes = tree.query_ball_point(positions, r=h, workers=-1)
        neighborhood_matrix = -np.ones((len(positions), MAX_NEIGHBORS), dtype=int)
        for i, neighbors in enumerate(neighbor_indexes):
            neighborhood_matrix[i, :min(MAX_NEIGHBORS, len(neighbors))] = neighbors[:MAX_NEIGHBORS]
        neighborhood_matrix = jnp.array(neighborhood_matrix)
        return neighborhood_matrix

    @partial(jit, static_argnums=0)
    def _get_particle_densities(self, positions, neighborhood_matrix, h):
        """
        positions: [n, d] array of particle positions
        neighborhood_matrix: [n, MAX_NEIGHBORS] array of neighbor indexes
        h: float, smoothing length
        returns: [n, ] array of density values
        """

        # Loop over each particle and compute the density
        densities = jnp.zeros((len(positions),), dtype=jnp.float32)

        def _compute_density(position, neighborhood, positions, h):
            # Get the positions of the neighbors
            neighbor_positions = positions[neighborhood]
            # Compute the kernel function for each neighbor
            position = jnp.repeat(position[None, :], len(neighborhood), axis=0)
            kernel_values = kernel_function_poly6(position, neighbor_positions, h)
            # Zero out -1 neighbors
            kernel_values = jnp.where(neighborhood == -1, 0.0, kernel_values)
            # Sum the kernel values to get the density
            return jnp.sum(kernel_values, axis=-1)
        
        # Loop over each particle and compute the density
        compute_density_vmap = jax.vmap(_compute_density, in_axes=(0, 0, None, None))
        densities = compute_density_vmap(positions, neighborhood_matrix, positions, h)
        return densities
    
    @partial(jit, static_argnums=0)
    def _get_particle_pressure_forces(self, positions, densities, pressures, neighborhood_matrix, h):
        """
        positions: [n, d] array of particle positions
        neighborhood_matrix: [n, MAX_NEIGHBORS] array of neighbor indexes
        densities: [n, ] array of particle densities
        pressures: [n, ] array of particle pressures
        h: float, smoothing length
        returns: [n, d] array of pressure forces
        """

        # Loop over each particle and compute the pressure force
        pressure_forces = jnp.zeros((len(positions), self.dimensions), dtype=jnp.float32)

        def _compute_pressure_force(particle_index, position, neighborhood, positions, pressures, h):
            # Get the positions of the neighbors
            neighbor_positions = positions[neighborhood]
            # Compute the kernel function for each neighbor
            position = jnp.repeat(position[None, :], len(neighborhood), axis=0)
            grad_kernel = kernel_function_gradient_spiky(position, neighbor_positions, h)

            # Zero out -1 neighbors
            zero_mask = jnp.where(neighborhood == -1, 0.0, 1.0)
            grad_kernel = grad_kernel * zero_mask[:, None]

            # Compute the pressure force for each neighbor
            pressures = -grad_kernel * (pressures[neighborhood] + pressures[particle_index])[:, None] / (2 * densities[neighborhood])[:, None]

            # Sum the pressure forces to get the total pressure force
            return jnp.sum(pressures, axis=0)
        
        # Loop over each particle and compute the pressure force
        compute_pressure_force_vmap = jax.vmap(_compute_pressure_force, in_axes=(0, 0, 0, None, None, None))
        pressure_forces = compute_pressure_force_vmap(jnp.arange(len(positions)), positions, neighborhood_matrix, positions, pressures, h)
        return pressure_forces


    @partial(jit, static_argnums=0)
    def _get_particle_viscosity_forces(self, positions, velocities, densities, neighborhood_matrix, h):
        """
        positions: [n, d] array of particle positions
        neighborhood_matrix: [n, MAX_NEIGHBORS] array of neighbor indexes
        velocities: [n, d] array of particle velocities
        densities: [n, ] array of particle densities
        h: float, smoothing length
        returns: [n, d] array of viscosity forces
        """

        # Loop over each particle and compute the viscosity force
        viscosity_forces = jnp.zeros((len(positions), self.dimensions), dtype=jnp.float32)

        def _compute_viscosity_force(particle_index, position, neighborhood, positions, velocities, densities, h):
            # Get the positions of the neighbors
            neighbor_positions = positions[neighborhood]
            # Compute the kernel function for each neighbor
            position = jnp.repeat(position[None, :], len(neighborhood), axis=0)
            grad_kernel = kernel_function_viscosity_laplacian(position, neighbor_positions, h)

            # Zero out -1 neighbors
            zero_mask = jnp.where(neighborhood == -1, 0.0, 1.0)
            grad_kernel = grad_kernel * zero_mask

            # Compute the viscosity force for each neighbor
            viscosity_forces = (self.config.viscosity_multiplier * grad_kernel)[:, None] * (velocities[neighborhood] - velocities[particle_index]) / (2 * densities[neighborhood])[:, None]
            
            # Sum the viscosity forces to get the total viscosity force
            return jnp.sum(viscosity_forces, axis=0)
        
        # Loop over each particle and compute the viscosity force
        compute_viscosity_force_vmap = jax.vmap(_compute_viscosity_force, in_axes=(0, 0, 0, None, None, None, None))
        viscosity_forces = compute_viscosity_force_vmap(jnp.arange(len(positions)), positions, neighborhood_matrix, positions, velocities, densities, h)
        return viscosity_forces
    
    @partial(jit, static_argnums=0)
    def _similation_step(self, positions, velocities, neighborhood_matrix):
        # Start computations
        densities = self._get_particle_densities(positions, neighborhood_matrix, self.config.kernel_radius)
        pressures = self.config.pressure_multiplier * (densities - self.config.rest_density)
        pressure_forces = self._get_particle_pressure_forces(positions, densities, pressures, neighborhood_matrix, self.config.kernel_radius)
        viscosity_forces = self._get_particle_viscosity_forces(positions, velocities, densities, neighborhood_matrix, self.config.kernel_radius)

        # Gravity
        if self.dimensions == 2:
            gravity = jnp.array([0.0, -self.config.gravity], dtype=jnp.float32)
        elif self.dimensions == 3:
            gravity = jnp.array([0.0, -self.config.gravity, 0.0], dtype=jnp.float32) # By Blender obj import convention

        gravity_forces = gravity[None, :]
        gravity_forces = jnp.repeat(gravity_forces, len(positions), axis=0)

        # Compute net force
        net_forces = jnp.zeros_like(positions)
        net_forces += pressure_forces
        net_forces += viscosity_forces
        net_forces += gravity_forces
        accelerations = net_forces / densities[:, None]

        # Update velocities and positions
        velocities += accelerations * self.config.time_step
        positions += velocities * self.config.time_step

        # Enforce Bounds
        left_mask = positions[:, 0] < self.config.kernel_radius
        right_mask = positions[:, 0] > (self.config.domain_size[0] - self.config.kernel_radius)
        top_mask = positions[:, 1] > (self.config.domain_size[1] - self.config.kernel_radius)
        bottom_mask = positions[:, 1] < self.config.kernel_radius

        if self.dimensions == 2:
            positions = jnp.where(left_mask[:, None], jnp.stack([self.config.kernel_radius * jnp.ones_like(positions[:,0]), positions[:, 1]], axis=-1), positions)
            velocities = jnp.where(left_mask[:, None], jnp.stack([self.config.bounds_factor * velocities[:, 0], velocities[:, 1]], axis=-1), velocities)
            positions = jnp.where(right_mask[:, None], jnp.stack([self.config.domain_size[0] - self.config.kernel_radius * jnp.ones_like(positions[:,0]), positions[:, 1]], axis=-1), positions)
            velocities = jnp.where(right_mask[:, None], jnp.stack([self.config.bounds_factor * velocities[:, 0], velocities[:, 1]], axis=-1), velocities)
            positions = jnp.where(top_mask[:, None], jnp.stack([positions[:, 0], self.config.domain_size[1] - self.config.kernel_radius * jnp.ones_like(positions[:,1])], axis=-1), positions)
            velocities = jnp.where(top_mask[:, None], jnp.stack([velocities[:, 0], self.config.bounds_factor * velocities[:, 1]], axis=-1), velocities)
            positions = jnp.where(bottom_mask[:, None], jnp.stack([positions[:, 0], self.config.kernel_radius * jnp.ones_like(positions[:,1])], axis=-1), positions)
            velocities = jnp.where(bottom_mask[:, None], jnp.stack([velocities[:, 0], self.config.bounds_factor * velocities[:, 1]], axis=-1), velocities)
        elif self.dimensions == 3:
            # 3D boundary enforcement
            front_mask = positions[:, 2] < self.config.kernel_radius
            back_mask = positions[:, 2] > (self.config.domain_size[2] - self.config.kernel_radius)

            positions = jnp.where(left_mask[:, None], jnp.stack([self.config.kernel_radius * jnp.ones_like(positions[:,0]), positions[:, 1], positions[:, 2]], axis=-1), positions)
            velocities = jnp.where(left_mask[:, None], jnp.stack([self.config.bounds_factor * velocities[:, 0], velocities[:, 1], velocities[:, 2]], axis=-1), velocities)
            positions = jnp.where(right_mask[:, None], jnp.stack([self.config.domain_size[0] - self.config.kernel_radius * jnp.ones_like(positions[:,0]), positions[:, 1], positions[:, 2]], axis=-1), positions)
            velocities = jnp.where(right_mask[:, None], jnp.stack([self.config.bounds_factor * velocities[:, 0], velocities[:, 1], velocities[:, 2]], axis=-1), velocities)
            positions = jnp.where(top_mask[:, None], jnp.stack([positions[:, 0], self.config.domain_size[1] - self.config.kernel_radius * jnp.ones_like(positions[:,1]), positions[:, 2]], axis=-1), positions)
            velocities = jnp.where(top_mask[:, None], jnp.stack([velocities[:, 0], self.config.bounds_factor * velocities[:, 1], velocities[:, 2]], axis=-1), velocities)
            positions = jnp.where(bottom_mask[:, None], jnp.stack([positions[:, 0], self.config.kernel_radius * jnp.ones_like(positions[:,1]), positions[:, 2]], axis=-1), positions)
            velocities = jnp.where(bottom_mask[:, None], jnp.stack([velocities[:, 0], self.config.bounds_factor * velocities[:, 1], velocities[:, 2]], axis=-1), velocities)
            positions = jnp.where(front_mask[:, None], jnp.stack([positions[:, 0], positions[:, 1], self.config.kernel_radius * jnp.ones_like(positions[:,2])], axis=-1), positions)
            velocities = jnp.where(front_mask[:, None], jnp.stack([velocities[:, 0], velocities[:, 1], self.config.bounds_factor * velocities[:, 2]], axis=-1), velocities)
            positions = jnp.where(back_mask[:, None], jnp.stack([positions[:, 0], positions[:, 1], self.config.domain_size[2] - self.config.kernel_radius * jnp.ones_like(positions[:,2])], axis=-1), positions)
            velocities = jnp.where(back_mask[:, None], jnp.stack([velocities[:, 0], velocities[:, 1], self.config.bounds_factor * velocities[:, 2]], axis=-1), velocities)

        return positions, velocities, pressure_forces, viscosity_forces

    def _spawn_particles(self, spawn_group_index: int):
        """
        Spawns particles for a given spawn group
        spawn_group_index : int Index of the spawn group to spawn particles for
        """
        if self.spawns_realized[spawn_group_index]:
            raise ValueError(f"Spawn group {spawn_group_index} already realized")

        spawn_group = self.config.spawn_groups[spawn_group_index]
        num_particles = spawn_group.num_particles
        spawn_position = jnp.array(spawn_group.spawn_position, dtype=jnp.float32)
        spawn_bounds = jnp.array(spawn_group.spawn_bounds, dtype=jnp.float32)
        spawn_pattern = spawn_group.spawn_pattern

        # Ensure spawn positions are within the domain bounds
        if spawn_position[0] - spawn_bounds[0] / 2 < 0 or spawn_position[0] + spawn_bounds[0] / 2 > self.config.domain_size[0]:
            raise ValueError("Spawn position out of bounds in x direction")
        if spawn_position[1] - spawn_bounds[1] / 2 < 0 or spawn_position[1] + spawn_bounds[1] / 2 > self.config.domain_size[1]:
            raise ValueError("Spawn position out of bounds in y direction")
        if self.dimensions == 3:
            if spawn_position[2] - spawn_bounds[2] / 2 < 0 or spawn_position[2] + spawn_bounds[2] / 2 > self.config.domain_size[2]:
                raise ValueError("Spawn position out of bounds in z direction")

        if spawn_pattern == "uniform_random":
            # Create a random distribution of particles
            positions = np.random.uniform(
                low=spawn_position - spawn_bounds / 2,
                high=spawn_position + spawn_bounds / 2,
                size=(num_particles, self.dimensions),
            )
            positions = jnp.array(positions, dtype=jnp.float32)
        else:
            raise ValueError(f"Unknown spawn pattern: {spawn_pattern}")
        
        velocities = jnp.tile(
            jnp.array(spawn_group.velocity, dtype=jnp.float32)[None, :],
            (num_particles, 1),
        )
        # Append the new particles to the existing particles
        self.positions = jnp.concatenate((self.positions, positions), axis=0)
        self.velocities = jnp.concatenate((self.velocities, velocities), axis=0)

        # Mark the spawn group as realized
        self.spawns_realized[spawn_group_index] = True

    @partial(jit, static_argnums=0)
    def _get_particle_density(self, pts_a, pt_b, h):
        """
        pts_a: [n, d] array of particle positions
        pts_b: [m, d] array of particle positions
        h: float, smoothing length
        Returns: [n, ] array of density values
        """
        pts_b = jnp.expand_dims(pt_b, axis=0)
        kernel_values = kernel_function_poly6(pts_a, pts_b, h)
        return jnp.sum(kernel_values, axis=-1)

    def _get_density_map_jit(self, neighborhood_matrix, positions, resolution):
        x = jnp.linspace(0, self.config.domain_size[0], resolution)
        y = jnp.linspace(0, self.config.domain_size[1], resolution)

        if self.dimensions == 2:
            # Perform a vmap
            # Create a grid of points
            grid_x, grid_y = jnp.meshgrid(x, y)
            grid_points = jnp.array([grid_x.flatten(), grid_y.flatten()]).T
            # Compute density for each grid point
            density_values = jax.vmap(self._get_particle_density, in_axes=(None, 0, None))(positions, grid_points, self.config.kernel_radius)
            # Reshape the density values to match the grid shape
            density_map = density_values.reshape(resolution, resolution)
            return density_map
        elif self.dimensions == 3:
            # Perform a vmap
            # Create a grid of points
            z = jnp.linspace(0, self.config.domain_size[2], resolution)
            grid_x, grid_y, grid_z = jnp.meshgrid(x, y, z)
            grid_points = jnp.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

            # Compute density for each grid point
            density_values = jax.vmap(self._get_particle_density, in_axes=(None, 0, None))(positions, grid_points, self.config.kernel_radius)

            # Reshape the density values to match the grid shape
            density_map = density_values.reshape(resolution, resolution, resolution)
            return density_map
        
    def get_density_map(self, resolution=100):
        """
        Get the density map of the simulation
        resolution: int, resolution of the density map
        returns: [resolution, resolution] array of density values
        """
        # Given the resolution, create a grid of points and compute the neighborhood matrix for each point on the grid
        if self.dimensions == 3:
            neighborhood_matrix = np.ones((resolution * resolution * resolution, MAX_NEIGHBORS), dtype=np.int32) * -1
            grid_points = np.meshgrid(
                np.linspace(0, self.config.domain_size[0], resolution),
                np.linspace(0, self.config.domain_size[1], resolution),
                np.linspace(0, self.config.domain_size[2], resolution),
                indexing='ij'
            )
            grid_points = np.array(grid_points).T.reshape(-1, 3)
            tree = KDTree(np.array(self.positions))
            neighbors = tree.query_ball_point(grid_points, self.config.kernel_radius)
            for i, neighbor in enumerate(neighbors):
                neighborhood_matrix[i, :len(neighbor)] = neighbor[:MAX_NEIGHBORS]
        else:
            neighborhood_matrix = np.ones((resolution * resolution, MAX_NEIGHBORS), dtype=np.int32) * -1
            grid_points = np.meshgrid(
                np.linspace(0, self.config.domain_size[0], resolution),
                np.linspace(0, self.config.domain_size[1], resolution),
                indexing='ij'
            )
            grid_points = np.array(grid_points).T.reshape(-1, 2)
            tree = KDTree(np.array(self.positions))
            neighbors = tree.query_ball_point(grid_points, self.config.kernel_radius)
            for i, neighbor in enumerate(neighbors):
                neighborhood_matrix[i, :len(neighbor)] = neighbor[:MAX_NEIGHBORS]

        return self._get_density_map_jit(neighborhood_matrix, self.positions, resolution)
    
    def get_voxel_representation(self, resolution=100, blur_sigma=0.25, pad=True):
        """
        Get the voxel representation of the simulation
        resolution: int, resolution of the voxel representation
        returns: [resolution, resolution] array of density values
        """
        if self.dimensions == 2:
            cells = np.zeros((resolution, resolution), dtype=np.float32)
            cell_affiliations = np.array(self.positions) * resolution / self.config.domain_size
            cell_affiliations = np.floor(cell_affiliations).astype(int)
            cells[cell_affiliations[:, 0], cell_affiliations[:, 1]] = 1.0
        elif self.dimensions == 3:
            cells = np.zeros((resolution, resolution, resolution), dtype=np.float32)
            cell_affiliations = np.array(self.positions) * resolution / self.config.domain_size
            cell_affiliations = np.floor(cell_affiliations).astype(int)
            cells[cell_affiliations[:, 0], cell_affiliations[:, 1], cell_affiliations[:, 2]] = 1.0

        if blur_sigma > 0.0:
            from scipy.ndimage import filters
            cells = filters.gaussian_filter(cells, sigma=blur_sigma)

        if pad:
            if self.dimensions == 2:
                cells = np.pad(cells, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            elif self.dimensions == 3:
                cells = np.pad(cells, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)

        return cells

    def step(self):
        """
        Perform a simulation step
        """
        # Check if any spawn groups need to be spawned
        for i, spawn_group in enumerate(self.config.spawn_groups):
            if spawn_group.spawn_time <= self.time and not self.spawns_realized[i]:
                self._spawn_particles(i)

        # Perform a simulation step
                # Perform a simulation step
        neighborhood_calulation_start = time.time()
        np_positions = np.array(self.positions)
        neighborhood_matrix = self._get_neighborhood_matrix(np_positions, self.config.kernel_radius) # Uses sklearn so we can't jit this
        neighborhood_calulation_end = time.time()
        step_start = time.time()
        self.positions, self.velocities, pressure_forces, viscosity_forces = self._similation_step(self.positions, self.velocities, neighborhood_matrix)
        step_end = time.time()

        # Profiling
        self.neighborhood_calculation_time += neighborhood_calulation_end - neighborhood_calulation_start
        self.step_time += step_end - step_start

        # Update the time
        self.time += self.config.time_step

        if self.time >= self.config.simulation_time:
            self.is_running = False

        return self.positions, self.velocities, pressure_forces, viscosity_forces
