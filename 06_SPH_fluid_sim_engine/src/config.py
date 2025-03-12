
from pydantic import BaseModel

DEFAULT_BOUNDS = 60.0

# TODO: At some point at geometry the fluid can collide with

class FluidSpawnGroup(BaseModel):
    """
    Configuration for a spawn group of particles
    """
    num_particles: int = 100
    spawn_time: float = 0.0
    spawn_position: list[float] = [0.0, 0.0]
    spawn_bounds: list[float] = [0.0, 0.0]
    velocity: list[float] = [0.0, 0.0]
    spawn_pattern: str = "uniform_random" # TODO: Add more spawn patterns

class FluidSimConfig2D(BaseModel):
    """
    Configuration for a 2D fluid simulation
    """
    simulation_time: float = 10.0
    time_step: float = 0.01
    kernel_radius: float = 5.0
    domain_size: list[float] = [DEFAULT_BOUNDS, DEFAULT_BOUNDS]
    bounds_factor: float = -0.6
    rest_density: float = 0.05
    gravity: float = 9.81
    viscosity_multiplier: float = 0.1
    pressure_multiplier: float = 1000.0

    # Particle spawn groups
    spawn_groups: list[FluidSpawnGroup] = [
        FluidSpawnGroup(
            num_particles=500,
            spawn_time=0.0,
            spawn_position=[DEFAULT_BOUNDS / 2, DEFAULT_BOUNDS / 2],
            spawn_bounds=[DEFAULT_BOUNDS, DEFAULT_BOUNDS],
            velocity=[0.0, 0.0],
            spawn_pattern="uniform_random"
        )
    ]

class FluidSimConfig3D(BaseModel):
    """
    Configuration for a 2D fluid simulation
    """
    simulation_time: float = 10.0
    time_step: float = 0.01
    kernel_radius: float = 5.0
    domain_size: list[float] = [DEFAULT_BOUNDS, DEFAULT_BOUNDS, DEFAULT_BOUNDS]
    bounds_factor: float = -0.6
    rest_density: float = 0.05
    gravity: float = 1.0
    viscosity_multiplier: float = 0.1
    pressure_multiplier: float = 1000.0

    # Particle spawn groups
    spawn_groups: list[FluidSpawnGroup] = [
        FluidSpawnGroup(
            num_particles=500,
            spawn_time=0.0,
            spawn_position=[DEFAULT_BOUNDS / 2, DEFAULT_BOUNDS / 2, DEFAULT_BOUNDS / 2],
            spawn_bounds=[DEFAULT_BOUNDS, DEFAULT_BOUNDS, DEFAULT_BOUNDS],
            velocity=[0.0, 0.0, 0.0],
            spawn_pattern="uniform_random"
        )
    ]