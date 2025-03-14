import sys
sys.path.append("..")

from importlib import reload
import src.engine
reload(src.engine)

from src.config import FluidSimConfig3D, FluidSpawnGroup
from src.engine import FluidSimulationEngine

import time

if __name__ == "__main__":
    BOUNDS = 60.0

    config = FluidSimConfig3D()
    config.simulation_time = 100.0
    config.time_step = 0.1
    config.domain_size = [BOUNDS, BOUNDS, BOUNDS]
    config.viscosity_multiplier = 7.0
    config.kernel_radius = 1.5
    config.rest_density = 0.8
    config.pressure_multiplier = 25
    config.spawn_groups= [
        FluidSpawnGroup(
            num_particles=20_000,
            spawn_time=0.0,
            spawn_position=[BOUNDS / 8, BOUNDS / 2, BOUNDS / 2],
            spawn_bounds=[BOUNDS / 4, BOUNDS, BOUNDS],
            velocity=[0.0, 0.0, 0.0],
            spawn_pattern="uniform_random"
        ),
        FluidSpawnGroup(
            num_particles=20_000,
            spawn_time=25.0,
            spawn_position=[BOUNDS / 8, BOUNDS / 2, BOUNDS / 2],
            spawn_bounds=[BOUNDS / 4, BOUNDS / 2, BOUNDS],
            velocity=[0.0, -5.0, 0.0],
            spawn_pattern="uniform_random"
        )
    ]

    engine = FluidSimulationEngine(config)

    _ = engine.step() # Jit compilation step


    i = 0
    while True:
        if engine.is_running:
            positions, _, _, _ = engine.step()
        else:
            break

        i += 1
        print(f"{i} / {1000} {round(100 * (i/1000),2)}% steps completed", end="\r") 

    print(f"Simulation completed in {engine.neighborhood_calculation_time + engine.step_time:.2f} seconds")
    print(f"Neighborhood calculation time: {engine.neighborhood_calculation_time:.2f} seconds")
    print(f"Step time: {engine.step_time:.2f} seconds")
    print(f"Simulation steps: {i}")
    print(f"Average time per step: {(engine.neighborhood_calculation_time + engine.step_time) / i:.2f} seconds")