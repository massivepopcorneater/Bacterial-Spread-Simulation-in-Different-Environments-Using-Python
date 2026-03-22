import numpy as np
import mesa
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import BacteriaAgent, STRAINS


# Helper functions for data collection 

def count_strain(model, name):
    total = 0
    for a in model.agents:
        if isinstance(a, BacteriaAgent) and a.strain == name:
            total += 1
    return total

def avg_growth_rate(model):
    agents = [a for a in model.agents if isinstance(a, BacteriaAgent)]
    if len(agents) == 0:
        return 0.0
    return sum(a.growth_rate for a in agents) / len(agents)

def avg_nutrients(model):
    return float(np.mean(model.nutrients))


# The main simulation model 

class BacteriaModel(mesa.Model):

    def __init__(self, width=60, height=60, initial_bacteria=80,
                 temperature=37.0, ph=7.0, initial_strain="E. coli"):
        super().__init__(seed=None)

        self.grid        = MultiGrid(width, height, torus=False)
        self.temperature = temperature
        self.ph          = ph
        self.step_count  = 0

        # Nutrient grid — starts at 100 everywhere (full petri dish)
        self.nutrients = np.full((width, height), 100.0)

        # Place bacteria at random positions on the grid
        all_cells = [(x, y) for x in range(width) for y in range(height)]
        self.random.shuffle(all_cells)
        for x, y in all_cells[:initial_bacteria]:
            b = BacteriaAgent(self, strain=initial_strain)
            self.grid.place_agent(b, (x, y))

        # Track data over time
        self.datacollector = DataCollector(
            model_reporters={
                "Total Population": lambda m: sum(1 for a in m.agents if isinstance(a, BacteriaAgent)),
                "E. coli Count":    lambda m: count_strain(m, "E. coli"),
                "Avg Growth Rate":  avg_growth_rate,
                "Avg Nutrients":    avg_nutrients,
                "Temperature":      lambda m: m.temperature,
                "pH":               lambda m: m.ph,
            }
        )
        self.datacollector.collect(self)

    def _diffuse(self, grid, diffusion_rate, decay_rate=0.0, replenish_rate=0.0, max_val=None):
        """
        Spreads nutrients across the grid using Fick's second law.
        Each cell moves toward the average of its 4 neighbours.
        """
        padded = np.pad(grid, 1, mode='edge')
        neighbor_avg = (
            padded[:-2, 1:-1] +
            padded[2:,  1:-1] +
            padded[1:-1, :-2] +
            padded[1:-1,  2:]
        ) / 4.0

        result = grid + diffusion_rate * (neighbor_avg - grid) - decay_rate * grid + replenish_rate
        result = np.clip(result, 0, max_val if max_val is not None else np.inf)
        return result

    def step(self):
        # All bacteria take their turn in random order
        self.agents.shuffle_do("step")

        # Nutrient diffusion — nutrients spread from full areas into depleted areas
        self.nutrients = self._diffuse(
            self.nutrients,
            diffusion_rate = 0.08,
            replenish_rate = 0.05,
            max_val        = 100.0,
        )

        self.datacollector.collect(self)
        self.step_count += 1

    def get_population(self):
        return sum(1 for a in self.agents if isinstance(a, BacteriaAgent))

    def get_grid_array(self):
        # Returns a 2D grid showing how many bacteria are in each cell
        arr = np.zeros((self.grid.width, self.grid.height))
        for cell_content, (x, y) in self.grid.coord_iter():
            arr[x][y] = len(cell_content)
        return arr

