"""
agent.py
--------
Bacteria agent for Mesa

The purpose of this file is to initialize the individual bacteria unit.
"""

import math
import mesa

"""
Effect of Temperature: Ratkowsky model 

The model:
sqrt(μ) = b × (T - T_min) × [1 - exp(c × (T - T_max))]

Squaring gives μ. This is then normalised to 0–1 by dividing by μ at T_opt.
"""
RATKOWSKY_b    = 0.0576
RATKOWSKY_c    = 0.0048


_NORM_T_OPT   = 37.0
_NORM_T_MIN   = 3.96
_NORM_T_MAX   = 49.0
_sq           = RATKOWSKY_b * (_NORM_T_OPT - _NORM_T_MIN) * (
                    1 - math.exp(RATKOWSKY_c * (_NORM_T_OPT - _NORM_T_MAX)))
RATKOWSKY_NORM = max(_sq, 0.0) ** 2   # μ_max for normalisation


def ratkowsky_temp_factor(T, T_min, T_max):
    """
    Effect of temperature based on Ratkowski's model

    Returns a normalised factor between 0 and 1.
      - 0 at or below T_min (no growth — membrane gels)
      - 1 at T_opt (~37 °C for E. coli)
      - 0 at or above T_max (upper cardinal temperature)
    """
    if T <= T_min or T >= T_max:
        return 0.0
    sq = RATKOWSKY_b * (T - T_min) * (1 - math.exp(RATKOWSKY_c * (T - T_max)))
    mu = max(sq, 0.0) ** 2
    return min(mu / RATKOWSKY_NORM, 1.0)


def gaussian_ph_factor(ph, ph_optimum, ph_width):
    """
    Effect of pH according to Zwietering et al.
    
    Symmetric Gaussian pH response. Returns 0–1.
    """
    return math.exp(-((ph - ph_optimum) ** 2) / (2 * ph_width ** 2))


# Different types of bacterias
STRAINS = { 
    "E. coli": {
        # Ratkowsky temperature parameters 
        "T_min":            7.0,    # °C  minimum cardinal temperature
        "T_max":           49.0,    # °C  maximum cardinal temperature
        # pH parameters 
        "ph_optimum":       7.0,
        "ph_width":         1.5,
        # Lifespan 
        "max_age":        200,
        # Lag phase 
        "q0_mean":          0.05,
        "q0_spread":        0.03,
        # Cell division
        "division_threshold": 2.0,
        "biomass_gain_rate":  0.12,
        # Thermal death
        "lethal_temp":     55.0,
    },
}

# Monod half-saturation constant
MONOD_Ks = 5.0

# Bacteria Unit
class BacteriaAgent(mesa.Agent):

    def __init__(self, model, strain="E. coli"):
        super().__init__(model)

        self.strain = strain
        base = STRAINS[strain]

        def vary(value, spread):
            return value + self.random.gauss(0, spread)

        # Each bacterium draws its own traits from the strain baseline.
        self.T_min      = vary(base["T_min"],      1.0)
        self.T_max      = vary(base["T_max"],      1.0)
        self.ph_optimum = vary(base["ph_optimum"], 0.3)
        self.ph_width   = max(0.5, vary(base["ph_width"], 0.2))
        self.max_age    = max(50,  int(vary(base["max_age"], 20.0)))

        # Ensure T_min < T_max 
        if self.T_min >= self.T_max:
            self.T_min = self.T_max - 5.0

        self.lethal_temp   = base["lethal_temp"]
        self.lethal_damage = 0

        # Lag phase implementation
        self.q = max(0.001, vary(base["q0_mean"], base["q0_spread"]))

        # Binary fission
        # Start at a random point in the cell cycle
        self.biomass = self.random.uniform(0.5, 1.5)

        self.division_threshold = max(1.5, vary(base["division_threshold"], 0.1))
        self.biomass_gain_rate  = max(0.02, vary(base["biomass_gain_rate"], 0.02))

        self.energy      = self.random.uniform(80, 120)
        self.age         = 0
        self.growth_rate = 0.0
        self.alpha       = 0.0

    # Environmental response 

    def compute_growth_rate(self):
        """
        Multiplicative gamma model: growth_rate = temp_factor x pH_factor x nutrient_factor
        """
        temp = self.model.temperature
        ph   = self.model.ph
        x, y = self.pos

        temp_factor = ratkowsky_temp_factor(temp, self.T_min, self.T_max)
        ph_factor   = gaussian_ph_factor(ph, self.ph_optimum, self.ph_width)

        S               = self.model.nutrients[x, y]
        nutrient_factor = S / (MONOD_Ks + S)

        return temp_factor * ph_factor * nutrient_factor

    # Lag phase update 
    def update_lag(self):
        """q accumulates, alpha = q/(1+q) gates division."""
        self.q    += self.growth_rate * self.q
        self.q     = min(self.q, 1000.0)
        self.alpha = self.q / (1.0 + self.q)

    # Nutrient consumption 
    def consume_nutrients(self):
        x, y = self.pos
        consumption = self.growth_rate * self.alpha * 1.5
        self.model.nutrients[x, y] = max(0.0, self.model.nutrients[x, y] - consumption)

    # Binary fission 
    def try_divide(self):
        """
        Biomass accumulates gated by growth_rate x alpha.
        Division fires when biomass >= division_threshold.
        Parent is removed; two daughter cells are created.
        """
        self.biomass += (self.growth_rate * self.alpha * self.biomass_gain_rate)

        if self.biomass < self.division_threshold:
            return False

        # Define neighbour cell and empty cell
        neighbours  = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        empty_cells = [c for c in neighbours if self.model.grid.is_cell_empty(c)]

        if not empty_cells:
            self.biomass = self.division_threshold * 0.99
            return False

        # Biomass division
        parent_pos = self.pos
        daughter_a = BacteriaAgent(self.model, strain=self.strain)
        daughter_a.biomass = self.biomass / 2
        daughter_b = BacteriaAgent(self.model, strain=self.strain)
        daughter_b.biomass = self.biomass / 2

        self.model.grid.remove_agent(self)
        self.remove()

        # Cell A takes parent's place, cell B takes a random empty neighbouring cell
        self.model.grid.place_agent(daughter_a, parent_pos)
        self.model.grid.place_agent(daughter_b, self.random.choice(empty_cells))

        return True

    # Movement 
    def move(self):
        neighbours = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True)
        self.model.grid.move_agent(self, self.random.choice(neighbours))

    # Main step 
    def step(self):
        self.age += 1
        self.growth_rate = self.compute_growth_rate()

        # Protein denaturation above lethal threshold
        if self.model.temperature >= self.lethal_temp:
            self.lethal_damage += 1
            if self.lethal_damage >= 3:
                self.model.grid.remove_agent(self)
                self.remove()
                return
        else:
            self.lethal_damage = max(0, self.lethal_damage - 1)

        self.update_lag()
        self.consume_nutrients()

        # Energy 
        base_change = (self.growth_rate - 0.15) * 20
        self.energy += base_change
        self.energy  = max(0, min(200, self.energy))

        # Cell death
        if self.age > self.max_age or self.energy <= 0 or self.growth_rate < 0.001:
            self.model.grid.remove_agent(self)
            self.remove()
            return

        if self.try_divide():
            return

        self.move()
