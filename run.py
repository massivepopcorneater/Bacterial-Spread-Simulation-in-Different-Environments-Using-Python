import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import numpy as np
from model import BacteriaModel

# Settings
GRID_WIDTH   = 60
GRID_HEIGHT  = 60
NUM_BACTERIA = 80

# Global variables
model  = None
paused = False

steps_history = []
pop_history   = []

# Create a new model
def create_model():
    return BacteriaModel(
        width            = GRID_WIDTH,
        height           = GRID_HEIGHT,
        initial_bacteria = NUM_BACTERIA,
        temperature      = temp_slider.val,
        ph               = ph_slider.val,
        initial_strain   = "E. coli",
    )

# Set up the figure
fig = plt.figure(figsize=(14, 8))
fig.canvas.manager.set_window_title("Bacteria Simulation")
fig.patch.set_facecolor("#1a1a2e")

# Left panel - petri dish
ax_dish = fig.add_axes([0.03, 0.28, 0.42, 0.65])
ax_dish.set_facecolor("#050510")
ax_dish.set_title("Petri Dish  (green = bacteria · red dots = nutrient depletion)",
                   color="white", fontsize=9)
ax_dish.set_xticks([])
ax_dish.set_yticks([])

# Right panel - population graph
ax_graph = fig.add_axes([0.55, 0.28, 0.42, 0.65])
ax_graph.set_facecolor("#0a0a1a")
ax_graph.set_title("Population Over Time", color="white", fontsize=12)
ax_graph.tick_params(colors="white")
ax_graph.spines[:].set_color("#444")

# Status text below the dish
ax_status = fig.add_axes([0.03, 0.18, 0.42, 0.08])
ax_status.axis("off")
status_text = ax_status.text(0, 0.5, "", color="white", fontsize=9,
                              verticalalignment="center", family="monospace")

# Sliders
ax_temp = fig.add_axes([0.10, 0.13, 0.35, 0.025])
temp_slider = Slider(ax_temp, "Temp (C)", 0, 80, valinit=37.0, color="#2d4a7a")
temp_slider.label.set_color("white")
temp_slider.valtext.set_color("white")

ax_ph = fig.add_axes([0.10, 0.07, 0.35, 0.025])
ph_slider = Slider(ax_ph, "pH", 0, 14, valinit=7.0, color="#2d4a7a")
ph_slider.label.set_color("white")
ph_slider.valtext.set_color("white")

# Buttons
ax_pause = fig.add_axes([0.60, 0.10, 0.12, 0.05])
ax_reset = fig.add_axes([0.75, 0.10, 0.12, 0.05])
pause_btn = Button(ax_pause, "Pause", color="#2a3a5a", hovercolor="#3a4a6a")
reset_btn = Button(ax_reset, "Reset", color="#5a2a2a", hovercolor="#7a3a3a")
pause_btn.label.set_color("white")
reset_btn.label.set_color("white")

# Create model
model = create_model()

# Bacteria layer (green)
bacteria_display = ax_dish.imshow(
    model.get_grid_array().T,
    origin="lower", cmap="Greens",
    vmin=0, vmax=5, interpolation="nearest"
)

# Nutrient depletion dots — one dot per cell, size shows how depleted it is
dot_x = [x for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)]
dot_y = [y for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)]
nutrient_dots = ax_dish.scatter(dot_x, dot_y, s=0, color="red", alpha=0.6, zorder=2, marker=".")

# Label that follows the colony
colony_label = ax_dish.text(
    0, 0, "E. coli",
    color="white", fontsize=9, fontweight="bold",
    ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="#00000088", edgecolor="none"),
    visible=False
)

# Population line on the graph
pop_line, = ax_graph.plot([], [], color="#50fa7b", linewidth=2, label="E. coli")
ax_graph.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)


# Slider callbacks
def slider_changed(val):
    model.temperature = temp_slider.val
    model.ph = ph_slider.val

temp_slider.on_changed(slider_changed)
ph_slider.on_changed(slider_changed)


# Button callbacks
def pause_clicked(event):
    global paused
    paused = not paused
    pause_btn.label.set_text("Resume" if paused else "Pause")
    fig.canvas.draw_idle()

def reset_clicked(event):
    global model
    model = create_model()
    steps_history.clear()
    pop_history.clear()
    colony_label.set_visible(False)

pause_btn.on_clicked(pause_clicked)
reset_btn.on_clicked(reset_clicked)


# Animation function (runs every frame)
def update(frame):
    if paused:
        return bacteria_display, nutrient_dots, pop_line, colony_label

    # Run one simulation step
    model.step()

    # Get all bacteria
    all_bacteria = list(model.agents)
    population   = len(all_bacteria)

    # Update bacteria layer
    grid = model.get_grid_array()
    bacteria_display.set_data(grid.T)
    bacteria_display.set_clim(0, max(grid.max(), 1))

    # Update nutrient dots — bigger dot = more depleted
    depletion = 100 - model.nutrients
    sizes = [model.nutrients[x, y] * 0.5 if model.nutrients[x, y] > 5 else 0 for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)]
    nutrient_dots.set_sizes(sizes)

    # Move E. coli label to centre of colony
    if population > 0:
        x_positions = [a.pos[0] for a in all_bacteria if a.pos]
        y_positions = [a.pos[1] for a in all_bacteria if a.pos]
        colony_label.set_position((sum(x_positions) / len(x_positions),
                                   sum(y_positions) / len(y_positions)))
        colony_label.set_visible(True)
    else:
        colony_label.set_visible(False)

    # Add current data to history
    steps_history.append(model.step_count)
    pop_history.append(population)

    # Update the population graph
    pop_line.set_data(steps_history, pop_history)

    if steps_history:
        ax_graph.set_xlim(steps_history[0], steps_history[-1] + 1)
        ax_graph.set_ylim(0, max(pop_history) * 1.10 + 10)

    # Update the status bar
    if population > 0:
        avg_growth = sum(a.growth_rate for a in all_bacteria) / population
    else:
        avg_growth = 0.0

    avg_nutrients = float(model.nutrients.mean())
    warning = "  WARNING: LETHAL TEMP" if model.temperature >= 55 else ""

    status_text.set_text(
        f"Step: {model.step_count}  |  Population: {population}  |  "
        f"Avg Growth Rate: {avg_growth:.2f}{warning}\n"
        f"Temp: {model.temperature:.1f}C   pH: {model.ph:.2f}   "
        f"Avg Nutrients: {avg_nutrients:.1f}"
    )

    return bacteria_display, nutrient_dots, pop_line, colony_label


# Start the animation
anim = animation.FuncAnimation(fig, update, interval=250,
                                blit=False, cache_frame_data=False)

plt.suptitle("Bacteria Spread Simulation - E. coli",
             color="white", fontsize=14, fontweight="bold", y=0.97)
plt.show()
