import os
import re
import difflib

import cantera as ct

# Optional imports if needed externally:
__all__ = [
    'MyClass', 'input_params', 'initialize_parameters',
    'plot_single', 'plot_multiple', 'plot_surface', 'generate_animation'

]


########################################################################################################################
# Initial Thermodynamic Condition Class
########################################################################################################################

class MyClass:
    def __init__(self):
        self.T = None
        self.P = None
        self.Phi = None
        self.Fuel = None
        self.mech = None
        self.species = None
        self.oxygenAmount = None
        self.nitrogenAmount = None
        self.X = {}

    def update_composition(self):
        if self.Fuel == "H2":
            self.oxygenAmount = 0.5
        elif self.Fuel == "C2H6":
            self.oxygenAmount = 3.5
        elif self.Fuel == "C4H10":
            self.oxygenAmount = 6.5
        else:
            raise ValueError(f"Unknown fuel type: {self.Fuel}")

        self.X = {
            self.Fuel: self.Phi,
            'O2': self.oxygenAmount,
            'N2': self.nitrogenAmount
        }

    def load_mechanism_species(self):
        if not self.mech:
            raise ValueError("Mechanism file not specified.")
        try:
            gas = ct.Solution(self.mech)
            self.species = gas.species_names
            del gas
        except Exception as e:
            raise RuntimeError(f"Failed to load mechanism file: {e}")


input_params = MyClass()


def initialize_parameters(T, P, Phi, Fuel, mech, nitrogenAmount=0):
    input_params.T = T
    input_params.P = P
    input_params.Phi = Phi
    input_params.Fuel = Fuel
    input_params.mech = mech
    input_params.nitrogenAmount = nitrogenAmount
    input_params.update_composition()
    input_params.load_mechanism_species()


########################################################################################################################
# Figure and Animation Utilities
########################################################################################################################

def plot_single(x, y, label=None, title=None, xlabel=None, ylabel=None, legend=True, output=None):
    """
    Plots a single line graph with optional labels and title.

    Parameters:
    - x: x-axis data.
    - y: y-axis data.
    - label: label for the line.
    - title: title of the plot.
    - xlabel: label for the x-axis.
    - ylabel: label for the y-axis.
    - legend: whether to show the legend.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y, label=label)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if legend and label:
        plt.legend()

    plt.grid(True)
    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()

def plot_multiple(x, y_list, labels=None, title=None, xlabel=None, ylabel=None, legend=True, colors=None,
                  linetypes=None, output=None):
    """
    Plots multiple lines on the same graph.

    Parameters:
    - x: x-axis data.
    - y_list: list of y-axis data for each line.
    - labels: list of labels for each line.
    - title: title of the plot.
    - xlabel: label for the x-axis.
    - ylabel: label for the y-axis.
    - legend: whether to show the legend.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt

    plt.figure()

    for i, y in enumerate(y_list):
        label = labels[i] if labels else None
        plt.plot(x, y, label=label,
                 color=colors[i] if colors else None,
                 linestyle=linetypes[i] if linetypes else '-')

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if legend and labels:
        plt.legend()

    plt.grid(True)
    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()

def plot_surface(x, y, z, title=None, xlabel=None, ylabel=None, zlabel=None, xlimits=None, ylimits=None, zlimits=None,
                 output=None):
    """
    Plots a 3D surface plot.

    Parameters:
    - x: x-axis data.
    - y: y-axis data.
    - z: z-axis data (2D array).
    - title: title of the plot.
    - xlabel: label for the x-axis.
    - ylabel: label for the y-axis.
    - zlabel: label for the z-axis.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, cmap='viridis')

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)

    plt.grid(True)
    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()

def generate_animation(source_dir, output_dir):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.animation as animation
    """
    Generates an animation from a series of PNG images in a specified directory.
    
    Parameters:
    - source_dir: Directory containing the PNG images.
    - output_dir: Path to save the generated animation (e.g., 'output.gif').
    """
    # Extract numeric parts of filenames for sorting
    def extract_frame_number(filename):
        match = re.search(r'plt(\d+)', filename)
        return int(match.group(1)) if match else -1

    # Filter and sort files by frame number
    image_files = [
        os.path.join(source_dir, f)
        for f in sorted(os.listdir(source_dir), key=extract_frame_number)
        if f.endswith('.png')  # Only include PNG files
    ]

    if not image_files:
        print("No image files found in the folder.")
        return

    # Load the first image to determine figure size
    first_image = mpimg.imread(image_files[0])
    fig = plt.figure(figsize=(first_image.shape[1] / 100, first_image.shape[0] / 100), dpi=100)
    plt.axis('off')
    img_display = plt.imshow(first_image)

    # Animation update function
    def update(frame):
        img_display.set_array(mpimg.imread(image_files[frame]))
        return img_display,

    # Create and save the animation
    ani = animation.FuncAnimation(fig, update, frames=len(image_files), blit=True)
    writer = animation.PillowWriter(fps=5)
    # writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    try:
        ani.save(output_dir, writer=writer)
    except Exception as e:
        print(f"Animation save failed: {e}")
    plt.close(fig)