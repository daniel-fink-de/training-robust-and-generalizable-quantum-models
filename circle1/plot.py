import matplotlib as mpl

# Set LaTex
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 16,               # LaTeX default is 10pt font.
    "font.size": 16,
    "legend.fontsize": 16,               # Make the legend/label fonts
    "xtick.labelsize": 16,               # a little smaller
    "ytick.labelsize": 16,
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{braket}",
        ])
    }
mpl.rcParams.update(pgf_with_latex)

def base():
    """
    This method is just used to load the global latex settings above.
    """
    return