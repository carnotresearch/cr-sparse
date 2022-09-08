from matplotlib import pyplot as plt


def one_plot(height=4):
    fig, ax = plt.subplots(figsize=(16, height))
    return ax

def h_plots(n, height=4):
    fig, ax = plt.subplots(n, 1, figsize=(16, height*n), 
        constrained_layout = True)
    return ax
