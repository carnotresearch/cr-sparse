from matplotlib import pyplot as plt


def one_plot(height=4):
    fig, ax = plt.subplots(figsize=(16, height))
    return ax

def h_plots(n):
    fig, ax = plt.subplots(n, 1, figsize=(16, 4.2*n))
    return ax
