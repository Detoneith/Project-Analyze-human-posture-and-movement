import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# Configuration des ondes sinusoïdales
num_waves = 1
max_waves = 10
sampling_rate = 100
total_time = 10
time = np.linspace(0, total_time, int(sampling_rate * total_time), endpoint=False)
amplitudes = np.ones(max_waves)
frequencies = np.arange(1, max_waves + 1)
phases = np.zeros(max_waves)

# Fonction pour calculer les coordonnées des ondes sinusoïdales en 3D
def calculate_waves():
    waves = np.zeros((num_waves, len(time)))
    for i in range(num_waves):
        waves[i] = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * time + phases[i])
    return waves

# Fonction de mise à jour des ondes
def update_waves(frame):
    global lines, num_waves
    waves = calculate_waves()
    for i in range(num_waves):
        lines[i].set_data(time[:frame], waves[i][:frame])
        lines[i].set_3d_properties(np.ones(frame) * i)
    return lines

# Fonction pour ajouter une courbe
def add_wave():
    global num_waves
    if num_waves < max_waves:
        num_waves += 1
        update_waves(len(time))
        fig.canvas.draw()

# Création de la fenêtre Tkinter
root = tk.Tk()
root.title("Ondes sinusoïdales en 3D")

# Création de la figure Matplotlib en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lines = [ax.plot([], [], [], lw=2)[0] for _ in range(max_waves)]
ax.set_xlim(0, total_time)
ax.set_ylim(-2, 2)
ax.set_zlim(0, max_waves)
ax.set_title("Ondes sinusoïdales en 3D")
ax.set_xlabel("Temps")
ax.set_ylabel("Amplitude")
ax.set_zlabel("Onde")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Bouton pour ajouter une courbe
button = tk.Button(root, text="Ajouter une courbe", command=add_wave)
button.pack()

# Mise à jour initiale des ondes
update_waves(0)

# Fonction d'animation
def animate(frame):
    return update_waves(frame)

# Lancement de l'animation
ani = FuncAnimation(fig, animate, frames=len(time), interval=20, blit=True)

# Boucle principale Tkinter
root.mainloop()

