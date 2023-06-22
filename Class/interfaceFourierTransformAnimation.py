import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np



class AnimationEpiCyclesTab(ttkb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=5, bootstyle=SECONDARY)
        self.pack(fill=BOTH, expand=YES)

        self.slider_value = 1
        self.time = 0
        self.wave = []
        self.rotation_direction = 1  # Direction de rotation : 1 pour le sens horaire, -1 pour le sens antihoraire

        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.set_xlim(-200, 600)
        self.ax.set_ylim(-200, 200)
        self.ax.set_aspect('equal')

        self.circles = []
        self.line, = self.ax.plot([], [], 'c-')

        self.is_running = False
        self.animation_id = None

        self.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.on_closing)

        self.create_buttons()
        self.create_canvas()

        #self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_canvas(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH, expand=YES)
        container.pack_propagate(False)

        separator = ttkb.Separator(container, orient=VERTICAL, bootstyle=PRIMARY)
        separator.pack(side=LEFT, fill=Y)

        self.anim_canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.anim_canvas.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=YES)
        # Supprimer les dessins des axes
        self.ax.axis('off')

    def create_buttons(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH)

        #add_folder_image = ImageTk.PhotoImage(Image.open('Interface/assets/icons8_folder_24px.png').resize((24,24), Image.LANCZOS))
        
        self.slider_label = ttkb.Label(container, text="Slider Value:")
        self.slider_label.pack(side=TOP)

        self.slider_entry = ttkb.Entry(container)
        self.slider_entry.pack(side=TOP)
        self.slider_entry.insert(ttkb.END, str(self.slider_value))


        start_button = ttkb.Button(
            master=container,
            text="START",
            #image=add_folder_image,
            command=self.start_animation,
            width=8,
            #compound=LEFT,
            bootstyle=SUCCESS,
        )
        start_button.pack()

        stop_button = ttkb.Button(
            master=container,
            text="STOP",
            bootstyle=DANGER,
            command=self.stop_animation,
            #state=DISABLED,
            width=8
        )
        stop_button.pack()

        update_button = ttkb.Button(
            master=container,
            text="Update",
            bootstyle=SECONDARY,
            command=self.update_slider_value,
            width=8
        )
        update_button.pack()

        direction_button = ttkb.Button(
            master=container,
            text="Direction",
            bootstyle=PRIMARY,
            command=self.change_direction,
            width=8
        )
        direction_button.pack()


    def init(self):
        for circle in self.circles:
            circle.remove()
        self.circles.clear()
        self.line.set_data([], [])
        return self.circles, self.line

    def animate(self):
        self.ax.cla()
        self.ax.set_xlim(-200, 600)
        self.ax.set_ylim(-200, 200)

        x = 0
        y = 0

        for i in range(self.slider_value):
            prevx = x
            prevy = y
            n = i * 2 + 1
            radius = 75 * (4 / (n * np.pi))
            x += radius * np.cos(self.rotation_direction * n * self.time)
            y += radius * np.sin(self.rotation_direction * n * self.time)

            circle = plt.Circle((prevx, prevy), radius, color='blue', ec='black', alpha=0.5)
            self.ax.add_patch(circle)
            self.circles.append(circle)

            self.ax.plot([prevx, x], [prevy, y], 'r-')

        self.ax.plot([x, 200], [y, y], 'black')
        self.ax.plot(200, y, 'o', color='orange')

        self.wave.insert(0, y)
        if len(self.wave) > 250:
            self.wave.pop()

        self.ax.plot(range(200, 200 + len(self.wave)), self.wave, 'b-')

        self.ax.axhline(0, color='black')
        self.ax.axvline(0, color='black')

        self.time += 0.09
        self.anim_canvas.draw()

        if self.is_running:
            self.animation_id = self.master.after(50, self.animate)

    def update_slider_value(self):
        try:
            self.slider_value = int(self.slider_entry.get())
        except ValueError:
            Messagebox.show_info("Invalid input", "Please enter an integer value.")
            return

    def start_animation(self):
        if self.is_running:
            Messagebox.show_info("Animation already running", "The animation is already running.")
            return

        self.is_running = True
        #self.start_animation.config(state=DISABLED)
        #self.start_animation.config(state=NORMAL)

        self.animate()

    def stop_animation(self):
        self.is_running = False

        if self.animation_id is not None and isinstance(self.animation_id, int):
            self.master.after_cancel(self.animation_id)
            self.animation_id = None


    def change_direction(self):
        self.rotation_direction *= -1  # Change la direction de rotation


    def on_closing(self):
        self.stop_animation()
        self.master.quit()


if __name__ == '__main__':
    app = ttkb.Window(title='My App', themename='cyborg', minsize=[1200, 800],
                      iconphoto='Interface\\assets\\icons8_spy_80px.png')
    AnimationEpiCyclesTab(app)
    app.mainloop()