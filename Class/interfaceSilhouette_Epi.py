import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad_vec
from tkinter import ttk
from tkinter import messagebox
import cv2
import mediapipe as mp

RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

class Silhouette():
    def __init__(self, name):
        self.image = cv2.imread(name)

        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

        self.get_silhouette()

    def get_mask(self, image):
        #détection de la silhouette par mediapipe
        res = self.segment.process(image)
        mask = res.segmentation_mask
        mask = mask > 0.8 #seuil à changer éventuellement
        mask = np.uint8(mask)
        return mask
    def getMaxContours(self, cont):
        max = -10000000000
        for c in cont:
            if max < len(c):
                max = len(c)
        n = 0
        for i in range(len(cont)):
            length = len(cont[i])
            if length == max:
                n = i
                break
        return n
    def get_silhouette(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        bin = self.get_mask(img)
        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        num = self.getMaxContours(contours)
        self.contour = contours[num]
        #self.contour = []
        # for c in contours:
        #     if len(c) > 10000:
        #         for p in c:
        #             self.contour.append(p)
        print('len contours = ', len(self.contour))
        cv2.polylines(self.image, self.contour, True, RED, 10)
        cv2.namedWindow("silhouette", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("silhouette", 800, 600)
        cv2.imshow("silhouette", self.image)
        #cv2.imshow('', self.image)
        cv2.waitKey(0)

class SilhouetteEpicycleAnimation(ttkb.Frame):
    #def __init__(self, order=100, frames=300):
    def __init__(self, master, order=100, frames=300):
        super().__init__(master, padding=5, bootstyle=SECONDARY)
        self.pack(fill=BOTH, expand=YES)

        self.order = order
        self.frames = frames
        self.fig = plt.Figure(figsize=(4, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.draw_x, self.draw_y = [], []
        self.circles = []
        self.circle_lines = []
        self.drawing = None
        self.orig_drawing = None
        self.xlim_data = None
        self.ylim_data = None
        self.t_list = None
        self.c = None
        self.progressbar = None
        self.ani = None
        self.is_running = False

        self.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.on_closing)


        self.create_buttons()
        self.create_progressbar()
        self.create_canvas()


    def create_canvas(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH, expand=YES)

        separator = ttkb.Separator(container, orient=VERTICAL, bootstyle=PRIMARY)
        separator.pack(side=LEFT, fill=Y)

        self.canvas_animation = FigureCanvasTkAgg(self.fig, master=container)
        self.canvas_animation.get_tk_widget().pack(fill=BOTH, expand=True)

        self.ax.axis('off')


    def create_progressbar(self):
        container = ttkb.Frame(self, padding=10)
        container.pack(side=BOTTOM, fill=BOTH)

        self.progressbar = ttkb.Progressbar(container, orient="horizontal", bootstyle="success-striped")
        self.progressbar.pack(fill=X, pady=10)

    def create_buttons(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH)

        open_button = ttkb.Button(
            master=container,
            text="OPEN",
            bootstyle=PRIMARY,
            command=self.open_image,
            width=8
        )
        open_button.pack()

        start_button = ttkb.Button(
            master=container,
            text="START",
            bootstyle=SUCCESS,
            command=self.start_animation,
            width=8
        )
        start_button.pack()

        stop_button = ttkb.Button(
            master=container,
            text="STOP",
            bootstyle=DANGER,
            command=self.stop_animation,
            width=8
        )
        stop_button.pack()


    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.generate_coefficients()
            self.animate()

    def generate_coefficients(self):
        # img = cv2.imread(self.image_path)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # #ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        # ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # contours = np.array(contours[1])

        cont = Silhouette(self.image_path)
        contours = cont.contour

        x_list, y_list = contours[:, :, 0].reshape(-1,), -contours[:, :, 1].reshape(-1,)
        x_list = x_list - np.mean(x_list)
        y_list = y_list - np.mean(y_list)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_list, y_list)
        self.xlim_data = plt.xlim()
        self.ylim_data = plt.ylim()
        plt.close(fig)

        t_list = np.linspace(0, tau, len(x_list))
        self.t_list = t_list
        self.x_list = x_list
        self.y_list = y_list

        order = self.order
        c = []
        self.progressbar["maximum"] = (order*2+1)
        for n in range(-order, order+1):
            coef = 1/tau*quad_vec(lambda t: self.f(t, t_list, x_list, y_list)*np.exp(-n*t*1j), 0, tau, limit=100, full_output=1)[0]
            c.append(coef)
            self.progressbar["value"] = n + order + 1
            self.progressbar.update()
        self.progressbar.stop()

        c = np.array(c)
        self.c = c

    def f(self, t, t_list, x_list, y_list):
        return np.interp(t, t_list, x_list + 1j*y_list)

    def sort_coeff(self, coeffs):
        new_coeffs = []
        new_coeffs.append(coeffs[self.order])
        for i in range(1, self.order+1):
            new_coeffs.extend([coeffs[self.order+i], coeffs[self.order-i]])
        return np.array(new_coeffs)
    
    def sort_amp(self, coeffs):
        arr_amp = []
        n = 0
        for (c) in coeffs:
            real = c.real
            img = c.imag
            amp = np.sqrt(real**2+img**2)
            amp = int(amp*1000)
            arr_amp.append([amp, n])
            n+=1

        arr_amp = np.array(arr_amp)
        arr = arr_amp[np.argsort(arr_amp[:, 0])]
        arr = arr[::-1]
        new_coef = []
        for e in arr:
            amp, n = e[0], e[1]
            new_coef.append(coeffs[n])
        return new_coef

    def make_frame(self, i, time, coeffs):
        t = time[i]

        exp_term = np.array([np.exp(n*t*1j) for n in range(-self.order, self.order+1)])
        coeffs = self.sort_amp(coeffs*exp_term)
        x_coeffs = np.real(coeffs)
        y_coeffs = np.imag(coeffs)

        center_x, center_y = 0, 0

        for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
            r = np.linalg.norm([x_coeff, y_coeff])
            theta = np.linspace(0, tau, num=50)
            x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
            self.circles[i].set_data(x, y)
            x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
            self.circle_lines[i].set_data(x, y)
            center_x, center_y = center_x + x_coeff, center_y + y_coeff

        self.draw_x.append(center_x)
        self.draw_y.append(center_y)
        self.drawing.set_data(self.draw_x, self.draw_y)
        self.orig_drawing.set_data(self.x_list, self.y_list)

    def animate(self):
        time = np.linspace(0, tau, num=self.frames)

        self.ax.clear()

        circles = [self.ax.plot([], [], 'r-')[0] for _ in range(-self.order, self.order+1)]
        circle_lines = [self.ax.plot([], [], 'b-')[0] for _ in range(-self.order, self.order+1)]
        drawing, = self.ax.plot([], [], 'k-', linewidth=2)
        orig_drawing, = self.ax.plot([], [], 'g-', linewidth=0.5)

        self.ax.set_xlim(self.xlim_data[0]-200, self.xlim_data[1]+200)
        self.ax.set_ylim(self.ylim_data[0]-200, self.ylim_data[1]+200)
        #self.ax.set_axis_off()
        self.ax.set_aspect('equal')

        self.circles = circles
        self.circle_lines = circle_lines
        self.drawing = drawing
        self.orig_drawing = orig_drawing

        self.ani = FuncAnimation(self.fig, self.make_frame, frames=self.frames, fargs=(time, self.c),
                                interval=20, repeat=False)

        self.canvas_animation.draw()  # Refresh the figure canvas

    def start_animation(self):
        if self.is_running:
            messagebox.showinfo("Animation already running", "The animation is already running.")
            return

        self.is_running = True
        self.start_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)

        self.animate()


    def stop_animation(self):
        if not self.is_running:
            Messagebox.show_info("Animation not running", "The animation is not running.")
            return

        self.is_running = False
        #self.start_button.config(state=NORMAL)
        #self.stop_button.config(state=DISABLED)

        if self.ani is not None:
            self.ani.event_source.stop()
            self.ani = None

    def on_closing(self):
        self.stop_animation()
        self.master.quit()


if __name__ == '__main__':
    app = ttkb.Window(title='My App', themename='cyborg', minsize=[1200, 800],
                      iconphoto='Interface\\assets\\icons8_spy_80px.png')
    SilhouetteEpicycleAnimation(app, order=100, frames=300)
    app.mainloop()