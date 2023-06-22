import cv2
import numpy as np
import imutils
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from PIL import Image, ImageTk


class DepthEstimation(ttkb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=5, bootstyle=SECONDARY)
        self.pack(fill=BOTH, expand=YES)
        self.cap_right = None
        self.cap_left = None
        self.dim = (640, 480)
        self.canvas_left = None
        self.canvas_right = None

        self.B = 9
        self.f = 6
        self.alpha = 56.6

        self.create_buttons()
        self.create_canvas()

        self.is_running = False

    def detect_circles(self, frame, camera):
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        if not camera:
            mask = cv2.inRange(hsv, np.array([176, 110, 50]), np.array([255, 255, 255]))
        else:
            mask = cv2.inRange(hsv, np.array([128, 110, 50]), np.array([255, 255, 255]))
        mask = cv2.dilate(mask, None, iterations=20)
        return mask

    def draw_circles(self, frame, mask):
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if len(contours) == 0:
            return None
        center = None
        if len(contours) > 0:
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            cv2.polylines(frame, c, True, (0, 255, 0), 4)
            M = cv2.moments(c)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 0), -1)

        h, w = frame.shape[1], frame.shape[0]
        c = [int(h * 0.5), int(w * 0.5)]
        cv2.circle(frame, c, 10, (0, 255, 0), -1)
        cv2.line(frame, c, center, (0, 255, 0), 2)
        return frame, center

    def find_depth(self, circle_right, circle_left, frame_right, frame_left, baseline, f, alpha):
        height_right, width_right, _ = frame_right.shape
        height_left, width_left, _ = frame_left.shape

        f_pixel = 0

        if width_right == width_left:
            f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 100)

        else:
            print('error')

        x_right = circle_right[0]
        x_left = circle_left[0]
        print(x_left, x_right)

        disparity = x_left - x_right
        zDepth = (baseline * f_pixel) / disparity

        return abs(zDepth)

    def update(self):
        if not self.is_running:
            return

        ret_right, frame_right = self.cap_right.read()
        ret_left, frame_left = self.cap_left.read()

        frame_right = cv2.resize(frame_right, self.dim, interpolation=cv2.INTER_AREA)
        frame_left = cv2.resize(frame_left, self.dim, interpolation=cv2.INTER_AREA)

        mask_right = self.detect_circles(frame_right, 1)
        mask_left = self.detect_circles(frame_left, 0)

        resR = self.draw_circles(frame_right, mask_right)
        resL = self.draw_circles(frame_left, mask_left)

        if resR and resL:
            frame_right, center_right = resR
            frame_left, center_left = resL

            if np.all(center_right) == None and np.all(center_left) == None:
                cv2.putText(frame_right, "Tracking lost", (72, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame_left, "Tracking lost", (72, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                depth = self.find_depth(center_right, center_left, frame_right, frame_left, self.B, self.f, self.alpha)
                cv2.putText(frame_right, 'Tracking', (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                cv2.putText(frame_left, 'Tracking', (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                cv2.putText(frame_right, 'Distance: ' + str(round(depth, 3)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (124, 252, 0), 2)
                cv2.putText(frame_left, 'Distance: ' + str(round(depth, 3)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (124, 252, 0), 2)

        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        image_right = ImageTk.PhotoImage(Image.fromarray(frame_right))
        image_left = ImageTk.PhotoImage(Image.fromarray(frame_left))

        self.canvas_right.create_image(0, 0, anchor=NW, image=image_right)
        self.canvas_left.create_image(0, 0, anchor=NW, image=image_left)

        self.canvas_right.image = image_right
        self.canvas_left.image = image_left

        self.after(1, self.update)

    def create_buttons(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH)

        start_button = ttkb.Button(
            master=container,
            text="START",
            bootstyle=SUCCESS,
            command=self.start_cameras,
            width=8
        )
        start_button.pack()

        stop_button = ttkb.Button(
            master=container,
            text="STOP",
            bootstyle=DANGER,
            command=self.stop_cameras,
            width=8
        )
        stop_button.pack()

    def create_canvas(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH, expand=YES)

        separator = ttkb.Separator(container, orient=VERTICAL, bootstyle=PRIMARY)
        separator.pack(side=LEFT, fill=Y)

        self.canvas_left = ttkb.Canvas(container, width=self.dim[0], height=self.dim[1], borderwidth=0)
        self.canvas_left.pack(side=LEFT, fill=BOTH, expand=YES)

        self.canvas_right = ttkb.Canvas(self, width=self.dim[0], height=self.dim[1], borderwidth=0)
        self.canvas_right.pack(side=LEFT, fill=BOTH, expand=YES)

    def start_cameras(self):
        self.cap_right = cv2.VideoCapture(0)
        self.cap_left = cv2.VideoCapture(1)
        self.is_running = True
        self.update()

    def stop_cameras(self):
        self.is_running = False
        self.cap_right.release()
        self.cap_left.release()
        self.canvas_right.delete("all")
        self.canvas_left.delete("all")


if __name__ == '__main__':
    app = ttkb.Window(title='My App', themename='cyborg', minsize=[1200, 800], iconphoto='Interface\\assets\\icons8_spy_80px.png')
    DepthEstimation(app)
    app.mainloop()


'''
import cv2
import numpy as np
import imutils
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from PIL import Image, ImageTk

class DepthEstimation:
    def __init__(self, root):
        self.root = root
        self.cap_right = None
        self.cap_left = None
        self.dim = (640, 480)
        self.canvas_left = None
        self.canvas_right = None

        self.start_button = None
        self.stop_button = None

        self.create_buttons()
        self.create_canvas()

        self.is_running = False

    def detect_circles(self, frame, camera):
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        if not camera:
            mask = cv2.inRange(hsv, np.array([176, 110, 50]), np.array([255, 255, 255]))
        else:
            mask = cv2.inRange(hsv, np.array([128, 110, 50]), np.array([255, 255, 255]))
        mask = cv2.dilate(mask, None, iterations=20)
        return mask

    def draw_circles(self, frame, mask):
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if len(contours) == 0:
            return None
        center = None
        if len(contours) > 0:
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            cv2.polylines(frame, c, True, (0, 255, 0), 4)
            M = cv2.moments(c)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 0), -1)

        h, w = frame.shape[1], frame.shape[0]
        c = [int(w * 0.5), int(w * 0.5)]
        cv2.circle(frame, c, 10, (0, 255, 0), -1)
        cv2.line(frame, c, center, (0, 255, 0), 2)
        return frame, center

    def update(self):
        if not self.is_running:
            return

        ret_right, frame_right = self.cap_right.read()
        ret_left, frame_left = self.cap_left.read()

        frame_right = cv2.resize(frame_right, self.dim, interpolation=cv2.INTER_AREA)
        frame_left = cv2.resize(frame_left, self.dim, interpolation=cv2.INTER_AREA)

        mask_right = self.detect_circles(frame_right, 1)
        mask_left = self.detect_circles(frame_left, 0)

        resR = self.draw_circles(frame_right, mask_right)
        resL = self.draw_circles(frame_left, mask_left)

        if resR and resL:
            frame_right, _ = resR
            frame_left, _ = resL

        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        image_right = ImageTk.PhotoImage(Image.fromarray(frame_right))
        image_left = ImageTk.PhotoImage(Image.fromarray(frame_left))

        self.canvas_right.create_image(0, 0, anchor=ttkb.NW, image=image_right)
        self.canvas_left.create_image(0, 0, anchor=ttkb.NW, image=image_left)

        self.canvas_right.image = image_right
        self.canvas_left.image = image_left

        self.root.after(1, self.update)

    def create_buttons(self):
        self.start_button = ttkb.Button(self.root, text="Start", command=self.start_cameras)
        self.start_button.pack(side=ttkb.LEFT, padx=10, pady=10)

        self.stop_button = ttkb.Button(self.root, text="Stop", command=self.stop_cameras)
        self.stop_button.pack(side=ttkb.LEFT, padx=10, pady=10)
        self.stop_button.configure(state=ttkb.DISABLED)


    def create_canvas(self):
        self.canvas_left = ttkb.Canvas(self.root, width=self.dim[0], height=self.dim[1])
        self.canvas_left.pack(side=ttkb.LEFT, padx=10, pady=10)

        self.canvas_right = ttkb.Canvas(self.root, width=self.dim[0], height=self.dim[1])
        self.canvas_right.pack(side=ttkb.LEFT, padx=10, pady=10)

    def start_cameras(self):
        self.cap_right = cv2.VideoCapture(0)
        self.cap_left = cv2.VideoCapture(1)
        self.is_running = True
        self.start_button.configure(state=ttkb.DISABLED)
        self.stop_button.configure(state=ttkb.NORMAL)
        self.update()

    def stop_cameras(self):
        self.is_running = False
        self.cap_right.release()
        self.cap_left.release()
        self.start_button.configure(state=ttkb.NORMAL)
        self.stop_button.configure(state=ttkb.DISABLED)
        self.canvas_right.delete("all")
        self.canvas_left.delete("all")

    def close(self):
        self.stop_cameras()
        self.root.destroy()


if __name__ == '__main__':
    app = ttkb.Window(title='My App', themename='cyborg', minsize=[1200, 800],
                      iconphoto='Interface\\assets\\icons8_spy_80px.png')
    DepthEstimation(app)
    app.mainloop()
'''