import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from PIL import Image, ImageTk
import cv2
import os
import mediapipe as mp


class VideoTab(ttkb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=5, bootstyle=SECONDARY)
        self.pack(fill=BOTH, expand=YES)

        self.file_path = ttkb.StringVar()

        self.create_buttons()
        self.create_canvas()

        self.video_capture1 = None
        self.video_capture2 = None
        self.is_capturing1 = False
        self.is_capturing2 = False

        #mpPose = mp.solutions.pose
        #self.mp_pose = mp.solutions.pose.Pose()
        self.mp_pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,  # Seuil de confiance minimum pour détecter une pose
        min_tracking_confidence=0.5    # Seuil de confiance minimum pour suivre une pose détectée
        )


        #self.pose = mpPose.Pose()

        self.num = 0
        self.num2 = 0
        #self.dim = (640, 480)
        self.dim = (640, 480)

    def create_canvas(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH, expand=YES)

        separator = ttkb.Separator(container, orient=VERTICAL, bootstyle=PRIMARY)
        separator.pack(side=LEFT, fill=Y)

        self.image_canvas1 = ttkb.Canvas(container, width=500, height=500, borderwidth=0)
        self.image_canvas1.pack(side=LEFT, fill=BOTH, expand=YES, padx=20)

        self.image_canvas2 = ttkb.Canvas(container, width=800, height=800, borderwidth=0)
        self.image_canvas2.pack(side=LEFT, fill=BOTH, expand=YES)

    def create_buttons(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH)

        start_camera_button = ttkb.Button(
            master=container,
            text="START",
            bootstyle=SUCCESS,
            command=self.start_video,
            width=8
        )
        start_camera_button.pack()

        stop_camera_button = ttkb.Button(
            master=container,
            text="STOP",
            bootstyle=DANGER,
            command=self.stop_video,
            width=8
        )
        stop_camera_button.pack()

        prise_images_button = ttkb.Button(
            master=container,
            text="Capture",
            bootstyle=PRIMARY,
            command=self.prise_images,
            width=8
        )
        prise_images_button.pack()

    def start_video(self):
        if not self.is_capturing1:
            self.is_capturing1 = True
            self.video_capture1 = cv2.VideoCapture(0)
            self.capture_frames(1)

        if not self.is_capturing2:
            self.is_capturing2 = True
            self.video_capture2 = cv2.VideoCapture(1)
            self.capture_frames(2)

    def stop_video(self):
        if self.is_capturing1:
            self.is_capturing1 = False
            self.video_capture1.release()
            self.image_canvas1.delete("all")

        if self.is_capturing2:
            self.is_capturing2 = False
            self.video_capture2.release()
            self.image_canvas2.delete("all")


    def capture_frames(self, camera_num):
        if camera_num == 1 and self.is_capturing1:
            ret, frame = self.video_capture1.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.mp_pose.process(frame_rgb)
                if res.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame_rgb, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                    for id, lm in enumerate(res.pose_landmarks.landmark):
                        h, w, c = frame_rgb.shape
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame_rgb, (x, y), 5, (0, 255, 0), cv2.FILLED)

                if frame_rgb.size > 0:
                    frame_rgb = cv2.resize(frame_rgb, self.dim, interpolation=cv2.INTER_AREA)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tk = ImageTk.PhotoImage(frame_pil)

                    self.image_canvas1.create_image(0, 0, anchor=NW, image=frame_tk)
                    self.image_canvas1.image = frame_tk

                    self.image_canvas1.update_idletasks()

            self.after(1, self.capture_frames, 1)

        if camera_num == 2 and self.is_capturing2:
            ret, frame = self.video_capture2.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.mp_pose.process(frame_rgb)
                if res.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame_rgb, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                    for id, lm in enumerate(res.pose_landmarks.landmark):
                        h, w, c = frame_rgb.shape
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame_rgb, (x, y), 5, (0, 255, 0), cv2.FILLED)

                if frame_rgb.size > 0:
                    frame_rgb = cv2.resize(frame_rgb, self.dim, interpolation=cv2.INTER_AREA)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tk = ImageTk.PhotoImage(frame_pil)

                    self.image_canvas2.create_image(0, 0, anchor=NW, image=frame_tk)
                    self.image_canvas2.image = frame_tk

                    self.image_canvas2.update_idletasks()

            self.after(1, self.capture_frames, 2)


    def prise_images(self):
        if self.is_capturing1:
            ret1, frame1 = self.video_capture1.read()
            if ret1:
                frame1 = cv2.resize(frame1, self.dim, interpolation=cv2.INTER_AREA)
                name = './CameraLeft/Left' + str(self.num) + '.jpg'
                #cv2.imwrite("photo_camera1.jpg", frame1)
                cv2.imwrite(name, frame1)
            self.num += 1

        if self.is_capturing2:
            ret2, frame2 = self.video_capture2.read()
            if ret2:
                frame2 = cv2.resize(frame2, self.dim, interpolation=cv2.INTER_AREA)
                name2 = './CameraRight/Right' + str(self.num2) + '.jpg'
                cv2.imwrite(name2, frame2)
            self.num2 += 1

    def create_folders(self):
        if not os.path.exists('./CameraLeft'):
            os.makedirs('./CameraLeft')
        if not os.path.exists('./CameraRight'):
            os.makedirs('./CameraRight')
        
        Messagebox.show_info('Folders created')

    def clear_images(self):
        paths = ['./CameraLeft/', './CameraRight/']
        for path in paths:
            files = os.listdir(path)
            for file_name in files:
                file_path = os.path.join(path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        Messagebox.show_info('All images cleared.')


if __name__ == '__main__':
    app = ttkb.Window(title='My App', themename='cyborg', minsize=[1200, 800], iconphoto='Interface\\assets\\icons8_spy_80px.png')
    VideoTab(app)
    app.mainloop()