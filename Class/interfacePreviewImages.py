import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import os


class ImageTab(ttkb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=5, bootstyle=SECONDARY)
        self.pack(fill=BOTH, expand=YES)

        self.file_path = ttkb.StringVar()

        self.left_files = []
        self.right_files = []
        self.current_image_index = 0

        self.create_buttons()
        self.create_canvas()


    def create_canvas(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH, expand=YES)

        separator = ttkb.Separator(container, orient=VERTICAL, bootstyle=PRIMARY)
        separator.pack(side=LEFT, fill=Y)

        self.image_canvas = ttkb.Canvas(container, width=500, height=500, borderwidth=0)
        self.image_canvas.pack(side=LEFT, fill=BOTH, expand=YES)

        self.image_canvas2 = ttkb.Canvas(container, width=500, height=500, borderwidth=0)
        self.image_canvas2.pack(side=LEFT, fill=BOTH, expand=YES)


    def create_buttons(self):
        container = ttkb.Frame(self, padding=10, borderwidth=0)
        container.pack(side=LEFT, fill=BOTH)

        #add_folder_image = ImageTk.PhotoImage(Image.open('assets/icons8_folder_24px.png').resize((24,24), Image.LANCZOS))
        
        open_image_button = ttkb.Button(
            master=container,
            text="Open",
            #image=add_folder_image,
            command=self.view_images,
            width=8,
            #compound=LEFT,
            bootstyle=SUCCESS,
        )
        open_image_button.pack()

        next_button = ttkb.Button(
            master=container,
            text="Next",
            bootstyle=PRIMARY,
            command=self.next_image,
            width=8
        )
        next_button.pack()

        prev_button = ttkb.Button(
            master=container,
            text="Prev",
            bootstyle=PRIMARY,
            command=self.prev_image,
            width=8
        )
        prev_button.pack()


    def view_images(self):
        left_folder = "./CameraLeft/"
        right_folder = "./CameraRight/"

        self.left_files = os.listdir(left_folder)
        self.right_files = os.listdir(right_folder)

        self.image_files = [os.path.join(left_folder, file) for file in self.left_files] + [os.path.join(right_folder, file) for file in self.right_files]
        self.current_image_index = 0

        self.show_current_image()


    def show_current_image(self):
        if 0 <= self.current_image_index < len(self.left_files):
            file_path = os.path.join("./CameraLeft/", self.left_files[self.current_image_index])
            self.file_path.set(file_path)
            image = Image.open(file_path).resize([400, 400])
            image_tk = ImageTk.PhotoImage(image)
            self.image_canvas.create_image(0, 0, anchor=NW, image=image_tk)
            self.image_canvas.image = image_tk
        else:
            self.image_canvas.delete("all")  # Clear the first canvas

        if 0 <= self.current_image_index < len(self.right_files):
            file_path = os.path.join("./CameraRight/", self.right_files[self.current_image_index])
            self.file_path.set(file_path)
            image2 = Image.open(file_path).resize([400, 400])
            image_tk2 = ImageTk.PhotoImage(image2)
            self.image_canvas2.create_image(0, 0, anchor=NW, image=image_tk2)
            self.image_canvas2.image2 = image_tk2
        else:
            self.image_canvas2.delete("all")  # Clear the second canvas

    def next_image(self):
        self.current_image_index += 1
        self.show_current_image()

    def prev_image(self):
        self.current_image_index -= 1
        self.show_current_image()


if __name__ == '__main__':
    app = ttkb.Window(title='My App', themename='cyborg', minsize=[1200, 800],
                      iconphoto='assets\\icons8_spy_80px.png')
    ImageTab(app)
    app.mainloop()