import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
from Class.interfaceVideoTab import VideoTab
from Class.interfacePreviewImages import ImageTab
from Class.interfaceCapsuleTTKB import DepthEstimation
from Class.interfaceFourierTransformAnimation import AnimationEpiCyclesTab
from Class.interfaceSilhouette_Epi import SilhouetteEpicycleAnimation



class AppInterface(ttkb.Frame):
    def __init__(self, master):
        super().__init__(master, padding=5, bootstyle=SECONDARY)
        self.pack(fill=BOTH, expand=YES)

        self.file_path = ttkb.StringVar()

        self.master.wm_protocol("WM_DELETE_WINDOW", self.on_closing)

        self.create_menu()
        self.create_tabs()

    def create_menu(self):
        menubar = ttkb.Menu(master=self.master)

        filemenu = ttkb.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Open')
        filemenu.add_command(label='Save')
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)

        helpmenu = ttkb.Menu(menubar, tearoff=0)
        helpmenu.add_command(label='About')

        menubar.add_cascade(label='File', menu=filemenu)
        menubar.add_cascade(label='Help', menu=helpmenu)

        self.master.config(menu=menubar)

    def create_tabs(self):
        tab_control = ttkb.Notebook(self, bootstyle=PRIMARY)

        video_tab = VideoTab(tab_control)
        tab_control.add(video_tab, text="Video Capture")

        depth_estimation_tab = DepthEstimation(tab_control)
        tab_control.add(depth_estimation_tab, text="Depth Estimation")

        image_tab = ImageTab(tab_control)
        tab_control.add(image_tab, text='Image View')

        epicycles_tab = AnimationEpiCyclesTab(tab_control)
        tab_control.add(epicycles_tab, text='Epicycles and Signal')  

        silhouette_tab = SilhouetteEpicycleAnimation(tab_control)
        tab_control.add(silhouette_tab, text='Silhouette')      

        tab_control.pack(fill=BOTH, expand=YES)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        self.file_path.set(file_path)
        self.image = Image.open(file_path).resize([400, 400])
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.image_canvas.create_image(0, 0, anchor=NW, image=self.image_tk)

    def on_closing(self):
        self.master.quit()


if __name__ == '__main__':
    app = ttkb.Window(title='My App', themename='cyborg', minsize=[1200, 800], iconphoto='Project-Body-Analyzer/assets/icons8_spy_80px.png')
    AppInterface(app)
    app.mainloop()