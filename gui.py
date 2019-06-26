import time
import tkinter

from PIL import Image, ImageDraw, ImageTk

from keras.models import load_model
import numpy as np
import copy
from utils import *
from tkinter import *
import PIL.Image
from projectParams import *
import asyncio
from PIL import Image, ImageTk
from tkinter import Tk, filedialog
from tkinter.ttk import Frame, Label, Style
import os

global text_file_num, e1, freq, glob_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Globals
model = load_model(modelPath)
model.load_weights(modelWeights)
dataColor = (0, 255, 0)
pred = ''
prevPred = ''
sentence = ""
defualt_freq = 15
count = defualt_freq
threshold = 0.8  # Between 0 and 1


async def predictImg(roi):
    global count, sentence
    global pred, prevPred, textForm

    img = cv2.resize(roi, (imgDim, imgDim))
    img = np.float32(img) / 255.
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    vec = model.predict(img)
    pred = convertEnglishToHebrewLetter(classes[np.argmax(vec[0])])
    maxVal = np.amax(vec)
    if maxVal < threshold or pred == '':
        pred = ''
        count = freq
    elif pred != prevPred:
        prevPred = pred
        count = freq
    else:  # maxVal >= Threshold && pred == prevPred
        count = count - 1
        if count == 0:
            count = freq
            if pred == 'del':
                sentence = sentence[:-1]
            else:
                sentence = sentence + pred
            if pred == ' ':
                pred = 'space'
            print(finalizeHebrewString(sentence))
            textForm.config(state=NORMAL)
            textForm.delete(0, END)
            textForm.insert(0, (finalizeHebrewString(sentence)))
            textForm.config(state=DISABLED)


class App:
    def __init__(self, window, window_title, video_source=0):
        global textForm, text_file_num, freq
        window.geometry("700x620+400+100")  # x:y
        text_file_num = 1
        freq = defualt_freq
        # create function add menu
        self.create_menu(window)
        window.resizable(False, False)
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoFrame(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=800, height=800)
        self.canvas.pack()

        # adding the stuff
        self.txt_label = tkinter.Label(window, text="The translated text :")
        self.txt_label.place(x=50, y=490)

        self.txt_box = tkinter.Entry(window, justify=RIGHT, font="Helvetica 18 bold")
        #   self.Entry1.place(relx=0.283, rely=0.422, height=144, relwidth=0.557)
        self.txt_box.place(x=180, y=490, height=90, width=350)
        self.txt_box.configure(width=334)
        textForm = self.txt_box
        textForm.config(state=DISABLED)

        image = Image.open("Resources\save_icon.png")
        img = ImageTk.PhotoImage(image)
        self.save_but = tkinter.Button(window, text="save text", width=50, height=50, image=img,
                                       command=self.click_on_save)
        self.save_but.place(x=555, y=510)

        del_img = Image.open("Resources\del_img.png")
        del_img = del_img.resize((20, 20), Image.ANTIALIAS)
        img_del = ImageTk.PhotoImage(del_img)
        self.clear_but = tkinter.Button(window, image=img_del, command=self.clear_txt_box)
        self.clear_but.place(x=155, y=556)
        self.clean_label = tkinter.Label(window, text="Clear text")
        self.clean_label.place(x=145, y=580)

        self.save_label = tkinter.Label(window, text="Save As Text")
        self.save_label.place(x=550, y=570)

        # Bind all keyboard pressed to keyPressed function.
        window.bind('<KeyPress>', self.keyPressed)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()
        self.window.mainloop()

    def create_menu(self, window):
        menu = Menu(window)
        window.config(menu=menu)
        filemenu = Menu(menu, tearoff=False)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Sign Language Alphabet", command=self.open_sign_win)
        filemenu.add_command(label="Set Capture Rate", command=self.set_capture_rate)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit_prog)
        helpmenu = Menu(menu, tearoff=False)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="Help guide", command=self.open_user_manual)

    def click_on_save(self):
        global textForm, text_file_num
        f = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        data = textForm.get()
        data.encode(encoding="UTF-8", errors='strict')
        f.write(data)
        f.close()

    def clear_txt_box(self):
        global textForm
        textForm.config(state=NORMAL)
        textForm.delete(0, END)
        textForm.config(state=DISABLED)

    def open_sign_win(self):
        root = tkinter.Toplevel()
        root.resizable(False, False)
        root.title("sign language alphabet")
        image = Image.open("Resources\Legend.jpg")
        image = image.resize((450, 500), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        panel = Label(root, image=img)
        panel.pack(side="bottom", fill="both")
        root.mainloop()

    def open_user_manual(self):
        os.startfile("user_m.pdf")

    def set_capture_rate(self):
        global e1, glob_root
        root = tkinter.Toplevel()
        root.resizable(False, False)
        glob_root = root
        root.geometry("320x100+400+100")  # x:y
        Label(root, text="Enter new capture rate vlaue").grid(row=0)
        root.title("Set rate value")
        e1 = Entry(root)
        e1.grid(row=0, column=1)
        Button(root, text='Set', command=self.check_valid_rate_input).grid(row=5, column=1, sticky=W, pady=4)
        Label(root, text="A value between 5 and 40 is required", foreground="red").grid(row=7)

    def check_valid_rate_input(self):
        global count, freq, glob_root
        input = e1.get()
        if input.isdigit():
            rate_in = int(input)
            if 5 <= rate_in <= 40:
                freq = rate_in
                count = rate_in
                glob_root.destroy()
            else:
                e1.delete(0, END)
                e1.insert(0, "WORNG RANGE")
        else:
            e1.delete(0, END)
            e1.insert(0, "WORNG INPUT!")

    def exit_prog(self):
        self.window.destroy()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)

    def keyPressed(self, event):
        if event.keycode == 27:  # Escape
            self.window.destroy()
        elif event.keycode == 37:  # Left
            self.vid.x0 = max((self.vid.x0 - 5, 0))
        elif event.keycode == 38:  # Up
            self.vid.y0 = max((self.vid.y0 - 5, 0))
        elif event.keycode == 39:  # Right
            self.vid.x0 = min((self.vid.x0 + 5, self.vid.frame.shape[1] - self.vid.predWidth))
        elif event.keycode == 40:  # Down
            self.vid.y0 = min((self.vid.y0 + 5, self.vid.frame.shape[0] - self.vid.predWidth))
        elif event.keycode == 77:  # 'M' - Binary Mask
            self.vid.showMask = not self.vid.showMask
        elif event.keycode == 80:  # 'P' - Prediction on
            self.vid.predict = not self.vid.predict


class VideoFrame:
    def __init__(self, video_source=0):

        # Open the video source
        self.vid = cv2.VideoCapture(video_source)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Capture parameters
        self.showMask, self.predict = 0, 0
        self.fx, self.fy, self.fh = 10, 50, 45
        self.x0, self.y0, self.predWidth = 400, 50, 224

    def get_frame(self):
        global dataColor
        global count, pred

        if self.vid.isOpened():
            ret, self.frame = self.vid.read()
            self.frame = cv2.flip(self.frame, 1)  # mirror
            frame = copy.deepcopy(self.frame)
            cv2.rectangle(frame, (self.x0, self.y0),
                          (self.x0 + self.predWidth - 1, self.y0 + self.predWidth - 1),
                          dataColor, 12)

            # get region of interest
            roi = self.frame[self.y0:self.y0 + self.predWidth, self.x0:self.x0 + self.predWidth]
            roi = binaryMask(roi)

            # apply processed roi in frame
            if self.showMask:
                img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                frame[self.y0:self.y0 + self.predWidth, self.x0:self.x0 + self.predWidth] = img

            # take data or apply predictions on ROI
            if self.predict:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(predictImg(roi))

            if self.predict:
                dataColor = (0, 250, 0)
                cv2.putText(frame, 'Strike ' + 'P' + ' to pause', (self.fx, self.fy - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, dataColor, 2, 1)
            else:
                dataColor = (0, 0, 250)
                cv2.putText(frame, 'Strike ' + 'P' + ' to start', (self.fx, self.fy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, dataColor, 2, 1)

            # Add Letter prediction
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((self.fx, self.fy + self.fh), "Prediction: %s" % pred, font=font, fill=dataColor)
            draw.text((self.fx, self.fy + 380), 'Sample Timer: %d ' % count, font=font, fill=dataColor)
            # noinspection PyAttributeOutsideInit
            self.frame = np.array(img_pil)

            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


# Create a window and pass it to the Application object
App(Tk(), "Israeli Sign Language Letters Recognition")
