from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, Label, Scale, HORIZONTAL
from PIL import Image, ImageTk
import numpy as np
import cv2
from threading import Thread
import time

# Initialize all global variables
scale = 1.0
video_path = None
cap = None
is_playing = False
current_frame = None
predicted_frame = None
frame_label = None
show_none = True
video_thread = None

# Load the YOLO model and get class names
model = YOLO("best_yolo11n_none.pt")
class_names = model.names
none_class_idx = None

for idx, name in class_names.items():
    if name.lower() == "none":
        none_class_idx = idx
        break

def zoom(event):
    global scale, frame_label
    if event.delta > 0:
        scale *= 1.1
    else:
        scale /= 1.1
    if current_frame is not None:
        update_frame(predicted_frame if predicted_frame is not None else current_frame)

def update_frame(frame):
    global frame_label, scale
    if frame is not None:
        height, width = frame.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        if len(resized_frame.shape) == 3:
            image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
            
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        frame_label.config(image=photo)
        frame_label.image = photo

def load_video():
    global video_path, cap, scale
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if video_path:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(video_path)
        scale = 1.0
        read_frame()
        toggle_play()

def toggle_none():
    global show_none
    show_none = not show_none
    if current_frame is not None:
        predict_frame()

def predict_frame():
    global current_frame, predicted_frame
    if current_frame is not None:
        conf_threshold = float(conf_slider.get())
        results = model(current_frame, conf=conf_threshold)
        
        for result in results:
            if not show_none and none_class_idx is not None:
                boxes = result.boxes
                mask = boxes.cls != none_class_idx
                result.boxes = boxes[mask]
            
            predicted_frame = result.plot(conf=False)
            update_frame(predicted_frame)

def read_frame():
    global cap, current_frame, predicted_frame
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            current_frame = frame
            predicted_frame = None
            update_frame(frame)
            return True
    return False

def play_video():
    global is_playing
    while is_playing and cap is not None:
        if read_frame():
            predict_frame()
            time.sleep(1/30)  # Adjust FPS as needed
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video

def toggle_play():
    global is_playing, video_thread
    is_playing = not is_playing
    play_button.config(text="Pause" if is_playing else "Play")
    
    if is_playing:
        video_thread = Thread(target=play_video)
        video_thread.daemon = True
        video_thread.start()

def reset():
    global frame_label, video_path, cap, is_playing, scale
    if cap is not None:
        cap.release()
    cap = None
    frame_label.config(image="")
    frame_label.image = None
    video_path = None
    is_playing = False
    scale = 1.0
    play_button.config(text="Play")

# Create Tkinter interface
root = tk.Tk()
root.title("YOLO Video Model Tester")

# Add confidence threshold slider
conf_label = tk.Label(root, text="Confidence Threshold:")
conf_label.pack()
conf_slider = Scale(root, from_=0.0, to=1.0, resolution=0.05, orient=HORIZONTAL)
conf_slider.set(0.5)
conf_slider.pack()

# Add buttons and frame display
load_button = tk.Button(root, text="Load Video", command=load_video)
load_button.pack()

play_button = tk.Button(root, text="Play", command=toggle_play)
play_button.pack()

reset_button = tk.Button(root, text="Reset", command=reset)
reset_button.pack()

toggle_none_btn = tk.Button(root, text=f"Toggle '{class_names.get(none_class_idx, 'None')}' Class", command=toggle_none)
toggle_none_btn.pack()

frame_label = Label(root)
frame_label.pack()
root.bind("<MouseWheel>", zoom)

root.mainloop()

# Cleanup
if cap is not None:
    cap.release()
