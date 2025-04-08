from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, Label, Scale, HORIZONTAL
from PIL import Image, ImageTk
import numpy as np

# Initialize all global variables
scale = 1.0
img_path = None
img = None
predicted_image = None
img_label = None
show_none = True

# Load the YOLO model and get class names
model = YOLO("best_yolo11n_none.pt")
class_names = model.names  # Get dictionary of class names
none_class_idx = None

# Find the index of "none" class
for idx, name in class_names.items():
    if name.lower() == "none":
        none_class_idx = idx
        break

def zoom(event):
    global scale, img_label, img_path
    if event.delta > 0:
        scale *= 1.1
    else:
        scale /= 1.1
    if img_path:
        # Use predicted image if available, otherwise use original
        update_image(predicted_image if predicted_image else None)

def update_image(image=None):
    global img_label, scale
    if image is None and img_path:
        if predicted_image:
            image = predicted_image
        else:
            image = Image.open(img_path)
    if image:
        # Calculate new size
        new_width = int(400 * scale)
        new_height = int(400 * scale)
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(resized_img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

def load_image():
    global img_label, img_path, img, scale
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if img_path:
        scale = 1.0  # Reset zoom
        img = Image.open(img_path)
        update_image(img)

def toggle_none():
    global show_none, predicted_image
    show_none = not show_none
    if img_path:
        predict()  # Re-run prediction with new visibility setting

def predict():
    global img_path, img_label, predicted_image
    if img_path:
        conf_threshold = float(conf_slider.get())
        results = model(img_path, conf=conf_threshold)
        
        for result in results:
            # Filter out none class if hidden
            if not show_none and none_class_idx is not None:
                boxes = result.boxes
                # Keep all classes except none
                mask = boxes.cls != none_class_idx
                result.boxes = boxes[mask]
            
            plotted_image = result.plot(conf=False)
            predicted_image = Image.fromarray(plotted_image[..., ::-1])
            update_image(predicted_image)

def reset():
    global img_label, img_path, predicted_image, scale
    img_label.config(image="")
    img_label.image = None
    img_path = None
    predicted_image = None
    scale = 1.0

# Create Tkinter interface
root = tk.Tk()
root.title("YOLO Model Tester")

# Add confidence threshold slider
conf_label = tk.Label(root, text="Confidence Threshold:")
conf_label.pack()
conf_slider = Scale(root, from_=0.0, to=1.0, resolution=0.05, orient=HORIZONTAL)
conf_slider.set(0.5)  # Set default value to 0.5
conf_slider.pack()

# Add buttons and image display
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

reset_button = tk.Button(root, text="Reset", command=reset)
reset_button.pack()

# Update toggle button text to show class name
toggle_none_btn = tk.Button(root, text=f"Toggle '{class_names.get(none_class_idx, 'None')}' Class", command=toggle_none)
toggle_none_btn.pack()

img_label = Label(root)
img_label.pack()
# Bind mouse wheel to zoom
root.bind("<MouseWheel>", zoom)

root.mainloop()
