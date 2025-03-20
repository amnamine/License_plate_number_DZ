import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os

class YOLOTester:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Model Tester")
        self.root.geometry("900x800")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use clam theme as base
        
        # Define colors
        self.colors = {
            'primary': '#2c3e50',      # Dark blue-gray
            'secondary': '#3498db',    # Bright blue
            'accent': '#e74c3c',       # Red
            'success': '#2ecc71',      # Green
            'warning': '#f1c40f',      # Yellow
            'background': '#ecf0f1',   # Light gray
            'text': '#2c3e50',         # Dark blue-gray
            'button_bg': '#3498db',    # Bright blue
            'button_fg': 'white',      # White
            'button_active': '#2980b9', # Darker blue
            'frame_bg': '#ffffff',     # White
            'border': '#bdc3c7'        # Gray
        }
        
        # Configure styles
        self.configure_styles()
        
        # Initialize model as None
        self.model = None
        self.model_name = None
        
        # Zoom variables
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        
        # Create GUI elements
        self.create_widgets()
        
        # Variables
        self.current_image = None
        self.photo = None
        self.original_image = None
        self.processed_image = None
        
        # Bind mouse wheel
        self.image_label.bind('<MouseWheel>', self.mouse_wheel)  # Windows
        self.image_label.bind('<Button-4>', self.mouse_wheel)    # Linux scroll up
        self.image_label.bind('<Button-5>', self.mouse_wheel)    # Linux scroll down
        
    def configure_styles(self):
        # Configure button style
        self.style.configure(
            'Custom.TButton',
            background=self.colors['button_bg'],
            foreground=self.colors['button_fg'],
            padding=5,
            font=('Helvetica', 10, 'bold')
        )
        
        # Configure label style
        self.style.configure(
            'Custom.TLabel',
            background=self.colors['frame_bg'],
            foreground=self.colors['text'],
            font=('Helvetica', 10)
        )
        
        # Configure frame style
        self.style.configure(
            'Custom.TFrame',
            background=self.colors['frame_bg']
        )
        
        # Configure scale style
        self.style.configure(
            'Custom.Horizontal.TScale',
            background=self.colors['frame_bg'],
            troughcolor=self.colors['background']
        )
        
    def create_widgets(self):
        # Set root background
        self.root.configure(bg=self.colors['background'])
        
        # Create main container with border
        main_container = tk.Frame(
            self.root,
            bg=self.colors['frame_bg'],
            highlightbackground=self.colors['border'],
            highlightthickness=1,
            padx=10,
            pady=10
        )
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create title label
        title_label = tk.Label(
            main_container,
            text="YOLO Object Detection",
            font=('Helvetica', 16, 'bold'),
            bg=self.colors['frame_bg'],
            fg=self.colors['primary']
        )
        title_label.pack(pady=(0, 10))
        
        # Create buttons frame with border
        button_frame = tk.Frame(
            main_container,
            bg=self.colors['frame_bg'],
            highlightbackground=self.colors['border'],
            highlightthickness=1,
            padx=5,
            pady=5
        )
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # Load Model button
        self.load_model_btn = tk.Button(
            button_frame,
            text="Load Model",
            command=self.load_model,
            bg=self.colors['button_bg'],
            fg=self.colors['button_fg'],
            activebackground=self.colors['button_active'],
            activeforeground=self.colors['button_fg'],
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        self.load_model_btn.pack(side=tk.LEFT, padx=5)
        
        # Model name label
        self.model_label = tk.Label(
            button_frame,
            text="No model loaded",
            fg=self.colors['accent'],
            bg=self.colors['frame_bg'],
            font=('Helvetica', 10)
        )
        self.model_label.pack(side=tk.LEFT, padx=5)
        
        # Select image button
        self.select_btn = tk.Button(
            button_frame,
            text="Select Image",
            command=self.select_image,
            state='disabled',
            bg=self.colors['button_bg'],
            fg=self.colors['button_fg'],
            activebackground=self.colors['button_active'],
            activeforeground=self.colors['button_fg'],
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # Delete image button
        self.delete_btn = tk.Button(
            button_frame,
            text="Delete Image",
            command=self.delete_image,
            state='disabled',
            bg=self.colors['accent'],
            fg=self.colors['button_fg'],
            activebackground='#c0392b',
            activeforeground=self.colors['button_fg'],
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Detect button
        self.detect_btn = tk.Button(
            button_frame,
            text="Detect",
            command=self.detect_objects,
            state='disabled',
            bg=self.colors['button_bg'],
            fg=self.colors['button_fg'],
            activebackground=self.colors['button_active'],
            activeforeground=self.colors['button_fg'],
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        self.reset_btn = tk.Button(
            button_frame,
            text="Reset",
            command=self.reset_image,
            state='disabled',
            bg=self.colors['button_bg'],
            fg=self.colors['button_fg'],
            activebackground=self.colors['button_active'],
            activeforeground=self.colors['button_fg'],
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Create zoom control frame with border
        zoom_frame = tk.Frame(
            main_container,
            bg=self.colors['frame_bg'],
            highlightbackground=self.colors['border'],
            highlightthickness=1,
            padx=5,
            pady=5
        )
        zoom_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # Zoom label
        zoom_label = tk.Label(
            zoom_frame,
            text="Zoom:",
            bg=self.colors['frame_bg'],
            fg=self.colors['text'],
            font=('Helvetica', 10)
        )
        zoom_label.pack(side=tk.LEFT, padx=5)
        
        # Zoom slider
        self.zoom_slider = ttk.Scale(
            zoom_frame,
            from_=self.min_zoom,
            to=self.max_zoom,
            value=1.0,
            orient=tk.HORIZONTAL,
            command=self.zoom_slider_changed,
            style='Custom.Horizontal.TScale'
        )
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Zoom percentage label
        self.zoom_percent = tk.Label(
            zoom_frame,
            text="100%",
            bg=self.colors['frame_bg'],
            fg=self.colors['text'],
            font=('Helvetica', 10)
        )
        self.zoom_percent.pack(side=tk.LEFT, padx=5)
        
        # Reset zoom button
        self.reset_zoom_btn = tk.Button(
            zoom_frame,
            text="Reset Zoom",
            command=self.reset_zoom,
            bg=self.colors['button_bg'],
            fg=self.colors['button_fg'],
            activebackground=self.colors['button_active'],
            activeforeground=self.colors['button_fg'],
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        self.reset_zoom_btn.pack(side=tk.LEFT, padx=5)
        
        # Create image container with border
        image_container = tk.Frame(
            main_container,
            bg=self.colors['frame_bg'],
            highlightbackground=self.colors['border'],
            highlightthickness=1,
            padx=5,
            pady=5
        )
        image_container.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for image
        self.canvas = tk.Canvas(
            image_container,
            highlightthickness=0,
            bg=self.colors['frame_bg']
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbars with custom style
        h_scrollbar = ttk.Scrollbar(
            image_container,
            orient=tk.HORIZONTAL,
            command=self.canvas.xview
        )
        v_scrollbar = ttk.Scrollbar(
            image_container,
            orient=tk.VERTICAL,
            command=self.canvas.yview
        )
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Create image label inside canvas
        self.image_label = tk.Label(
            self.canvas,
            bg=self.colors['frame_bg']
        )
        self.canvas.create_window((0, 0), window=self.image_label, anchor="nw")
        
        # Configure disabled button states
        self.configure_disabled_states()
        
    def configure_disabled_states(self):
        """Configure the appearance of disabled buttons"""
        disabled_style = {
            'bg': '#bdc3c7',
            'fg': '#95a5a6',
            'activebackground': '#bdc3c7',
            'activeforeground': '#95a5a6'
        }
        
        for btn in [self.select_btn, self.detect_btn, self.reset_btn, self.delete_btn]:
            btn.configure(**disabled_style)
    
    def enable_buttons(self):
        """Enable buttons and restore their normal appearance"""
        for btn in [self.select_btn, self.detect_btn, self.reset_btn, self.delete_btn]:
            if btn == self.delete_btn:
                btn.configure(
                    state='normal',
                    bg=self.colors['accent'],
                    fg=self.colors['button_fg'],
                    activebackground='#c0392b',
                    activeforeground=self.colors['button_fg']
                )
            else:
                btn.configure(
                    state='normal',
                    bg=self.colors['button_bg'],
                    fg=self.colors['button_fg'],
                    activebackground=self.colors['button_active'],
                    activeforeground=self.colors['button_fg']
                )
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        self.zoom_slider.set(self.zoom_factor)
        self.update_zoom()
        
    def mouse_wheel(self, event):
        # Handle mouse wheel zoom
        if event.num == 5 or event.delta < 0:  # Zoom out
            self.zoom_factor = max(self.min_zoom, self.zoom_factor - 0.1)
        if event.num == 4 or event.delta > 0:  # Zoom in
            self.zoom_factor = min(self.max_zoom, self.zoom_factor + 0.1)
            
        self.zoom_slider.set(self.zoom_factor)
        self.update_zoom()
        
    def zoom_slider_changed(self, value):
        self.zoom_factor = float(value)
        self.update_zoom()
        
    def update_zoom(self):
        if self.processed_image is not None:
            # Update zoom percentage label
            self.zoom_percent.config(text=f"{int(self.zoom_factor * 100)}%")
            
            # Calculate new dimensions
            new_width = int(640 * self.zoom_factor)
            new_height = int(640 * self.zoom_factor)
            
            # Resize the image
            resized_image = cv2.resize(self.processed_image, (new_width, new_height))
            
            # Update the display
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            self.image_label.config(image=self.photo)
            
            # Update canvas scrollregion
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Reset scroll position when zoom is reset
            if self.zoom_factor == 1.0:
                self.canvas.xview_moveto(0)
                self.canvas.yview_moveto(0)
        
    def process_image(self, image):
        """Process image to 640x640 size"""
        # Resize image to 640x640
        image_resized = cv2.resize(image, (640, 640))
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        return rgb_image
        
    def load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PT files", "*.pt")]
        )
        
        if file_path:
            try:
                self.model = YOLO(file_path)
                self.model_name = os.path.basename(file_path)
                self.model_label.config(
                    text=f"Model: {self.model_name}",
                    fg=self.colors['success']
                )
                
                # Enable buttons after model is loaded
                self.enable_buttons()
                
                messagebox.showinfo("Success", f"Model '{self.model_name}' loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.model_label.config(
                    text="No model loaded",
                    fg=self.colors['accent']
                )
                self.model = None
                self.configure_disabled_states()
        
    def select_image(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                # Read and display the image
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise Exception("Failed to load image")
                
                # Store original image for reset functionality
                self.original_image = self.current_image.copy()
                
                # Process image to 640x640
                self.processed_image = self.process_image(self.current_image)
                
                # Reset zoom and update display
                self.reset_zoom()
                
                # Enable delete button
                self.delete_btn.config(state='normal')
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.processed_image = self.process_image(self.current_image)
            self.reset_zoom()
    
    def detect_objects(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        try:
            # Ensure image is 640x640 for detection
            resized_image = cv2.resize(self.current_image, (640, 640))
            
            # Run detection
            results = self.model(resized_image)
            
            # Get the first result
            result = results[0]
            
            # Draw boxes on the image without confidence scores
            annotated_image = result.plot(conf=False)
            
            # Convert BGR to RGB
            self.processed_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Update display with zoom
            self.update_zoom()
            
            # Show detection results
            boxes = result.boxes
            if len(boxes) > 0:
                messagebox.showinfo("Detection Results", f"Found {len(boxes)} objects")
            else:
                messagebox.showinfo("Detection Results", "No objects detected")
                
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
    
    def delete_image(self):
        """Delete the current image and reset the display"""
        if self.current_image is not None:
            # Clear all image-related variables
            self.current_image = None
            self.original_image = None
            self.processed_image = None
            self.photo = None
            
            # Clear the display
            self.image_label.config(image='')
            
            # Reset zoom
            self.reset_zoom()
            
            # Disable detect and reset buttons
            self.detect_btn.config(state='disabled')
            self.reset_btn.config(state='disabled')
            self.delete_btn.config(state='disabled')
            
            # Update disabled button styles
            self.configure_disabled_states()
            
            messagebox.showinfo("Success", "Image deleted successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOTester(root)
    root.mainloop() 