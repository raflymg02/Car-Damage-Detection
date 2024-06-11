import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import filedialog, Tk, Label, Button, Canvas, Frame
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import os

# Load the model
model_path = 'classificationModel.keras'
model = load_model(model_path)

# Define the class names and corresponding colors
CATEGORIES = ["01-minor", "02-moderate", "03-severe"]
CATEGORY_COLORS = {"01-minor": "green", "02-moderate": "orange", "03-severe": "red"}

# Function to preprocess the image
def preprocess_image(img_path, target_size):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch of images
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Function to get color based on confidence
def get_confidence_color(confidence):
    if confidence > 0.75:
        return "green"
    elif confidence > 0.5:
        return "orange"
    else:
        return "red"

# Function to upload and process the image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        img_array = preprocess_image(file_path, target_size=(224, 224))

        # Perform prediction using the given prediction code
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions)

        # Convert predicted label to class name
        predicted_class = CATEGORIES[predicted_label]

        # Calculate confidence
        confidence = predictions[0][predicted_label]

        # Display the uploaded image
        img = Image.open(file_path)
        img.thumbnail((400, 300))
        img = ImageTk.PhotoImage(img)
        image_canvas.create_image(200, 150, image=img)
        image_canvas.image = img

        # Display the results
        damages_label.config(text=predicted_class, fg=CATEGORY_COLORS[predicted_class])
        confidence_label.config(text=f"{confidence * 100:.2f}%", fg=get_confidence_color(confidence))

# Function to create the main window
def create_main_window():
    root = Tk()
    root.title("Car Damage Detector")
    return root

# Function to create the main frame
def create_main_frame(root):
    main_frame = Frame(root, bg="white")
    main_frame.pack(fill="both", expand=True)
    return main_frame

# Function to create the header
def create_header(main_frame):
    header_frame = Frame(main_frame, bg="#2a5280")
    header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
    header = Label(header_frame, text="Car Damage Detector", font=("Helvetica", 20), bg="#2a5280", fg="white")
    header.pack(pady=10)

# Function to create the image display area
def create_image_display(main_frame):
    image_frame = Frame(main_frame, width=400, height=300, bg="lightgrey")
    image_frame.grid(row=1, column=0, padx=10, pady=10)
    global image_canvas
    image_canvas = Canvas(image_frame, width=400, height=300, bg="lightgrey")
    image_canvas.pack()

# Function to create the upload button
def create_upload_button(main_frame):
    upload_button = Button(main_frame, text="Upload Image", command=upload_image, font=("Helvetica", 12))
    upload_button.grid(row=2, column=0, pady=10)

# Function to create the results section
def create_results_section(main_frame):
    results_frame = Frame(main_frame, bg="white")
    results_frame.grid(row=1, column=1, padx=10, pady=10)
    results_label = Label(results_frame, text="Results", font=("Helvetica", 14), bg="white")
    results_label.grid(row=0, column=0, columnspan=2, pady=5)

    damages_text_label = Label(results_frame, text="Damages:", font=("Helvetica", 12), bg="white")
    damages_text_label.grid(row=1, column=0, sticky="w")
    global damages_label
    damages_label = Label(results_frame, text="...", font=("Helvetica", 12, "bold"), bg="white")
    damages_label.grid(row=1, column=1, sticky="w")

    confidence_text_label = Label(results_frame, text="Confidence:", font=("Helvetica", 12), bg="white")
    confidence_text_label.grid(row=2, column=0, sticky="w")
    global confidence_label
    confidence_label = Label(results_frame, text="...", font=("Helvetica", 12, "bold"), bg="white")
    confidence_label.grid(row=2, column=1, sticky="w")

# Main function to run the application
def main():
    root = create_main_window()
    main_frame = create_main_frame(root)
    create_header(main_frame)
    create_image_display(main_frame)
    create_upload_button(main_frame)
    create_results_section(main_frame)
    root.mainloop()

if __name__ == "__main__":
    main()
