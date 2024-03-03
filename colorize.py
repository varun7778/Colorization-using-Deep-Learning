import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import os

# Paths to load the model
DIR = r"D:/Projects/Colorization-using-Deep-Learning"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")


def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net

# Function to colorize image
def colorize_image():
    # Open file dialog to select image
    file_path = filedialog.askopenfilename()
    if file_path:
        net = load_model()

        # Preprocess the input image
        image = cv2.imread(file_path)
        L, shape, lab = preprocess_frame(image)

        # Colorize the image
        colorized = colorize(net, L, shape, lab)

        # Display results
        display_frame(image, colorized)

        mse, psnr, ssim_value = calculate_metrics(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY))
        print("Mean Squared Error (MSE):", mse)
        print("Peak Signal-to-Noise Ratio (PSNR):", psnr)
        print("Structural Similarity Index (SSIM):", ssim_value)


def colorize_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        net = load_model()

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            L, shape, lab = preprocess_frame(frame)
            colorized = colorize(net, L, shape, lab)
            cv2.imshow("frame",colorized)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()

def colorize_webcam():
    net = load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        L, shape, lab = preprocess_frame(frame)
        colorized = colorize(net, L, shape, lab)
        cv2.imshow("frame",colorized)

        key = cv2.waitKey(1)
        if key == 27:
            break

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

    cap.release()


def preprocess_frame(image):
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    return L, image.shape, lab

def colorize(net, L, shape, lab):
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (shape[1], shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    return colorized
    
# Function to display the colorized image
def display_frame(original, colorized):
    # Convert images from OpenCV BGR format to RGB format
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

    # Convert images to PIL format
    original_pil = Image.fromarray(original_rgb)
    colorized_pil = Image.fromarray(colorized_rgb)

    # Resize images to fit in the window
    original_pil = original_pil.resize((300, 300))
    colorized_pil = colorized_pil.resize((300, 300))

    # Create image objects from PIL images
    original_img = ImageTk.PhotoImage(original_pil)
    colorized_img = ImageTk.PhotoImage(colorized_pil)

    # Create labels to display images
    original_label = tk.Label(panel, image=original_img)
    original_label.image = original_img
    original_label.grid(row=0, column=0, padx=10)

    colorized_label = tk.Label(panel, image=colorized_img)
    colorized_label.image = colorized_img
    colorized_label.grid(row=0, column=1, padx=10)

def calculate_metrics(ground_truth, colorized):
    # Mean Squared Error (MSE)
    mse = np.mean((ground_truth - colorized) ** 2)

    # Peak Signal-to-Noise Ratio (PSNR)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    # Structural Similarity Index (SSIM)
    ssim_value, _ = ssim(ground_truth, colorized, full=True)

    return mse, psnr, ssim_value

# Create the main window
root = tk.Tk()
root.title("Image Colorization")

button_frame = tk.Frame(root)
button_frame.pack()

# Create a button to select and colorize an image
colorize_button = tk.Button(button_frame, text="Colorize Image", command=colorize_image)
colorize_button.pack(side=tk.LEFT, padx=5, pady=10)

colorize_button = tk.Button(button_frame, text="Colorize Video", command=colorize_video)
colorize_button.pack(side=tk.LEFT, padx=5, pady=10)

# Create a button to save the colorized image
save_button = tk.Button(button_frame, text="Webcam", command=colorize_webcam)
save_button.pack(side=tk.LEFT, padx=5, pady=10)

# Create a panel to display the colorized image
panel = tk.Label(root)
panel.pack()

# Run the Tkinter event loop
root.mainloop()
