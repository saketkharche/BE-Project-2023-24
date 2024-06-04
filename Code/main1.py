import os
import time
import cv2
import easyocr
import imutils
import mysql.connector
import numpy as np
import pytesseract
import re
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_table():
    # Connect to the MySQL database
    conn = mysql.connector.connect(
        host="localhost", user="root", password="root", database="be_project"
    )
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS vehicles (vehicle_number TEXT, bike_image_path TEXT)"
    )
    conn.commit()
    conn.close()


def insert_record(vehicle_number, bike_image_path):
    # Connect to the MySQL database
    conn = mysql.connector.connect(
        host="localhost", user="root", password="root", database="be_project"
    )
    c = conn.cursor()
    c.execute("INSERT INTO vehicles VALUES (%s, %s)", (vehicle_number, bike_image_path))
    conn.commit()
    conn.close()


def generate_receipt(vehicle_number, bike_image_path):
    c = canvas.Canvas("receipt.pdf", pagesize=letter)

    # Write vehicle number
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Vehicle Number: {}".format(vehicle_number))

    # Draw bike image
    bike_image = cv2.imread(bike_image_path)
    bike_image_height, bike_image_width = bike_image.shape[:2]
    aspect_ratio = bike_image_width / bike_image_height

    # Define maximum width and height for the image in PDF
    max_width = 400  # in points (1 inch = 72 points)
    max_height = 400  # in points

    # Calculate image width and height based on aspect ratio and maximum dimensions
    if aspect_ratio > 1:  # Landscape orientation
        image_width = max_width
        image_height = max_width / aspect_ratio
        if image_height > max_height:
            image_height = max_height
            image_width = max_height * aspect_ratio
    else:  # Portrait orientation
        image_height = max_height
        image_width = max_height * aspect_ratio
        if image_width > max_width:
            image_width = max_width
            image_height = max_width / aspect_ratio

    # Calculate position to center the image
    x_offset = (max_width - image_width) / 2
    y_offset = (max_height - image_height) / 2

    # Draw the image on the PDF canvas
    c.drawInlineImage(
        bike_image_path,
        100 + x_offset,
        50,  # Adjust y-position to place image at the bottom of the page
        width=image_width,
        height=image_height,
    )

    c.save()


# Define functions for GUI components
def select_video_file():
    filename = filedialog.askopenfilename(title="Select Video File")
    entry_video_file.delete(0, tk.END)
    entry_video_file.insert(0, filename)


def start_processing():
    video_file = entry_video_file.get()
    if not video_file:
        messagebox.showerror("Error", "Please select a video file.")
        return

    video_capture = cv2.VideoCapture(video_file)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect person on a bike
        person_bike_results = person_bike_model.predict(img)

        # Process each detection result
        for r in person_bike_results:
            boxes = r.boxes
            # Filter detections for person on a bike
            for box in boxes:
                cls = box.cls
                if person_bike_model.names[int(cls)] == "Person_Bike":
                    # Crop person on a bike image
                    x1, y1, x2, y2 = box.xyxy[0]
                    person_bike_image = frame[int(y1) : int(y2), int(x1) : int(x2)]

                    # Detect helmet on the person
                    helmet_results = helmet_model.predict(person_bike_image)

                    # Process each helmet detection result
                    for hr in helmet_results:
                        h_boxes = hr.boxes
                        # Filter detections for no helmet
                        for h_bo in h_boxes:
                            h_cls = h_bo.cls
                            if not helmet_model.names[int(h_cls)] == "With Helmet":
                                # Extract number plate from the person bike image
                                gray = cv2.cvtColor(
                                    person_bike_image, cv2.COLOR_BGR2GRAY
                                )
                                bfilter = cv2.bilateralFilter(
                                    gray, 11, 17, 17
                                )  # Noise reduction
                                edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

                                keypoints = cv2.findContours(
                                    edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                                )
                                contours = imutils.grab_contours(keypoints)
                                contours = sorted(
                                    contours, key=cv2.contourArea, reverse=True
                                )[:10]

                                location = None
                                for contour in contours:
                                    approx = cv2.approxPolyDP(contour, 10, True)
                                    if len(approx) == 4:
                                        location = approx
                                        break

                                # Check if 'location' is not None before drawing contours
                                if location is not None:
                                    mask = np.zeros(gray.shape, np.uint8)
                                    new_image = cv2.drawContours(
                                        mask, [location], 0, 255, -1
                                    )
                                    new_image = cv2.bitwise_and(
                                        person_bike_image, person_bike_image, mask=mask
                                    )

                                    (x, y) = np.where(mask == 255)
                                    (x1, y1) = (np.min(x), np.min(y))
                                    (x2, y2) = (np.max(x), np.max(y))
                                    cropped_image = gray[x1 : x2 + 1, y1 : y2 + 1]

                                    reader = easyocr.Reader(["en"])
                                    result = reader.readtext(cropped_image)

                                    # Extracted text from EasyOCR result
                                    if result:
                                        text = result[0][-2]
                                        # Remove special characters from the text
                                        text = re.sub(r"[^\w\s]", "", text)
                                        # Save the cropped number plate image
                                        image_file = str(int(time.time())) + ".jpg"
                                        output_file = f"person_violation_{image_file}"
                                        output_path = os.path.join(
                                            output_dir, output_file
                                        )
                                        cv2.imwrite(output_path, person_bike_image)

                                        # Insert the extracted number plate into the database
                                        create_table()
                                        insert_record(text, output_path)
                                        # Show the violation information and generate receipt
                                        show_results(text, output_path)


def show_results(vehicle_number, bike_image_path):
    # Show violation information in a message box
    messagebox.showinfo(
        "Violation Detected",
        "Vehicle Number: {}\nBike Image Path: {}".format(
            vehicle_number, bike_image_path
        ),
    )

    # Generate receipt
    generate_receipt(vehicle_number, bike_image_path)


# Initialize YOLO models and Tesseract
person_bike_model = YOLO(
    r"C:\Users\saket\OneDrive\Desktop\BE Project Code\BE Project Code\BE Project Code\Results\Models\First Model to detect motorcyclists\best.pt"
)
helmet_model = YOLO(
    r"C:\Users\saket\OneDrive\Desktop\BE Project Code\BE Project Code\BE Project Code\Results\Models\Second Model to detect helmet\best.pt"
)
number_plate_model = YOLO(
    r"C:\Users\saket\OneDrive\Desktop\BE Project Code\BE Project Code\BE Project Code\Results\Models\Third Model to detect number plate\best.pt"
)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR"
tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'

# Initialize database parameters
output_dir = r"C:\Users\saket\Downloads\Code\Code\output"

# Create Tkinter GUI
root = tk.Tk()
root.title("Video Processing Application")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

label_video_file = tk.Label(frame, text="Video File:")
label_video_file.grid(row=0, column=0, sticky="e")

entry_video_file = tk.Entry(frame, width=50)
entry_video_file.grid(row=0, column=1, padx=5, pady=5)

button_browse = tk.Button(frame, text="Browse", command=select_video_file)
button_browse.grid(row=0, column=2, padx=5, pady=5)

button_start = tk.Button(frame, text="Start Processing", command=start_processing)
button_start.grid(row=1, columnspan=3, pady=10)

root.mainloop()
