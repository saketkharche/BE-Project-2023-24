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


# Function to create the table in the database
def create_table():
    conn = mysql.connector.connect(
        host="localhost", user="root", password="root", database="be_project"
    )
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS violators (id INT AUTO_INCREMENT PRIMARY KEY, vehicle_number VARCHAR(20) NOT NULL, bike_image_path TEXT)"
    )
    conn.commit()
    conn.close()


# Function to insert a record into the database
def insert_record(vehicle_number, bike_image_path):
    conn = mysql.connector.connect(
        host="localhost", user="root", password="root", database="be_project"
    )
    c = conn.cursor()
    c.execute(
        "INSERT INTO violators (vehicle_number, bike_image_path) VALUES (%s, %s)",
        (vehicle_number, bike_image_path),
    )
    conn.commit()
    conn.close()


# Function to generate a receipt PDF
def generate_receipt(vehicle_number, bike_image_path):
    c = canvas.Canvas("receipt_{}.pdf".format(vehicle_number), pagesize=letter)
    c.drawString(100, 750, "Vehicle Number: {}".format(vehicle_number))
    c.drawString(100, 730, "Bike Image Path: {}".format(bike_image_path))
    c.save()


# Function to show results
def show_results(vehicle_number, bike_image_path):
    # Show violation information in a message box
    messagebox.showinfo(
        "Violation Detected",
        f"Vehicle Number: {vehicle_number}\nBike Image Path: {bike_image_path}",
    )

    # Generate receipt
    generate_receipt(vehicle_number, bike_image_path)


# Function to process the video file
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
                                    edged.copy(),
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE,
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
                                        person_bike_image,
                                        person_bike_image,
                                        mask=mask,
                                    )

                                    (x, y) = np.where(mask == 255)
                                    (x1, y1) = (np.min(x), np.min(y))
                                    (x2, y2) = (np.max(x), np.max(y))
                                    cropped_image = gray[x1 : x2 + 1, y1 : y2 + 1]

                                    reader = easyocr.Reader(["en"])
                                    result = reader.readtext(cropped_image)

                                    # Extracted text from EasyOCR result
                                    if result:
                                        # Preprocess extracted text to remove spaces
                                        extracted_vehicle_number = result[0][
                                            -2
                                        ].replace(" ", "")
                                        # Remove special characters from the text
                                        extracted_vehicle_number = re.sub(
                                            r"[^\w\s]", "", extracted_vehicle_number
                                        )

                                        # Save the cropped number plate image
                                        image_file = str(int(time.time())) + ".jpg"
                                        output_file = f"person_violation_{image_file}"
                                        output_path = os.path.join(
                                            output_dir, output_file
                                        )
                                        cv2.imwrite(output_path, person_bike_image)

                                        # Insert the extracted number plate into the database
                                        create_table()
                                        insert_record(
                                            extracted_vehicle_number, output_path
                                        )
                                        # Show the violation information and generate receipt
                                        show_results(
                                            extracted_vehicle_number, output_path
                                        )


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

# Create Tkinter GUI
root = tk.Tk()
root.title("Video Processing Application")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

label_video_file = tk.Label(frame, text="Video File:")
label_video_file.grid(row=0, column=0, sticky="e")

entry_video_file = tk.Entry(frame, width=50)
entry_video_file.grid(row=0, column=1, padx=5, pady=5)


def select_video_file():
    video_file = filedialog.askopenfilename(
        initialdir="/",
        title="Select Video File",
        filetypes=(("Video files", "*.mp4"), ("All files", "*.*")),
    )
    entry_video_file.delete(0, tk.END)
    entry_video_file.insert(0, video_file)


button_browse = tk.Button(frame, text="Browse", command=select_video_file)
button_browse.grid(row=0, column=2, padx=5, pady=5)

button_start = tk.Button(frame, text="Start Processing", command=start_processing)
button_start.grid(row=1, columnspan=3, pady=10)

root.mainloop()
