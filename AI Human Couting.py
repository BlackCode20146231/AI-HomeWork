import cv2 as cv
import tensorflow as tf
import numpy as np
from keras.models import load_model
from tkinter import *
from PIL import Image, ImageTk

# Load model
model = tf.keras.models.load_model('CNN_Human_Rec.h5', compile=False)
print("Import model complete!")

cap = cv.VideoCapture("crownd (2).mp4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()
count = 0
running = False
text =''
def start_video():
    global running
    running = True
    video_loop()


def stop_video():
    global running
    running = False


def exit_program():
    cap.release()
    cv.destroyAllWindows()
    root.quit()


def video_loop():
    global running
    global frame
    global count
    global text

    if running:
        ret, frame = cap.read()
        global dets
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            exit()

        frame = cv.resize(frame, (720, 480))
        frame = cv.GaussianBlur(frame, (3, 3), 1)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        b_img = cv.threshold(gray, 90, 255, cv.THRESH_BINARY_INV)[1]
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        b_img = cv.morphologyEx(b_img, cv.MORPH_OPEN, kernel, iterations=1)
        contours, hierarchy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        weight, height, _ = frame.shape
        center_y = int((weight+40) / 2)
        dets = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 100:

                x, y, w, h = cv.boundingRect(cnt)
                y_medium = int((y + y + h) / 2)
                
                dets.append([x, y, x+w, y+h])
                crop_img = frame[y:y + (h + 20), x:x + w]

                center_x = int(x + (w / 2))
                center_point = (center_x, y_medium)
                
                gray_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
                # Resize image to (28, 28)
                img_resized = cv.resize(gray_img, (28, 28))
                # Normalize image
                img_resized = img_resized.astype(float)  # Grayscale image with values in the range [0, 1]
                img_normalized = cv.normalize(img_resized, img_resized, 0, 1.0, cv.NORM_MINMAX)
                img_reshaped = np.reshape(img_normalized, (1, 28, 28, 1))
                # Check if center point touches the red line
                if center_y == y_medium:
                    print("Start Detect")
                    result = np.argmax(model.predict(img_reshaped))
                    print (result)
                    # If detection is successful
                    if result == 1 or result==2:
                        global text
                        count += 1
                        count_label.config(text=f"People pass the line: {count}")
                        text = "Human"
                cv.putText(frame, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.circle(frame, center_point, 2, (0, 0, 255), 2)
                print(count)
                # Draw bounding box and center point
        cv.line(frame, (0, center_y), (870, center_y), (0, 0, 255), 1)
        img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(60, video_loop)


# Create Tkinter window
root = Tk()
root.title("Human Detection")
root.geometry("1000x500")

# Create buttons
start_button = Button(root, text="Start", command=start_video, width=10)
start_button.place(x=820, y=50)

stop_button = Button(root, text="Stop", command=stop_video, width=10)
stop_button.place(x=820, y=100)

exit_button = Button(root, text="Exit", command=exit_program, width=10)
exit_button.place(x=820, y=150)

# Create video label
video_label = Label(root)
video_label.place(x=10, y=10)

# Create count label
count_label = Label(root, text="People pass the line: 0", font=("Courier", 14))
count_label.place(x=670, y=200)

root.after(60, video_loop)
root.mainloop()


