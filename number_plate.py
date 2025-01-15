import cv2
import os
from flask import Flask, render_template, send_from_directory, request, jsonify
import threading
import easyocr

app = Flask(__name__, static_folder='static')

# Directory to save captured plates
PLATE_DIR = "number_plates"
if not os.path.exists(PLATE_DIR):
    os.makedirs(PLATE_DIR)

# The Haarcascade for Russian number plates
harcascade = r"model\haarcascade_russian_plate_number.xml"

# Threading variables
capture_event = threading.Event()
captured_image = ""  # Store the path of the captured image
extracted_text = ""  # Store the extracted text

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def capture_image():
    """Function to capture an image from the webcam and detect plates."""
    global captured_image, extracted_text
    capture_event.clear()  # Reset event before starting capture

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        captured_image = ""
        extracted_text = ""
        capture_event.set()
        return

    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    plate_cascade = cv2.CascadeClassifier(harcascade)
    if plate_cascade.empty():
        print("Error: Haarcascade file not found.")
        captured_image = ""
        extracted_text = ""
        capture_event.set()
        return

    print("Camera started. Press 's' to save the detected plate.")

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            captured_image = ""
            extracted_text = ""
            capture_event.set()
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            img_roi = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            if len(plates) > 0:
                filename = os.path.join(PLATE_DIR, f"plate_{len(os.listdir(PLATE_DIR))}.jpg")
                cv2.imwrite(filename, img_roi)
                print(f"Captured and saved image: {filename}")
                captured_image = os.path.basename(filename)  # Only send the base name

                # Perform OCR on the captured image
                try:
                    results = reader.readtext(filename)
                    extracted_text = " ".join([text[1] for text in results])
                    print(f"Extracted Text: {extracted_text}")
                except Exception as e:
                    print(f"Error during OCR: {e}")
                    extracted_text = "OCR Failed"

                capture_event.set()  # Notify capture complete
                cap.release()
                cv2.destroyAllWindows()
                return
            else:
                print("No plate detected. Press 's' to try again or 'q' to quit.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture aborted by the user.")
            captured_image = ""
            extracted_text = ""
            capture_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route('/capture', methods=['POST'])
def capture():
    """Handle the capture image request."""
    global captured_image, extracted_text
    threading.Thread(target=capture_image).start()
    capture_event.wait()  # Wait for the capture thread to signal completion

    if captured_image:
        return jsonify(filename=captured_image, text=extracted_text)
    else:
        return jsonify(error="Failed to capture image. Check the logs."), 500

@app.route('/show/<filename>')
def show_image(filename):
    """Serve the captured image to the client."""
    return send_from_directory(PLATE_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)