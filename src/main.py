import cv2
import easyocr
from ultralytics import YOLO


# --- Validation Logic ---

def is_valid_plate(text, region="Europe"):
    """Validates the plate format based on the region."""
    clean_text = text.strip().upper()

    if not clean_text.isalnum():
        return False, clean_text

    if len(clean_text) != 7:
        return False, clean_text

    if region == "Europe":
        # Example logic: 4 alnum followed by 3 alpha
        first_part = all(c.isalnum() for c in clean_text[:4])
        second_part = all(c.isalpha() for c in clean_text[4:])
        return (first_part and second_part), clean_text

    elif region == "Philippines":
        # Standard PH: 3 Alpha + 4 Numeric (or vice versa for newer/older plates)
        # Your original logic had a conflict; I've simplified it to 3 alpha + 4 digits
        is_alpha_start = clean_text[:3].isalpha() and clean_text[3:].isdigit()
        return is_alpha_start, clean_text

    return False, clean_text


# --- Image Processing ---

def get_cropped_plate(frame, box, scale_ratio):
    """Crops the detected plate area from the original high-res frame."""
    x1, y1, x2, y2 = box
    # Scale coordinates back to original frame size
    start_y = int(round(y1 * scale_ratio))
    end_y = int(round(y2 * scale_ratio))
    start_x = int(round(x1 * scale_ratio))
    end_x = int(round(x2 * scale_ratio))

    return frame[start_y:end_y, start_x:end_x]


# --- Main Execution ---

def main():
    # Configuration
    video_path = r"D:\Users\Strix\PycharmProjects\PlateNumberDetection\test.mp4"
    model_path = r"D:\Users\Strix\PycharmProjects\PlateNumberDetection\runs\detect\train6\weights\best.pt"
    output_path = r"D:\Users\Strix\PycharmProjects\PlateNumberDetection\prediction_output\detect\test_result.txt"
    region = "Europe"

    # Initialize Models
    reader = easyocr.Reader(['en'])
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    plate_detected = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for YOLO inference
        h, w = frame.shape[:2]
        new_height = 640
        scale_ratio = h / new_height
        new_width = int((w / h) * new_height)
        resized_image = cv2.resize(frame, (new_width, new_height))

        # Inference
        results = model.predict(resized_image, save=False, imgsz=640, conf=0.25, verbose=False)
        annotated_img = results[0].plot()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            cropped_img = get_cropped_plate(frame, box, scale_ratio)

            # OCR Processing
            ocr_results = reader.readtext(cropped_img)

            for (bbox, text, prob) in ocr_results:
                valid, clean_text = is_valid_plate(text, region)

                if valid and clean_text not in plate_detected:
                    print(f"New Plate Detected: {clean_text} (Conf: {prob:.2f})")
                    plate_detected.append(clean_text)

            # Visual Feedback
            cv2.imshow("Cropped Plate", cropped_img)

        cv2.imshow("Detection Stream", annotated_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalize results
    cap.release()
    cv2.destroyAllWindows()

    with open(output_path, "w") as f:
        f.write(f"Detected: {', '.join(plate_detected)}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()