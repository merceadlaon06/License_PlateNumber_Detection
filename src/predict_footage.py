from ultralytics import YOLO
import cv2
import easyocr

reader = easyocr.Reader(['en'])
# Load a pretrained YOLO26n model
model = YOLO(r"D:\Users\Strix\PycharmProjects\PlateNumberDetection\runs\detect\train6\weights\best.pt")
cap = cv2.VideoCapture(r"D:\Users\Strix\PycharmProjects\PlateNumberDetection\test videos\test2.mp4")
place = "Europe"
# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully!")

# Get video properties (e.g., frame count and frame width)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames in the video
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second (FPS)
print(f"Total frames: {frame_count}, FPS: {fps}")
# Read and display each frame of the video
plate_detected = []
outFileName = r"D:\Users\Strix\PycharmProjects\PlateNumberDetection\prediction_output\detect\test_result.txt"
outFile = open(outFileName, "w")
outFile.write("Detected: ")
while True:
    ret, frame = cap.read()
    if ret ==False:
        break
    #print("SIZE",frame.shape)
    ratio = frame.shape[1]/ frame.shape[0]
    #print("RATIO", ratio)
    new_height = 640
    new_width = int(ratio*new_height)
    resized_image = cv2.resize(frame, (new_width, new_height))
    #cv2.imshow("Video Frame", resized_image)
    result = model.predict(resized_image, save=False, imgsz=640, conf=0.25)
    annotated_img = result[0].plot()
    #print(result[0].boxes)
    #print(result[0].boxes.xyxy)
    box_position = result[0].boxes.xyxy
    print("Box position", box_position)
    for x in range(len(box_position)):
        cv2.imshow("Prediction", annotated_img)
        cv2.waitKey(1)
        x_start = int(box_position[x][0])
        y_start = int(box_position[x][1])
        x_end = int(box_position[x][2])
        y_end = int(box_position[x][3])
        b_h = frame.shape[0]
        b_w = frame.shape[1]
        ratio = b_w / b_h
        scale_ratio = b_h / 640
        #print(scale_ratio)
        cropped_img = frame[round(y_start*scale_ratio):round(y_end*scale_ratio), round(x_start*scale_ratio):round(x_end*scale_ratio)]
        cv2.imshow("Cropped image", cropped_img)
        cv2.waitKey(1)
        result_text = reader.readtext(cropped_img)
        print(result_text) #TODO: NOT RETURNING ANYTHING
        for (bbox, text, prob) in result_text:
            is_alnum = text.isalnum()
            #is_upper =text.isupper()
            if is_alnum: #== True and is_upper == True:
                ok_text = text.strip( )
                if place == "Europe":
                    is_correct = True
                    if len(ok_text) != 7:
                        is_correct = False
                    for char in ok_text[0:4]:
                        if not char.isalnum():
                            is_correct = False
                    for char in ok_text[4:]:
                        if not char.isalpha():
                            is_correct =False
                    if is_correct:
                        if ok_text not in plate_detected:
                            plate_detected.append(ok_text)

                if place == "Philippines":
                    is_correct = True
                    if len(ok_text) != 7:
                        is_correct = False
                    for char in ok_text[0:3]:
                        if not char.isalpha():
                            is_correct = False
                    for char in ok_text[3:]:
                        if not char.isdigit():
                            is_correct = False
                    for char in ok_text[0:3]:
                        if not char.isdigit():
                            is_correct = False
                    for char in ok_text[3:]:
                        if not char.isalpha():
                            is_correct = False
                    if is_correct:
                        if ok_text not in plate_detected:
                            plate_detected.append(ok_text)

            else:
                text = " "
            print(f'Text: {text}, Probability: {prob}')


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if not ret:
        print("End of video or error occurred.")
        break

plate_numbers = ", ".join(plate_detected).upper()
outFile.write(plate_numbers)
outFile.close()

cap.release()
cv2.destroyAllWindows()


