import cv2
from imageai.Detection import ObjectDetection

# Шлях до моделі
model_path = "./models/yolo-tiny.h5"

# Ініціалізація детектора
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

# Відео
input_video_path = './input/input/video.mp4'
output_video_path = './output/output_video.mp4'

# Відкриття відео файлу
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Could not open video file")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Відео записувач
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Збереження поточного кадру у тимчасовий файл
    temp_input_path = "temp_frame.jpg"
    temp_output_path = "temp_frame_output.jpg"
    cv2.imwrite(temp_input_path, frame)

    # Детекція об'єктів на кадрі
    detections = detector.detectObjectsFromImage(
        input_image=temp_input_path,
        output_image_path=temp_output_path
    )

    # Зчитування обробленого кадру
    output_frame = cv2.imread(temp_output_path)

    # Запис обробленого кадру у відео
    out.write(output_frame)

    # Виведення детекції у консолі
    for eachItem in detections:
        print(f"{eachItem['name']}:{eachItem['percentage_probability']}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закриття відео файлів
cap.release()
out.release()
cv2.destroyAllWindows()
