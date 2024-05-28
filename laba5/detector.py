from imageai.Detection import ObjectDetection

model_path = "./models/yolo-tiny.h5"
input_path = "./input/test2.jpg"
output_path = "./output/test2.jpg"

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(
    input_image=input_path,
    output_image_path=output_path
)

for eachItem in detection:
    print(f"{eachItem['name']}:{eachItem['percentage_probability']}")

