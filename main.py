import torch
import torchvision.transforms as T
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
import numpy as np
import cv2

from threading import Thread


def get_points():
    image_pts = np.load("sample_images/pts.npy")
    idx = np.load("sample_images/idx.npy")

    image_pts = image_pts[:, :2]

    line_set = []
    for tri in idx:
        line_set.append([tri[0], tri[1]])
        line_set.append([tri[1], tri[2]])
        line_set.append([tri[2], tri[0]])
    return image_pts, line_set


class Record:
    def __init__(self):
        # Load a pre-trained DeepLabV3 models with a ResNet-50 backbone
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        model.eval()

        self.cap = cv2.VideoCapture(0)
        self.input_image = None
        self.mask = np.zeros((480, 640), np.uint8)

        self.running = True
        self.points = []

    def predict(self):
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        model.eval()

        while self.running:
            if self.input_image is None:
                continue

            new_image = cv2.resize(self.input_image, None, fx=0.2, fy=0.2)

            # Apply transformations to the input image
            preprocess = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(new_image).unsqueeze(0)

            # Make predictions on the input image
            with torch.no_grad():
                output = model(input_tensor)['out'][0]

            mask = (output.argmax(0) == 15).float()  # Assuming 15 is the class ID for the object of interest

            # Convert the mask to a NumPy array
            mask = mask.numpy().astype(np.uint8)

            self.mask = (cv2.resize(mask, None, fx=5, fy=5) != 0).astype(np.uint8)

    def hands(self):
        model = torch.hub.load('yolov5', 'custom', path='weights/best.pt', source='local')
        model.eval()

        while self.running:
            if self.input_image is None:
                continue

            result = model(self.input_image)
            pts = result.pandas().xyxy[0]

            self.points = []
            for item in pts.iterrows():
                self.points.append([
                    int(item[1]['xmin']), int(item[1]['ymin']),
                    int(item[1]['xmax']), int(item[1]['ymax'])
                ])

    def run(self):
        bg = cv2.imread('sample_images/img.png')
        points, indices = get_points()

        while self.running:
            background = cv2.resize(bg, None, fx=640 / bg.shape[1], fy=480 / bg.shape[0])
            _, img = self.cap.read()

            self.input_image = np.zeros((480, 640, 3), dtype=np.uint8)

            dx, dy = 480 - img.shape[0], 640 - img.shape[1]
            self.input_image[dx // 2:img.shape[0] + dx // 2, dy // 2:img.shape[1] + dy // 2, :] = img

            output_image = cv2.bitwise_and(self.input_image, self.input_image, mask=self.mask)
            background = cv2.bitwise_and(background, background, mask=1 - self.mask)
            output_image = cv2.bitwise_or(output_image, background)

            if self.points != []:
                translation = np.array([
                    (self.points[0][0] + self.points[0][2]) / 2,
                    (self.points[0][1] + self.points[0][3]) / 2
                ])

                show_image_pts = ((self.points[0][2] - self.points[0][0]) * points * 3 / 8 + translation - np.array([0, 75])).astype(int)

                for line in indices:
                    cv2.line(output_image, show_image_pts[line[0]], show_image_pts[line[1]], [0, 255, 0], 1)
                # cv2.rectangle(output_image, point[0], point[1], (0, 255, 0), 5)

            # Save the resulting image with the background removed
            cv2.imshow("img", output_image)

            if cv2.waitKey(1) == ord('q'):
                cv2.imwrite('sample_images/out.png', output_image)
                self.running = False


if __name__ == '__main__':
    a = Record()
    t1 = Thread(target=a.predict)
    t1.start()

    t2 = Thread(target=a.hands)
    t2.start()

    a.run()


