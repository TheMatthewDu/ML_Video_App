import torch
import torchvision.transforms as T
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
import numpy as np
import cv2

from threading import Thread


class Record:
    def __init__(self):
        # Load a pre-trained DeepLabV3 models with a ResNet-50 backbone
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        model.eval()

        self.cap = cv2.VideoCapture(1)
        self.input_image = None
        self.mask = np.zeros((480, 640), np.uint8)
        self.frames = []

        self.running = True
        self.points = []

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
                if item[1]['confidence'] >= 0.9:
                    self.points.append([
                        int(item[1]['xmin']), int(item[1]['ymin']),
                        int(item[1]['xmax']), int(item[1]['ymax']),
                        int(item[1]['class'])
                    ])

    def run(self):
        # bg = cv2.imread('sample_images/img.png')
        while self.running:
            _, img = self.cap.read()

            self.input_image = np.zeros((480, 640, 3), dtype=np.uint8)

            dx, dy = 480 - img.shape[0], 640 - img.shape[1]
            self.input_image[dx // 2:img.shape[0] + dx // 2, dy // 2:img.shape[1] + dy // 2, :] = img

            output_image = self.input_image
            self.frames.append(output_image)

            if self.points:  # != []
                for point in self.points:
                    cv2.rectangle(output_image, (point[0], point[1]), (point[2], point[3]), (0, 255 * (point[4]), 255 * (1 - point[4])), 5)

            # Save the resulting image with the background removed
            cv2.imshow("img", output_image)

            if cv2.waitKey(1) == ord('q'):
                self.running = False


if __name__ == '__main__':
    a = Record()
    t2 = Thread(target=a.hands)
    t2.start()

    a.run()


