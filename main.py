import torch
import torchvision.transforms as T
import tqdm
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
import numpy as np
import cv2

from threading import Thread
from typing import Optional

DEBUG = False


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
    """
    :ivar cap:
    """
    cap: cv2.VideoCapture
    input_image: Optional[np.ndarray]
    foreground_mask: Optional[np.ndarray]
    running: bool
    hand_box_points: list[list[int]]
    frames: list[np.ndarray]
    bg_model: torch.nn.Module

    def __init__(self):
        # Load a pre-trained DeepLabV3 models with a ResNet-50 backbone
        self.bg_model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.bg_model.eval()

        self.hand_model = torch.hub.load('yolov5', 'custom', path='weights/best.pt', source='local')
        self.hand_model.eval()

        self.cap = cv2.VideoCapture(1)
        self.input_image = None
        self.foreground_mask = None

        self.running = True
        self.hand_box_points = []

        # Apply transformations to the input image
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.frames = []
        self.processed = []

    def find_foreground(self):
        """ Function to predict the hand position and remove the background

        :return: Nothing
        """
        # If the other thread has not finished collecting pictures, block
        while self.input_image is None:
            pass

        assert self.input_image is not None

        while self.running:
            # Resize the image to ...
            new_image = cv2.resize(self.input_image, None, fx=0.2, fy=0.2)

            mask = self.process_background(new_image)

            self.foreground_mask = (cv2.resize(mask, None, fx=5, fy=5) != 0).astype(np.uint8)

    def process_background(self, img):
        input_tensor = self.preprocess(img).unsqueeze(0)

        # Make predictions on the input image
        with torch.no_grad():
            output = self.bg_model(input_tensor)['out'][0]
        mask = (output.argmax(0) == 15).float()  # Assuming 15 is the class ID for the object of interest

        # Convert the mask to a NumPy array
        return mask.numpy().astype(np.uint8)

    def find_hands(self):
        # If the other thread has not finished collecting pictures, block
        while self.input_image is None:
            pass

        assert self.input_image is not None

        while self.running:
            result = self.hand_model(self.input_image)
            pts = result.pandas().xyxy[0]

            # Get the points for the hand
            self.hand_box_points = [
                [
                    int(point[1]['xmin']), int(point[1]['ymin']),
                    int(point[1]['xmax']), int(point[1]['ymax'])
                ]
                for point in pts.iterrows() if point[1]['confidence'] >= 0.5
            ]

    def run(self):
        bg = cv2.imread('sample_images/img.png')
        points, indices = get_points()

        while self.running:
            _, img = self.cap.read()

            # Resize and create a copy of the background to prevent aliasing
            background = cv2.resize(bg, None, fx=640 / bg.shape[1], fy=480 / bg.shape[0])

            # Pad the image with zeroes so that it is 480 by 640 sized.
            self.input_image = np.zeros((480, 640, 3), dtype=np.uint8)
            dx, dy = 480 - img.shape[0], 640 - img.shape[1]
            self.input_image[dx // 2:img.shape[0] + dx // 2, dy // 2:img.shape[1] + dy // 2, :] = img

            output_image = np.copy(self.input_image)
            self.frames.append(self.input_image)

            # Apply the background remover mask
            if self.foreground_mask is not None:
                # Apply the background image
                output_image = cv2.bitwise_and(self.input_image, self.input_image, mask=self.foreground_mask)
                background = cv2.bitwise_and(background, background, mask=1 - self.foreground_mask)
                output_image = cv2.bitwise_or(output_image, background)

            # Show the positions of the hand
            if self.hand_box_points:  # != []
                # Find the center of the hand
                translation = np.array([
                    (self.hand_box_points[0][0] + self.hand_box_points[0][2]) / 2,
                    (self.hand_box_points[0][1] + self.hand_box_points[0][3]) / 2
                ]) - np.array([0, 75])
                scale = 3 / 8 * (self.hand_box_points[0][2] - self.hand_box_points[0][0])

                show_image_pts = (scale * points + translation).astype(int)

                for line in indices:
                    cv2.line(output_image, show_image_pts[line[0]], show_image_pts[line[1]], [0, 255, 0], 1)

                    if DEBUG:
                        cv2.rectangle(output_image, show_image_pts[0], show_image_pts[1], (0, 255, 0), 5)

            # Show the resulting image
            cv2.imshow("img", output_image)

            if cv2.waitKey(1) == ord('q'):
                self.running = False

    def post_process(self):
        bg = cv2.imread('sample_images/img.png')
        points, indices = get_points()

        video_writer = cv2.VideoWriter('video.avi',
                                       cv2.VideoWriter_fourcc(*'XVID'),
                                       20.0, (640, 480))

        for img in tqdm.tqdm(self.frames):
            # Resize and create a copy of the background to prevent aliasing
            background = cv2.resize(bg, None, fx=640 / bg.shape[1], fy=480 / bg.shape[0])

            new_img = cv2.resize(img, None, fx=1/2, fy=1/2)
            foreground_mask = self.process_background(new_img)
            foreground_mask = cv2.resize(foreground_mask, None, fx=2, fy=2)

            # Apply the background image
            output_image = cv2.bitwise_and(img, img, mask=foreground_mask)
            background = cv2.bitwise_and(background, background, mask=1 - foreground_mask)
            output_image = cv2.bitwise_or(output_image, background)

            hand_result = self.hand_model(img)
            pts = hand_result.pandas().xyxy[0]

            # Get the points for the hand
            hand_box_points = [
                [
                    int(point[1]['xmin']), int(point[1]['ymin']),
                    int(point[1]['xmax']), int(point[1]['ymax'])
                ]
                for point in pts.iterrows() if point[1]['confidence'] >= 0.5
            ]

            if hand_box_points:  # != []
                # Find the center of the hand
                translation = np.array([
                    (hand_box_points[0][0] + hand_box_points[0][2]) / 2,
                    (hand_box_points[0][1] + hand_box_points[0][3]) / 2
                ]) - np.array([0, 75])
                scale = 3 / 8 * (hand_box_points[0][2] - hand_box_points[0][0])

                show_image_pts = (scale * points + translation).astype(int)

                for line in indices:
                    cv2.line(output_image, show_image_pts[line[0]], show_image_pts[line[1]], [0, 255, 0], 1)

                    if DEBUG:
                        cv2.rectangle(output_image, show_image_pts[0], show_image_pts[1], (0, 255, 0), 5)

            video_writer.write(output_image)


def main():
    recorder = Record()
    t1 = Thread(target=recorder.find_foreground)
    t2 = Thread(target=recorder.find_hands)

    t1.start()
    t2.start()

    recorder.run()
    recorder.post_process()

if __name__ == '__main__':
    main()


