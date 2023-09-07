import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import numpy as np
import cv2

# Load a pre-trained DeepLabV3 models with a ResNet-50 backbone
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
model.eval()

cap = cv2.VideoCapture(0)
while True:
    _, input_image = cap.read()

    new_image = cv2.resize(input_image, None, fx=0.25, fy=0.25)

    # Apply transformations to the input image
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(new_image).unsqueeze(0)

    # Make predictions on the input image
    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    # Apply thresholding to create a binary mask
    threshold = 0.5
    mask = (output.argmax(0) == 15).float()  # Assuming 15 is the class ID for the object of interest

    # Convert the mask to a NumPy array
    mask_np = mask.numpy().astype(np.uint8)

    mask_np = (cv2.resize(mask_np, None, fx=4, fy=4) != 0).astype(np.uint8)

    background = np.zeros(input_image.shape, np.uint8)
    background[:, :, 0] = 255
    background[:, :, 1] = 128

    output_image = cv2.bitwise_and(input_image, input_image, mask=mask_np)
    background = cv2.bitwise_and(background, background, mask=1 - mask_np)
    output_image = cv2.bitwise_or(output_image, background)

    # Save the resulting image with the background removed
    cv2.imshow("img", output_image)

    if cv2.waitKey(1) == ord('q'):
        break
