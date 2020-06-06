import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN

from mobilenetv3 import mobilenetv3_large


def detect_and_align_face(img):
    mtcnn = MTCNN(post_process=False, device="cpu")
    _, _, landmarks = mtcnn.detect(img, landmarks=True)
    assert len(landmarks), "no face detected"
    lmk = landmarks[0]
    # print(lmk)
    src = np.array([lmk[0], lmk[1]], dtype=np.float32)
    dst = np.array([(68, 112), (108, 112)], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    return cv2.warpAffine(
        img, M, (178, 218), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    net = mobilenetv3_large(num_classes=2)
    net.load_state_dict(
        torch.load("baldnet_large_0605.mdl", map_location=torch.device("cpu"))
    )
    net.eval()

    img = cv2.imread(args.image, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_show = detect_and_align_face(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    result = net(img)
    print("result:", result)
    print("秃" if result[0, 0] > result[0, 1] else "不秃")
    cv2.imshow("face", cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


main()
