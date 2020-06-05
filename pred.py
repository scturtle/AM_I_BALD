import cv2
import face_alignment
import numpy as np
import torch
import torchvision.transforms as transforms

from mobilenetv3 import mobilenetv3_large


def detect_and_align_face(img):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')
    preds = fa.get_landmarks_from_image(img)
    assert preds, 'no face detected'
    pred = preds[0]
    src = np.array([pred[37], pred[43]], dtype=np.float32)
    dst = np.array([(64, 110), (106, 110)], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    return cv2.warpAffine(img, M, (178, 218), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()

    net = mobilenetv3_large(num_classes=2)
    net.load_state_dict(torch.load('baldnet_large_0605.mdl', map_location=torch.device('cpu')))
    net.eval()

    img = cv2.imread(args.image, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_show = detect_and_align_face(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    result = net(img)
    print('result:', result)
    print('秃' if result[0, 0] > result[0, 1] else '不秃')
    cv2.imshow('face', cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

main()
