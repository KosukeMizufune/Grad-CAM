import tempfile

import cv2
import requests
from chainercv import transforms


def imread_web(url):
    # 画像をリクエストする
    res = requests.get(url)
    img = None
    # Tempfileを作成して即読み込む
    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
    return img


def transform(inputs, mean, std, output_size=(224, 224)):
    x, lab = inputs
    x = x.copy()
    x -= mean[:, None, None]
    x /= std[:, None, None]
    x = x[::-1]
    x = transforms.resize(x, output_size)
    return x, lab
