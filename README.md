Am I bald?
==========

Test if you are bald with big data.

training code: [ShusenTang/Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)<br/>
dataset: [Bald Classification OR Detection 200K Images](https://www.kaggle.com/ashishjangra27/bald-classification-200k-images-celeba)<br/>
network: [d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)<br/>
landmark: [face-alignment](https://github.com/1adrianb/face-alignment)

Usage:

<img src="wyf.jpg" width="200"/>

```
$ python pred.py wyf.jpg
result: tensor([[ 1.0036, -1.0202]])
秃
```

<img src="wyf2.jpg" width="200"/>

```
$ python pred.py wyf2.jpg
result: tensor([[-6.0963,  6.3362]])
不秃
```
