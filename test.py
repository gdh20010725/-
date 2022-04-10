import numpy as np
import imageio
import cv2
def test(a):
    return (a > 0).astype(np.float32) * 3

gif_images = []
for i in range(0, 200):
    gif_images.append(imageio.imread(str(i+1)+".jpg"))   # 读取多张图片
imageio.mimsave("W.gif", gif_images, fps=5)