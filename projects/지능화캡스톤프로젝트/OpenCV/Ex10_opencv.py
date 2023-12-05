###############################################################################
# OpenCV (1) - 이미지 파일 불러오기
###############################################################################
"""
import time
import cv2
import matplotlib.pyplot as plt

# Color image
img_bgr = cv2.imread('cat.bmp')     # cv2.IMREAD_COLOR (default)
type(img_bgr)       # <class 'numpy.ndarray'>
img_bgr.dtype       # dtype('uint8')
img_bgr.shape       # (480, 640, 3)
len(img_bgr.shape)  # 3


# Gray image
img_gray = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)
type(img_gray)       # <class 'numpy.ndarray'>
img_gray.dtype       # dtype('uint8')
img_gray.shape       # (480, 640)
len(img_gray.shape)  # 2

h, w = img_bgr.shape[:2]    # h = 480, w = 640

t0 = time.time()
for j in range(h):
    for i in range(w):
        img_gray[j, i] = 255            # intensity
        img_bgr[j, i] = [0, 0, 255]   # [B, G, R]

# Vectorization : 140배 더 빠름 (1ms)
img_gray[:, :] = 255
img_bgr[:, :] = [0, 0, 255]

cv2.imshow('Color', img_bgr)
cv2.imshow('Gray', img_gray)
cv2.waitKey()
cv2.destroyAllWindows()


###############################################################################
# OpenCV (2) - Matplotlib에서 영상 출력
###############################################################################
import cv2
import matplotlib.pyplot as plt


img_bgr = cv2.imread('cat.bmp')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)

fig, ax = plt.subplots(3, 1, figsize=(4, 7))

# BGR image 출력
ax[0].imshow(img_bgr)
ax[0].axis('off')
ax[0].set_title('BGR image')

# RGB image 출력
ax[1].imshow(img_rgb)
ax[1].axis('off')
ax[1].set_title('RGB image')

# Grayscale image 출력
ax[2].imshow(img_gray, cmap='gray')
ax[2].axis('off')
ax[2].set_title('Gray image')

plt.tight_layout()
plt.show()

# Gray 영상 출력
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.show()


###############################################################################
# OpenCV (3) - 영상 초기화
###############################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = np.empty((480, 640, 3), np.uint8)                    # random image
img2 = np.zeros((480, 640, 3), np.uint8)                    # black image
img3 = np.ones((480, 640, 3), np.uint8) * 255               # white image
img4 = np.full((480, 640, 3), (255, 255, 0), np.uint8)      # color image
imgs = [img1, img2, img3, img4]

fig, axes = plt.subplots(1, 4, figsize=(10, 4))

for i, (ax, img) in enumerate(zip(axes, imgs)):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('img' + str(i+1))

plt.show()


###############################################################################
# OpenCV (4) - 부분 영상 추출
###############################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('cat.bmp')

img_assign = img        # shallow copy (reference) : 이름만 다를 뿐, 같은 변수
img_copy = img.copy()   # deep copy : 완전히 다른 변수

img[100:200, 200:300] = 0                   # black
img_assign[200:300, 300:400] = 255          # white
img_copy[300:400, 400:500] = [0, 0, 255]    # red

id(img)             # 1557621973392 (원본)
id(img_assign)      # 1557621973392 (원본과 동일)
id(img_copy)        # 1557606882192

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

imgs = [img, img_assign, img_copy]
imgs_title = ['img', 'img_assign', 'img_copy']
for i, (ax, img) in enumerate(zip(axes, imgs)):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(imgs_title[i])
plt.show()


###############################################################################
# OpenCV (5) - 그리기 함수
###############################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt


# color 설정
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

img = np.full((500, 700, 3), 255, np.uint8)

cv2.line(img, (50, 50), (200, 50), RED, 5)          # red
cv2.line(img, (50, 60), (150, 160), BLUE, 2)        # red

cv2.rectangle(img, (50, 200, 150, 100), GREEN, 2)
cv2.rectangle(img, (70, 220), (180, 280), BLUE, -1)     # 두께가 음수이면 내부를 채움

cv2.circle(img, (300, 100), 60, BLUE, 3, cv2.LINE_AA)
cv2.circle(img, (300, 100), 30, GREEN, -1, cv2.LINE_AA)

cv2.ellipse(img, (550, 100), (100, 60), 0, 0, 360, GREEN, 2)
cv2.ellipse(img, (550, 100), (100, 60), 30, 0, 360, RED, 2)
cv2.ellipse(img, (550, 300), (100, 60), 30, 0, 270, BLUE, -1)

pts = np.array([[250, 200], [300, 200], [350, 300], [250, 300]])
cv2.polylines(img, [pts], True, RED, 2)

text = 'Industrial AI (INDAI) Research Center'
cv2.putText(img, text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(img, text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_4)


cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()
"""
###############################################################################
# OpenCV (6) - 동영상 읽고 쓰기
###############################################################################
import sys
import time
import cv2


cap = cv2.VideoCapture(0)       # 기본 카메라 장치 열기
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'divx')    # *'DIVX' == 'D', 'I', 'V', 'X'
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('동영상 재생 종료')
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)  # 대칭 : 0(상하), 1(좌우)
    text = f'{frame_count} frame'
    cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('Webcam: {} x {} x {}fps'.format(w, h, fps), frame)

    inversed_img = ~frame       # 반전 -> 저장
    out.write(inversed_img)

    key = cv2.waitKey(1)
    if key == 27 or frame_count == 300:   # ESC를 누르면 while 루프 종료
        time.sleep(3)
        break

cap.release()
out.release()

# 저장한 동영상 불러오기
cap = cv2.VideoCapture('output.avi')
if not cap.isOpened():
    sys.exit('동영상 읽기 실패')

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(cap.get(cv2.CAP_PROP_FPS))
delay = int(1000 / fps)
total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득 실패')
        break

    text = '{}/{} frames'.format(round(cap.get(cv2.CAP_PROP_POS_FRAMES)), total_frames)
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Video: {} x {} x {}fps'.format(w, h, fps), frame)

    key = cv2.waitKey(delay)
    if key == ord('s'):         # 잠시 멈춤
        cv2.waitKey()

cap.release()
time.sleep(3)
cv2.destroyAllWindows()

"""
###############################################################################
# OpenCV (7) - 히스토그램 그리기
###############################################################################
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

img = cv2.imread('lenna_color.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
colors = ['k:', 'r', 'g', 'b']
imgs = [img_gray, img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]]

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 2)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(img_gray, cmap='gray')
ax.set_title('Grayscale')
ax.axis('off')

ax = fig.add_subplot(gs[0, 1])
ax.imshow(img_rgb)
ax.set_title('Color')
ax.axis('off')

ax = fig.add_subplot(gs[1, :])
for p, c in zip(imgs, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])
    ax.plot(hist, c, label=c)
ax.legend()
ax.set_title('Histogram')
ax.set_xlabel('Pixel value')

plt.show()

###############################################################################
# OpenCV (8) - 이진화 + 트랙바
###############################################################################
import cv2


img = cv2.imread('cells.png', cv2.IMREAD_GRAYSCALE)


# 트랙바 콜백함수
def on_change(pos):
    _, dst = cv2.threshold(img, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binarization', dst)


cv2.namedWindow('Binarization')
cv2.createTrackbar('Threshold', 'Binarization', 0, 255, on_change)      # 트랙바 생성: 초기값 0
th_ostu, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu: 206
cv2.setTrackbarPos('Threshold', 'Binarization', round(th_ostu))         # 트랙바 초기값 설정


cv2.waitKey()
cv2.destroyAllWindows()


###############################################################################
# OpenCV (9) - 에지 검출 + 트랙바
###############################################################################
import cv2


low_thresh = 50
high_thresh = 150

img = cv2.imread('lenna_color.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('lenna_color.png')


def on_change_low(pos):
    dst = cv2.Canny(img, pos, high_thresh)
    cv2.imshow('Canny', dst)


def on_change_high(pos):
    dst = cv2.Canny(img, low_thresh, pos)
    cv2.imshow('Canny', dst)


cv2.namedWindow('Canny')
cv2.createTrackbar('Low_th', 'Canny', 0, 255, on_change_low)
cv2.createTrackbar('High_th', 'Canny', 0, 255, on_change_high)
cv2.setTrackbarPos('Low_th', 'Canny', low_thresh)
cv2.setTrackbarPos('High_th', 'Canny', high_thresh)


cv2.waitKey()
cv2.destroyAllWindows()

"""
