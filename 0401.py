import cv2 as cv # OpenCV 라이브러리 임포트
import matplotlib.pyplot as plt # 시각화 라이브러리 임포트

# 1. 이미지 로드 및 그레이스케일 변환
img = cv.imread('mot_color70.jpg') # 이미지 읽기
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # SIFT 처리를 위해 흑백 변환

# 2. SIFT 객체 생성 (nfeatures로 특징점 개수 제한 가능)
sift = cv.SIFT_create(nfeatures=500) # 특징점 최대 개수를 500개로 제한하여 객체 생성

# 3. 특징점 검출 및 기술자 계산
kp, des = sift.detectAndCompute(gray, None) # 이미지에서 특징점(kp)과 기술자(des)를 추출

# 4. 특징점 시각화 (방향과 크기 표시)
img_kp = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 특징점의 크기와 방향을 포함해 그리기

# 5. 결과 출력
plt.figure(figsize=(12,6)) # 출력창 크기 설정
plt.subplot(1, 2, 1), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('Original') # 원본 출력
plt.subplot(1, 2, 2), plt.imshow(cv.cvtColor(img_kp, cv.COLOR_BGR2RGB)), plt.title('SIFT Keypoints') # 특징점 출력
plt.show() # 시각화