import cv2 as cv # OpenCV 임포트
import matplotlib.pyplot as plt # 시각화 임포트

# 1. 두 이미지 로드
img1 = cv.imread('mot_color70.jpg') # 첫 번째 이미지
img2 = cv.imread('mot_color83.jpg') # 두 번째 이미지

# 2. SIFT 특징점 및 기술자 추출
sift = cv.SIFT_create() # SIFT 객체 생성
kp1, des1 = sift.detectAndCompute(img1, None) # 첫 번째 이미지 특징 추출
kp2, des2 = sift.detectAndCompute(img2, None) # 두 번째 이미지 특징 추출

# 3. BFMatcher 객체 생성 및 매칭 수행
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True) # L2 거리를 사용하고 서로 일치하는 점만 찾는 매처 생성
matches = bf.match(des1, des2) # 두 기술자 집합 간의 최적 매칭 수행

# 4. 거리에 따라 매칭 결과 정렬 (정확도 높은 순)
matches = sorted(matches, key=lambda x: x.distance) # 매칭 거리가 짧은(유사한) 순서대로 정렬

# 5. 매칭 결과 그리기 (상위 50개만)
img_match = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # 매칭 쌍 시각화

# 6. 결과 출력
plt.figure(figsize=(15,7)) # 출력 크기 설정
plt.imshow(img_match) # 매칭 결과 표시
plt.title('SIFT Matching (Top 50)') # 제목 설정
plt.show() # 출력