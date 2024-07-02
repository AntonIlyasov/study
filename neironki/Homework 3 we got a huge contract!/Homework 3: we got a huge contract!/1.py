import cv2
import numpy as np

# Загрузите изображение полки и изображение коробки сока
shelf_image = cv2.imread('shelf.jpg')
juice_box_image = cv2.imread('juice_box.jpg')

# Преобразуйте изображения в оттенки серого
shelf_gray = cv2.cvtColor(shelf_image, cv2.COLOR_BGR2GRAY)
juice_box_gray = cv2.cvtColor(juice_box_image, cv2.COLOR_BGR2GRAY)

# Создайте объект SIFT
sift = cv2.SIFT_create()

# Найдите ключевые точки и дескрипторы с помощью SIFT
keypoints_shelf, descriptors_shelf = sift.detectAndCompute(shelf_gray, None)
keypoints_juice_box, descriptors_juice_box = sift.detectAndCompute(juice_box_gray, None)

# Создайте объект для сопоставления дескрипторов
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Сопоставьте дескрипторы
matches = bf.knnMatch(descriptors_juice_box, descriptors_shelf, k=2)

# Отсортируйте совпадения по расстоянию
# matches = sorted(matches, key=lambda x: x.distance)

# Нарисуйте первые 10 совпадений
# matched_image = cv2.drawMatches(juice_box_image, keypoints_juice_box, shelf_image, keypoints_shelf, matches[:10], shelf_image, flags=2)
# cv2.imshow('matched_image', matched_image)
# cv2.waitKey(0)

# Найдите гомографию между ключевыми точками
src_pts = np.float32([keypoints_juice_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_shelf[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Используйте RANSAC для вычисления гомографии
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Получите размеры изображения коробки сока
h, w = juice_box_image.shape[:2]

# Используйте гомографию для проецирования углов изображения коробки сока на изображение полки
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)

# Нарисуйте ограничивающую рамку вокруг коробки сока на изображении полки
shelf_image = cv2.polylines(shelf_image, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

# Покажите результат
cv2.imshow('Juice Box Detection', shelf_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

