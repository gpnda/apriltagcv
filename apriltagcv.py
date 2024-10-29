import numpy as np
import cv2
import pupil_apriltags


image = cv2.imread("./02.png")
color = (0,255,0)
thickness = 3


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# options = pupil_apriltags.DetectorOptions(families="tag36h11")  # Так опции задаются для библиотеки apriltags,
# но мы используем другую
detector = pupil_apriltags.Detector(families = "tag36h11")
results = detector.detect(gray)

print("[INFO] {} total AprilTags detected".format(len(results)))

# print(results)













# Здесь мы уже получили результаты (по 4 угла каждой метки apriltag, координаты на 2D изображении)
# Все метки лежат в массиве results

point1 = (int(results[0].corners[0][0]), int(results[0].corners[0][1]))
point2 = (int(results[0].corners[1][0]), int(results[0].corners[1][1]))
point3 = (int(results[0].corners[2][0]), int(results[0].corners[2][1]))
point4 = (int(results[0].corners[3][0]), int(results[0].corners[3][1]))

image = cv2.line(image, point1, point2, color, thickness)
image = cv2.line(image, point2, point3, color, thickness)
image = cv2.line(image, point3, point4, color, thickness)
image = cv2.line(image, point4, point1, color, thickness)



# Создаем 3D представление метки, ну это просто одна грань, дно куба в 3D координатах, с гранью = 1
object_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0]
], dtype=np.float32)
 

# Создаем объект камеры (по умолчанию)
# с этим не разобрался пока, но работает
camera_matrix = np.array([
    [1000, 0, 640 / 2],
    [0, 1000, 480 / 2],
    [0, 0, 1]
], dtype=np.float32)


# Создаем вектор искажений (по умолчанию)
# с этим не разобрался пока, но работает
distortion_coefficients = np.zeros((4, 1), dtype=np.float32)


# Find the rotation and translation vectors.
# что такое ret - не понял, но эта переменная дальше и не используется
# тут вобщем то и происходит основная магия решается уравнение, сопоставляя 4 точки в 2D с ними-же в 3D, мы получаем 
# трансформационные матрицы rvecs, tvecs. В дальнейшем с их помощью мы получим проекции любой точки 3D -> на 2D
ret, rvecs, tvecs = cv2.solvePnP(object_points, results[0].corners, camera_matrix, distortion_coefficients)

# project 3D points to image plane
# Вот эту одну точку для начала - проецируем на 2D, используя rvecs, tvecs
point_3d = np.array([0.5, 0.5, 0.5, 1]) # 3D-точка, только не понял зачем 4й элемент, все равно массив потом обрезается

# Далее опять магическая магия. Сказано: В этом примере мы проецируем только одну точку. Если вам нужно проецировать 
# несколько точек, просто передайте массив 3D-точек в cv2.projectPoints()
point_2d, _ = cv2.projectPoints(
    point_3d[None, :3], 
    rvecs, 
    tvecs, 
    camera_matrix, 
    distortion_coefficients
)
point_2d = point_2d[0][0] # Извлекаем координаты 2D-точки

# Вывод результата
print(f"Матрица трансформации:\n{tvecs}")
print(f"Проекция 3D-точки на 2D:\n{point_2d}")

# превратим список 2D float координат в кортеж (x:int, y:int)
point_center = (int(point_2d[0]), int(point_2d[1]))

image = cv2.line(image, point1, point_center, color, thickness)
image = cv2.line(image, point2, point_center, color, thickness)
image = cv2.line(image, point3, point_center, color, thickness)
image = cv2.line(image, point4, point_center, color, thickness)


# все на том-же канвасе рисуем
cv2.imshow('image',image)
k = cv2.waitKey(0)


