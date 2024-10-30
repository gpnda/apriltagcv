import numpy as np
import cv2 as cv
import pupil_apriltags

thickness = 3
color = (0, 0, 255)

# Инициализация видеопотока
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = pupil_apriltags.Detector(families="tag36h11")

def initialize_camera_matrix(width, height):
    """Инициализация матрицы камеры."""
    return np.array([
        [1000, 0, width / 2],
        [0, 1000, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)

def main():
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    camera_matrix = initialize_camera_matrix(width, height)
    distortion_coefficients = np.zeros((4, 1), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        results = detector.detect(gray)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        if results:
            points = np.array(results[0].corners, dtype=np.float32)

            # 3D точки для куба (грань)
            object_points = np.array([
                [0, 0, 0],  # нижний левый
                [1, 0, 0],  # нижний правый
                [1, 1, 0],  # верхний правый
                [0, 1, 0],  # верхний левый
                [0, 0, 1],  # нижний левый (высота)
                [1, 0, 1],  # нижний правый (высота)
                [1, 1, 1],  # верхний правый (высота)
                [0, 1, 1]   # верхний левый (высота)
            ], dtype=np.float32)

            # Найти векторы вращения и трансляции
            _, rvecs, tvecs = cv.solvePnP(object_points[:4], points, camera_matrix, distortion_coefficients)

            # Проекция 3D точек на 2D
            points_2d, _ = cv.projectPoints(object_points, rvecs, tvecs, camera_matrix, distortion_coefficients)

            # Рисуем грани куба
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # нижняя грань
                [4, 5], [5, 6], [6, 7], [7, 4],  # верхняя грань
                [0, 4], [1, 5], [2, 6], [3, 7]   # вертикальные грани
            ]

            for edge in edges:
                start_point = tuple(map(int, points_2d[edge[0]][0]))
                end_point = tuple(map(int, points_2d[edge[1]][0]))
                cv.line(image, start_point, end_point, color, thickness)

        out = cv.addWeighted(frame, 1.0, image, 1.0, 0)
        cv.imshow('frame', out)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
