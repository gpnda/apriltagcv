# OpenCV AprilTag minimal example

Итак, первый день знакомства с OpenCV.
День приятных открытий.
**Открытие 1:** Легкая в использовани, простая и приятная библиотека apriltags. Вообще ничего не надо делать, только написать вызов метода. Правда под виндой пришлось использовать какой-то форк.

**Открытие 2.** cv2.solvePnP() Оказывается можно отдать в метод массив с 3D точками и массив с их проекциями на плоскость, и без программирования и танцев с бубнами, получить трансформационные вектора.

**Открытие 3.** cv2.projectPoints() Можно отдать коодинаты в виде (0.5, 0.5, 0.5) в специальный магический метод, вместе с трансформационными векторами, и на выходе получить проекцию этой точки на 2D плоскость. solvePnP()+projectPoints() = <3
