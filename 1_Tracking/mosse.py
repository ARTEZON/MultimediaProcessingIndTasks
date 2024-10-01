import numpy as np
import cv2


# Для просмотра float матриц надо поделить на 255
# cv2.imshow('Debug', init_frame / 255)


def linear_mapping(matrix):
    # Функция для линейного отображения матрицы
    # Каждый элемент матрицы переводится в диапазон [0, 1],
    # минимальное значение становится нулём, максимальное - единицей
    minimum, maximum = matrix.min(), matrix.max()
    return (matrix - minimum) / (maximum - minimum)


def pre_process(img):
    # Функция для предварительной обработки изображения
    height, width = img.shape
    # Логарифмирование пикселей (помогает в ситуациях с низким контрастом изображения)
    img = np.log(img + 1)  # логарифм нуля не определён, поэтому + 1
    # Значения пикселей нормализуются, чтобы иметь среднее значение 0 и норму 1
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # Умножаем на окно Хэннинга для минимизации краевых эффектов
    window = window_func_2d(height, width)
    img = img * window
    return img


def window_func_2d(height, width):
    # Функция для создания маски из двумерного окна Хэннинга
    # Создаём одномерные окна Ханнинга для строк и столбцов
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    # Создаём двумерную маску
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    win = mask_col * mask_row
    return win


class Mosse:
    def __init__(self):
        # Задаём темп обучения, параметр сигма и количество шагов предобучения
        self.learning_rate = 0.125  # темп обучения
        self.sigma = 200  # стандартное отклонение для гауссовского отклика
        self.num_pretrain = 128  # количество шагов предобучения

        # Инициализация переменных для хранения состояния трекера
        self.init_bbox_size = None  # размер исходной области интереса
        self.pos = None             # прямоугольник объекта в текущем кадре
        self.clip_pos = None        # прямоугольник объекта в текущем кадре, не вылезающий за границы экрана
        self.Ai = None              # матрица, накапливающая значения для фильтра
        self.Bi = None              # вторая матрица, накапливающая значения для фильтра
        self.G = None               # гауссовский отклик в частотной области

    def init(self, init_frame, init_bbox):
        # Аргументы:
        # init_frame - кадр при инициализации трекинга
        # init_bbox - заданный вручную прямоугольник, содержащий объект (ground truth)

        # Преобразуем исходный кадр в градации серого
        init_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        # Преобразуем тип данных изображения в float32 для дальнейших вычислений
        init_frame = init_frame.astype(np.float32)
        # Получаем гауссовский отклик для исходного кадра и позиции объекта
        response_map = self._get_gauss_response(init_frame, init_bbox)
        # Вырезаем участок отклика, соответствующий объекту (размытая белая точка будет посередине)
        g = response_map[init_bbox[1]:init_bbox[1] + init_bbox[3], init_bbox[0]:init_bbox[0] + init_bbox[2]]
        # Вырезаем участок изображения с объектом
        fi = init_frame[init_bbox[1]:init_bbox[1] + init_bbox[3], init_bbox[0]:init_bbox[0] + init_bbox[2]]
        # Преобразуем гауссовский отклик в частотную область с помощью быстрого преобразования Фурье
        self.G = np.fft.fft2(g)
        # Выполняем предобучение трекера на первом кадре
        self.Ai, self.Bi = self._pre_training(fi, self.G)
        self.Ai = self.learning_rate * self.Ai
        self.Bi = self.learning_rate * self.Bi
        # Сохраняем размер исходного ограничивающего прямоугольника
        self.init_bbox_size = (init_bbox[2], init_bbox[3])
        # Сохраняем начальную позицию объекта
        self.pos = list(init_bbox)
        self.clip_pos = np.array([self.pos[0], self.pos[1], self.pos[0] + self.pos[2], self.pos[1] + self.pos[3]])

    def update(self, current_frame):
        # Аргументы:
        # current_frame - текущий кадр

        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_frame = current_frame.astype(np.float32)

        # Рассчитываем MOSSE-фильтр Hi как отношение коэффициентов Ai и Bi
        Hi = self.Ai / self.Bi
        # Вырезаем участок изображения с объектом
        fi = current_frame[self.clip_pos[1]:self.clip_pos[3], self.clip_pos[0]:self.clip_pos[2]]
        if fi.size == 0:
            return False, None
        # Меняем размер участка изображения и применяем предварительную обработку
        fi = pre_process(cv2.resize(fi, self.init_bbox_size))
        # Применяем фильтр Hi к преобразованному в частотную область изображению fi
        Gi = Hi * np.fft.fft2(fi)  # по формуле
        # Преобразуем отклик из частотной области обратно в пространственную
        gi = linear_mapping(np.fft.ifft2(Gi))  # обратное дискретное преобразование Фурье
        # Максимум отклика gi указывает на вероятное местоположение объекта
        max_value = np.max(gi)
        max_pos = np.where(gi == max_value)
        # Смещение объекта
        dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

        # Обновление положения объекта
        self.pos[0] = self.pos[0] + dx
        self.pos[1] = self.pos[1] + dy
        # Корректируем координаты объекта, чтобы они не выходили за границы изображения
        self.clip_pos[0] = np.clip(self.pos[0], 0, current_frame.shape[1])
        self.clip_pos[1] = np.clip(self.pos[1], 0, current_frame.shape[0])
        self.clip_pos[2] = np.clip(self.pos[0] + self.pos[2], 0, current_frame.shape[1])
        self.clip_pos[3] = np.clip(self.pos[1] + self.pos[3], 0, current_frame.shape[0])

        # Вырезаем участок изображения из нового прямоугольника
        fi = current_frame[self.clip_pos[1]:self.clip_pos[3], self.clip_pos[0]:self.clip_pos[2]]
        if fi.size == 0:
            return False, None
        fi = pre_process(cv2.resize(fi, self.init_bbox_size))
        # Обновляем матрицы Ai и Bi
        self.Ai = self.learning_rate * (self.G * np.conjugate(np.fft.fft2(fi))) + (1 - self.learning_rate) * self.Ai
        self.Bi = self.learning_rate * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.learning_rate) * self.Bi

        return True, self.pos

    def _pre_training(self, init_frame_region, G):
        # Функция для предобучения фильтра на первом кадре
        # G - гауссовский отклик в частотной области
        fi = pre_process(init_frame_region)  # предобработка
        # Вычисляем начальные значения для Ai и Bi
        # Функция np.conjugate возвращает комплексно-сопряжённое значение для каждого элемента массива
        # Комплексно-сопряжённое число получается изменением знака его мнимой части
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame_region) * np.conjugate(np.fft.fft2(init_frame_region))
        for _ in range(self.num_pretrain):
            Ai += G * np.conjugate(np.fft.fft2(fi))
            Bi += np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        return Ai, Bi

    def _get_gauss_response(self, img, ground_truth_bbox):
        # Функция для генерации гауссовского отклика
        # Получаем размеры изображения
        height, width = img.shape
        # Создаём сетку координат
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # Находим центр объекта
        center_x = ground_truth_bbox[0] + 0.5 * ground_truth_bbox[2]
        center_y = ground_truth_bbox[1] + 0.5 * ground_truth_bbox[3]
        # Вычисляем квадрат расстояния до центра для каждого пикселя
        dist = np.square(xx - center_x) + np.square(yy - center_y)
        # Генерируем гауссовский отклик
        response = np.exp(-dist / self.sigma)
        # Нормализуем его
        response = linear_mapping(response)
        return response
