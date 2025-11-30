"""Названия классов COCO и утилиты для работы с ними."""

# Названия классов датасета YOLO COCO (80 штук).
YOLO_COCO_CLASS_NAMES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

# Соответствие ID → названию.
CLASS_ID_TO_NAME: dict[int, str] = dict(enumerate(YOLO_COCO_CLASS_NAMES))

# Обратная карта: название → ID.
CLASS_NAME_TO_ID: dict[str, int] = {
    name: i for i, name in enumerate(YOLO_COCO_CLASS_NAMES)
}

# Русские переводы названий классов.
CLASS_NAMES_RU: dict[str, str] = {
    'person': 'Человек',
    'bicycle': 'Велосипед',
    'car': 'Автомобиль',
    'motorcycle': 'Мотоцикл',
    'airplane': 'Самолет',
    'bus': 'Автобус',
    'train': 'Поезд',
    'truck': 'Грузовик',
    'boat': 'Лодка',
    'traffic light': 'Светофор',
    'fire hydrant': 'Пожарный гидрант',
    'stop sign': 'Стоп-знак',
    'parking meter': 'Парковочный счетчик',
    'bench': 'Скамейка',
    'bird': 'Птица',
    'cat': 'Кошка',
    'dog': 'Собака',
    'horse': 'Лошадь',
    'sheep': 'Овца',
    'cow': 'Корова',
    'elephant': 'Слон',
    'bear': 'Медведь',
    'zebra': 'Зебра',
    'giraffe': 'Жираф',
    'backpack': 'Рюкзак',
    'umbrella': 'Зонт',
    'handbag': 'Сумка',
    'tie': 'Галстук',
    'suitcase': 'Чемодан',
    'frisbee': 'Фрисби',
    'skis': 'Лыжи',
    'snowboard': 'Сноуборд',
    'sports ball': 'Спортивный мяч',
    'kite': 'Воздушный змей',
    'baseball bat': 'Бейсбольная бита',
    'baseball glove': 'Бейсбольная перчатка',
    'skateboard': 'Скейтборд',
    'surfboard': 'Серфборд',
    'tennis racket': 'Теннисная ракетка',
    'bottle': 'Бутылка',
    'wine glass': 'Бокал для вина',
    'cup': 'Чашка',
    'fork': 'Вилка',
    'knife': 'Нож',
    'spoon': 'Ложка',
    'bowl': 'Миска',
    'banana': 'Банан',
    'apple': 'Яблоко',
    'sandwich': 'Бутерброд',
    'orange': 'Апельсин',
    'broccoli': 'Брокколи',
    'carrot': 'Морковь',
    'hot dog': 'Хот-дог',
    'pizza': 'Пицца',
    'donut': 'Пончик',
    'cake': 'Торт',
    'chair': 'Стул',
    'couch': 'Диван',
    'potted plant': 'Комнатное растение',
    'bed': 'Кровать',
    'dining table': 'Обеденный стол',
    'toilet': 'Унитаз',
    'tv': 'Телевизор',
    'laptop': 'Ноутбук',
    'mouse': 'Мышь',
    'remote': 'Пульт',
    'keyboard': 'Клавиатура',
    'cell phone': 'Мобильный телефон',
    'microwave': 'Микроволновка',
    'oven': 'Духовка',
    'toaster': 'Тостер',
    'sink': 'Раковина',
    'refrigerator': 'Холодильник',
    'book': 'Книга',
    'clock': 'Часы',
    'vase': 'Ваза',
    'scissors': 'Ножницы',
    'teddy bear': 'Плюшевый мишка',
    'hair drier': 'Фен',
    'toothbrush': 'Зубная щетка',
}


def get_class_name_ru(class_name: str) -> str:
    """Возвращает перевод названия класса на русский язык."""
    return CLASS_NAMES_RU.get(class_name.lower(), class_name)


def get_class_name(class_id: int) -> str:
    """Возвращает имя класса по его ID."""
    return CLASS_ID_TO_NAME.get(class_id, f'class_{class_id}')


def get_class_id(class_name: str) -> int | None:
    """Возвращает ID класса по имени (или None, если не найден)."""
    return CLASS_NAME_TO_ID.get(class_name.lower())


def normalize_class_key(class_key: str | int) -> str:
    """Нормализует ключ класса в строку (имя или ID)."""
    if isinstance(class_key, int):
        return str(class_key)
    # Пробуем конвертировать строку в число, чтобы найти имя класса.
    try:
        class_id = int(class_key)
        return get_class_name(class_id)
    except (ValueError, TypeError):
        # Use as-is (assume it's a class name)
        return class_key.lower()
