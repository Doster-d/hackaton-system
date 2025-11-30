# 🚂 Industrial Activity Analytics

> Компьютерное зрение, которое превращает видеопотоки с депо в управляемые метрики: headcount, простои, нарушения и экономический эффект.

---

## 📌 Быстрые ссылки

- 📄 **Business case** – [`evidence/business_case.md`](evidence/business_case.md)
- 🎨 **Дашборд и визуальные приемы** – [`evidence/dashboard_showcase.md`](evidence/dashboard_showcase.md)
- 🧭 **План презентации** – [`evidence/presentation_plan.md`](evidence/presentation_plan.md)
- ✅ **Чек-лист критериев** – [`evidence/criteria.md`](evidence/criteria.md)

---

## 👥 Команда и роли

| Участник | Роль и вклад |
| --- | --- |
| **Алексеев Дмитрий Александрович (ИКБО-10-22)** | Построил модуль извлечения контекста с видео: компьютерное зрение + VideoLLaMA3, выделение действий сотрудников и объектов заказчика. |
| **Ушаков Иван Алексеевич (ИВБО-08-22)** | Обновил UI, добавил загрузку/обработку через Streamlit, вынес настройки в общий конфиг, расширил `video_processor.py` (детекция людей, поездов/номеров, логирование, превью, patch inference, ReID), расширил README и сценарии запуска (CLI + Streamlit). |
| **Бондаренко Вячеслав Васильевич (КМБО-03-23)** | Проработал архитектуру и структуру проекта, отвечал за разработку дашборда. |
| **Набиев Рашидхон Равшанович (ИКМО-04-25)** | Исследовательская работа, проверка гипотез и моделей, расчёт экономического эффекта, подготовка презентации. |
| **Дюкина Элина Маратовна (УИБО-03-22)** | Участвовала в дизайне и разработке дашборда, подготовила экономическое обоснование и фирменный стиль. |

---

## 🧠 Архитектура решения

```
📹 Видео или RTSP поток
   │  загрузка через UI/CLI
   ▼
🧮 VideoProcessor (YOLO + ByteTrack + ReID + VideoLLaMA3)
   │  сохраняет треки, эпизоды и метаданные
   ▼
🗃️ SQLite/PostgreSQL (SQLAlchemy модели)
   │
   ├─ CLI (Typer) — подготовка БД, сидинг, обработка партий
   └─ 📊 Streamlit Dashboard — визуализация headcount, активностей, нарушений
```

Разработка разделена на три слоя:

1. **pipeline** – модуль `VideoProcessor` отвечает за детекцию, трекинг, действия и запись в БД.
2. **db** – модели SQLAlchemy, сидеры демо-данных, миграции.
3. **dashboard** – Streamlit UI с переключением тем, фильтрами и загрузкой новых роликов.

---

## 🔁 Подробный пайплайн

| Шаг | Что происходит | Ключевые файлы/переменные |
| --- | --- | --- |
| 1. Ingest | `cli.py process-video` или кнопка в Streamlit сохраняет ролик в `video/`. | `hackaton_system/dashboard/app.py`, `video_processor.py` |
| 2. Препроцессинг | Опциональное шумоподавление/контраст, разморозка кадров. | `Settings.preprocess_*` |
| 3. Детекция + трекинг | Ultralytics YOLO (patch inference по необходимости) + ByteTrack. | `Settings.detection_*`, `use_patch_inference`, `tracker_*` |
| 4. ReID | Torchreid/ResNet18 для сшивки ID и роли по зонам. | `reid_*`, `role_assignment_threshold`, `zones` |
| 5. Поезда и OCR | Детекция поездов (COCO class 6), распознавание номеров EasyOCR. | `enable_train_detection`, `train_number_*`, `train_event_output_dir` |
| 6. Аналитика действий | Heuristic скорости или VideoLLaMA3 (промпты). | `action_backend`, `action_prompts`, `class_prompts` |
| 7. Экспорт | Склеиваем эпизоды, headcount, нарушения → таблицы `videos`, `persons`, `activities`, `detections`, `pose_keypoints`. | `hackaton_system/db/*` |
| 8. Визуализация | Дашборд читает данные, строит графики (headcount, bar, heatmap, timeline, таблицы) и показывает превью. | `hackaton_system/dashboard/*` |

## 🗄️ Структура хранилища

```sql
videos(video_id, filename, fps, duration_sec)
persons(person_id, video_id, track_id, person_type)
detections(id, video_id, person_id, frame_id, time_sec, x_min, y_min, x_max, y_max, confidence)
activities(id, video_id, person_id, activity_class, t_start_sec, t_end_sec, activity_conf)
pose_keypoints(id, video_id, person_id, frame_id, time_sec, keypoints, pose_conf)
```

- `videos` — реестр роликов и базовые параметры.
- `persons` — агрегированные треки (track_id) с ролью.
- `detections` — сырые рамки (используются для отладки и визуализаций).
- `activities` — интервалы активности для headcount/теплокарты.
- `pose_keypoints` — опционально: хранение ключевых точек для пост-аналитики.
- при включённой детекции поездов создаём JSON в `runs/train_events`.

---

## 🧱 Проектная структура

```
hackaton_system/
├── config.py              # Pydantic Settings + ENV
├── cli.py                 # Typer-команды (seed, process-video…)
├── pipeline/
│   ├── video_processor.py # основной пайплайн CV
│   └── videollama3.py     # zero-shot распознавание действий
├── db/
│   ├── models.py          # SQLAlchemy
│   ├── seed_demo.py       # генерация демо-данных
│   └── __init__.py        # init_db, get_session
└── dashboard/
    ├── app.py             # Streamlit UI + сценарии
    ├── data_access.py     # слои запросов в БД
    └── theme.py           # палитры и CSS для светлой/тёмной тем
```

---

## 📁 Структура репозитория

### Корень
- `CHANGES.md` — заметки о релизах и ключевых апдейтах.
- `compose.yaml` / `Dockerfile` — окружение для запуска Streamlit и пайплайна в контейнере.
- `Makefile` — вспомогательные команды (линт, формат, запуск контейнера).
- `pyproject.toml` + `uv.lock` — зависимости проекта и конфигурация `uv`.
- `README.md` — текущий документ.
- `evidence/` — бизнес-документы и материалы для презентации.
- `video/` — папка для загружаемых пользователем роликов.
- `runs/` — артефакты пайплайна (превью, json событий поездов и т.д.).
- `tests/` — заготовки для автотестов и unit-тестов пайплайна.
- `notebooks/` — эксперименты (EDA, прототипы моделей).

### `hackaton_system/`
- `cli.py` — команды Typer (`seed-demo`, `process-video`).
- `config.py` — Pydantic Settings и все `HACKATON_*` переменные.
- `models/` — вспомогательные датаклассы/модели.
- `pipeline/` — сердце CV-процессинга: `video_processor.py`, модули детекции/ролей/активностей.
- `db/` — ORM-модели, управление сессией и сидеры.
- `dashboard/` — Streamlit UI, темы и слой доступа к данным.

### `hackaton_system/pipeline/`
- `video_processor.py` — оркестратор (детекция → трекинг → действия → запись в БД).
- `detection.py`, `activity.py`, `roles.py`, `train_detection.py`, `reid.py`, `rendering.py` — специализированные блоки.
- `videollama3.py`, `yolo_classes.py` — поддержка zero-shot и словари классов.
- `utils.py` — утилиты (геометрия, работа с видео, общие функции).

### `hackaton_system/db/`
- `models.py` — определения таблиц `Video`, `Person`, `Detection`, `Activity`, `PoseKeypoints`.
- `session.py` — фабрика подключений/сессий.
- `seeder.py` — генерация демо-данных.
- `__init__.py` — экспорт `init_db`, `get_session`, моделей.

### `hackaton_system/dashboard/`
- `app.py` — основной Streamlit UI.
- `data_access.py` — функции загрузки агрегаций из БД.
- `theme.py` — палитры и CSS для светлой/тёмной тем.
- `__main__.py` — запуск дашборда как модуля.

---


## 💡 Идеи для развития

- Learning loop: собирать обратную связь от технологов, дообучать YOLO/VideoLLaMA3 на новых сменах.
- Аналитика смен: гибкие фильтры по сменам/участкам, экспорт отчётов (PDF/Excel).
- Интеграция с MES/ERP: автоматическое создание задач при простоях/нарушениях.
- Онлайн-мониторинг: адаптер для RTSP и оповещения в мессенджеры.
- Расширение моделей: детекция техники, анализ позы, прогноз headcount/простоев.
- API для партнёров: REST/gRPC сервис, который отдаёт headcount/инциденты без Streamlit.

---

## 📊 Dashboard и темы

- Сайдбар объединяет загрузку роликов, запуск пайплайна и выбор обработанных видео.
- Метрики/графики оформлены в фирменной палитре (см. `hackaton_system/dashboard/theme.py`), есть быстрый переключатель светлая ↔ тёмная.
- Визуализации: area chart headcount, bar активности, heatmap «роль × активность», таймлайн эпизодов, таблица с фильтрами.
- Раздел «Управление промптами» позволяет редактировать подсказки VideoLLaMA3 прямо из UI, настройки берутся из `.env`.
- Все стили вынесены в CSS-генератор, поэтому адаптация под другой бренд сводится к изменениям палитры.

---

## 🧰 Стек технологий

- `ultralytics` (YOLOv11) + ByteTrack/BOT-SORT.
- Torch, torchvision, torchreid (ReID), VideoLLaMA3 (HuggingFace) для понимания действий.
- OpenCV + ffmpeg для работы с кадрами и превью.
- SQLAlchemy + SQLite/PostgreSQL.
- Streamlit + Plotly + pandas для аналитики.
- Docker/Compose + `uv` для воспроизводимого окружения.

---

## ⚙️ Переменные окружения

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `HACKATON_DATABASE_URL` | `sqlite:///data/hackaton.db` | Хранилище данных (можно задать PostgreSQL). |
| `HACKATON_DATA_DIR` | `data/` | Папка артефактов и SQLite. |
| `HACKATON_DETECTION_MODEL_PATH` | `./yolo11x.pt` | Вес YOLO для людей/поездов. |
| `HACKATON_USE_PATCH_INFERENCE` | `false` | Детекция в патчах для высоких разрешений. |
| `HACKATON_TRACKER_TRACK_BUFFER` | `120` | Сколько кадров держать потерянный трек. |
| `HACKATON_ENABLE_REID` (`reid_enabled`) | `true` | Включает сшивку ID по эмбеддингам. |
| `HACKATON_ENABLE_VIDEO_RENDER` | `true` | Генерация mp4 превью (`runs/visualizations`). |
| `HACKATON_PREVIEW_FFMPEG_PATH` | `ffmpeg` | Путь к ffmpeg для сборки видео. |
| `HACKATON_ENABLE_TRAIN_DETECTION` | `true` | Выделяем приближение поездов и читаем номера. |
| `HACKATON_ACTION_BACKEND` | `heuristic` или `videollama3` | Метод распознавания активности. |
| `HACKATON_CLASS_PROMPTS` | пусто | JSON словарь промптов для VideoLLaMA3 (читается дашбордом). |

👉 Полный перечень смотрите в `hackaton_system/config.py` (докстринги описывают каждое поле).

---

## ▶️ Как запустить

### 1. Локально (uv)
```bash
uv sync
uv run python hackaton_system/cli.py seed-demo --force
uv run python hackaton_system/cli.py process-video video/factory.mp4
streamlit run hackaton_system/dashboard/app.py
```

### 2. Docker Compose
```bash
docker compose up dashboard
# pipeline/CLI внутри контейнера
docker compose run --rm processor uv run python hackaton_system/cli.py seed-demo --force
```
Приложение поднимется на `http://127.0.0.1:8501`.

### 3. Общий сценарий демо
1. Загрузите ролик в сайдбаре, нажмите «Запустить обработку».
2. Дождитесь уведомления, переключайтесь между светлой/тёмной темой.
3. Покажите headcount, матрицу «роль×активность», таймлайн и таблицу эпизодов.
4. Перейдите в раздел управления промптами и продемонстрируйте настройку VideoLLaMA3 без перезапуска backend-а.