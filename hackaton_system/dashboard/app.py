from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, TypedDict, cast

import pandas as pd
import plotly.express as _px  # type: ignore[import]
import streamlit as _st  # type: ignore[import]

from hackaton_system.config import get_settings
from hackaton_system.dashboard.data_access import (
    delete_video_and_related,
    list_available_videos,
    load_activity_summary,
    load_headcount,
    load_person_episodes,
    load_role_activity_matrix,
)
from hackaton_system.dashboard.theme import THEME_PALETTES, build_theme_css
from hackaton_system.pipeline import VideoProcessor
from hackaton_system.pipeline.yolo_classes import (
    CLASS_ID_TO_NAME,
    YOLO_COCO_CLASS_NAMES,
    get_class_name_ru,
    normalize_class_key,
)

px = cast(Any, _px)
st = cast(Any, _st)
settings = get_settings()
PREVIEW_DIR = getattr(settings, "video_output_dir", Path("runs/visualizations"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)


def _process_video_file(target_path: Path) -> None:
    """Запускает VideoProcessor для файла и выводит статус в сайдбар."""
    status_box = st.sidebar.empty()
    try:
        with status_box, st.spinner("Запускаем пайплайн..."):
            processor = VideoProcessor()
            video_id = processor.process_video(target_path)
        preview_candidate = PREVIEW_DIR / f"{video_id}_{target_path.stem}.mp4"
        if preview_candidate.exists():
            status_box.success(
                f"Готово! video_id={video_id}. Визуализация: {preview_candidate.name}"
            )
        else:
            status_box.warning(
                f"Готово! video_id={video_id}, но превью не найдено. Проверьте журналы консоли."
            )
        st.rerun()
    except Exception as exc:  # pragma: no cover - интерактивная ошибка
        status_box.error(f"Ошибка обработки: {exc}")


class VideoRecord(TypedDict):
    """Описание видеофайла, отображаемого в интерфейсе."""
    id: int | None
    filename: str
    duration_sec: float
    fps: float


st.set_page_config(
    page_title="Industrial Activity Analytics",
    page_icon="🛠️",
    layout="wide",
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

st.sidebar.markdown("### Настройки отображения")
use_dark_theme = st.sidebar.toggle(
    "Тёмная тема", value=st.session_state["theme"] == "dark"
)
st.session_state["theme"] = "dark" if use_dark_theme else "light"
active_theme = THEME_PALETTES[st.session_state["theme"]]
st.sidebar.divider()

CUSTOM_CSS = build_theme_css(active_theme)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar — сначала даём возможность запустить обработку, потом выбор готового ролика
RAW_VIDEO_DIR = Path("video")
RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.sidebar.header("Обработка видео")
uploaded_video = st.sidebar.file_uploader(
    "Загрузите файл", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False
)
process_clicked = st.sidebar.button("Запустить обработку", width="stretch")
if process_clicked:
    if uploaded_video is None:
        st.sidebar.warning("Сначала загрузите файл, затем запускайте обработку.")
    else:
        target_path = RAW_VIDEO_DIR / uploaded_video.name
        with open(target_path, "wb") as dst:
            dst.write(uploaded_video.getbuffer())
        _process_video_file(target_path)

st.sidebar.header("Файлы на сервере")
raw_files = sorted(
    [p for p in RAW_VIDEO_DIR.glob("*") if p.is_file()],
    key=lambda p: p.name.lower(),
)
if raw_files:
    raw_names = [file.name for file in raw_files]
    selected_raw = st.sidebar.selectbox("Загруженные файлы", raw_names)
    selected_raw_path = RAW_VIDEO_DIR / selected_raw
    size_mb = selected_raw_path.stat().st_size / (1024 * 1024)
    st.sidebar.caption(f"Размер: {size_mb:.2f} MB")
    if st.sidebar.button("Обработать выбранный файл", key="process_existing_raw"):
        _process_video_file(selected_raw_path)
    if st.sidebar.button("Удалить файл", key="delete_raw_file"):
        try:
            selected_raw_path.unlink()
            st.sidebar.success("Файл удалён.")
            st.rerun()
        except Exception as exc:  # pragma: no cover - UI feedback
            st.sidebar.error(f"Не удалось удалить файл: {exc}")
else:
    st.sidebar.info("Нет загруженных роликов. Добавьте файл через форму выше.")

st.sidebar.header("Просмотр результатов")
videos_df: pd.DataFrame = list_available_videos()
video_records: List[VideoRecord] = cast(
    List[VideoRecord],
    cast(Any, videos_df).to_dict(orient="records"),
)
video_options: Dict[str, int | None] = {
    record["filename"]: record.get("id") for record in video_records
}
selected_filename: str = st.sidebar.selectbox(
    "Выберите ролик", list(video_options.keys())
)
selected_video_id: int | None = video_options[selected_filename]

selected_meta = next(
    (record for record in video_records if record["filename"] == selected_filename),
    None,
)
if selected_meta:
    minutes = selected_meta["duration_sec"] / 60
    st.sidebar.metric("Длительность", f"{minutes:.1f} мин")
    st.sidebar.metric("FPS", f"{selected_meta['fps']:.1f}")
    if selected_video_id is not None and st.sidebar.button(
        "Удалить обработанные данные", type="secondary"
    ):
        removed = delete_video_and_related(selected_video_id)
        if removed:
            st.sidebar.success("Видео и связанные данные удалены.")
        else:
            st.sidebar.warning("Запись не найдена или уже удалена.")
        st.rerun()
else:
    st.sidebar.info("Пока нет обработанных видео — отображаются демо-данные.")


def _render_metric(title: str, value: str) -> None:
    """Рисует карточку с показателем в соответствии с активной темой."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>🏭 Панель мониторинга активности</h1>
        <p>Аналитика по людям, ролям и действиям на производственной площадке.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

headcount_df = load_headcount(selected_video_id)
activity_summary = load_activity_summary(selected_video_id)
role_matrix = load_role_activity_matrix(selected_video_id)
episodes_df = load_person_episodes(selected_video_id)

working_minutes = (
    float(
        activity_summary.loc[
            activity_summary["activity_class"] == "working", "duration_min"
        ].sum()
    )
    if not activity_summary.empty
    else 0.0
)
idle_minutes = (
    float(
        activity_summary.loc[
            activity_summary["activity_class"] == "idle_at_station", "duration_min"
        ].sum()
    )
    if not activity_summary.empty
    else 0.0
)
restricted_minutes = (
    float(
        activity_summary.loc[
            activity_summary["activity_class"] == "in_restricted_zone", "duration_min"
        ].sum()
    )
    if not activity_summary.empty
    else 0.0
)
unique_people = episodes_df["person_id"].nunique() if not episodes_df.empty else 0

col1, col2, col3 = st.columns(3)
with col1:
    _render_metric("Работа", f"{working_minutes:.1f} мин")
with col2:
    _render_metric("Простой", f"{idle_minutes:.1f} мин")
with col3:
    summary_caption = f"{unique_people} чел"
    if restricted_minutes > 0:
        summary_caption += f" · {restricted_minutes:.1f} мин в запрете"
    _render_metric("Всего людей / нарушения", summary_caption)

preview_path: Path | None = None
if selected_video_id:
    candidate = PREVIEW_DIR / f"{selected_video_id}_{Path(selected_filename).stem}.mp4"
    if candidate.exists():
        preview_path = candidate

st.markdown('<div class="section-title">Видео с разметкой</div>', unsafe_allow_html=True)
if preview_path and preview_path.exists():
    with open(preview_path, "rb") as preview_file:
        st.video(preview_file.read(), format="video/mp4")
else:
    st.info("Пока нет визуализации для этого ролика.")

# Раздел с графиками и визуализациями.
st.markdown(
    '<div class="section-title">Динамика людей на объекте</div>', unsafe_allow_html=True
)
if not headcount_df.empty:
    headcount_fig = px.area(
        headcount_df,
        x="time_sec",
        y="headcount",
        color_discrete_sequence=[active_theme["chart_primary"]],
        labels={"time_sec": "Секунды", "headcount": "Люди"},
    )
    headcount_fig.update_layout(
        template=active_theme["plotly_template"],
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            showgrid=False,
            linecolor=active_theme["muted_text"],
            color=active_theme["muted_text"],
        ),
        yaxis=dict(
            showgrid=False,
            linecolor=active_theme["muted_text"],
            color=active_theme["muted_text"],
        ),
        font=dict(color=active_theme["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(headcount_fig, width="stretch")
else:
    st.info("Нет данных по трекам — запустите пайплайн обработки.")

left, right = st.columns(2)

with left:
    st.markdown(
        '<div class="section-title">Время по активностям</div>', unsafe_allow_html=True
    )
    if not activity_summary.empty:
        duration_fig = px.bar(
            activity_summary,
            x="activity_class",
            y="duration_min",
            color="activity_class",
            text_auto=".1f",
            color_discrete_sequence=active_theme["chart_sequence"],
            labels={"activity_class": "Активность", "duration_min": "Минуты"},
        )
        duration_fig.update_layout(
            template=active_theme["plotly_template"],
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(color=active_theme["text"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        duration_fig.update_xaxes(
            color=active_theme["muted_text"], linecolor=active_theme["muted_text"]
        )
        duration_fig.update_yaxes(
            color=active_theme["muted_text"], linecolor=active_theme["muted_text"]
        )
        st.plotly_chart(duration_fig, width="stretch")
    else:
        st.info("Добавьте активности, чтобы увидеть распределение времени.")

with right:
    st.markdown(
        '<div class="section-title">Матрица “роль × активность”</div>',
        unsafe_allow_html=True,
    )
    if not role_matrix.empty:
        heatmap_source = role_matrix.set_index("person_type")
        heatmap_fig = px.imshow(
            heatmap_source,
            color_continuous_scale=active_theme["heatmap_scale"],
            labels=dict(x="Активность", y="Тип сотрудника", color="Минуты"),
            aspect="auto",
        )
        heatmap_fig.update_layout(
            template=active_theme["plotly_template"],
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(color=active_theme["text"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        heatmap_fig.update_xaxes(color=active_theme["muted_text"])
        heatmap_fig.update_yaxes(color=active_theme["muted_text"])
        st.plotly_chart(heatmap_fig, width="stretch")
    else:
        st.info("Пока нет данных для построения матрицы.")

st.markdown('<div class="section-title">Хронология движения по трекам</div>', unsafe_allow_html=True)
motion_df = episodes_df[episodes_df["activity_class"] == "moving"].copy()
if motion_df.empty and not episodes_df.empty:
    motion_df = episodes_df.copy()
if not motion_df.empty:
    motion_df = motion_df.rename(columns={"t_start_sec": "start_sec", "t_end_sec": "end_sec"})
    motion_df["duration_sec"] = motion_df["end_sec"] - motion_df["start_sec"]
    motion_df["person_label"] = motion_df.apply(
        lambda row: f"{row['person_type']} · ID {row['track_id']}"
        if isinstance(row.get("person_type"), str)
        else f"Track {row['track_id']}",
        axis=1,
    )
    timeline_fig = px.bar(
        motion_df,
        x="duration_sec",
        y="person_label",
        base="start_sec",
        color="activity_class",
        color_discrete_sequence=active_theme["chart_sequence"],
        orientation="h",
        labels={
            "duration_sec": "Длительность, сек",
            "start_sec": "Начало, сек",
            "person_label": "Сотрудник / Track",
            "activity_class": "Активность",
        },
    )
    timeline_fig.update_layout(
        template=active_theme["plotly_template"],
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Секунды от начала видео",
        font=dict(color=active_theme["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    timeline_fig.update_xaxes(
        color=active_theme["muted_text"], linecolor=active_theme["muted_text"]
    )
    timeline_fig.update_yaxes(
        color=active_theme["muted_text"], linecolor=active_theme["muted_text"]
    )
    st.plotly_chart(timeline_fig, width="stretch")
else:
    st.info("Нет данных о движении — обработайте ролик, чтобы увидеть таймлайн.")

st.markdown('<div class="section-title">Таблица эпизодов</div>', unsafe_allow_html=True)
if not episodes_df.empty:
    role_options = sorted(episodes_df["person_type"].dropna().unique().tolist()) or [
        "unknown"
    ]
    activity_options = sorted(
        episodes_df["activity_class"].dropna().unique().tolist()
    ) or ["walking"]
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_roles = st.multiselect(
            "Тип сотрудника",
            role_options,
            default=role_options,
        )
    with filter_col2:
        selected_activities = st.multiselect(
            "Активность",
            activity_options,
            default=activity_options,
        )
    filtered = episodes_df[
        episodes_df["person_type"].isin(selected_roles)
        & episodes_df["activity_class"].isin(selected_activities)
    ].copy()
    filtered["duration_sec"] = filtered["duration_sec"].round(1)
    filtered = filtered.rename(
        columns={
            "person_id": "Person ID",
            "track_id": "Track ID",
            "person_type": "Тип",
            "activity_class": "Активность",
            "t_start_sec": "Начало, сек",
            "t_end_sec": "Конец, сек",
            "duration_sec": "Длительность, сек",
        }
    )
    display_columns = [
        "Person ID",
        "Track ID",
        "Тип",
        "Активность",
        "Начало, сек",
        "Конец, сек",
        "Длительность, сек",
    ]
    styled_table = filtered[display_columns].style.set_properties(
        **{
            "background-color": active_theme["dataframe_bg"],
            "color": active_theme["dataframe_text"],
            "border-color": active_theme["card_border"],
        }
    )
    st.dataframe(
        styled_table,
        width="stretch",
        hide_index=True,
    )
else:
    st.info(
        "Нет записанных эпизодов — выполните обработку видео или загрузите демо-данные."
    )

st.markdown("---")

# Управление промптами для классов и UI настроек.
st.markdown(
    '<div class="section-title">Управление промптами для классов YOLO</div>',
    unsafe_allow_html=True,
)
settings = get_settings()

# Загружаем текущие настройки class_prompts.
class_prompts = getattr(settings, "class_prompts", {})
if not isinstance(class_prompts, dict):
    class_prompts = {}

# Выводим список сохранённых промптов.
st.subheader("Текущие промпты по классам")
if class_prompts:
    for class_key, prompts in class_prompts.items():
        # Показываем русское название для удобства.
        class_name_ru = get_class_name_ru(class_key)
        display_name = (
            f"{class_name_ru} ({class_key})"
            if class_name_ru != class_key
            else class_key
        )
        with st.expander(f"Класс: {display_name} ({len(prompts)} промптов)"):
            for i, prompt in enumerate(prompts):
                st.text(f"{i + 1}. {prompt}")
else:
    st.info("Нет настроенных промптов. Добавьте промпты для классов ниже.")

# Форма добавления и редактирования промптов.
st.subheader("Добавить/Редактировать промпты")
col1, col2 = st.columns([1, 2])

with col1:
    # Селектор класса с отображением русского названия.
    class_options_ru = ["Выберите класс..."] + [
        f"{get_class_name_ru(name)} ({name})" for name in YOLO_COCO_CLASS_NAMES
    ]
    selected_class_display = st.selectbox(
        "Выберите класс YOLO COCO",
        class_options_ru,
        key="class_selector",
    )

    # Выделяем английское имя класса из выпадающего списка.
    if selected_class_display != "Выберите класс...":
        # Извлекаем имя из строки вида «русское название (english_name)».
        if " (" in selected_class_display and selected_class_display.endswith(")"):
            selected_class_name = selected_class_display.split(" (")[-1].rstrip(")")
        else:
            selected_class_name = selected_class_display
    else:
        selected_class_name = "Выберите класс..."

    # Альтернативно можно указать числовой ID класса.
    class_id_input = st.number_input(
        "Или введите ID класса (0-79)",
        min_value=0,
        max_value=79,
        value=0,
        key="class_id_input",
    )

    # Определяем, какое значение использовать при сохранении.
    if selected_class_name != "Выберите класс...":
        target_class = selected_class_name
    elif class_id_input is not None:
        target_class = CLASS_ID_TO_NAME.get(
            class_id_input, f"class_{class_id_input}"
        )
    else:
        target_class = None

with col2:
    if target_class:
        # Загружаем уже сохранённые промпты для выбранного класса.
        normalized_key = normalize_class_key(target_class)
        existing_prompts = class_prompts.get(normalized_key, [])

        # Рассчитываем русское название для заголовка формы.
        class_name_ru = get_class_name_ru(target_class)
        display_name = (
            f"{class_name_ru} ({target_class})"
            if class_name_ru != target_class
            else target_class
        )

        # Поле для ввода промптов (один на строку).
        prompts_text = st.text_area(
            f'Промпты для класса "{display_name}" (по одному на строку)',
            value="\n".join(existing_prompts) if existing_prompts else "",
            height=200,
            key="prompts_textarea",
            help="Введите промпты, по одному на строку. Каждый промпт будет использован для классификации действий через VideoLLaMA3.",
        )

        if st.button("Сохранить промпты", key="save_prompts"):
            if prompts_text.strip():
                # Разбираем текстовое поле по строкам.
                new_prompts = [
                    p.strip() for p in prompts_text.strip().split("\n") if p.strip()
                ]
                if new_prompts:
                    # Обновляем записи для выбранного класса.
                    class_prompts[normalized_key] = new_prompts
                    display_name_success = (
                        f"{class_name_ru} ({target_class})"
                        if class_name_ru != target_class
                        else target_class
                    )
                    st.success(
                        f'Сохранено {len(new_prompts)} промптов для класса "{display_name_success}". '
                        "Для применения изменений перезапустите приложение или обновите переменные окружения."
                    )
                    st.info(
                        "Примечание: Изменения сохраняются в памяти. Для постоянного сохранения "
                        "обновите переменную окружения HACKATON_CLASS_PROMPTS или файл .env"
                    )
                else:
                    st.warning("Введите хотя бы один промпт")
            else:
                st.warning("Промпты не могут быть пустыми")
    else:
        st.info("Выберите класс для редактирования промптов")
