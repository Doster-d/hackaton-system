"""Конфигурация приложения и вспомогательные модели настроек."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DETECTION_CLASSES_DEFAULT = [0, 6]


class ZoneDefinition(BaseModel):
    """Прямоугольная зона на кадре с заданным смыслом и типом персонала."""

    name: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    zone_category: Literal["station", "corridor", "restricted", "visitor"]
    person_type: str


class Settings(BaseSettings):
    """Глобальные настройки проекта, считываемые из окружения и .env."""

    # Используем модельную конфигурацию Pydantic, чтобы подтягивать значения из .env
    # c префиксом HACKATON_. Эта секция по сути описывает «откуда» приходят данные.
    model_config = SettingsConfigDict(
        env_prefix="HACKATON_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # URL подключения к БД; по умолчанию используем локальный SQLite в каталоге data.
    database_url: str = Field(
        default="sqlite:///data/hackaton.db",
        description="SQLAlchemy-compatible database URL.",
    )
    # Путь к каталогу с данными — через него создаём директории при инициализации БД.
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for processed artifacts and SQLite database.",
    )
    ultralytics_config_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "HACKATON_ULTRALYTICS_CONFIG_DIR", "ULTRALYTICS_CONFIG_DIR"
        ),
        description="Cache dir for Ultralytics settings/models.",
    )
    torch_home: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("HACKATON_TORCH_HOME", "TORCH_HOME"),
        description="Cache dir for Torch hub/downloads.",
    )
    easyocr_module_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("HACKATON_EASYOCR_MODULE_PATH", "EASYOCR_MODULE_PATH"),
        description="Local directory for EasyOCR models.",
    )
    # Опциональные пути к весам: оставляем None, чтобы брать значения по умолчанию из Ultralytics.
    detection_model_path: str | None = Field(
        default="./yolo11x.pt", description="./yolo11n.pt"
    )
    enable_pose_capture: bool = Field(
        default=True,
        description="Persist pose keypoints when the underlying YOLO weights expose them.",
    )
    enable_video_render: bool = Field(
        default=True,
        description="Render annotated video previews for processed runs.",
    )
    preview_max_side: int = Field(
        default=1920,
        description="Max width/height for rendered previews (frames are downscaled preserving aspect ratio).",
    )
    preview_frame_step: int = Field(
        default=1,
        description="Write every N-th frame to preview to reduce size (1 = every frame).",
    )
    preview_ffmpeg_path: str | None = Field(
        default="ffmpeg",
        description="Path to ffmpeg binary (if None, assumes ffmpeg available in PATH).",
    )
    preview_draw_poses: bool = Field(
        default=True,
        description="Overlay pose skeletons on preview videos when pose data is available.",
    )
    pose_conf_threshold: float = Field(
        default=0.2,
        description="Minimum pose keypoint confidence for visualization and analytics.",
    )
    preprocess_enable: bool = Field(
        default=False,
        description="Apply denoise/contrast/gamma preprocessing before detection.",
    )
    preprocess_denoise_h: int = Field(
        default=0,
        description="Strength for fastNlMeansDenoisingColored (0 disables).",
    )
    preprocess_contrast_alpha: float = Field(
        default=1.15,
        description="Contrast gain factor (1.0 = no change).",
    )
    preprocess_brightness_beta: float = Field(
        default=5.0,
        description="Brightness shift added after contrast.",
    )
    preprocess_gamma: float = Field(
        default=1.05,
        description="Gamma correction (1.0 = no change).",
    )
    use_patch_inference: bool = Field(
        default=False,
        description="Enable YOLO patch-based inference for high-recall detection.",
    )
    patch_shape_x: int = Field(
        default=960,
        description="Patch width when using patch-based inference.",
    )
    patch_shape_y: int = Field(
        default=960,
        description="Patch height when using patch-based inference.",
    )
    patch_overlap_x: int = Field(
        default=200,
        description="Horizontal overlap (pixels) between neighboring patches.",
    )
    patch_overlap_y: int = Field(
        default=200,
        description="Vertical overlap (pixels) between neighboring patches.",
    )
    patch_frame_skip: int = Field(
        default=1,
        description="Process every Nth frame in patch mode (1=every frame, 2=every 2nd frame, etc). Higher values speed up processing but reduce temporal resolution.",
    )
    patch_nms_threshold: float = Field(
        default=0.25,
        description="Patch-based NMS threshold when combining detections.",
    )
    tracker_track_buffer: int = Field(
        default=120,
        description="Number of frames to keep lost tracks in ByteTrack-like logic. Higher values reduce track fragmentation but may cause ID switches.",
    )
    tracker_match_threshold: float = Field(
        default=0.6,
        description="Matching threshold for tracker associations. Lower values allow more matches but may cause ID switches.",
    )
    enable_train_detection: bool = Field(
        default=True,
        description="Detect approaching trains (COCO class 6) and read their numbers.",
    )
    train_detection_conf: float = Field(
        default=0.45,
        description="Confidence threshold for train bounding boxes.",
    )
    train_number_langs: list[str] = Field(
        default_factory=lambda: ["en", "ru"],
        description="Languages passed to OCR reader for train numbers.",
    )
    train_number_min_length: int = Field(
        default=3,
        description="Minimum number of characters for accepted train identifiers.",
    )
    train_event_output_dir: Path = Field(
        default=Path("runs/train_events"),
        description="Folder for JSON reports with detected train numbers.",
    )
    reid_enabled: bool = Field(
        default=True, description="Enable ReID-assisted track stitching."
    )
    reid_similarity_threshold: float = Field(
        default=0.8, description="Cosine similarity threshold for merging tracks."
    )
    reid_time_gap_sec: float = Field(
        default=1.5,
        description="Maximum time gap (sec) between tracks considered for stitching.",
    )
    reid_allow_overlapping_merge: bool = Field(
        default=True,
        description="Allow merging overlapping tracks based on embeddings (useful when people stand close and tracker confuses them).",
    )
    reid_overlap_similarity_threshold: float = Field(
        default=0.85,
        description="Stricter similarity threshold for merging overlapping tracks (higher = more conservative).",
    )
    train_track_merge_gap_sec: float = Field(
        default=5.0,
        description="Maximum time gap (seconds) between train tracks to merge them.",
    )
    train_track_merge_max_distance: float = Field(
        default=200.0,
        description="Maximum distance (pixels) between train track centers to merge them.",
    )
    reid_model_name: str = Field(
        default="osnet_x1_0", description="Torchreid model name for feature extraction."
    )
    reid_model_path: str | None = Field(
        default=None, description="Optional custom checkpoint for Torchreid model."
    )
    video_output_dir: Path = Field(
        default=Path("runs/visualizations"),
        description="Directory where annotated preview videos will be stored.",
    )
    tracker_config_path: str = Field(
        default="bytetrack.yaml",
        description="Tracker configuration name/path for Ultralytics track API.",
    )
    detection_conf: float = Field(
        default=0.25, description="Confidence threshold for YOLO detections."
    )
    detection_iou: float = Field(
        default=0.6, description="IoU threshold for YOLO tracker associations."
    )
    detection_imgsz: int = Field(
        default=1280, description="Image size (short side) passed to YOLO track."
    )
    detection_tta: bool = Field(
        default=False,
        description="Enable light test-time augmentation during detection.",
    )
    activity_model_path: str | None = Field(
        default="./yolo11n-pose.pt", description="Path to the action recognition weights."
    )
    activity_velocity_move_thresh: float = Field(
        default=140.0,
        description="Speed (px/sec) above which a person is confidently moving.",
    )
    activity_velocity_idle_thresh: float = Field(
        default=40.0,
        description="Speed (px/sec) below which a person is confidently idle.",
    )
    activity_min_interval_sec: float = Field(
        default=1.5,
        description="Shortest interval duration that will be recorded as activity.",
    )
    role_assignment_threshold: float = Field(
        default=0.6,
        description="Share of time in dominant zone required to fix person_type.",
    )
    zones: list[ZoneDefinition] = Field(
        default_factory=lambda: [
            ZoneDefinition(
                name="station_left",
                x_min=50,
                y_min=220,
                x_max=640,
                y_max=700,
                zone_category="station",
                person_type="operator",
            ),
            ZoneDefinition(
                name="station_right",
                x_min=650,
                y_min=220,
                x_max=1230,
                y_max=700,
                zone_category="station",
                person_type="operator",
            ),
            ZoneDefinition(
                name="corridor_main",
                x_min=0,
                y_min=80,
                x_max=1280,
                y_max=210,
                zone_category="corridor",
                person_type="supervisor",
            ),
            ZoneDefinition(
                name="visitor_strip",
                x_min=20,
                y_min=0,
                x_max=1260,
                y_max=70,
                zone_category="visitor",
                person_type="visitor",
            ),
            ZoneDefinition(
                name="restricted_zone",
                x_min=900,
                y_min=400,
                x_max=1200,
                y_max=650,
                zone_category="restricted",
                person_type="restricted",
            ),
        ],
        description="Rectangular production zones that drive roles and activities.",
    )

    @field_validator("zones", mode="before")
    @classmethod
    def parse_zones(cls, v: Any) -> Any:
        """Позволяет описывать зоны в разных форматах и превращает их в словари."""
        # Если уже пришёл список ZoneDefinition (например, из default_factory), превратим их в словари.
        if isinstance(v, list):
            # Проверяем, что все элементы уже являются ZoneDefinition.
            if all(isinstance(item, ZoneDefinition) for item in v):
                # Конвертируем объекты в словари, чтобы Pydantic мог их валидировать.
                return [item.model_dump() if hasattr(item, 'model_dump') else item.dict() for item in v]
            # Иначе пытаемся обработать каждый элемент по отдельности.
            result = []
            for item in v:
                if isinstance(item, ZoneDefinition):
                    # Превращаем в словарь.
                    result.append(item.model_dump() if hasattr(item, 'model_dump') else item.dict())
                elif isinstance(item, dict):
                    result.append(item)
                else:
                    # Пробуем привести произвольную структуру к dict.
                    result.append(dict(item))
            return result
        # Если прилетел одиночный ZoneDefinition, оборачиваем в список и превращаем в dict.
        if isinstance(v, ZoneDefinition):
            return [v.model_dump() if hasattr(v, 'model_dump') else v.dict()]
        # Если видим dict, выясняем, это одна зона или структура с ключом zones.
        if isinstance(v, dict):
            # Может быть либо самой зоной, либо словарём с ключом zones.
            if "zones" in v:
                return cls.parse_zones(v["zones"])
            return [v]
        return v
    action_backend: Literal["heuristic", "videollama3"] = Field(
        default="heuristic",
        description="Backend for action inference: heuristic or VideoLLaMA3 zero-shot.",
    )
    action_prompts: list[str] = Field(
        default_factory=lambda: [
            "person is standing idle without doing anything",
            "person is walking or moving to a location",
            "person is repairing the train",
            "person is working on train maintenance",
            "person is inspecting the train",
            "person is waiting near the train",
        ],
        description="Text prompts for zero-shot action classification via VideoLLaMA3 (person actions).",
    )
    action_clip_frames: int = Field(
        default=16,
        description="Number of frames per clip segment for VideoLLaMA3 action inference (legacy, used if take/skip not set).",
    )
    action_clip_stride: int = Field(
        default=2,
        description="Frame stride between clip segments for VideoLLaMA3 (legacy, used if take/skip not set).",
    )
    action_clip_take_duration_sec: float | None = Field(
        default=None,
        description="Duration in seconds to take frames for each segment (e.g., 0.2 for 200ms). If set, enables take-skip pattern.",
    )
    action_clip_skip_duration_sec: float | None = Field(
        default=None,
        description="Duration in seconds to skip between segments (e.g., 0.3 for 300ms). Used with action_clip_take_duration_sec.",
    )
    action_min_confidence: float = Field(
        default=0.2,
        description="Minimum confidence threshold for VideoLLaMA3 action predictions.",
    )
    videollama3_model_name: str = Field(
        default="DAMO-NLP-SG/VideoLLaMA3-2B",
        description="HuggingFace model name for VideoLLaMA3.",
    )
    videollama3_device: str = Field(
        default="auto",
        description="Device for VideoLLaMA3 ('auto', 'cuda', or 'cpu').",
    )
    videollama3_max_frames: int = Field(
        default=32,
        description="Maximum number of frames to process per video segment in VideoLLaMA3.",
    )
    detection_classes: list[int] = Field(
        default_factory=lambda: DETECTION_CLASSES_DEFAULT.copy(),
        description="YOLO COCO class IDs to detect (0=person, 6=train, etc.).",
    )
    class_prompts: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Action prompts per class (key: class name or ID, value: list of prompts).",
    )
    train_min_frames_present: int = Field(
        default=10,
        description="Minimum consecutive frames with train to confirm arrival.",
    )
    train_max_frames_absent: int = Field(
        default=30,
        description="Maximum frames without train to confirm departure.",
    )
    train_videollama3_enabled: bool = Field(
        default=True,
        description="Use VideoLLaMA3 for train state classification.",
    )
    train_prompts: list[str] = Field(
        default_factory=lambda: [
            "train is arriving at the station",
            "train is departing from the station",
            "train is present and stationary at the station",
            "train is moving slowly near the station",
            "no train visible in the scene",
        ],
        description="Text prompts for VideoLLaMA3 train state classification.",
    )

    @field_validator("detection_classes", mode="before")
    @classmethod
    def parse_detection_classes(cls, v: Any) -> list[int]:
        """Parse detection_classes from JSON string or list."""
        if isinstance(v, list):
            return [int(x) for x in v]
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [int(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
        return DETECTION_CLASSES_DEFAULT.copy()

    @field_validator("class_prompts", mode="before")
    @classmethod
    def parse_class_prompts(cls, v: Any) -> dict[str, list[str]]:
        """Parse class_prompts from JSON string or dict."""
        if isinstance(v, dict):
            result = {}
            for k, v_val in v.items():
                if isinstance(v_val, list):
                    result[str(k)] = [str(p) for p in v_val]
                else:
                    result[str(k)] = [str(v_val)]
            return result
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    result = {}
                    for k, v_val in parsed.items():
                        if isinstance(v_val, list):
                            result[str(k)] = [str(p) for p in v_val]
                        else:
                            result[str(k)] = [str(v_val)]
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    @field_validator("train_prompts", mode="before")
    @classmethod
    def parse_train_prompts(cls, v: Any) -> list[str]:
        """Разбирает список подсказок для поездов из строк, JSON и массивов."""
        if isinstance(v, list):
            return [str(item) for item in v]
        if isinstance(v, str):
            # Пустая строка превращается в пустой список.
            if not v.strip():
                return []
            # Сначала пробуем прочитать как JSON.
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (json.JSONDecodeError, TypeError):
                # Если JSON невалидный, пытаемся вытащить элементы вручную.
                try:
                    v_normalized = v.replace("\n", " ").replace("\r", " ")
                    if "[" in v_normalized and "]" in v_normalized:
                        start = v_normalized.index("[") + 1
                        end = v_normalized.rindex("]")
                        content = v_normalized[start:end]
                        items = [
                            item.strip().strip('"').strip("'")
                            for item in content.split(",")
                            if item.strip()
                        ]
                        if items:
                            return items
                except Exception:
                    pass
            # Далее пробуем интерпретировать как список через запятую.
            if "," in v:
                return [item.strip() for item in v.split(",") if item.strip()]
            # В остальных случаях считаем, что это одиночная строка.
            return [v.strip()]
        return []

    @field_validator("action_prompts", mode="before")
    @classmethod
    def parse_action_prompts(cls, v: Any) -> list[str]:
        """Нормализует список текстовых подсказок для действий."""
        if isinstance(v, list):
            return [str(item) for item in v]
        if isinstance(v, str):
            # Пустая строка превращается в пустой список.
            if not v.strip():
                return []
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (json.JSONDecodeError, TypeError):
                try:
                    v_normalized = v.replace("\n", " ").replace("\r", " ")
                    if "[" in v_normalized and "]" in v_normalized:
                        start = v_normalized.index("[") + 1
                        end = v_normalized.rindex("]")
                        content = v_normalized[start:end]
                        items = [
                            item.strip().strip('"').strip("'")
                            for item in content.split(",")
                            if item.strip()
                        ]
                        if items:
                            return items
                except Exception:
                    pass
            if "," in v:
                return [item.strip() for item in v.split(",") if item.strip()]
            # Если ничего из вышеперечисленного не подошло — возвращаем одиночную строку.
            return [v.strip()]
        return []

    # Поля, оставленные ради совместимости со старым X-CLIP API.
    @property
    def xclip_model_name(self) -> str:
        """Устарело: вместо этого используйте videollama3_model_name."""
        return self.videollama3_model_name

    @property
    def xclip_device(self) -> str:
        """Устарело: вместо этого используйте videollama3_device."""
        return self.videollama3_device

    @property
    def train_xclip_enabled(self) -> bool:
        """Устарело: вместо этого используйте train_videollama3_enabled."""
        return self.train_videollama3_enabled


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает кешированный экземпляр Settings."""

    # lru_cache гарантирует, что настройки считываются один раз за запуск процесса,
    # а дальнейшие импорты получают готовый объект без повторного чтения .env.
    return Settings()
