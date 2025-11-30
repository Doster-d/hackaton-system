"""CLI-приложение для управления системой видеоаналитики."""

import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import typer

from hackaton_system.config import get_settings
from hackaton_system.db import init_db
from hackaton_system.db.seeder import seed_demo_data
from hackaton_system.pipeline import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

app = typer.Typer(help='Hackathon video analytics toolkit')


@app.command()
def setup_database() -> None:
    """Создаёт структуру базы данных и все необходимые директории."""

    # Создаём таблицы и директории; вызывается один раз при развёртывании.
    init_db()
    typer.echo('Database initialized.')


@app.command()
def process_video(
    video_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help='Path to the video file to process',
    ),
) -> None:
    """Запускает полный пайплайн обработки указанного видео."""

    # Лениво создаём экземпляр пайплайна (внутри инициализируется БД и загрузчик YOLO).
    processor = VideoProcessor()
    video_id = processor.process_video(video_path)
    typer.echo(f'Video stored with id={video_id}')


@app.command()
def seed_demo(
    force: bool = typer.Option(
        False, '--force', help='Override existing data'
    ),
) -> None:
    """Заполняет базу фиксированными демонстрационными данными."""

    # Загружаем предопределённые данные, чтобы интерфейс Streamlit сразу что-то показывал.
    rows = seed_demo_data(force=force)
    if rows:
        typer.echo(f'Inserted {rows} demo rows.')
    else:
        typer.echo('Demo data already present. Use --force to overwrite.')


@app.command()
def test_segment(
    video_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help='Path to the video file',
    ),
    start_sec: float = typer.Argument(
        ...,
        help='Start time in seconds',
    ),
    end_sec: float = typer.Argument(
        ...,
        help='End time in seconds',
    ),
) -> None:
    """Проверяет работу YOLO и VideoLLaMA3 на выбранном отрезке видео."""
    settings = get_settings()
    video_path = Path(video_path)

    if not video_path.exists():
        typer.echo(f'Error: Video file not found: {video_path}', err=True)
        raise typer.Exit(1)

    if start_sec < 0:
        typer.echo('Error: start_sec must be >= 0', err=True)
        raise typer.Exit(1)

    if end_sec <= start_sec:
        typer.echo('Error: end_sec must be > start_sec', err=True)
        raise typer.Exit(1)

    typer.echo(f'Testing segment: {start_sec:.2f}s - {end_sec:.2f}s')

    # Вырезаем сегмент из исходного файла, чтобы работать только с нужным окном.
    typer.echo('Extracting video segment...')
    segment_path = _extract_video_segment(video_path, start_sec, end_sec)
    if segment_path is None:
        typer.echo('Error: Failed to extract video segment', err=True)
        raise typer.Exit(1)

    try:
        # Считываем параметры сегмента, чтобы правильно обрабатывать частоту кадров.
        cap = cv2.VideoCapture(str(segment_path))
        if not cap.isOpened():
            typer.echo('Error: Failed to open segment video', err=True)
            raise typer.Exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        typer.echo(f'Segment: {frame_count} frames, {fps:.2f} fps')

        # Используем тот же VideoProcessor, что и в основном пайплайне.
        typer.echo('\nRunning YOLO detection...')
        processor = VideoProcessor(settings)
        
        typer.echo('Running detection...')
        # Запускаем обнаружение объектов тем же методом, что и в process_video.
        detections, poses, train_events = processor._run_detection(segment_path, fps=fps)

        # Показываем все detections YOLO, сгруппированные по классам.
        typer.echo('\n' + '=' * 80)
        typer.echo('YOLO DETECTION RESULTS (by class):')
        
        # Группируем детекции по class_id.
        detections_by_class: dict[int, list] = defaultdict(list)
        for det in detections:
            class_id = getattr(det, 'class_id', 0)
            detections_by_class[class_id].append(det)
        
        # Логируем статистику отдельно для каждого класса.
        for class_id in sorted(detections_by_class.keys()):
            class_dets = detections_by_class[class_id]
            try:
                from hackaton_system.pipeline.yolo_classes import get_class_name
                class_name = get_class_name(class_id)
            except ImportError:
                class_name = f'class_{class_id}'
            
            typer.echo(f'  {class_name} (class_id={class_id}) detections: {len(class_dets)}')
            if class_dets:
                tracks_dict = processor._group_detections_by_track(class_dets)
                typer.echo(f'    {class_name} tracks: {len(tracks_dict)}')
                for track_id, track_dets in sorted(tracks_dict.items()):
                    time_range = f'{track_dets[0].time_sec:.2f}s - {track_dets[-1].time_sec:.2f}s'
                    avg_conf = sum(d.confidence for d in track_dets) / len(track_dets)
                    typer.echo(
                        f'      Track ID {track_id}: {len(track_dets)} detections, '
                        f'time: {time_range}, avg confidence: {avg_conf:.3f}'
                    )
        
        # Отдельно фиксируем поезд, потому что используем TrainDetectionResult.
        typer.echo(f'  Train (class_id=6) detections: {len(train_events)}')
        if train_events:
            train_by_track: dict[int, list] = defaultdict(list)
            for train in train_events:
                train_by_track[train.track_id].append(train)
            typer.echo(f'    Train tracks: {len(train_by_track)}')
            for track_id, track_trains in sorted(train_by_track.items()):
                time_range = f'{track_trains[0].time_sec:.2f}s - {track_trains[-1].time_sec:.2f}s'
                avg_conf = sum(t.confidence for t in track_trains) / len(track_trains)
                numbers = [t.number for t in track_trains if t.number]
                number_info = f', numbers: {set(numbers)}' if numbers else ''
                typer.echo(
                    f'      Train Track ID {track_id}: {len(track_trains)} detections, '
                    f'time: {time_range}, avg confidence: {avg_conf:.3f}{number_info}'
                )
        
        typer.echo(f'  Pose detections: {len(poses)}')
        typer.echo('=' * 80)

        # Дальше запускаем инференс VideoLLaMA3 теми же методами, что и пайплайн.
        typer.echo('\nRunning VideoLlama inference...')
        
        if not detections and not train_events:
            typer.echo('No detections found, skipping VideoLlama inference')
        else:
            # Для людей нужно сгруппировать детекции по track_id.
            tracks_dict = defaultdict(list)
            for det in detections:
                tracks_dict[det.track_id].append(det)
            
            # Получаем активности людей.
            activities = []
            if tracks_dict:
                activities = processor._infer_activities(
                    tracks_dict, fps=fps, video_path=segment_path
                )
            
            # Получаем активности поездов.
            train_activities = []
            if train_events:
                train_activities = processor._infer_train_activities(
                    train_events, segment_path, fps
                )
            
            # Собираем все активности в один список.
            all_activities = activities + train_activities
            
            typer.echo('VideoLlama Results:')
            typer.echo(f'  - Total activities detected: {len(all_activities)}')
            typer.echo(f'    Person activities: {len(activities)}')
            typer.echo(f'    Train activities: {len(train_activities)}')

            if all_activities:
                # Показываем активности людей.
                if activities:
                    typer.echo('\n  Person activities:')
                    for activity in activities:
                        typer.echo(
                            f'    Track {activity.track_id}: '
                            f'{activity.activity_class} '
                            f'(confidence: {activity.confidence:.3f}, '
                            f'time: {activity.start_sec:.2f}s - {activity.end_sec:.2f}s)'
                        )
                
                # Показываем активности поездов.
                if train_activities:
                    typer.echo('\n  Train activities:')
                    for activity in train_activities:
                        typer.echo(
                            f'    Track {activity.track_id}: '
                            f'{activity.activity_class} '
                            f'(confidence: {activity.confidence:.3f}, '
                            f'time: {activity.start_sec:.2f}s - {activity.end_sec:.2f}s)'
                        )
            else:
                typer.echo(
                    '  - No activities detected above confidence threshold'
                )

        typer.echo('\nTest completed successfully!')

    finally:
        # Чистим временный файл сегмента, если он ещё существует.
        if segment_path.exists():
            try:
                segment_path.unlink()
            except Exception as e:
                typer.echo(
                    f'Warning: Failed to delete temp file: {e}', err=True
                )


def _extract_video_segment(
    video_path: Path, start_sec: float, end_sec: float
) -> Path | None:
    """Вырезает временной промежуток из видео и сохраняет его во временный файл."""
    try:
        cap = cv2.VideoCapture(str(video_path))
    except Exception as e:
        logging.error('Failed to create VideoCapture: %s', e)
        return None

    if not cap.isOpened():
        logging.error('Failed to open video: %s', video_path)
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception as e:
        logging.error('Failed to get video properties: %s', e)
        cap.release()
        return None

    # Переводим секунды в номера кадров.
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Создаём временный файл, куда будем писать фрагмент.
    try:
        with tempfile.NamedTemporaryFile(
            suffix='.mp4', delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
    except Exception as e:
        logging.error('Failed to create temporary file: %s', e)
        cap.release()
        return None

    # Последовательно записываем кадры в новый файл.
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
    except Exception as e:
        logging.error('Failed to create VideoWriter: %s', e)
        cap.release()
        if temp_path.exists():
            temp_path.unlink()
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
    except Exception as e:
        logging.error('Failed to write frames: %s', e)
        out.release()
        cap.release()
        if temp_path.exists():
            temp_path.unlink()
        return None

    try:
        out.release()
        cap.release()
    except Exception as e:
        logging.warning('Failed to release resources: %s', e)

    return temp_path


def main() -> None:
    """Точка входа CLI-приложения."""
    app()


if __name__ == '__main__':
    main()
