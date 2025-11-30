"""Набор тем и CSS-конструктор для интерфейса Streamlit."""

from __future__ import annotations

from typing import Any, Dict


THEME_PALETTES: Dict[str, Dict[str, Any]] = {
    "light": {
        "name": "Светлая",
        "app_bg": "linear-gradient(180deg, #f2f2f2 0%, #e2e2e2 35%, #ffffff 100%)",
        "sidebar_bg": "#e2e2e2",
        "sidebar_text": "#000000",
        "text": "#000000",
        "muted_text": "#475b85",
        "input_bg": "#ffffff",
        "input_border": "#bababa",
        "input_text": "#000000",
        "button_bg": "linear-gradient(90deg, #FFD200, #FCAF17)",
        "button_text": "#000000",
        "button_border": "#ef6b01",
        "button_hover": "linear-gradient(90deg, #FFE066, #FFC12D)",
        "hero_bg": "linear-gradient(120deg, rgba(255,210,0,0.9), rgba(249,157,28,0.9))",
        "hero_border": "#f99d1c",
        "hero_text": "#000000",
        "card_bg": "#ffffff",
        "card_border": "#bababa",
        "card_shadow": "0px 8px 25px rgba(71,91,133,0.15)",
        "section_title": "#475b85",
        "metric_value": "#ef6b01",
        "info_bg": "rgba(255,210,0,0.1)",
        "info_border": "#FFD200",
        "warning_bg": "rgba(239,107,1,0.12)",
        "warning_border": "#EF6B01",
        "success_bg": "rgba(16,185,129,0.12)",
        "success_border": "#10B981",
        "dataframe_bg": "#ffffff",
        "dataframe_text": "#000000",
        "dropdown_bg": "#fffefa",
        "dropdown_border": "#d6d6d6",
        "table_header_bg": "#f3f4f7",
        "table_row_bg": "#ffffff",
        "grid_bg": "#ffffff",
        "grid_header": "#4c4c4c",
        "grid_text": "#000000",
        "tab_bg": "#ffffff",
        "tab_border": "#dcdcdc",
        "tab_active_bg": "#ffd200",
        "tab_active_text": "#000000",
        "expander_bg": "#ffffff",
        "plotly_template": "plotly_white",
        "chart_primary": "#ef6b01",
        "chart_sequence": [
            "#FFD200",
            "#FCAF17",
            "#F99D1C",
            "#EF6B01",
            "#C20937",
            "#006BB2",
        ],
        "heatmap_scale": [
            [0.0, "#F2F2F2"],
            [0.5, "#FCAF17"],
            [1.0, "#C20937"],
        ],
    },
    "dark": {
        "name": "Тёмная",
        "app_bg": "linear-gradient(160deg, #475b85 0%, #2b3658 45%, #05070b 100%)",
        "sidebar_bg": "#1c2640",
        "sidebar_text": "#f2f2f2",
        "text": "#f2f2f2",
        "muted_text": "#95a0b2",
        "input_bg": "rgba(28,38,64,0.8)",
        "input_border": "rgba(255,210,0,0.4)",
        "input_text": "#f2f2f2",
        "dropdown_bg": "#1c2640",
        "dropdown_border": "rgba(255,210,0,0.3)",
        "button_bg": "linear-gradient(90deg, #FCAF17, #EF6B01)",
        "button_text": "#05070b",
        "button_border": "rgba(255,210,0,0.6)",
        "button_hover": "linear-gradient(90deg, #FFD200, #FCAF17)",
        "hero_bg": "linear-gradient(140deg, rgba(255,210,0,0.25), rgba(239,107,1,0.35))",
        "hero_border": "rgba(255,210,0,0.35)",
        "hero_text": "#f2f2f2",
        "card_bg": "rgba(28,38,64,0.9)",
        "card_border": "rgba(255,210,0,0.25)",
        "card_shadow": "0px 12px 35px rgba(5,7,11,0.8)",
        "section_title": "#FFD200",
        "metric_value": "#FFD200",
        "info_bg": "rgba(71,91,133,0.45)",
        "info_border": "rgba(255,210,0,0.5)",
        "warning_bg": "rgba(194,9,55,0.35)",
        "warning_border": "rgba(255,210,0,0.6)",
        "success_bg": "rgba(16,185,129,0.3)",
        "success_border": "rgba(71,133,91,0.8)",
        "dataframe_bg": "rgba(5,7,11,0.5)",
        "dataframe_text": "#f2f2f2",
        "table_header_bg": "#1f2a45",
        "table_row_bg": "rgba(15,18,30,0.8)",
        "grid_bg": "#0e1117",
        "grid_header": "rgba(250,250,250,0.6)",
        "grid_text": "#fafafa",
        "tab_bg": "#1c2640",
        "tab_border": "rgba(255,210,0,0.2)",
        "tab_active_bg": "#ef6b01",
        "tab_active_text": "#05070b",
        "expander_bg": "rgba(15,18,30,0.65)",
        "plotly_template": "plotly_dark",
        "chart_primary": "#ffd200",
        "chart_sequence": [
            "#FFD200",
            "#FCAF17",
            "#F99D1C",
            "#EF6B01",
            "#C20937",
            "#95A0B2",
        ],
        "heatmap_scale": [
            [0.0, "#1c2640"],
            [0.5, "#475b85"],
            [1.0, "#FFD200"],
        ],
    },
}


def build_theme_css(theme: Dict[str, Any]) -> str:
    """Генерирует CSS-строку для переданной цветовой палитры."""

    return f"""
<style>
    body, .stApp, .block-container {{
        background: {theme["app_bg"]} !important;
        color: {theme["text"]} !important;
    }}
    [data-testid=\"stAppViewContainer\"] {{
        background: {theme["app_bg"]};
        color: {theme["text"]};
    }}
    header[data-testid=\"stHeader\"] {{
        background: {theme["card_bg"]};
        border-bottom: 1px solid {theme["card_border"]};
        color: {theme["text"]};
    }}
    [data-testid=\"stSidebar\"] {{
        background-color: {theme["sidebar_bg"]};
        color: {theme["sidebar_text"]};
    }}
    [data-testid=\"stSidebar\"] * {{
        color: {theme["sidebar_text"]};
    }}
    .hero {{
        padding: 1.5rem;
        border-radius: 1rem;
        background: {theme["hero_bg"]};
        border: 1px solid {theme["hero_border"]};
        backdrop-filter: blur(6px);
        margin-bottom: 1.5rem;
        color: {theme["hero_text"]};
    }}
    .hero h1 {{
        color: {theme["hero_text"]};
        margin-bottom: 0.5rem;
    }}
    .metric-card {{
        padding: 1rem 1.25rem;
        border-radius: 0.9rem;
        border: 1px solid {theme["card_border"]};
        background: {theme["card_bg"]};
        box-shadow: {theme["card_shadow"]};
    }}
    .metric-title {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        color: {theme["muted_text"]};
        margin-bottom: 0.35rem;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 600;
        color: {theme["metric_value"]};
    }}
    .section-title {{
        font-size: 1.1rem;
        letter-spacing: 0.05rem;
        color: {theme["section_title"]};
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .stButton>button {{
        background: {theme["button_bg"]};
        color: {theme["button_text"]};
        border: 1px solid {theme["button_border"]};
        border-radius: 999px;
        padding: 0.35rem 1.5rem;
        font-weight: 600;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }}
    .stButton>button:hover {{
        background: {theme["button_hover"]};
        border-color: {theme["button_border"]};
    }}
    .stButton>button:focus-visible {{
        outline: 2px solid {theme["button_border"]};
    }}
    input, textarea, select,
    .stTextInput>div>div>input,
    .stTextArea textarea,
    .stNumberInput input,
    .stSelectbox div[data-baseweb=\"select\"]>div,
    .stMultiSelect div[data-baseweb=\"select\"]>div {{
        background-color: {theme["input_bg"]} !important;
        color: {theme["input_text"]} !important;
        border-radius: 0.65rem;
        border: 1px solid {theme["input_border"]} !important;
        box-shadow: none !important;
    }}
    .stSelectbox div[role=\"button\"],
    .stMultiSelect div[role=\"button\"] {{
        background-color: {theme["card_bg"]};
        color: {theme["text"]};
        border: 1px solid {theme["input_border"]};
    }}
    div[data-baseweb=\"select\"] ul[role=\"listbox\"],
    div[data-baseweb=\"popover\"] ul[role=\"listbox\"] {{
        background: {theme["dropdown_bg"]};
        color: {theme["text"]};
        border: 1px solid {theme["dropdown_border"]};
        border-radius: 0.5rem;
    }}
    div[data-baseweb=\"select\"] ul[role=\"listbox\"] li {{
        color: {theme["text"]};
        background: {theme["dropdown_bg"]};
    }}
    div[data-baseweb=\"select\"] ul[role=\"listbox\"] li:hover {{
        background: {theme["button_hover"]};
        color: {theme["button_text"]};
    }}
    label, .stCaption, div[data-testid=\"stMarkdownContainer\"] p {{
        color: {theme["text"]} !important;
    }}
    div[data-testid=\"stMarkdownContainer\"] h1,
    div[data-testid=\"stMarkdownContainer\"] h2,
    div[data-testid=\"stMarkdownContainer\"] h3,
    div[data-testid=\"stMarkdownContainer\"] h4,
    div[data-testid=\"stMarkdownContainer\"] h5,
    div[data-testid=\"stMarkdownContainer\"] h6 {{
        color: {theme["text"]} !important;
    }}
    code, pre {{
        background: {theme["card_bg"]} !important;
        color: {theme["text"]} !important;
        border: 1px solid {theme["card_border"]} !important;
    }}
    div[data-testid=\"stAlert\"] {{
        border-radius: 0.85rem;
        border: 1px solid {theme["info_border"]};
        background: {theme["info_bg"]};
        color: {theme["text"]};
    }}
    div[data-testid=\"stAlert\"].stAlertWarning {{
        border-color: {theme["warning_border"]};
        background: {theme["warning_bg"]};
    }}
    div[data-testid=\"stAlert\"].stAlertSuccess {{
        border-color: {theme["success_border"]};
        background: {theme["success_bg"]};
    }}
    div[data-testid=\"stAlert\"] p {{
        color: {theme["text"]} !important;
    }}
    div[data-testid=\"stDataFrame\"] {{
        border-radius: 1rem;
        border: 1px solid {theme["card_border"]};
        background: {theme["dataframe_bg"]};
        box-shadow: {theme["card_shadow"]};
        padding: 0.5rem;
    }}
    div[data-testid=\"stDataFrame\"] table thead tr th {{
        background: {theme["table_header_bg"]};
        color: {theme["text"]};
        border-bottom: 1px solid {theme["card_border"]};
    }}
    div[data-testid=\"stDataFrame\"] table tbody tr td {{
        background: {theme["table_row_bg"]};
        color: {theme["text"]};
        border-bottom: 1px solid {theme["card_border"]};
    }}
    div[data-testid=\"stFileUploaderDropzone\"] {{
        background: {theme["card_bg"]};
        border: 2px dashed {theme["button_border"]};
        color: {theme["text"]};
        border-radius: 1rem;
        box-shadow: inset 0 0 0 1px {theme["card_border"]};
    }}
    div[data-testid=\"stFileUploader\"] button {{
        background: {theme["button_bg"]};
        color: {theme["button_text"]};
        border: 1px solid {theme["button_border"]};
        position: relative;
    }}
    div[data-testid=\"stFileUploader\"] button span {{
        color: transparent !important;
    }}
    div[data-testid=\"stFileUploader\"] button::after {{
        content: "Выбрать файлы";
        color: {theme["button_text"]};
        font-weight: 600;
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: none;
    }}
    div[data-testid=\"stFileUploaderDropzoneInstructions\"] span,
    div[data-testid=\"stFileUploaderDropzoneInstructions\"] div {{
        color: {theme["text"]} !important;
    }}
    .stNumberInput button {{
        background: {theme["card_bg"]};
        color: {theme["text"]};
        border: 1px solid {theme["card_border"]};
    }}
    .stNumberInput button:hover {{
        background: {theme["button_bg"]};
        color: {theme["button_text"]};
    }}
    div[data-testid=\"stExpander\"] > div {{
        background: {theme["expander_bg"]};
        border: 1px solid {theme["card_border"]};
        border-radius: 0.8rem;
    }}
    div[data-testid=\"stTabs\"] [role=\"tablist\"] button {{
        background: {theme["tab_bg"]};
        color: {theme["muted_text"]};
        border-bottom: 2px solid {theme["tab_border"]};
    }}
    div[data-testid=\"stTabs\"] [role=\"tablist\"] button[aria-selected=\"true\"] {{
        background: {theme["tab_active_bg"]};
        color: {theme["tab_active_text"]};
        border-bottom: 2px solid {theme["button_border"]};
    }}
    .st-emotion-cache-1sswg7i, .stDataFrameGlideDataEditor {{
        --gdg-bg-cell: {theme["grid_bg"]} !important;
        --gdg-bg-cell-medium: {theme["grid_bg"]} !important;
        --gdg-bg-header: {theme["table_header_bg"]} !important;
        --gdg-bg-header-has-focus: rgba(0,0,0,0.05);
        --gdg-bg-header-hovered: rgba(0,0,0,0.05);
        --gdg-text-header: {theme["grid_header"]} !important;
        --gdg-text-header-selected: {theme["grid_text"]} !important;
        --gdg-text-dark: {theme["grid_text"]} !important;
        --gdg-text-medium: rgba(0,0,0,0.75) !important;
        --gdg-text-light: rgba(0,0,0,0.45) !important;
        --gdg-bg-group-header: {theme["table_header_bg"]} !important;
        --gdg-border-color: {theme["card_border"]} !important;
        --gdg-horizontal-border-color: {theme["card_border"]} !important;
    }}
</style>
"""


__all__ = ["THEME_PALETTES", "build_theme_css"]
