import json
from collections import Counter
from io import BytesIO
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


METRIC_CARDS = [
    ("Total Call Volume", "total"),
    ("Abandoned Within Amelia", "abandoned_amelia"),
    ("Contained Within Amelia", "contained"),
    ("Escalated to Agent", "escalated"),
    ("Patient Abandoned Before Escal.", "patient_abandoned_before_esc"),
    ("Resolved", "resolved"),
    ("Partially Resolved", "partially_resolved"),
    ("Unresolved", "unresolved"),
    ("Authentication Started", "auth_started"),
    ("Authentication Success", "auth_success"),
    ("Authentication %", "auth_percentage"),
    ("Under 18", "under_age_true"),
    ("User Language", "user_language"),
    ("Avg Handle Time", "avg_handle_time"),
    ("Avg Satisfaction", "avg_satisfaction"),
    ("Avg Answer Speed", "avg_answer_speed"),
]

CATEGORY_LABELS = {k: v for v, k in METRIC_CARDS}
CATEGORY_OPTIONS = [("All", "all")] + METRIC_CARDS
COMPUTED_KEYS = {
    "avg_handle_time",
    "avg_satisfaction",
    "avg_answer_speed",
    "auth_started",
    "auth_success",
    "auth_percentage",
    "under_age_true",
    "user_language",
}


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def normalize_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    mins, secs = divmod(seconds, 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins}m {secs}s"


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def transcript_to_text(conv: dict) -> str:
    transcript = conv.get("transcript", [])
    if not isinstance(transcript, list) or not transcript:
        return "No transcript available."

    lines = []
    for msg in transcript:
        if not isinstance(msg, dict):
            continue
        speaker = msg.get("userName", "Unknown")
        user_type = msg.get("userType", "")
        text = msg.get("messageText", "")
        created = msg.get("created", "")
        prefix = f"[{user_type}] {speaker}" if user_type else speaker
        if created:
            prefix += f" ({created})"
        lines.append(f"{prefix}:\n{text}\n")
    return "\n".join(lines) if lines else "No transcript available."


def build_metric_maps(conv: dict):
    raw_metrics = conv.get("metrics", {})
    raw_custom = conv.get("customMetrics", {})

    metrics_map = {}
    custom_map = {}

    if isinstance(raw_metrics, dict):
        metrics_map.update(raw_metrics)
    elif isinstance(raw_metrics, list):
        for item in raw_metrics:
            if not isinstance(item, dict):
                continue
            code = item.get("code")
            value = item.get("value")
            if code is None:
                continue
            metrics_map[code] = value
            if item.get("custom") is True:
                custom_map[code] = value

    if isinstance(raw_custom, dict):
        custom_map.update(raw_custom)

    return metrics_map, custom_map


def get_metric_values(conv: dict, keys: list[str]) -> list[Any]:
    values: list[Any] = []
    raw_metrics = conv.get("metrics", {})
    raw_custom = conv.get("customMetrics", {})

    if isinstance(raw_metrics, list):
        for item in raw_metrics:
            if not isinstance(item, dict):
                continue
            if item.get("code") in keys:
                values.append(item.get("value"))
    elif isinstance(raw_metrics, dict):
        for key in keys:
            if key in raw_metrics:
                values.append(raw_metrics.get(key))

    if isinstance(raw_custom, dict):
        for key in keys:
            if key in raw_custom:
                values.append(raw_custom.get(key))

    return values


def normalize_string_list(values: list[Any]) -> list[str]:
    out = []
    for value in values:
        if value is None:
            continue
        out.append(str(value).strip().lower())
    return out


def get_first(mapping: dict, keys: list[str], default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def parse_uploaded_json(uploaded_file) -> list[dict]:
    data = json.load(uploaded_file)
    if isinstance(data, dict):
        for key in ("conversations", "data", "results", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON format: expected a list of conversations or an object containing one.")


def parse_date_safe(value):
    if value is None or value == "":
        return None

    # Handles epoch timestamps (seconds or ms)
    if isinstance(value, (int, float)):
        parsed = pd.to_datetime(value, unit="s", errors="coerce", utc=False)
        if pd.isna(parsed):
            parsed = pd.to_datetime(value, unit="ms", errors="coerce", utc=False)
    else:
        parsed = pd.to_datetime(value, errors="coerce", utc=False)

    if pd.isna(parsed):
        return None

    try:
        return parsed.date()
    except Exception:
        return None


def parse_metrics(data: list[dict]) -> dict:
    metric_buckets = {key: [] for _, key in METRIC_CARDS if key not in COMPUTED_KEYS}

    handle_times = []
    satisfaction_scores = []
    answer_speeds = []
    intent_counts = Counter()
    resolution_counts = Counter()
    channel_counts = Counter()
    day_counts = Counter()
    under_age_counts = Counter()
    auth_value_counts = Counter()
    language_counts = Counter()
    all_entries = []

    for conv in data:
        if not isinstance(conv, dict):
            continue

        metrics_map, custom_map = build_metric_maps(conv)

        abandoned = get_first(metrics_map, ["abandoned", "Abandoned"], None)
        escalated = get_first(
            custom_map,
            ["ESCALATED", "Escalated", "escalated"],
            get_first(metrics_map, ["escalated", "Escalated"], False),
        )
        escalated = normalize_bool(escalated)

        caller = get_first(
            custom_map,
            ["Caller Number", "CallerNumber", "caller_number"],
            get_first(metrics_map, ["Caller Number", "CallerNumber", "caller_number"], ""),
        )

        # Correct fields: use exact custom metric keys requested by the user
        queue_code = get_first(
            custom_map,
            ["Queue Code"],
            get_first(metrics_map, ["Queue Code"], ""),
        )
        escalation_number = get_first(
            custom_map,
            ["Escalation Number"],
            get_first(metrics_map, ["Escalation Number"], ""),
        )

        handle_time = normalize_number(
            get_first(
                metrics_map,
                ["totalHandleTime", "total_handle_time", "TotalHandleTime", "total_handleTime", "total_handle_time"],
                0,
            ),
            0.0,
        )
        satisfaction = normalize_number(
            get_first(metrics_map, ["satisfaction_score"], get_first(custom_map, ["satisfactionScore"], 0)),
            0.0,
        )
        answer_speed = normalize_number(get_first(metrics_map, ["amelia_answer_speed"], 0), 0.0)

        resolution_status = str(get_first(metrics_map, ["resolution_status"], "UNKNOWN")).strip().upper()
        user_intent = str(
            get_first(custom_map, ["userInitialIntent"], get_first(metrics_map, ["userInitialIntent"], ""))
        ).strip()
        channel = str(conv.get("initialChannel", "")).strip()
        conversation_id = str(conv.get("conversationId", "")).strip()
        created_at = conv.get("conversationCreated", "")
        created_date = parse_date_safe(created_at)

        abandoned_num = normalize_number(abandoned, -999)
        if float(abandoned_num).is_integer():
            abandoned_num = int(abandoned_num)

        auth_values = normalize_string_list(get_metric_values(conv, ["authentication", "Authentication"]))
        auth_started = "start" in auth_values
        auth_success = "success" in auth_values
        auth_status = "success" if auth_success else ("start" if auth_started else (auth_values[-1] if auth_values else ""))
        auth_completed_after_start = auth_started and auth_success

        under_age_values = normalize_string_list(get_metric_values(conv, ["underAge", "underage", "UnderAge"]))
        under_age_known = any(value in {"true", "false"} for value in under_age_values)
        under_age = any(value == "true" for value in under_age_values)
        under_age_label = "Under 18" if under_age else ("18+" if under_age_known else "Unknown")

        user_language_values = normalize_string_list(get_metric_values(conv, ["userLanguage", "UserLanguage", "user_language"]))
        if user_language_values:
            user_language_raw = user_language_values[-1]
            if user_language_raw == "spanish":
                user_language = "Spanish"
            elif user_language_raw == "english":
                user_language = "English"
            else:
                user_language = user_language_raw.title()
        else:
            user_language = "Unknown"

        handle_times.append(handle_time)
        satisfaction_scores.append(satisfaction)
        answer_speeds.append(answer_speed)

        if user_intent:
            intent_counts[user_intent] += 1
        if resolution_status:
            resolution_counts[resolution_status] += 1
        if channel:
            channel_counts[channel] += 1
        if created_date:
            day_counts[str(created_date)] += 1
        if under_age_known:
            under_age_counts[under_age_label] += 1
        if auth_status:
            auth_value_counts[auth_status] += 1
        if user_language:
            language_counts[user_language] += 1

        entry = {
            "caller": str(caller).strip() if caller is not None else "",
            "conversation_id": conversation_id,
            "queue_code": str(queue_code).strip() if queue_code is not None else "",
            "escalation_number": str(escalation_number).strip() if escalation_number is not None else "",
            "handle_time": handle_time,
            "handle_time_fmt": fmt_time(handle_time),
            "satisfaction_score": satisfaction,
            "answer_speed": answer_speed,
            "answer_speed_fmt": f"{answer_speed:.1f}s",
            "resolution_status": resolution_status,
            "intent": user_intent,
            "channel": channel,
            "created_at": str(created_at),
            "created_date": created_date,
            "auth_values": auth_values,
            "auth_value": auth_status,
            "auth_started": auth_started,
            "auth_success": auth_success,
            "auth_completed_after_start": auth_completed_after_start,
            "under_age": under_age,
            "under_age_known": under_age_known,
            "under_age_label": under_age_label,
            "user_language": user_language,
            "raw": conv,
        }

        all_entries.append(entry)
        metric_buckets["total"].append(entry)

        if abandoned_num == 1 and not escalated:
            metric_buckets["abandoned_amelia"].append(entry)
        if abandoned_num == 0 and not escalated:
            metric_buckets["contained"].append(entry)
        if escalated:
            metric_buckets["escalated"].append(entry)
        if abandoned_num == 1 and escalated:
            metric_buckets["patient_abandoned_before_esc"].append(entry)

        if resolution_status == "RESOLVED":
            metric_buckets["resolved"].append(entry)
        elif resolution_status == "PARTIALLY_RESOLVED":
            metric_buckets["partially_resolved"].append(entry)
        elif resolution_status == "UNRESOLVED":
            metric_buckets["unresolved"].append(entry)

    available_dates = sorted({entry["created_date"] for entry in all_entries if entry.get("created_date")})

    return {
        "metrics": metric_buckets,
        "avg_handle_time": sum(handle_times) / len(handle_times) if handle_times else 0.0,
        "avg_satisfaction": sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0.0,
        "avg_answer_speed": sum(answer_speeds) / len(answer_speeds) if answer_speeds else 0.0,
        "total": len(all_entries),
        "intent_counts": dict(intent_counts),
        "resolution_counts": dict(resolution_counts),
        "channel_counts": dict(channel_counts),
        "day_counts": dict(sorted(day_counts.items())),
        "under_age_counts": dict(under_age_counts),
        "auth_value_counts": dict(auth_value_counts),
        "language_counts": dict(language_counts),
        "available_dates": available_dates,
        "all_entries": all_entries,
    }


def compute_auth_metrics(entries: list[dict]) -> dict:
    auth_started = sum(1 for entry in entries if entry.get("auth_started"))
    auth_success = sum(1 for entry in entries if entry.get("auth_success"))
    auth_completed_after_start = sum(1 for entry in entries if entry.get("auth_completed_after_start"))
    under_age_true = sum(1 for entry in entries if entry.get("under_age"))
    under_age_known = sum(1 for entry in entries if entry.get("under_age_known"))
    language_counts = Counter(entry.get("user_language", "Unknown") or "Unknown" for entry in entries)
    auth_percentage = (auth_success / auth_started * 100.0) if auth_started else 0.0
    under_age_percentage = (under_age_true / under_age_known * 100.0) if under_age_known else 0.0
    top_language = language_counts.most_common(1)[0][0] if language_counts else "Unknown"
    return {
        "auth_started": auth_started,
        "auth_success": auth_success,
        "auth_completed_after_start": auth_completed_after_start,
        "auth_percentage": auth_percentage,
        "under_age_true": under_age_true,
        "under_age_known": under_age_known,
        "under_age_percentage": under_age_percentage,
        "language_counts": dict(language_counts),
        "top_language": top_language,
    }


def entries_for_selected_key(parsed: dict, selected_key: str) -> list[dict]:
    if selected_key == "all":
        return list(parsed["all_entries"])
    if selected_key == "avg_handle_time":
        return sorted(parsed["all_entries"], key=lambda entry: entry.get("handle_time", 0), reverse=True)
    if selected_key == "avg_satisfaction":
        return sorted(parsed["all_entries"], key=lambda entry: entry.get("satisfaction_score", 0), reverse=True)
    if selected_key == "avg_answer_speed":
        return sorted(parsed["all_entries"], key=lambda entry: entry.get("answer_speed", 0), reverse=True)
    if selected_key == "auth_started":
        return [entry for entry in parsed["all_entries"] if entry.get("auth_started")]
    if selected_key == "auth_success":
        return [entry for entry in parsed["all_entries"] if entry.get("auth_success")]
    if selected_key == "auth_percentage":
        return [entry for entry in parsed["all_entries"] if entry.get("auth_started")]
    if selected_key == "under_age_true":
        return [entry for entry in parsed["all_entries"] if entry.get("under_age")]
    if selected_key == "user_language":
        return [entry for entry in parsed["all_entries"] if (entry.get("user_language") or "Unknown") != "Unknown"]
    return list(parsed["metrics"].get(selected_key, []))


def make_table(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        rows.append(
            {
                "Caller Number": entry.get("caller") or "—",
                "Conversation ID": entry.get("conversation_id") or "—",
                "Queue Code": entry.get("queue_code") or "—",
                "Escalation Number": entry.get("escalation_number") or "—",
                "Intent": (entry.get("intent") or "—").replace("_", " "),
                "Resolution": entry.get("resolution_status") or "—",
                "Channel": entry.get("channel") or "—",
                "Authentication": entry.get("auth_value") or "—",
                "Auth Started": "Yes" if entry.get("auth_started") else "No",
                "Auth Success": "Yes" if entry.get("auth_success") else "No",
                "Under 18": entry.get("under_age_label") or "Unknown",
                "Language": entry.get("user_language") or "Unknown",
                "Handle Time": entry.get("handle_time_fmt") or "—",
                "Satisfaction": round(float(entry.get("satisfaction_score", 0)), 3),
                "Answer Speed": entry.get("answer_speed_fmt") or "—",
                "Created": str(entry.get("created_date") or entry.get("created_at") or "—"),
            }
        )
    return pd.DataFrame(rows)


def filter_entries(
    entries: list[dict],
    search: str,
    resolution_filter: list[str],
    channel_filter: list[str],
    intent_filter: list[str],
    under_age_filter: list[str],
    auth_filter: list[str],
    language_filter: list[str],
    selected_date,
) -> list[dict]:
    search = search.strip().lower()
    out = []
    for entry in entries:
        if resolution_filter and entry.get("resolution_status") not in resolution_filter:
            continue
        if channel_filter and entry.get("channel") not in channel_filter:
            continue
        if intent_filter and entry.get("intent") not in intent_filter:
            continue
        if under_age_filter and entry.get("under_age_label") not in under_age_filter:
            continue
        if auth_filter and (entry.get("auth_value") or "none") not in auth_filter:
            continue
        if language_filter and (entry.get("user_language") or "Unknown") not in language_filter:
            continue
        if selected_date and entry.get("created_date") != selected_date:
            continue

        haystack = " ".join(
            [
                str(entry.get("caller", "")),
                str(entry.get("conversation_id", "")),
                str(entry.get("queue_code", "")),
                str(entry.get("escalation_number", "")),
                str(entry.get("intent", "")),
                str(entry.get("resolution_status", "")),
                str(entry.get("channel", "")),
                str(entry.get("created_date", "")),
                str(entry.get("auth_value", "")),
                str(entry.get("under_age_label", "")),
                str(entry.get("user_language", "")),
            ]
        ).lower()

        if search and search not in haystack:
            continue

        out.append(entry)
    return out


def metric_value(parsed: dict, key: str, computed: dict | None = None):
    computed = computed or {}

    if key == "avg_handle_time":
        return fmt_time(parsed["avg_handle_time"])
    if key == "avg_satisfaction":
        return f"{parsed['avg_satisfaction']:.3f}"
    if key == "avg_answer_speed":
        return f"{parsed['avg_answer_speed']:.1f}s"
    if key == "auth_started":
        return str(computed.get("auth_started", 0))
    if key == "auth_success":
        return str(computed.get("auth_success", 0))
    if key == "auth_percentage":
        return f"{computed.get('auth_percentage', 0.0):.1f}%"
    if key == "under_age_true":
        return str(computed.get("under_age_true", 0))
    if key == "user_language":
        return computed.get("top_language", "Unknown")
    if key == "all":
        return str(len(parsed["all_entries"]))
    return str(len(parsed["metrics"].get(key, [])))


def metric_help() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Total Call Volume", "Total number of conversations analyzed", "Count of all conversations"],
            ["Abandoned Within Amelia", "User dropped while Amelia was handling the conversation", "abandoned = 1 AND not escalated"],
            ["Contained Within Amelia", "Conversation completed within Amelia without escalation", "abandoned = 0 AND not escalated"],
            ["Escalated to Agent", "Conversation was transferred to a human agent", "escalated = True"],
            ["Patient Abandoned Before Escal.", "User dropped after escalation was flagged", "abandoned = 1 AND escalated = True"],
            ["Resolved", "Conversation fully resolved", 'resolution_status = "RESOLVED"'],
            ["Partially Resolved", "Conversation partially resolved", 'resolution_status = "PARTIALLY_RESOLVED"'],
            ["Unresolved", "Conversation did not resolve the issue", 'resolution_status = "UNRESOLVED"'],
            ["Authentication Started", 'Conversation contains authentication = "start"', 'authentication contains "start"'],
            ["Authentication Success", 'Conversation contains authentication = "success"', 'authentication contains "success"'],
            ["Authentication %", "Percent of authentication starts that reached success", "authentication_success / authentication_started"],
            ["Under 18", 'Conversation contains underAge = true', 'underAge = "true"'],
            ["User Language", 'Language captured from userLanguage custom metric', 'Most common value such as English or Spanish'],
            ["Avg Handle Time", "Average conversation duration", "sum(handle_time) / total conversations"],
            ["Avg Satisfaction", "Average user satisfaction score", "sum(satisfaction_score) / total conversations"],
            ["Avg Answer Speed", "Average Amelia response speed", "sum(amelia_answer_speed) / total conversations"],
        ],
        columns=["Metric", "Meaning", "Calculation"],
    )


def kpi_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="padding:14px 16px;border:1px solid #2A2D3E;border-radius:14px;background:#1A1D27;min-height:104px;">
            <div style="font-size:12px;color:#A0A4B8;margin-bottom:8px;">{label}</div>
            <div style="font-size:28px;font-weight:700;color:#FFFFFF;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_theme():
    st.set_page_config(page_title="Amelia Intelligence Dashboard", page_icon="🤖", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #0F1117; color: white; }
        [data-testid="stSidebar"] { background: #141824; }
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 12px; }
        .stTabs [data-baseweb="tab"] {
            background: #1A1D27; border-radius: 10px; padding: 8px 14px; border: 1px solid #2A2D3E;
        }
        div[data-testid="stMetric"] {
            background: #1A1D27; border: 1px solid #2A2D3E; padding: 10px; border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_pdf_report(
    parsed: dict,
    filtered_entries: list[dict],
    selected_label: str,
    selected_date,
    computed: dict,
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleBlue",
        parent=styles["Title"],
        textColor=colors.HexColor("#4F8EF7"),
        fontSize=22,
        spaceAfter=10,
    )

    story = []
    story.append(Paragraph("Amelia Conversation Intelligence Report", title_style))
    story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph(f"Metric Focus: {selected_label}", styles["Normal"]))
    story.append(Paragraph(f"Date Filter: {selected_date if selected_date else 'All dates'}", styles["Normal"]))
    story.append(Paragraph(f"Filtered Conversations: {len(filtered_entries)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    summary_rows = [["Metric", "Value"]]
    for label, key in METRIC_CARDS:
        summary_rows.append([label, metric_value(parsed, key, computed)])

    summary_table = Table(summary_rows, colWidths=[10 * cm, 5 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F8EF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#2A2D3E")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FB"), colors.white]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 14))

    auth_rows = [
        ["Summary", "Value"],
        ["Authentication Started", str(computed.get("auth_started", 0))],
        ["Authentication Success", str(computed.get("auth_success", 0))],
        ["Authentication %", f"{computed.get('auth_percentage', 0.0):.1f}%"],
        ["Under 18", str(computed.get("under_age_true", 0))],
        ["Under 18 %", f"{computed.get('under_age_percentage', 0.0):.1f}%"],
        ["Top User Language", str(computed.get("top_language", "Unknown"))],
    ]
    auth_table = Table(auth_rows, colWidths=[8.5 * cm, 4.0 * cm])
    auth_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A1D27")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7DFEA")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FB")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ]
        )
    )
    story.append(Paragraph("Authentication and Underage Summary", styles["Heading2"]))
    story.append(auth_table)
    story.append(Spacer(1, 12))

    table_df = make_table(filtered_entries)
    max_rows = min(len(table_df), 60)
    data_rows = [[
        "Caller Number",
        "Conversation ID",
        "Queue Code",
        "Escalation Number",
        "Intent",
        "Resolution",
        "Authentication",
        "Under 18",
        "Language",
        "Handle Time",
        "Created",
    ]]
    for _, row in table_df.head(max_rows).iterrows():
        data_rows.append(
            [
                str(row["Caller Number"]),
                str(row["Conversation ID"]),
                str(row["Queue Code"]),
                str(row["Escalation Number"]),
                str(row["Intent"]),
                str(row["Resolution"]),
                str(row["Authentication"]),
                str(row["Under 18"]),
                str(row["Language"]),
                str(row["Handle Time"]),
                str(row["Created"]),
            ]
        )

    convo_table = Table(
        data_rows,
        repeatRows=1,
        colWidths=[2.3 * cm, 3.3 * cm, 2.3 * cm, 4.0 * cm, 2.8 * cm, 2.4 * cm, 2.2 * cm, 1.9 * cm, 2.2 * cm, 1.9 * cm, 2.2 * cm],
    )
    convo_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A1D27")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7DFEA")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FB")]),
                ("FONTSIZE", (0, 0), (-1, -1), 7.0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(Paragraph("Filtered Conversations", styles["Heading2"]))
    story.append(convo_table)

    if len(table_df) > max_rows:
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                f"Note: PDF includes first {max_rows} filtered conversations out of {len(table_df)}.",
                styles["Italic"],
            )
        )

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def main():
    add_theme()
    st.title("🤖 Amelia Conversation Intelligence")
    st.caption(
        "Upload a JSON file, filter conversations, export a PDF summary, and explore transcripts in the browser."
    )

    with st.sidebar:
        st.header("Upload & Filters")
        uploaded = st.file_uploader("Upload conversation JSON", type=["json"])
        st.markdown("---")
        st.write(
            "The dashboard supports metric focus, authentication and under-18 tracking, search, date filtering, transcript exploration, CSV download, and PDF export."
        )

    if not uploaded:
        st.info("Upload a JSON file from the sidebar to generate the dashboard.")
        st.dataframe(metric_help(), use_container_width=True, hide_index=True)
        return

    try:
        conversations = parse_uploaded_json(uploaded)
        parsed = parse_metrics(conversations)
    except Exception as exc:
        st.error(f"Failed to parse JSON: {exc}")
        return

    all_entries = parsed["all_entries"]
    resolution_options = sorted({entry.get("resolution_status", "") for entry in all_entries if entry.get("resolution_status")})
    channel_options = sorted({entry.get("channel", "") for entry in all_entries if entry.get("channel")})
    intent_options = sorted({entry.get("intent", "") for entry in all_entries if entry.get("intent")})
    under_age_options = [
        option for option in ["Under 18", "18+", "Unknown"] if any(entry.get("under_age_label") == option for entry in all_entries)
    ]
    auth_options = sorted({entry.get("auth_value") or "none" for entry in all_entries})
    language_options = sorted({entry.get("user_language") or "Unknown" for entry in all_entries})
    available_dates = parsed.get("available_dates", [])

    with st.sidebar:
        search = st.text_input(
            "Search",
            placeholder="Phone, conversation ID, intent, queue code, escalation number..."
        )
        selected_key = st.selectbox(
            "Metric focus",
            options=[key for _, key in CATEGORY_OPTIONS],
            format_func=lambda key: "All" if key == "all" else CATEGORY_LABELS[key],
        )
        date_mode = st.radio("Date filter", options=["All dates", "Single date"], horizontal=True)
        selected_date = None
        if date_mode == "Single date":
            if available_dates:
                selected_date = st.selectbox(
                    "Choose date",
                    options=available_dates,
                    format_func=lambda value: value.strftime("%Y-%m-%d"),
                )
            else:
                st.caption("No valid conversation dates found in the uploaded JSON.")
        resolution_filter = st.multiselect("Resolution", options=resolution_options)
        channel_filter = st.multiselect("Channel", options=channel_options)
        intent_filter = st.multiselect("Intent", options=intent_options)
        under_age_filter = st.multiselect("Under 18", options=under_age_options)
        auth_filter = st.multiselect("Authentication", options=auth_options)
        language_filter = st.multiselect("Language", options=language_options)

    base_entries = entries_for_selected_key(parsed, selected_key)
    filtered_entries = filter_entries(
        base_entries,
        search,
        resolution_filter,
        channel_filter,
        intent_filter,
        under_age_filter,
        auth_filter,
        language_filter,
        selected_date,
    )
    computed = compute_auth_metrics(filtered_entries)
    selected_label = "All" if selected_key == "all" else CATEGORY_LABELS[selected_key]

    tab_dashboard, tab_explorer, tab_definitions = st.tabs(
        ["Dashboard", "Conversation Explorer", "Metric Definitions"]
    )

    with tab_dashboard:
        for start in range(0, len(METRIC_CARDS), 4):
            cols = st.columns(4)
            for idx, (label, key) in enumerate(METRIC_CARDS[start : start + 4]):
                with cols[idx]:
                    kpi_card(label, metric_value(parsed, key, computed))

        st.markdown("### Active Filter")
        st.info(
            f"Showing **{selected_label}** conversations | "
            f"Date: **{selected_date if selected_date else 'All dates'}** | "
            f"Rows: **{len(filtered_entries)}**"
        )

        st.markdown("### Authentication, Underage, and Language Summary")
        info_col1, info_col2, info_col3, info_col4, info_col5 = st.columns(5)
        info_col1.metric("Authentication Started", computed["auth_started"])
        info_col2.metric("Authentication Success", computed["auth_success"])
        info_col3.metric("Authentication %", f"{computed['auth_percentage']:.1f}%")
        info_col4.metric("Under 18", computed["under_age_true"])
        info_col5.metric("Top User Language", computed["top_language"])
        st.caption(
            "Authentication % = authentication success count divided by authentication started count. Older rows without start are safely ignored in the denominator."
        )

        chart_col1, chart_col2 = st.columns(2)
        call_dist = pd.DataFrame(
            {
                "Category": [
                    "Abandoned (Amelia)",
                    "Contained",
                    "Escalated",
                    "Patient Abandoned",
                ],
                "Count": [
                    len(parsed["metrics"]["abandoned_amelia"]),
                    len(parsed["metrics"]["contained"]),
                    len(parsed["metrics"]["escalated"]),
                    len(parsed["metrics"]["patient_abandoned_before_esc"]),
                ],
            }
        )

        with chart_col1:
            fig = px.pie(call_dist, names="Category", values="Count", title="Call Distribution", hole=0.35)
            fig.update_traces(hovertemplate="%{label}: %{value}<extra></extra>")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            fig = px.bar(call_dist, x="Category", y="Count", title="Volume by Category", text_auto=True)
            fig.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            auth_df = pd.DataFrame(
                {
                    "Stage": ["Started", "Success", "Success %"],
                    "Value": [
                        computed["auth_started"],
                        computed["auth_success"],
                        round(computed["auth_percentage"], 1),
                    ],
                }
            )
            fig = px.bar(auth_df, x="Stage", y="Value", title="Authentication Funnel", text_auto=True)
            fig.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with chart_col4:
            language_df = pd.DataFrame(
                sorted(computed["language_counts"].items(), key=lambda item: item[1], reverse=True),
                columns=["Language", "Count"],
            )
            if not language_df.empty:
                fig = px.pie(language_df, names="Language", values="Count", title="Language Distribution", hole=0.35)
                fig.update_traces(hovertemplate="%{label}: %{value}<extra></extra>")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No language data found.")

        chart_col5, chart_col6 = st.columns(2)
        with chart_col5:
            resolution_df = pd.DataFrame(
                {
                    "Resolution": ["RESOLVED", "PARTIALLY_RESOLVED", "UNRESOLVED"],
                    "Count": [
                        parsed["resolution_counts"].get("RESOLVED", 0),
                        parsed["resolution_counts"].get("PARTIALLY_RESOLVED", 0),
                        parsed["resolution_counts"].get("UNRESOLVED", 0),
                    ],
                }
            )
            fig = px.bar(resolution_df, x="Resolution", y="Count", title="Resolution Status", text_auto=True)
            fig.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with chart_col6:
            top_intents = sorted(parsed["intent_counts"].items(), key=lambda item: item[1], reverse=True)[:8]
            intent_df = pd.DataFrame(top_intents, columns=["Intent", "Count"])
            if not intent_df.empty:
                fig = px.bar(intent_df, x="Intent", y="Count", title="Top Intent Distribution", text_auto=True)
                fig.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No intent data found.")

        trend_df = pd.DataFrame(list(parsed["day_counts"].items()), columns=["Day", "Conversations"])
        if not trend_df.empty:
            fig = px.line(trend_df, x="Day", y="Conversations", markers=True, title="Conversation Trend by Day")
            fig.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Filtered Conversations — {selected_label}")
        st.caption(
            "Use the sidebar to switch between All or a specific metric, then optionally narrow by date, resolution, channel, intent, authentication state, under-18 flag, queue code, or escalation number through search."
        )
        df = make_table(filtered_entries)
        st.dataframe(df, use_container_width=True, hide_index=True)

        action_col1, action_col2 = st.columns([1, 1])
        with action_col1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered CSV",
                data=csv_bytes,
                file_name="amelia_filtered_conversations.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with action_col2:
            pdf_bytes = build_pdf_report(
                parsed,
                filtered_entries,
                selected_label,
                selected_date,
                computed,
            )
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name="amelia_dashboard_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with tab_explorer:
        st.subheader("Conversation Explorer")
        explorer_search = st.text_input(
            "Find a conversation by ID, phone number, queue code, or escalation number",
            key="explorer_search"
        )
        matching_entries = (
            filter_entries(all_entries, explorer_search, [], [], [], [], [], [], None)
            if explorer_search
            else all_entries
        )
        explorer_df = make_table(matching_entries)
        st.dataframe(explorer_df, use_container_width=True, hide_index=True, height=320)

        default_idx = 0
        if explorer_search:
            for idx, entry in enumerate(matching_entries):
                searchable = " ".join(
                    [
                        str(entry.get("conversation_id", "")),
                        str(entry.get("caller", "")),
                        str(entry.get("queue_code", "")),
                        str(entry.get("escalation_number", "")),
                    ]
                ).lower()
                if explorer_search.lower() in searchable:
                    default_idx = idx
                    break

        selected_entry = None
        if matching_entries:
            selected_index = st.selectbox(
                "Select conversation",
                options=list(range(len(matching_entries))),
                index=default_idx,
                format_func=lambda idx: (
                    f"{matching_entries[idx].get('conversation_id') or 'No ID'} | "
                    f"{matching_entries[idx].get('caller') or 'No phone'} | "
                    f"{matching_entries[idx].get('intent') or 'No intent'}"
                ),
            )
            selected_entry = matching_entries[selected_index]

        if selected_entry:
            top1, top2, top3, top4 = st.columns(4)
            top1.metric("Conversation ID", selected_entry.get("conversation_id") or "—")
            top2.metric("Caller Number", selected_entry.get("caller") or "—")
            top3.metric("Queue Code", selected_entry.get("queue_code") or "—")
            top4.metric("Escalation Number", selected_entry.get("escalation_number") or "—")

            top5, top6, top7, top8, top9 = st.columns(5)
            top5.metric("Resolution", selected_entry.get("resolution_status") or "—")
            top6.metric("Handle Time", selected_entry.get("handle_time_fmt") or "—")
            top7.metric("Authentication", selected_entry.get("auth_value") or "—")
            top8.metric("Under 18", selected_entry.get("under_age_label") or "Unknown")
            top9.metric("Language", selected_entry.get("user_language") or "Unknown")

            detail_tabs = st.tabs(["Transcript", "Summary", "Metrics", "Topics", "Raw JSON"])
            with detail_tabs[0]:
                st.text_area("Transcript", transcript_to_text(selected_entry["raw"]), height=420)
            with detail_tabs[1]:
                st.json(
                    {
                        "conversation_id": selected_entry.get("conversation_id"),
                        "caller": selected_entry.get("caller"),
                        "queue_code": selected_entry.get("queue_code"),
                        "escalation_number": selected_entry.get("escalation_number"),
                        "intent": selected_entry.get("intent"),
                        "resolution_status": selected_entry.get("resolution_status"),
                        "channel": selected_entry.get("channel"),
                        "authentication_values": selected_entry.get("auth_values"),
                        "auth_started": selected_entry.get("auth_started"),
                        "auth_success": selected_entry.get("auth_success"),
                        "under_age": selected_entry.get("under_age"),
                        "under_age_known": selected_entry.get("under_age_known"),
                        "user_language": selected_entry.get("user_language"),
                        "handle_time": selected_entry.get("handle_time"),
                        "satisfaction_score": selected_entry.get("satisfaction_score"),
                        "answer_speed": selected_entry.get("answer_speed"),
                        "created_at": selected_entry.get("created_at"),
                    }
                )
            with detail_tabs[2]:
                st.code(pretty_json(selected_entry["raw"].get("metrics", [])), language="json")
            with detail_tabs[3]:
                st.code(pretty_json(selected_entry["raw"].get("topics", [])), language="json")
            with detail_tabs[4]:
                st.code(pretty_json(selected_entry["raw"]), language="json")
        else:
            st.info("No matching conversations found.")

    with tab_definitions:
        st.subheader("Metric Definitions")
        st.dataframe(metric_help(), use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Notes**
            - **All** shows every conversation in the uploaded file.
            - **Metric focus** narrows the table to only that category, such as *Contained Within Amelia*, *Authentication Started*, or *Under 18*.
            - **Single date** shows only conversations whose `conversationCreated` falls on that date.
            - *Patient Abandoned Before Escalation* follows: `abandoned = 1 AND escalated = True`.
            - **Authentication Started** counts rows where `authentication = start` exists.
            - **Authentication Success** counts rows where `authentication = success` exists.
            - **Authentication %** uses `authentication success / authentication started`.
            - **Under 18** counts rows where `underAge = true`.
            - **User Language** comes from the `userLanguage` custom metric and is shown as a top language KPI plus a language distribution chart.
            - **Queue Code** and **Escalation Number** are shown only in the conversation table and search, not in the main KPI dashboard.
            - Older conversations that do not contain the newer authentication metrics are still supported.
            """
        )


if __name__ == "__main__":
    main()
