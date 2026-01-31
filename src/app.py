import streamlit as st
import pandas as pd
import io
import time
import uuid
import json
from streamlit_echarts import st_echarts

import export_core

st.set_page_config(
    page_title="赛博朋克业务数据看板生成器",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session_state
if "saved_charts" not in st.session_state:
    st.session_state.saved_charts = []

if "preview_chart" not in st.session_state:
    st.session_state.preview_chart = None

if "chart_type" not in st.session_state:
    st.session_state.chart_type = "趋势图"

if "chart_style" not in st.session_state:
    st.session_state.chart_style = "默认"

if "chart_dimension2" not in st.session_state:
    st.session_state.chart_dimension2 = "不选择"

if "gauge_target_value" not in st.session_state:
    st.session_state.gauge_target_value = 100.0

if "gauge_agg_mode" not in st.session_state:
    st.session_state.gauge_agg_mode = "汇总"

if "radar_topn" not in st.session_state:
    st.session_state.radar_topn = 5

if "map_geojson" not in st.session_state:
    st.session_state.map_geojson = None

if "map_lon_col" not in st.session_state:
    st.session_state.map_lon_col = "不选择"

if "map_lat_col" not in st.session_state:
    st.session_state.map_lat_col = "不选择"

if "export_task_id" not in st.session_state:
    st.session_state.export_task_id = None

if "export_task_format" not in st.session_state:
    st.session_state.export_task_format = None

if "export_task_email" not in st.session_state:
    st.session_state.export_task_email = ""

if "export_file_bytes" not in st.session_state:
    st.session_state.export_file_bytes = None

if "export_file_sha256" not in st.session_state:
    st.session_state.export_file_sha256 = None

# 删除图表的函数
def delete_chart(chart_id):
    st.session_state.saved_charts = [c for c in st.session_state.saved_charts if c["id"] != chart_id]


def clear_preview():
    st.session_state.preview_chart = None


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

    :root {
        --bg-color: #E6F3FF;
        --card-bg: rgba(255, 255, 255, 0.95);
        --primary-blue: #1277D1;
        --accent-blue: #2FA3FF;
        --accent-orange: #FF8A3D;
        --text-main: #0B1B33;
        --text-sub: #3A4A63;
        --border-color: #A7C9FF;
    }

    .stApp {
        background-color: var(--bg-color);
        background-image: linear-gradient(rgba(18, 119, 209, 0.06) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(18, 119, 209, 0.06) 1px, transparent 1px);
        background-size: 28px 28px;
        color: var(--text-main);
    }

    section[data-testid="stSidebar"] {
        background-color: #F3F8FF;
        border-right: 1px solid var(--border-color);
        box-shadow: 6px 0 18px rgba(18, 119, 209, 0.08);
    }

    h1, h2, h3, .stMetricLabel {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.8px;
    }

    h1 {
        color: var(--primary-blue) !important;
        text-shadow: 0 0 10px rgba(18, 119, 209, 0.35);
        border-bottom: 2px solid var(--primary-blue);
        padding-bottom: 8px;
        background: linear-gradient(90deg, rgba(18, 119, 209, 0.15) 0%, transparent 100%);
    }

    h3 {
        color: var(--text-main) !important;
        border-left: 4px solid var(--accent-blue);
        padding-left: 10px;
    }

    .tech-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 14px;
        margin-bottom: 18px;
        position: relative;
        box-shadow: inset 0 0 18px rgba(18, 119, 209, 0.08), 0 8px 20px rgba(12, 35, 66, 0.06);
        clip-path: polygon(0 0, 100% 0, 100% calc(100% - 14px), calc(100% - 14px) 100%, 0 100%);
    }

    .tech-card::before,
    .tech-card::after {
        content: '';
        position: absolute;
        width: 10px;
        height: 10px;
    }

    .tech-card::before {
        top: 0;
        left: 0;
        border-top: 2px solid var(--primary-blue);
        border-left: 2px solid var(--primary-blue);
    }

    .tech-card::after {
        bottom: 0;
        right: 0;
        border-bottom: 2px solid var(--primary-blue);
        border-right: 2px solid var(--primary-blue);
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary-blue) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700;
        font-size: 2.2rem !important;
        text-shadow: 0 0 8px rgba(18, 119, 209, 0.35);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-sub) !important;
        font-size: 0.9rem !important;
    }

    .stButton>button {
        background: white;
        color: var(--primary-blue);
        border: 1px solid var(--primary-blue);
        font-family: 'Orbitron', sans-serif;
        transition: all 0.3s;
        box-shadow: 0 0 10px rgba(18, 119, 209, 0.15);
    }

    .stButton>button:hover {
        background: rgba(18, 119, 209, 0.12);
        box-shadow: 0 0 16px rgba(18, 119, 209, 0.28);
        color: #0B1B33;
    }

    .stSelectbox, .stMultiSelect {
        color: var(--text-main);
    }

    div[data-baseweb="select"] > div {
        background-color: white !important;
        border-color: #C7D9F5 !important;
        color: var(--text-main) !important;
    }

    canvas {
        filter: drop-shadow(0 0 4px rgba(18, 119, 209, 0.25));
    }
</style>
""",
    unsafe_allow_html=True
)

st.title("赛博朋克业务数据看板生成器")
st.markdown('<div class="tech-card">数据可视化系统 V1.0</div>', unsafe_allow_html=True)

st.sidebar.header("1. 数据接入")
uploaded_file = st.sidebar.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])


@st.cache_data
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None


def detect_date_columns(dataframe):
    date_cols = []
    for col in dataframe.columns:
        # 严禁将数值类型误判为日期
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            continue
            
        try:
            parsed = pd.to_datetime(dataframe[col], errors="coerce")
            # 只有当解析成功率超过 80% 时才认为是日期列
            if parsed.notna().mean() >= 0.8:
                date_cols.append(col)
        except Exception:
            continue
    return date_cols


def apply_date_highlight(dataframe, date_cols):
    def style_column(col):
        if col.name in date_cols:
            return ["background-color: #FFF4CC; color: #1F3B5B; font-weight: 600;"] * len(col)
        return [""] * len(col)

    return dataframe.style.apply(style_column)


def format_number(value):
    try:
        v = float(value)
    except Exception:
        return str(value)
    if pd.isna(v):
        return "—"
    if abs(v) >= 1_0000_0000:
        return f"{v/1_0000_0000:.2f}亿"
    if abs(v) >= 1_0000:
        return f"{v/1_0000:.2f}万"
    return f"{v:,.2f}"


def get_preview_title(chart_conf):
    ctype = chart_conf.get("type")
    metrics = chart_conf.get("metrics") or []
    dimension = chart_conf.get("dimension")
    if ctype == "趋势图":
        if metrics:
            return " · ".join(metrics) + " 趋势"
        return "趋势图"
    if ctype == "占比图":
        if metrics and dimension and dimension != "不选择":
            return f"{metrics[0]} 按 {dimension} 占比"
        return "占比图"
    if ctype == "排名图":
        if metrics and dimension and dimension != "不选择":
            return f"{metrics[0]} 按 {dimension} 排名"
        return "排名图"
    if ctype == "漏斗图":
        if len(metrics) > 1:
            return " → ".join(metrics) + " 转化漏斗"
        if metrics and dimension and dimension != "不选择":
            return f"{metrics[0]} 按 {dimension} 漏斗"
        return "漏斗图"
    if ctype == "箱线图":
        if metrics and dimension and dimension != "不选择":
            return f"{metrics[0]} 按 {dimension} 分布（箱线）"
        return "箱线图"
    if ctype == "散点图":
        if len(metrics) >= 2:
            return f"{metrics[0]} vs {metrics[1]} 散点"
        return "散点图"
    if ctype == "热力图":
        dim2 = chart_conf.get("dimension2")
        if metrics and dimension and dim2 and dimension != "不选择" and dim2 != "不选择":
            return f"{metrics[0]} 在 {dimension}×{dim2} 热力"
        return "热力图"
    if ctype == "雷达图":
        if metrics and dimension and dimension != "不选择":
            return f"{dimension} 多指标雷达"
        return "雷达图"
    if ctype == "仪表盘":
        if metrics:
            return f"{metrics[0]} 仪表盘"
        return "仪表盘"
    if ctype == "桑基图":
        dim2 = chart_conf.get("dimension2")
        if metrics and dimension and dim2 and dimension != "不选择" and dim2 != "不选择":
            return f"{metrics[0]} 从 {dimension} 到 {dim2} 流向"
        return "桑基图"
    if ctype == "地图":
        if metrics and dimension and dimension != "不选择":
            return f"{metrics[0]} 地图分布"
        return "地图"
    return str(ctype)


def style_candidates_for(chart_type: str) -> list[str]:
    chart_type = str(chart_type)
    if chart_type == "趋势图":
        return ["大屏发光", "面积", "折线"]
    if chart_type == "排名图":
        return ["实时排名赛", "横向", "竖向"]
    if chart_type == "占比图":
        return ["环形", "玫瑰", "饼图"]
    if chart_type == "散点图":
        return ["涟漪", "普通"]
    if chart_type == "仪表盘":
        return ["半圆", "全圆"]
    if chart_type == "雷达图":
        return ["对比", "单体"]
    if chart_type == "热力图":
        return ["矩阵"]
    if chart_type == "桑基图":
        return ["默认"]
    if chart_type == "地图":
        return ["散点叠加", "航线叠加", "区域填色"]
    return ["默认"]


def _geojson_feature_name(feature: dict) -> str:
    props = feature.get("properties") or {}
    for k in ("name", "NAME", "Name", "fullname", "FULLNAME", "fullName"):
        v = props.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _geojson_feature_centroid(feature: dict):
    geom = feature.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None

    pts = []
    try:
        if gtype == "Polygon":
            rings = coords
            for ring in rings:
                for p in ring:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append((float(p[0]), float(p[1])))
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for p in ring:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            pts.append((float(p[0]), float(p[1])))
    except Exception:
        return None

    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def render_map_chart(
    df,
    metrics,
    dimension,
    *,
    style,
    dimension2,
    lon_col,
    lat_col,
    map_geojson,
):
    if not metrics:
        st.warning("地图需要选择【一个核心指标】")
        return
    metric = metrics[0]
    if metric not in df.columns:
        st.warning("地图配置无效")
        return

    style = str(style or "散点叠加")

    if isinstance(map_geojson, dict) and map_geojson.get("type") == "FeatureCollection" and map_geojson.get("features"):
        map_name = "CUSTOM_MAP"
        features = map_geojson.get("features") or []
        name_to_centroid = {}
        for f in features:
            n = _geojson_feature_name(f)
            c = _geojson_feature_centroid(f)
            if n and c:
                name_to_centroid[n] = c

        g = None
        if dimension and dimension != "不选择" and dimension in df.columns:
            tmp = df[[dimension, metric]].dropna().copy()
            tmp[dimension] = tmp[dimension].astype(str)
            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
            tmp = tmp.dropna(subset=[metric])
            g = tmp.groupby(dimension)[metric].sum(numeric_only=True).reset_index()
        else:
            g = pd.DataFrame({dimension: ["总计"], metric: [pd.to_numeric(df[metric], errors="coerce").sum()]})

        g = g.sort_values(by=metric, ascending=False)
        vmax = float(g[metric].max() or 0)
        if vmax <= 0:
            vmax = 1.0

        base_geo = {
            "map": map_name,
            "roam": True,
            "label": {"show": False},
            "itemStyle": {"areaColor": "#E6F3FF", "borderColor": "#A7C9FF"},
            "emphasis": {"itemStyle": {"areaColor": "#CCE5FF"}},
        }

        series = []
        if style == "区域填色":
            series.append(
                {
                    "type": "map",
                    "map": map_name,
                    "geoIndex": 0,
                    "data": [
                        {"name": str(r[dimension]), "value": float(r[metric] or 0)}
                        for _, r in g.iterrows()
                        if str(r[dimension])
                    ],
                }
            )

        if style in {"散点叠加", "航线叠加"}:
            points = []
            for _, r in g.head(80).iterrows():
                name = str(r[dimension])
                c = name_to_centroid.get(name)
                if not c:
                    continue
                points.append({"name": name, "value": [c[0], c[1], float(r[metric] or 0)]})

            if points:
                series.append(
                    {
                        "type": "effectScatter" if style == "散点叠加" else "scatter",
                        "coordinateSystem": "geo",
                        "symbolSize": 10,
                        "data": points,
                        "itemStyle": {"color": "#1277D1"},
                        "rippleEffect": {"scale": 3, "brushType": "stroke"} if style == "散点叠加" else None,
                    }
                )

        if style == "航线叠加" and dimension2 and dimension2 != "不选择" and dimension2 in df.columns and dimension in df.columns:
            tmp = df[[dimension, dimension2, metric]].dropna().copy()
            tmp[dimension] = tmp[dimension].astype(str)
            tmp[dimension2] = tmp[dimension2].astype(str)
            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
            tmp = tmp.dropna(subset=[metric])
            gg = tmp.groupby([dimension, dimension2])[metric].sum(numeric_only=True).reset_index()
            gg = gg.sort_values(by=metric, ascending=False).head(50)
            lines = []
            for _, r in gg.iterrows():
                s = str(r[dimension])
                t = str(r[dimension2])
                cs = name_to_centroid.get(s)
                ct = name_to_centroid.get(t)
                if not cs or not ct:
                    continue
                lines.append({"fromName": s, "toName": t, "coords": [[cs[0], cs[1]], [ct[0], ct[1]]], "value": float(r[metric] or 0)})

            if lines:
                series.append(
                    {
                        "type": "lines",
                        "coordinateSystem": "geo",
                        "zlevel": 2,
                        "effect": {"show": True, "symbol": "arrow", "symbolSize": 8},
                        "lineStyle": {"width": 1.2, "opacity": 0.45, "curveness": 0.2, "color": "#2FA3FF"},
                        "data": lines,
                    }
                )

        series = [s for s in series if s]
        for s in series:
            if s.get("rippleEffect") is None:
                s.pop("rippleEffect", None)

        option = {
            "title": {"text": f"{metric} 地图", "textStyle": {"color": "#1277D1", "fontSize": 16}},
            "tooltip": {"trigger": "item"},
            "geo": base_geo,
            "visualMap": {
                "min": 0,
                "max": vmax,
                "left": "left",
                "bottom": 0,
                "inRange": {"color": ["#CCE5FF", "#2FA3FF", "#1277D1"]},
                "textStyle": {"color": "#3A4A63"},
                "calculable": True,
                "show": style == "区域填色",
            },
            "series": series,
        }
        st_echarts(options=option, height="420px", key=f"chart_{uuid.uuid4()}", map={"geoJSON": map_geojson, "name": map_name})
        return

    if lon_col and lat_col and lon_col != "不选择" and lat_col != "不选择" and lon_col in df.columns and lat_col in df.columns:
        tmp = df[[lon_col, lat_col, metric] + ([dimension] if dimension and dimension != "不选择" and dimension in df.columns else [])].dropna().copy()
        tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
        tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna(subset=[lon_col, lat_col, metric]).head(3000)
        data = tmp[[lon_col, lat_col, metric]].values.tolist()
        option = {
            "title": {"text": f"{metric} 坐标分布", "textStyle": {"color": "#1277D1", "fontSize": 16}},
            "tooltip": {"trigger": "item"},
            "xAxis": {"type": "value", "name": lon_col, "axisLabel": {"color": "#3A4A63"}},
            "yAxis": {"type": "value", "name": lat_col, "axisLabel": {"color": "#3A4A63"}},
            "series": [
                {
                    "type": "scatter",
                    "data": data,
                    "symbolSize": 8,
                    "itemStyle": {"color": "#1277D1", "opacity": 0.75},
                }
            ],
        }
        st_echarts(options=option, height="420px", key=f"chart_{uuid.uuid4()}")
        return

    st.warning("地图需要上传 GeoJSON（可选）或选择经纬度列（lon/lat）")


def generate_description(chart_conf, df):
    ctype = chart_conf.get("type")
    metrics = chart_conf.get("metrics") or []
    dimension = chart_conf.get("dimension")
    x_axis_mode = chart_conf.get("x_axis_mode")

    lines = []

    if ctype == "趋势图" and metrics:
        metric = metrics[0]
        if "时间列:" in str(x_axis_mode):
            time_col = str(x_axis_mode).split(": ", 1)[1]
            if time_col in df.columns:
                tmp = df[[time_col, metric]].dropna().copy()
                if pd.api.types.is_datetime64_any_dtype(tmp[time_col]):
                    tmp["__t"] = tmp[time_col].dt.strftime("%Y-%m-%d")
                else:
                    tmp["__t"] = tmp[time_col].astype(str)
                g = tmp.groupby("__t")[metric].sum().reset_index().sort_values("__t")
                if len(g) >= 2:
                    peak_row = g.iloc[g[metric].values.argmax()]
                    min_row = g.iloc[g[metric].values.argmin()]
                    lines.append(
                        f"{metric} 在 {peak_row['__t']} 达到峰值，数值为 {format_number(peak_row[metric])}，随后呈现回落/平稳趋势。"
                    )
                    lines.append(
                        f"{metric} 在 {min_row['__t']} 触及阶段低点，数值为 {format_number(min_row[metric])}，建议结合活动节奏核对原因。"
                    )
        if not lines:
            total = df[metric].sum() if metric in df.columns else None
            if total is not None:
                lines.append(f"{metric} 汇总为 {format_number(total)}，整体呈现波动变化，需关注峰值与拐点。")

    if ctype == "漏斗图" and metrics:
        if len(metrics) > 1:
            sums = df[metrics].sum()
            rates = []
            for i in range(len(metrics) - 1):
                a = float(sums.get(metrics[i], 0) or 0)
                b = float(sums.get(metrics[i + 1], 0) or 0)
                r = (b / a) if a > 0 else 0.0
                rates.append((metrics[i], metrics[i + 1], r))
            if rates:
                worst = min(rates, key=lambda x: x[2])
                lines.append(
                    f"用户在 {worst[0]} 到 {worst[1]} 的流转过程中出现断崖式下跌，转化率仅为 {worst[2]*100:.1f}%，是核心转化瓶颈。"
                )
                overall = float(sums.get(metrics[-1], 0) or 0) / float(sums.get(metrics[0], 0) or 1)
                lines.append(
                    f"从 {metrics[0]} 到 {metrics[-1]} 的全链路转化率为 {overall*100:.1f}%，建议优先优化瓶颈环节的加载/交互门槛。"
                )
        elif dimension and dimension != "不选择":
            metric = metrics[0]
            g = df.groupby(dimension)[metric].sum().sort_values(ascending=False)
            if len(g) >= 2:
                top = g.index[0]
                second = g.index[1]
                lines.append(
                    f"{top} 是本期主力阵地，{metric} 贡献显著高于其他分组；与 {second} 的差距体现出明显头部效应。"
                )

    if ctype in {"排名图", "占比图"} and metrics and dimension and dimension != "不选择":
        metric = metrics[0]
        g = df.groupby(dimension)[metric].sum().sort_values(ascending=False)
        if len(g) >= 1:
            total = float(g.sum() or 1)
            top_name = g.index[0]
            top_val = float(g.iloc[0] or 0)
            top_share = top_val / total
            lines.append(
                f"{top_name} 贡献了 {top_share*100:.1f}% 的 {metric}，是本期活动的主力阵地。"
            )
            if len(g) >= 5:
                head_share = float(g.iloc[:3].sum() or 0) / total
                lines.append(
                    f"Top3 合计贡献 {head_share*100:.1f}% 的 {metric}，呈现明显头部效应与二八原则特征。"
                )

    if ctype == "箱线图" and metrics and dimension and dimension != "不选择":
        metric = metrics[0]
        g = df.groupby(dimension)[metric]
        med = g.median().sort_values(ascending=False)
        if len(med) >= 2:
            lines.append(
                f"不同 {dimension} 的 {metric} 分布差异显著，{med.index[0]} 的中位数最高（{format_number(med.iloc[0])}），{med.index[-1]} 最低（{format_number(med.iloc[-1])}）。"
            )
            lines.append(
                f"建议关注波动区间较大的分组，其稳定性可能影响整体指标表现（需结合IQR与异常点进一步验证）。"
            )

    return lines


def render_trend_chart(df, metrics, x_axis_mode, style):
    if not metrics:
        st.warning("趋势图需要选择【核心指标】")
        return

    if "时间列:" in x_axis_mode:
        time_col = x_axis_mode.split(": ")[1]
        df_trend = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_trend[time_col]):
            x_data = df_trend[time_col].dt.strftime("%Y-%m-%d").tolist()
        else:
            x_data = df_trend[time_col].astype(str).tolist()
    elif "类别列:" in x_axis_mode:
        cat_col = x_axis_mode.split(": ")[1]
        df_trend = df.groupby(cat_col)[metrics].sum().reset_index()
        x_data = df_trend[cat_col].astype(str).tolist()
    else:
        df_trend = df.reset_index(drop=True)
        x_data = df_trend.index.tolist()

    style = str(style or "面积")
    series = []
    colors = ["#1277D1", "#2FA3FF", "#FF8A3D", "#FF5252", "#8C52FF"]
    for idx, metric in enumerate(metrics):
        area_style = None
        line_style = {"width": 2}
        item_style = {"color": colors[idx % len(colors)]}
        if style == "面积":
            area_style = {
                "color": {
                    "type": "linear",
                    "x": 0,
                    "y": 0,
                    "x2": 0,
                    "y2": 1,
                    "colorStops": [
                        {"offset": 0, "color": f"{colors[idx % len(colors)]}88"},
                        {"offset": 1, "color": f"{colors[idx % len(colors)]}00"},
                    ],
                }
            }
        if style == "大屏发光":
            line_style = {"width": 3, "shadowBlur": 12, "shadowColor": colors[idx % len(colors)]}
            item_style = {"color": colors[idx % len(colors)], "shadowBlur": 12, "shadowColor": colors[idx % len(colors)]}
        series.append(
            {
                "name": metric,
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "data": df_trend[metric].tolist(),
                "itemStyle": item_style,
                "lineStyle": line_style,
                "areaStyle": area_style,
            }
        )

    option = {
        "title": {"text": "趋势分析", "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": metrics, "textStyle": {"color": "#3A4A63"}, "top": 25},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": x_data,
            "axisLabel": {"color": "#3A4A63"},
            "axisLine": {"lineStyle": {"color": "#A7C9FF"}},
        },
        "yAxis": {
            "type": "value",
            "splitLine": {
                "lineStyle": {"color": "rgba(167, 201, 255, 0.3)", "type": "dashed"}
            },
            "axisLabel": {"color": "#3A4A63"},
        },
        "series": series,
    }
    st_echarts(options=option, height="300px", key=f"chart_{uuid.uuid4()}")


def render_pie_chart(df, metrics, dimension, style):
    if not metrics or dimension == "不选择":
        st.warning("占比图需要选择【核心指标】和【分组/对比维度】")
        return

    metric = metrics[0]
    pie_data = df.groupby(dimension)[metric].sum().reset_index().sort_values(by=metric, ascending=False)
    pie_chart_data = [{"value": row[metric], "name": str(row[dimension])} for _, row in pie_data.iterrows()]

    pie_style = str(style or "环形")
    rose = None
    radius = ["40%", "70%"]
    if pie_style == "玫瑰":
        rose = "radius"
        radius = ["20%", "70%"]
    if pie_style == "饼图":
        rose = None
        radius = "65%"

    option = {
        "title": {"text": f"{metric} 占比", "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "tooltip": {"trigger": "item"},
        "legend": {
            "orient": "vertical",
            "left": "left",
            "textStyle": {"color": "#3A4A63"},
            "top": 25,
        },
        "series": [
            {
                "name": metric,
                "type": "pie",
                "radius": radius,
                "roseType": rose,
                "avoidLabelOverlap": False,
                "itemStyle": {"borderRadius": 5, "borderColor": "#fff", "borderWidth": 2},
                "label": {"show": False, "position": "center"},
                "emphasis": {"label": {"show": True, "fontSize": "18", "fontWeight": "bold"}},
                "labelLine": {"show": False},
                "data": pie_chart_data,
            }
        ],
        "color": ["#1277D1", "#2FA3FF", "#66B2FF", "#99CCFF", "#CCE5FF", "#FF8A3D"],
    }
    st_echarts(options=option, height="300px", key=f"chart_{uuid.uuid4()}")


def render_bar_chart(df, metrics, dimension, style):
    if not metrics or dimension == "不选择":
        st.warning("排名图需要选择【核心指标】和【分组/对比维度】")
        return

    metric = metrics[0]
    bar_data = df.groupby(dimension)[metric].sum().reset_index()
    top_data = bar_data.sort_values(by=metric, ascending=True).tail(10)

    bar_style = str(style or "横向")
    categories = top_data[dimension].astype(str).tolist()
    values = top_data[metric].tolist()

    if bar_style == "竖向":
        x_axis = {"type": "category", "data": categories, "axisLabel": {"color": "#3A4A63"}}
        y_axis = {
            "type": "value",
            "axisLabel": {"color": "#3A4A63"},
            "splitLine": {"lineStyle": {"color": "rgba(167, 201, 255, 0.3)", "type": "dashed"}},
        }
        series_data = values
        axis_series = {"xAxis": x_axis, "yAxis": y_axis}
    else:
        x_axis = {
            "type": "value",
            "axisLabel": {"color": "#3A4A63"},
            "splitLine": {"lineStyle": {"color": "rgba(167, 201, 255, 0.3)", "type": "dashed"}},
        }
        y_axis = {"type": "category", "data": categories, "axisLabel": {"color": "#3A4A63"}}
        series_data = values
        axis_series = {"xAxis": x_axis, "yAxis": y_axis}

    realtime_sort = bar_style == "实时排名赛" and bar_style != "竖向"

    option = {
        "title": {"text": f"{metric} Top 10", "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        **axis_series,
        "animationDuration": 800,
        "animationDurationUpdate": 800,
        "series": [
            {
                "name": metric,
                "type": "bar",
                "data": series_data,
                "realtimeSort": realtime_sort,
                "label": {"show": realtime_sort, "position": "right", "color": "#0B1B33"},
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0,
                        "y": 0,
                        "x2": 1,
                        "y2": 0,
                        "colorStops": [{"offset": 0, "color": "#2FA3FF"}, {"offset": 1, "color": "#1277D1"}],
                    },
                    "borderRadius": [0, 4, 4, 0],
                },
            }
        ],
    }
    st_echarts(options=option, height="300px", key=f"chart_{uuid.uuid4()}")


def render_funnel_chart(df, metrics, dimension):
    data = []
    title_text = "转化漏斗"

    if dimension != "不选择" and metrics:
        metric = metrics[0]
        funnel_data = df.groupby(dimension)[metric].sum().reset_index().sort_values(by=metric, ascending=False)
        data = [{"value": row[metric], "name": str(row[dimension])} for _, row in funnel_data.iterrows()]
        title_text = f"{metric} 按 {dimension} 漏斗"
    elif len(metrics) > 1:
        sums = df[metrics].sum().sort_values(ascending=False)
        data = [{"value": val, "name": name} for name, val in sums.items()]
        title_text = "多指标转化漏斗"
    else:
        st.warning("漏斗图需选择【多个指标】作为层级，或选择【一个指标+一个维度】")
        return

    option = {
        "title": {"text": title_text, "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b} : {c}"},
        "legend": {"data": [d["name"] for d in data], "textStyle": {"color": "#3A4A63"}, "top": 25},
        "series": [
            {
                "name": "漏斗",
                "type": "funnel",
                "left": "10%",
                "top": 60,
                "bottom": 60,
                "width": "80%",
                "min": 0,
                "max": data[0]["value"] if data else 100,
                "minSize": "0%",
                "maxSize": "100%",
                "sort": "descending",
                "gap": 2,
                "label": {"show": True, "position": "inside"},
                "labelLine": {"length": 10, "lineStyle": {"width": 1, "type": "solid"}},
                "itemStyle": {"borderColor": "#fff", "borderWidth": 1},
                "emphasis": {"label": {"fontSize": 20}},
                "data": data,
            }
        ],
        "color": ["#1277D1", "#2FA3FF", "#66B2FF", "#99CCFF", "#FF8A3D"],
    }
    st_echarts(options=option, height="300px", key=f"chart_{uuid.uuid4()}")


def render_boxplot_chart(df, metrics, dimension):
    if not metrics or dimension == "不选择":
        st.warning("箱线图需要选择【核心指标】和【分组/对比维度】")
        return
    metric = metrics[0]
    if metric not in df.columns or dimension not in df.columns:
        st.warning("箱线图配置无效")
        return
    grouped = df[[dimension, metric]].dropna().copy()
    grouped[dimension] = grouped[dimension].astype(str)
    medians = grouped.groupby(dimension)[metric].median().sort_values(ascending=False)
    cats = medians.index.tolist()[:20]
    grouped = grouped[grouped[dimension].isin(cats)]

    data = []
    for c in cats:
        values = grouped[grouped[dimension] == c][metric]
        if len(values) == 0:
            continue
        q1 = float(values.quantile(0.25))
        q2 = float(values.quantile(0.50))
        q3 = float(values.quantile(0.75))
        vmin = float(values.min())
        vmax = float(values.max())
        data.append([vmin, q1, q2, q3, vmax])

    option = {
        "title": {"text": f"{metric} 分布（箱线）", "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "tooltip": {"trigger": "item"},
        "grid": {"left": "3%", "right": "4%", "bottom": "10%", "containLabel": True},
        "xAxis": {"type": "category", "data": cats, "axisLabel": {"color": "#3A4A63", "rotate": 25}},
        "yAxis": {
            "type": "value",
            "axisLabel": {"color": "#3A4A63"},
            "splitLine": {"lineStyle": {"color": "rgba(167, 201, 255, 0.3)", "type": "dashed"}},
        },
        "series": [
            {
                "name": metric,
                "type": "boxplot",
                "data": data,
                "itemStyle": {"color": "#2FA3FF", "borderColor": "#1277D1"},
            }
        ],
    }
    st_echarts(options=option, height="300px", key=f"chart_{uuid.uuid4()}")


def render_scatter_chart(df, metrics, dimension, style):
    if len(metrics) < 2:
        st.warning("散点图需要选择【两个核心指标】")
        return
    x_metric, y_metric = metrics[0], metrics[1]
    if x_metric not in df.columns or y_metric not in df.columns:
        st.warning("散点图配置无效")
        return

    scatter_style = str(style or "普通")
    series_type = "effectScatter" if scatter_style == "涟漪" else "scatter"

    tmp = df[[x_metric, y_metric] + ([dimension] if dimension and dimension != "不选择" else [])].dropna().copy()
    tmp[x_metric] = pd.to_numeric(tmp[x_metric], errors="coerce")
    tmp[y_metric] = pd.to_numeric(tmp[y_metric], errors="coerce")
    tmp = tmp.dropna(subset=[x_metric, y_metric])
    tmp = tmp.head(2000)

    series = []
    colors = ["#1277D1", "#2FA3FF", "#FF8A3D", "#FF5252", "#8C52FF"]
    if dimension and dimension != "不选择" and dimension in tmp.columns:
        top_cats = tmp[dimension].astype(str).value_counts().head(8).index.tolist()
        for idx, c in enumerate(top_cats):
            sub = tmp[tmp[dimension].astype(str) == c]
            series.append(
                {
                    "name": str(c),
                    "type": series_type,
                    "data": sub[[x_metric, y_metric]].values.tolist(),
                    "symbolSize": 9,
                    "itemStyle": {"color": colors[idx % len(colors)]},
                    "rippleEffect": {"scale": 3, "brushType": "stroke"} if series_type == "effectScatter" else None,
                }
            )
    else:
        series.append(
            {
                "name": f"{x_metric} vs {y_metric}",
                "type": series_type,
                "data": tmp[[x_metric, y_metric]].values.tolist(),
                "symbolSize": 9,
                "itemStyle": {"color": "#1277D1"},
                "rippleEffect": {"scale": 3, "brushType": "stroke"} if series_type == "effectScatter" else None,
            }
        )

    series = [s for s in series if s.get("data")]
    for s in series:
        if s.get("rippleEffect") is None:
            s.pop("rippleEffect", None)

    option = {
        "title": {"text": f"{x_metric} vs {y_metric}", "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "tooltip": {"trigger": "item"},
        "legend": {"show": len(series) > 1, "textStyle": {"color": "#3A4A63"}, "top": 25},
        "grid": {"left": "3%", "right": "4%", "bottom": "8%", "containLabel": True},
        "xAxis": {"type": "value", "name": x_metric, "axisLabel": {"color": "#3A4A63"}},
        "yAxis": {"type": "value", "name": y_metric, "axisLabel": {"color": "#3A4A63"}},
        "series": series,
    }
    st_echarts(options=option, height="320px", key=f"chart_{uuid.uuid4()}")


def render_heatmap_chart(df, metrics, dimension, dimension2):
    if not metrics or dimension == "不选择" or dimension2 == "不选择":
        st.warning("热力图需要选择【核心指标】和【两个维度】")
        return
    metric = metrics[0]
    if metric not in df.columns or dimension not in df.columns or dimension2 not in df.columns:
        st.warning("热力图配置无效")
        return

    tmp = df[[dimension, dimension2, metric]].dropna().copy()
    tmp[dimension] = tmp[dimension].astype(str)
    tmp[dimension2] = tmp[dimension2].astype(str)

    top_x = tmp.groupby(dimension)[metric].sum().sort_values(ascending=False).head(20).index.tolist()
    top_y = tmp.groupby(dimension2)[metric].sum().sort_values(ascending=False).head(20).index.tolist()
    tmp = tmp[tmp[dimension].isin(top_x) & tmp[dimension2].isin(top_y)]

    pivot = tmp.groupby([dimension2, dimension])[metric].sum().reset_index()
    x_labels = top_x
    y_labels = top_y
    x_index = {v: i for i, v in enumerate(x_labels)}
    y_index = {v: i for i, v in enumerate(y_labels)}

    data = []
    vmax = 0.0
    for _, row in pivot.iterrows():
        x = x_index.get(str(row[dimension]))
        y = y_index.get(str(row[dimension2]))
        v = float(row[metric] or 0)
        vmax = max(vmax, v)
        if x is not None and y is not None:
            data.append([x, y, v])

    option = {
        "title": {"text": f"{metric} 热力分布", "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "tooltip": {"position": "top"},
        "grid": {"left": "7%", "right": "4%", "bottom": "12%", "containLabel": True},
        "xAxis": {"type": "category", "data": x_labels, "axisLabel": {"color": "#3A4A63", "rotate": 25}},
        "yAxis": {"type": "category", "data": y_labels, "axisLabel": {"color": "#3A4A63"}},
        "visualMap": {
            "min": 0,
            "max": max(1.0, vmax),
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
            "inRange": {"color": ["#CCE5FF", "#2FA3FF", "#1277D1"]},
        },
        "series": [
            {
                "name": metric,
                "type": "heatmap",
                "data": data,
                "label": {"show": False},
                "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.25)"}},
            }
        ],
    }
    st_echarts(options=option, height="340px", key=f"chart_{uuid.uuid4()}")


def render_radar_chart(df, metrics, dimension, style, topn: int):
    if not metrics or dimension == "不选择":
        st.warning("雷达图需要选择【多个核心指标】和【分组/对比维度】")
        return
    use_metrics = metrics[:6]
    if any(m not in df.columns for m in use_metrics) or dimension not in df.columns:
        st.warning("雷达图配置无效")
        return

    tmp = df[[dimension] + use_metrics].dropna().copy()
    tmp[dimension] = tmp[dimension].astype(str)
    g = tmp.groupby(dimension)[use_metrics].sum(numeric_only=True).reset_index()
    sort_metric = use_metrics[0]
    g = g.sort_values(by=sort_metric, ascending=False).head(max(1, int(topn or 5)))

    indicators = []
    for m in use_metrics:
        vmax = float(g[m].max() or 0)
        indicators.append({"name": m, "max": max(1.0, vmax * 1.2)})

    radar_style = str(style or "对比")
    series_data = []
    for _, row in g.iterrows():
        series_data.append({"name": str(row[dimension]), "value": [float(row[m] or 0) for m in use_metrics]})

    if radar_style == "单体" and series_data:
        series_data = [series_data[0]]

    option = {
        "title": {"text": "多维指标对比", "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "tooltip": {},
        "legend": {"data": [d["name"] for d in series_data], "textStyle": {"color": "#3A4A63"}, "top": 25},
        "radar": {
            "indicator": indicators,
            "splitArea": {"areaStyle": {"color": ["rgba(18, 119, 209, 0.06)"]}},
            "axisLine": {"lineStyle": {"color": "rgba(167, 201, 255, 0.7)"}},
            "splitLine": {"lineStyle": {"color": "rgba(167, 201, 255, 0.45)"}},
        },
        "series": [
            {
                "type": "radar",
                "data": series_data,
                "areaStyle": {"opacity": 0.12},
            }
        ],
        "color": ["#1277D1", "#2FA3FF", "#FF8A3D", "#8C52FF", "#FF5252"],
    }
    st_echarts(options=option, height="340px", key=f"chart_{uuid.uuid4()}")


def render_gauge_chart(df, metrics, *, target_value: float, agg_mode: str, style: str):
    if not metrics:
        st.warning("仪表盘需要选择【一个核心指标】")
        return
    metric = metrics[0]
    if metric not in df.columns:
        st.warning("仪表盘配置无效")
        return

    series = pd.to_numeric(df[metric], errors="coerce")
    v = float(series.mean() if str(agg_mode) == "均值" else series.sum())
    if pd.isna(v):
        v = 0.0
    max_v = float(target_value or 0)
    if max_v <= 0:
        max_v = max(1.0, abs(v) * 1.2)

    gauge_style = str(style or "半圆")
    start_angle, end_angle = (180, 0) if gauge_style == "半圆" else (90, -270)
    ratio = min(max(v / max_v, 0.0), 1.0)

    option = {
        "title": {"text": f"{metric} 完成度", "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "series": [
            {
                "type": "gauge",
                "startAngle": start_angle,
                "endAngle": end_angle,
                "min": 0,
                "max": max_v,
                "progress": {"show": True, "width": 16},
                "axisLine": {
                    "lineStyle": {
                        "width": 16,
                        "color": [[ratio, "#1277D1"], [1, "#CCE5FF"]],
                    }
                },
                "axisTick": {"show": False},
                "splitLine": {"show": False},
                "axisLabel": {"color": "#3A4A63"},
                "pointer": {"show": True, "width": 4},
                "title": {"show": True, "offsetCenter": [0, "65%"], "color": "#3A4A63"},
                "detail": {"valueAnimation": True, "formatter": "{value}", "color": "#0B1B33"},
                "data": [{"value": round(v, 4), "name": str(agg_mode or "汇总")}],
            }
        ],
    }
    st_echarts(options=option, height="320px", key=f"chart_{uuid.uuid4()}")


def render_sankey_chart(df, metrics, dimension, dimension2):
    if not metrics or dimension == "不选择" or dimension2 == "不选择":
        st.warning("桑基图需要选择【核心指标】和【两个维度】")
        return
    metric = metrics[0]
    if metric not in df.columns or dimension not in df.columns or dimension2 not in df.columns:
        st.warning("桑基图配置无效")
        return

    tmp = df[[dimension, dimension2, metric]].dropna().copy()
    tmp[dimension] = tmp[dimension].astype(str)
    tmp[dimension2] = tmp[dimension2].astype(str)
    g = tmp.groupby([dimension, dimension2])[metric].sum(numeric_only=True).reset_index()
    g = g.sort_values(by=metric, ascending=False).head(40)

    nodes = sorted(set(g[dimension].astype(str).tolist() + g[dimension2].astype(str).tolist()))
    links = [
        {"source": str(r[dimension]), "target": str(r[dimension2]), "value": float(r[metric] or 0)}
        for _, r in g.iterrows()
    ]

    option = {
        "title": {"text": f"{metric} 流向", "textStyle": {"color": "#1277D1", "fontSize": 16}},
        "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
        "series": [
            {
                "type": "sankey",
                "data": [{"name": n} for n in nodes],
                "links": links,
                "emphasis": {"focus": "adjacency"},
                "lineStyle": {"color": "source", "curveness": 0.5, "opacity": 0.45},
                "label": {"color": "#0B1B33"},
            }
        ],
        "color": ["#1277D1", "#2FA3FF", "#FF8A3D", "#8C52FF", "#FF5252", "#66B2FF"],
    }
    st_echarts(options=option, height="360px", key=f"chart_{uuid.uuid4()}")


def render_chart(chart_conf, df_data, *, is_preview=False):
    chart_id = chart_conf["id"]
    chart_type = chart_conf["type"]
    metrics = chart_conf["metrics"]
    dimension = chart_conf["dimension"]
    x_axis_mode = chart_conf["x_axis_mode"]
    style = chart_conf.get("chart_style")
    dimension2 = chart_conf.get("dimension2")
    gauge_target_value = float(chart_conf.get("target_value") or 0)
    gauge_agg_mode = str(chart_conf.get("agg_mode") or "汇总")
    radar_topn = int(chart_conf.get("radar_topn") or 5)
    map_geojson = chart_conf.get("map_geojson")
    lon_col = chart_conf.get("lon_col")
    lat_col = chart_conf.get("lat_col")

    st.markdown('<div class="tech-card" style="position: relative;">', unsafe_allow_html=True)

    type_options = ["趋势图", "排名图", "占比图", "漏斗图", "箱线图", "散点图", "热力图", "雷达图", "仪表盘", "桑基图", "地图"]
    if chart_type not in type_options:
        chart_type = type_options[0]

    header_cols = st.columns([2.4, 1.1, 0.9])
    with header_cols[0]:
        selected_type = st.selectbox(
            "图形",
            options=type_options,
            index=type_options.index(chart_type),
            key=f"inline_type_{chart_id}",
            label_visibility="collapsed",
        )
    with header_cols[1]:
        if is_preview:
            if st.button("锁定并保存", key=f"inline_save_{chart_id}", use_container_width=True):
                conf = dict(chart_conf)
                conf["id"] = str(uuid.uuid4())
                conf["locked"] = True
                st.session_state.saved_charts.append(conf)
                st.rerun()
        else:
            locked_now = bool(chart_conf.get("locked", True))
            locked_new = st.checkbox("锁定", value=locked_now, key=f"inline_lock_{chart_id}")
            if locked_new != locked_now:
                for i, c in enumerate(st.session_state.saved_charts):
                    if c.get("id") == chart_id:
                        updated = dict(c)
                        updated["locked"] = locked_new
                        st.session_state.saved_charts[i] = updated
                        break
                st.rerun()
    with header_cols[2]:
        if is_preview:
            if st.button("清除", key=f"inline_clear_{chart_id}", use_container_width=True):
                clear_preview()
                st.rerun()
        else:
            if st.button("删除", key=f"inline_del_{chart_id}", use_container_width=True):
                delete_chart(chart_id)
                st.rerun()

    if selected_type != chart_type:
        updated = dict(chart_conf)
        updated["type"] = selected_type
        updated["chart_style"] = style_candidates_for(selected_type)[0]
        if is_preview:
            st.session_state.preview_chart = updated
        else:
            for i, c in enumerate(st.session_state.saved_charts):
                if c.get("id") == chart_id:
                    st.session_state.saved_charts[i] = updated
                    break
        st.rerun()

    style_options = style_candidates_for(chart_type)
    style_cols = st.columns(len(style_options))
    for idx, s in enumerate(style_options):
        btn_type = "primary" if str(style or "") == s else "secondary"
        if style_cols[idx].button(s, key=f"inline_style_{chart_id}_{s}", type=btn_type, use_container_width=True):
            updated = dict(chart_conf)
            updated["chart_style"] = s
            if is_preview:
                st.session_state.preview_chart = updated
            else:
                for i, c in enumerate(st.session_state.saved_charts):
                    if c.get("id") == chart_id:
                        st.session_state.saved_charts[i] = updated
                        break
            st.rerun()

    title = get_preview_title(chart_conf)
    st.markdown(f"**{title}**")

    if chart_type == "趋势图":
        render_trend_chart(df_data, metrics, x_axis_mode, style)
    elif chart_type == "占比图":
        render_pie_chart(df_data, metrics, dimension, style)
    elif chart_type == "排名图":
        render_bar_chart(df_data, metrics, dimension, style)
    elif chart_type == "漏斗图":
        render_funnel_chart(df_data, metrics, dimension)
    elif chart_type == "箱线图":
        render_boxplot_chart(df_data, metrics, dimension)
    elif chart_type == "散点图":
        render_scatter_chart(df_data, metrics, dimension, style)
    elif chart_type == "热力图":
        render_heatmap_chart(df_data, metrics, dimension, str(dimension2 or "不选择"))
    elif chart_type == "雷达图":
        render_radar_chart(df_data, metrics, dimension, style, radar_topn)
    elif chart_type == "仪表盘":
        render_gauge_chart(
            df_data,
            metrics,
            target_value=gauge_target_value,
            agg_mode=gauge_agg_mode,
            style=str(style or "半圆"),
        )
    elif chart_type == "桑基图":
        render_sankey_chart(df_data, metrics, dimension, str(dimension2 or "不选择"))
    elif chart_type == "地图":
        render_map_chart(
            df_data,
            metrics,
            dimension,
            style=style,
            dimension2=str(dimension2 or "不选择"),
            lon_col=str(lon_col or "不选择"),
            lat_col=str(lat_col or "不选择"),
            map_geojson=map_geojson,
        )

    desc = generate_description(chart_conf, df_data)
    if desc:
        st.markdown("\n".join([f"- {d}" for d in desc]))

    st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        st.sidebar.success(f"成功加载 {len(df)} 行数据")

        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        date_cols = detect_date_columns(df)
        time_candidates = ["不选择（使用行号）"] + date_cols + [c for c in all_cols if c not in date_cols]
        dimension_candidates = ["不选择"] + all_cols

        st.sidebar.header("2. 指标与维度配置")
        
        # X轴模式选择
        x_axis_mode = st.sidebar.selectbox(
            "X轴列 / 汇总模式",
            options=["-- 显示汇总结果 --"] + [f"时间列: {c}" for c in date_cols] + [f"类别列: {c}" for c in all_cols if c not in date_cols],
            index=0
        )
        
        # 核心指标配置
        metrics = st.sidebar.multiselect(
            "核心数值指标（多选）",
            options=numeric_cols,
            default=numeric_cols[:2] if numeric_cols else None
        )
        
        # 分组/对比维度
        dimension = st.sidebar.selectbox("分组/对比维度", options=["不选择"] + all_cols, index=0)

        st.sidebar.markdown("---")
        st.sidebar.header("3. 指标卡展示")
        preferred = []
        for c in numeric_cols:
            lc = str(c).lower()
            if "pv" in lc or "uv" in lc:
                preferred.append(c)
        default_cards = (preferred + [c for c in numeric_cols if c not in preferred])[:3]
        card_metrics = st.sidebar.multiselect(
            "选择指标卡（最多3个）",
            options=numeric_cols,
            default=default_cards if default_cards else None,
        )
        card_metrics = (card_metrics or [])[:3]

        st.sidebar.markdown("---")
        st.sidebar.header("4. 快速图形样式")

        with st.sidebar.expander("商务大屏核心图表推荐", expanded=True):
            rec1 = st.columns(2)
            if rec1[0].button("发光折线", use_container_width=True):
                st.session_state.chart_type = "趋势图"
                st.session_state.chart_style = "大屏发光"
            if rec1[1].button("实时排名赛", use_container_width=True):
                st.session_state.chart_type = "排名图"
                st.session_state.chart_style = "实时排名赛"

            rec2 = st.columns(2)
            if rec2[0].button("环形占比", use_container_width=True):
                st.session_state.chart_type = "占比图"
                st.session_state.chart_style = "环形"
            if rec2[1].button("玫瑰占比", use_container_width=True):
                st.session_state.chart_type = "占比图"
                st.session_state.chart_style = "玫瑰"

            rec3 = st.columns(2)
            if rec3[0].button("漏斗转化", use_container_width=True):
                st.session_state.chart_type = "漏斗图"
                st.session_state.chart_style = "默认"
            if rec3[1].button("涟漪散点", use_container_width=True):
                st.session_state.chart_type = "散点图"
                st.session_state.chart_style = "涟漪"

            rec4 = st.columns(2)
            if rec4[0].button("KPI仪表盘", use_container_width=True):
                st.session_state.chart_type = "仪表盘"
                st.session_state.chart_style = "半圆"
            if rec4[1].button("多维雷达", use_container_width=True):
                st.session_state.chart_type = "雷达图"
                st.session_state.chart_style = "对比"

            rec5 = st.columns(2)
            if rec5[0].button("热力矩阵", use_container_width=True):
                st.session_state.chart_type = "热力图"
                st.session_state.chart_style = "矩阵"
            if rec5[1].button("桑基流向", use_container_width=True):
                st.session_state.chart_type = "桑基图"
                st.session_state.chart_style = "默认"

            rec6 = st.columns(2)
            if rec6[0].button("地图散点", use_container_width=True):
                st.session_state.chart_type = "地图"
                st.session_state.chart_style = "散点叠加"
            if rec6[1].button("地图航线", use_container_width=True):
                st.session_state.chart_type = "地图"
                st.session_state.chart_style = "航线叠加"

        base_cols = st.sidebar.columns(2)
        if base_cols[0].button("清空已保存", use_container_width=True):
            st.session_state.saved_charts = []
            st.session_state.preview_chart = None
            st.rerun()
        if base_cols[1].button("清除预览", use_container_width=True):
            clear_preview()
            st.rerun()

        chart_type = st.sidebar.selectbox(
            "当前图形",
            options=["趋势图", "排名图", "占比图", "漏斗图", "箱线图", "散点图", "热力图", "雷达图", "仪表盘", "桑基图", "地图"],
            index=["趋势图", "排名图", "占比图", "漏斗图", "箱线图", "散点图", "热力图", "雷达图", "仪表盘", "桑基图", "地图"].index(
                st.session_state.chart_type
            )
            if st.session_state.chart_type in ["趋势图", "排名图", "占比图", "漏斗图", "箱线图", "散点图", "热力图", "雷达图", "仪表盘", "桑基图", "地图"]
            else 0,
        )
        st.session_state.chart_type = chart_type

        style_options = style_candidates_for(chart_type)
        current_style = st.session_state.chart_style
        style_index = style_options.index(current_style) if current_style in style_options else 0
        chart_style = st.sidebar.selectbox("图形样式", style_options, index=style_index)
        st.session_state.chart_style = chart_style

        dimension2 = st.session_state.chart_dimension2
        if chart_type in {"热力图", "桑基图", "地图"}:
            label = "第二维度"
            if chart_type == "地图" and str(st.session_state.chart_style or "") == "航线叠加":
                label = "目标维度（用于航线）"
            dimension2 = st.sidebar.selectbox(
                label,
                options=["不选择"] + all_cols,
                index=(
                    (["不选择"] + all_cols).index(dimension2)
                    if dimension2 in (["不选择"] + all_cols)
                    else 0
                ),
            )
            st.session_state.chart_dimension2 = dimension2

        map_geojson = st.session_state.map_geojson
        map_lon_col = st.session_state.map_lon_col
        map_lat_col = st.session_state.map_lat_col
        if chart_type == "地图":
            with st.sidebar.expander("地图配置", expanded=True):
                geojson_file = st.file_uploader("上传 GeoJSON（可选）", type=["geojson", "json"], key="map_geojson_upload")
                if geojson_file is not None:
                    try:
                        map_geojson = json.loads(geojson_file.getvalue().decode("utf-8"))
                        st.session_state.map_geojson = map_geojson
                        st.success("GeoJSON 已加载")
                    except Exception as e:
                        st.session_state.map_geojson = None
                        st.error(f"GeoJSON 解析失败：{e}")

                coord_cols = ["不选择"] + all_cols
                map_lon_col = st.selectbox(
                    "经度列（lon，可选）",
                    options=coord_cols,
                    index=coord_cols.index(map_lon_col) if map_lon_col in coord_cols else 0,
                )
                st.session_state.map_lon_col = map_lon_col
                map_lat_col = st.selectbox(
                    "纬度列（lat，可选）",
                    options=coord_cols,
                    index=coord_cols.index(map_lat_col) if map_lat_col in coord_cols else 0,
                )
                st.session_state.map_lat_col = map_lat_col

        gauge_target_value = float(st.session_state.gauge_target_value or 0)
        gauge_agg_mode = str(st.session_state.gauge_agg_mode or "汇总")
        if chart_type == "仪表盘":
            gauge_agg_mode = st.sidebar.selectbox("指标口径", options=["汇总", "均值"], index=0 if gauge_agg_mode == "汇总" else 1)
            st.session_state.gauge_agg_mode = gauge_agg_mode
            gauge_target_value = float(
                st.sidebar.number_input(
                    "目标值（用于完成度）",
                    min_value=0.0,
                    value=float(gauge_target_value or 100.0),
                    step=1.0,
                )
            )
            st.session_state.gauge_target_value = gauge_target_value

        radar_topn = int(st.session_state.radar_topn or 5)
        if chart_type == "雷达图":
            radar_topn = int(st.sidebar.slider("对比对象 TopN", min_value=1, max_value=10, value=radar_topn))
            st.session_state.radar_topn = radar_topn

        act_cols = st.sidebar.columns(2)
        preview_click = act_cols[0].button("预览", type="primary", use_container_width=True)
        save_click = act_cols[1].button("保存并锁定", use_container_width=True)

        # 全局数据预处理（筛选逻辑）
        df_processed = df.copy()
        
        # 解析时间列（如果有）
        selected_time_col = None
        if "时间列:" in x_axis_mode:
            selected_time_col = x_axis_mode.split(": ")[1]
            parsed_time = pd.to_datetime(df_processed[selected_time_col], errors="coerce")
            if parsed_time.notna().any():
                df_processed[selected_time_col] = parsed_time
                df_processed = df_processed.sort_values(by=selected_time_col)
            else:
                st.sidebar.warning(f"无法解析所选时间列：{selected_time_col}")

        st.sidebar.markdown("---")
        st.sidebar.header("5. 筛选过滤")
        filter_dimension = st.sidebar.selectbox("筛选维度", options=["不选择"] + all_cols, index=0)
        if filter_dimension != "不选择":
            unique_vals = df_processed[filter_dimension].astype(str).unique().tolist()
            selected_vals = st.sidebar.multiselect(f"筛选 {filter_dimension}", options=unique_vals, default=unique_vals)
            if selected_vals:
                df_filtered = df_processed[df_processed[filter_dimension].astype(str).isin(selected_vals)]
            else:
                df_filtered = df_processed
        else:
            df_filtered = df_processed

        if preview_click:
            if not metrics:
                st.sidebar.error("请至少选择一个核心指标")
            else:
                st.session_state.preview_chart = {
                    "id": str(uuid.uuid4()),
                    "type": chart_type,
                    "x_axis_mode": x_axis_mode,
                    "metrics": metrics,
                    "dimension": dimension,
                    "dimension2": dimension2,
                    "card_metrics": card_metrics,
                    "chart_style": chart_style,
                    "target_value": gauge_target_value,
                    "agg_mode": gauge_agg_mode,
                    "radar_topn": radar_topn,
                    "map_geojson": map_geojson,
                    "lon_col": map_lon_col,
                    "lat_col": map_lat_col,
                }
                st.rerun()

        if save_click:
            if st.session_state.preview_chart is None:
                st.sidebar.error("请先点击“预览”生成预览效果")
            else:
                conf = dict(st.session_state.preview_chart)
                conf["id"] = str(uuid.uuid4())
                conf["locked"] = True
                st.session_state.saved_charts.append(conf)
                st.sidebar.success("已保存并锁定")
                st.rerun()

        top_cols = st.columns(4)
        with top_cols[0]:
            st.markdown('<div class="tech-card">数据接入</div>', unsafe_allow_html=True)
        with top_cols[1]:
            st.markdown('<div class="tech-card">指标卡</div>', unsafe_allow_html=True)
        with top_cols[2]:
            st.markdown('<div class="tech-card">图表预览</div>', unsafe_allow_html=True)
        with top_cols[3]:
            st.markdown('<div class="tech-card">已保存图表</div>', unsafe_allow_html=True)

        if card_metrics:
            mcols = st.columns(len(card_metrics))
            for i, m in enumerate(card_metrics):
                if m in df_filtered.columns:
                    total_val = df_filtered[m].sum()
                    avg_val = df_filtered[m].mean()
                    with mcols[i]:
                        st.markdown(
                            f"""
<div class="tech-card" style="margin-bottom: 12px;">
  <div style="color:#3A4A63;font-size:0.9em;">{m}</div>
  <div style="color:#1277D1;font-family:'Rajdhani',sans-serif;font-size:2.1em;font-weight:700;">{format_number(total_val)}</div>
  <div style="color:#3A4A63;font-size:0.85em;">均值：{format_number(avg_val)}</div>
</div>
""",
                            unsafe_allow_html=True,
                        )

        body_cols = st.columns([2.2, 1])
        with body_cols[0]:
            st.markdown("### 预览区")
            if st.session_state.preview_chart is not None:
                render_chart(st.session_state.preview_chart, df_filtered, is_preview=True)
            else:
                st.info("在左侧选好指标后点击“预览”，这里会出现预览效果。")

        with body_cols[1]:
            st.markdown("### 已保存（锁定）")
            if st.session_state.saved_charts:
                for chart_conf in st.session_state.saved_charts:
                    render_chart(chart_conf, df_filtered, is_preview=False)
            else:
                st.info("暂无已保存图表")
            
        st.markdown("---")
        with st.expander("数据预览", expanded=False):
            df_display = df_filtered.head(50).copy()
            for col in date_cols:
                if col in df_display.columns:
                    parsed = pd.to_datetime(df_display[col], errors="coerce")
                    df_display[col] = parsed.dt.strftime("%Y-%m-%d").fillna(df_display[col].astype(str))
            st.dataframe(apply_date_highlight(df_display, date_cols), use_container_width=True, height=320)

        st.sidebar.markdown("---")
        st.sidebar.header("6. 导出")
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "下载筛选后数据（CSV）",
            csv,
            "dashboard_data.csv",
            "text/csv",
            key="download-csv",
        )

        st.sidebar.markdown("#### 已保存看板导出（MVP）")
        user_email = st.sidebar.text_input("用户邮箱（用于水印）", value=st.session_state.export_task_email)
        st.session_state.export_task_email = user_email
        export_format_label = st.sidebar.selectbox("导出格式", options=["可交互 HTML", "Excel (.xlsx)"], index=0)
        export_format = "html" if export_format_label.startswith("可交互") else "xlsx"

        saved_chart_confs = list(st.session_state.saved_charts or [])

        with st.sidebar.expander("导出选择（可混合勾选）", expanded=True):
            sel_cols = st.columns(2)
            if sel_cols[0].button("全选", use_container_width=True, key="export_select_all"):
                for m in card_metrics:
                    st.session_state[f"export_card_{m}"] = True
                for c in saved_chart_confs:
                    st.session_state[f"export_chart_{c['id']}"] = True
                st.rerun()
            if sel_cols[1].button("全不选", use_container_width=True, key="export_select_none"):
                for m in card_metrics:
                    st.session_state[f"export_card_{m}"] = False
                for c in saved_chart_confs:
                    st.session_state[f"export_chart_{c['id']}"] = False
                st.rerun()

            st.markdown("**指标卡**")
            selected_cards: list[str] = []
            for m in card_metrics:
                key = f"export_card_{m}"
                if key not in st.session_state:
                    st.session_state[key] = True
                if st.checkbox(f"指标卡：{m}", key=key):
                    selected_cards.append(m)

            st.markdown("**已保存图表**")
            selected_chart_ids: list[str] = []
            for c in saved_chart_confs:
                title = get_preview_title(c)
                key = f"export_chart_{c['id']}"
                if key not in st.session_state:
                    st.session_state[key] = True
                if st.checkbox(f"图表：{title}", key=key):
                    selected_chart_ids.append(c["id"])

        num_cards = len(selected_cards)
        num_charts = len(selected_chart_ids)
        estimate = export_core.estimate_export(num_cards, num_charts, export_format)
        with st.sidebar.expander("导出预览（估算）", expanded=False):
            st.write(f"预计文件大小：{estimate.size_bytes/1024/1024:.2f} MB")
            st.write(f"预计导出耗时：{estimate.estimate_seconds:.1f} s")

        run_cols = st.sidebar.columns(2)
        start_export = run_cols[0].button("开始导出", type="primary", use_container_width=True)
        cancel_export = run_cols[1].button("取消任务", use_container_width=True)

        if cancel_export and st.session_state.export_task_id and user_email:
            export_core.TASK_MANAGER.cancel(st.session_state.export_task_id, user_email=user_email)
            st.rerun()

        if start_export:
            if not user_email:
                st.sidebar.error("请先填写用户邮箱（用于水印）")
            elif num_cards == 0 and num_charts == 0:
                st.sidebar.error("请至少勾选一个导出项")
            else:
                chart_confs = [c for c in saved_chart_confs if c["id"] in set(selected_chart_ids)]
                df_for_export = df_filtered.copy()
                product_name = "赛博朋克业务数据看板生成器"
                language_default = "zh"

                def runner(set_progress, is_cancelled):
                    set_progress(5)
                    if is_cancelled():
                        raise RuntimeError("cancelled")
                    cards = export_core.build_card_summaries(df_for_export, selected_cards)
                    set_progress(25)
                    if is_cancelled():
                        raise RuntimeError("cancelled")
                    charts = export_core.build_chart_bundles(df_for_export, chart_confs)
                    set_progress(55)
                    if is_cancelled():
                        raise RuntimeError("cancelled")
                    if export_format == "html":
                        res = export_core.export_dashboard_html(
                            product_name=product_name,
                            user_email=user_email,
                            language_default=language_default,
                            cards=cards,
                            charts=charts,
                        )
                    else:
                        res = export_core.export_dashboard_xlsx(
                            product_name=product_name,
                            user_email=user_email,
                            cards=cards,
                            charts=charts,
                        )
                    set_progress(95)
                    return res

                task = export_core.TASK_MANAGER.submit(
                    user_email=user_email,
                    export_format=export_format,
                    estimate=estimate,
                    runner=runner,
                )

                st.session_state.export_task_id = task.task_id
                st.session_state.export_task_format = export_format
                st.session_state.export_file_bytes = None
                st.session_state.export_file_sha256 = None
                st.rerun()

        if st.session_state.export_task_id and user_email:
            task = export_core.TASK_MANAGER.get(st.session_state.export_task_id)
            if task and task.user_email == user_email:
                st.sidebar.progress(task.progress / 100)
                st.sidebar.caption(f"任务状态：{task.status} | traceId={task.trace_id}")
                if task.status == "succeeded":
                    result = export_core.TASK_MANAGER.get_result(task.task_id, user_email=user_email)
                    if result and st.session_state.export_file_bytes is None:
                        st.session_state.export_file_bytes = result.content
                        st.session_state.export_file_sha256 = result.sha256
                    st.sidebar.code(f"SHA-256: {st.session_state.export_file_sha256}")

                    if st.session_state.export_file_bytes is not None:
                        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
                        if st.session_state.export_task_format == "html":
                            st.sidebar.download_button(
                                "下载导出文件（HTML）",
                                st.session_state.export_file_bytes,
                                f"dashboard_{ts}.html",
                                "text/html",
                                key="download-export-html",
                            )
                        else:
                            st.sidebar.download_button(
                                "下载导出文件（Excel）",
                                st.session_state.export_file_bytes,
                                f"dashboard_{ts}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download-export-xlsx",
                            )
                elif task.status in {"failed", "cancelled"}:
                    if task.error_message:
                        st.sidebar.error(f"导出失败：{task.error_message}（traceId={task.trace_id}）")

                auto_refresh = st.sidebar.checkbox("自动刷新导出进度", value=True)
                if auto_refresh and task.status in {"queued", "running"}:
                    time.sleep(0.6)
                    st.rerun()

        st.sidebar.info("提示：导出文件默认带水印（产品名/邮箱/UTC时间）。")

else:
    st.info("欢迎使用，请在左侧上传数据文件开始。")
    st.markdown(
        """
### 快速开始指南
1. 准备数据：确保你的 Excel 或 CSV 有表头。
2. 上传文件：拖拽文件到左侧上传区域。
3. 配置指标：选择时间列和核心指标。
4. 即刻呈现：生成可视化数据看板。
"""
    )

    if st.button("使用演示数据"):
        data = {
            "日期": pd.date_range(start="2024-01-01", periods=30, freq="D"),
            "渠道": ["自然", "投放", "社交"] * 10,
            "销售额": [100 + i * 5 for i in range(30)],
            "访问量": [200 + i * 10 for i in range(30)]
        }
        df_demo = pd.DataFrame(data)
        csv_buffer = io.BytesIO()
        df_demo.to_csv(csv_buffer, index=False)
        st.download_button("下载演示数据 CSV", csv_buffer.getvalue(), "demo_data.csv", "text/csv")
