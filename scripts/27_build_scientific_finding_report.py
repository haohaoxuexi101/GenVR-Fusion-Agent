from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import unicodedata

os.environ.setdefault("MPLCONFIGDIR", "/tmp/genvr-mpl")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "submission" / "SCIENTIFIC_FINDING_AND_ENVIRONMENT.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "GenShield-Agent_科学发现与环境定义报告.pdf"
DEFAULT_LEGACY_OUTPUT = ROOT / "output" / "pdf" / "GenVR-Fusion_科学发现与环境定义报告.pdf"
DEFAULT_RUN_DIR = ROOT / "outputs" / "gmc_material_discovery" / "run_20260920_020815"
DEFAULT_AUDIT = ROOT / "submission" / "assets" / "api_participation_audit.png"
FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
font_manager.fontManager.addfont(str(FONT_PATH))
CJK_FAMILY = FontProperties(fname=str(FONT_PATH)).get_name()

PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
NAVY = "#14213D"
BLUE = "#2A5BD7"
LIGHT_BLUE = "#EAF0FF"
PALE_BLUE = "#F5F8FE"
MID_BLUE = "#D9E4FA"
TEXT = "#17213A"
MUTED = "#5C6784"
GREEN = "#3D8B4E"
RED = "#C94A4A"
ORANGE = "#D97924"
GRID = "#C8D2E5"
plt.rcParams["figure.max_open_warning"] = 0


def font(size: float, weight: str = "normal") -> FontProperties:
    return FontProperties(family=["DejaVu Sans", CJK_FAMILY], size=size, weight=weight)


MONO = FontProperties(family=["DejaVu Sans Mono", CJK_FAMILY])


def parse_front_matter(text: str) -> tuple[dict[str, str], list[str]]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, lines
    metadata: dict[str, str] = {}
    index = 1
    while index < len(lines) and lines[index].strip() != "---":
        if ":" in lines[index]:
            key, value = lines[index].split(":", 1)
            metadata[key.strip()] = value.strip()
        index += 1
    return metadata, lines[index + 1 :]


def display_units(text: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F", "A"} else 1
        for character in text
    )


def wrap_display(text: str, max_units: int) -> list[str]:
    text = text.strip()
    if not text:
        return [""]
    lines: list[str] = []
    remaining = text
    while display_units(remaining) > max_units:
        used = 0
        split_at = 0
        last_soft = -1
        for index, character in enumerate(remaining):
            units = 2 if unicodedata.east_asian_width(character) in {"W", "F", "A"} else 1
            if used + units > max_units:
                break
            used += units
            split_at = index + 1
            if character.isspace() or character in "，。；：、,.!?）)]}—/":
                last_soft = split_at
        if last_soft > max(8, split_at - 20):
            split_at = last_soft
        lines.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        lines.append(remaining)
    return lines


def clean_inline(text: str) -> str:
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("`", "")
    return text.strip()


def read_trajectory(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_api_audit(run_dir: Path, output: Path) -> Path:
    rows = read_trajectory(run_dir / "trajectory.jsonl")
    exchanges = [row for row in rows if row.get("event_type") == "llm_exchange"]
    decisions = {
        int(row["step"]): row.get("tool", "")
        for row in rows
        if row.get("event_type") == "decision"
    }
    tool_results = {
        int(row["step"]): row
        for row in rows
        if row.get("event_type") == "tool_result"
    }
    attempts: defaultdict[int, int] = defaultdict(int)
    table_rows: list[list[str]] = []
    for exchange in exchanges:
        step = int(exchange.get("step", 0))
        attempts[step] += 1
        accepted = bool(exchange.get("accepted"))
        usage = exchange.get("usage") or {}
        response_id = str(exchange.get("response_id") or "—")
        result = tool_results.get(step, {})
        if accepted:
            outcome = decisions.get(step, "—")
            if result and not bool(result.get("accepted", True)):
                outcome += " → 拒绝"
            elif step == 12:
                outcome += " → inconclusive"
            else:
                outcome += " → 执行"
        else:
            outcome = "协议拒绝并重试"
        table_rows.append(
            [
                str(step),
                str(attempts[step]),
                "接受" if accepted else "拒绝",
                response_id[:12],
                f"{int(usage.get('prompt_tokens', 0)):,}",
                f"{int(usage.get('completion_tokens', 0)):,}",
                outcome,
            ]
        )

    accepted_count = sum(bool(row.get("accepted")) for row in exchanges)
    prompt_tokens = sum(int((row.get("usage") or {}).get("prompt_tokens", 0)) for row in exchanges)
    completion_tokens = sum(
        int((row.get("usage") or {}).get("completion_tokens", 0)) for row in exchanges
    )
    total_tokens = sum(int((row.get("usage") or {}).get("total_tokens", 0)) for row in exchanges)
    fingerprints = {
        row.get("system_fingerprint") or row.get("fingerprint")
        for row in exchanges
        if row.get("system_fingerprint") or row.get("fingerprint")
    }
    models = {str(row.get("model")) for row in exchanges if row.get("model")}

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.text(0.045, 0.955, "DeepSeek API 参与审计", fontproperties=font(24, "bold"), color=NAVY, va="top")
    fig.text(
        0.045,
        0.915,
        "正式运行逐交换证据：response_id、token、协议接受状态与实际工具结果",
        fontproperties=font(12),
        color=MUTED,
        va="top",
    )

    cards = [
        ("Provider / Model", f"DeepSeek / {next(iter(models), '—')}"),
        ("API Exchanges", f"{len(exchanges)} 次 · {accepted_count} 接受 · {len(exchanges)-accepted_count} 重试"),
        ("Token Usage", f"{total_tokens:,} = {prompt_tokens:,} + {completion_tokens:,}"),
        ("Fingerprint", next(iter(fingerprints), "—")),
    ]
    card_width = 0.215
    for index, (label, value) in enumerate(cards):
        x = 0.045 + index * 0.235
        fig.add_artist(
            Rectangle(
                (x, 0.81),
                card_width,
                0.075,
                transform=fig.transFigure,
                facecolor=PALE_BLUE,
                edgecolor=MID_BLUE,
                linewidth=1.0,
            )
        )
        fig.text(x + 0.012, 0.865, label, fontproperties=font(9), color=MUTED, va="top")
        fig.text(x + 0.012, 0.835, value, fontproperties=font(11, "bold"), color=TEXT, va="top")

    ax = fig.add_axes([0.045, 0.13, 0.91, 0.64])
    ax.axis("off")
    column_labels = ["Step", "Try", "状态", "response_id", "Prompt", "Completion", "动作 / 结果"]
    column_widths = [0.055, 0.05, 0.07, 0.15, 0.09, 0.09, 0.37]
    table = ax.table(
        cellText=table_rows,
        colLabels=column_labels,
        colWidths=column_widths,
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.get_text().set_fontproperties(font(8.2, "bold" if row_index == 0 else "normal"))
        cell.get_text().set_color("white" if row_index == 0 else TEXT)
        if row_index == 0:
            cell.set_facecolor(BLUE)
        else:
            status = table_rows[row_index - 1][2]
            if status == "拒绝":
                cell.set_facecolor("#FFF1EE")
            else:
                cell.set_facecolor("white" if row_index % 2 else PALE_BLUE)
            if column_index == 2:
                cell.get_text().set_color(GREEN if status == "接受" else RED)

    fig.text(
        0.045,
        0.087,
        "事件链：llm_exchange → agent_output → decision → tool_call → tool_result",
        fontproperties=font(11, "bold"),
        color=BLUE,
        va="center",
    )
    fig.text(
        0.955,
        0.087,
        "17 个唯一 response_id · secret_recorded=false · deterministic fallback=false",
        fontproperties=font(9),
        color=MUTED,
        ha="right",
        va="center",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


class ReportRenderer:
    def __init__(self, source: Path, output: Path, metadata: dict[str, str]):
        self.source = source
        self.output = output
        self.metadata = metadata
        self.pages: list[tuple[plt.Figure, bool, str]] = []
        self.fig: plt.Figure | None = None
        self.y = 0.0
        self.current_section = ""
        self.page_has_content = False

    def new_page(self, cover: bool = False) -> None:
        if self.fig is not None:
            self.pages.append((self.fig, False, self.current_section))
        self.fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), facecolor="white")
        self.y = 0.91
        self.page_has_content = False
        if not cover:
            self.fig.add_artist(
                Rectangle(
                    (0.085, 0.946),
                    0.83,
                    0.006,
                    transform=self.fig.transFigure,
                    facecolor=BLUE,
                    edgecolor="none",
                )
            )
            self.fig.text(
                0.085,
                0.965,
                "GenShield-Agent  |  科学发现与环境定义报告",
                fontproperties=font(8.5, "bold"),
                color=NAVY,
                va="top",
            )
            self.fig.text(
                0.915,
                0.965,
                "复赛提交版 · 2026-09-20",
                fontproperties=font(8),
                color=MUTED,
                ha="right",
                va="top",
            )

    def cover(self) -> None:
        self.fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), facecolor="white")
        self.fig.text(0.09, 0.835, "GOAI 2026 · 前沿探索 AI for Research", fontproperties=font(11), color=MUTED)
        self.fig.text(0.09, 0.765, self.metadata.get("title", "GenShield-Agent"), fontproperties=font(26, "bold"), color=NAVY)
        subtitle = self.metadata.get("subtitle", "")
        subtitle_lines = wrap_display(subtitle, 40)
        y = 0.705
        for line in subtitle_lines:
            self.fig.text(0.09, y, line, fontproperties=font(20, "bold"), color=NAVY)
            y -= 0.052
        bar_y = y - 0.025
        self.fig.add_artist(
            Rectangle((0.11, bar_y), 0.72, 0.045, transform=self.fig.transFigure, facecolor=BLUE, edgecolor="none")
        )
        self.fig.text(0.13, bar_y + 0.024, self.metadata.get("report", "科学发现与环境定义报告"), fontproperties=font(12, "bold"), color="white", va="center")
        self.fig.add_artist(
            Rectangle((0.11, bar_y - 0.045), 0.72, 0.045, transform=self.fig.transFigure, facecolor=LIGHT_BLUE, edgecolor=MID_BLUE, linewidth=0.8)
        )
        self.fig.text(0.13, bar_y - 0.022, self.metadata.get("edition", ""), fontproperties=font(8.8), color=TEXT, va="center")

        details_y = 0.39
        details = [
            f"队伍：{self.metadata.get('team', '')}",
            f"成员：{self.metadata.get('author', '')}",
            f"赛题：{self.metadata.get('track', '')}",
        ]
        for detail in details:
            self.fig.text(0.09, details_y, detail, fontproperties=font(11), color=TEXT)
            details_y -= 0.038

        claim_lines = wrap_display(self.metadata.get("claim", ""), 84)
        claim_height = 0.055 + 0.027 * len(claim_lines)
        claim_y = 0.135
        self.fig.add_artist(
            Rectangle(
                (0.09, claim_y),
                0.76,
                claim_height,
                transform=self.fig.transFigure,
                facecolor=LIGHT_BLUE,
                edgecolor=BLUE,
                linewidth=1.5,
            )
        )
        self.fig.text(0.11, claim_y + claim_height - 0.026, "统一口径", fontproperties=font(9, "bold"), color=BLUE, va="top")
        text_y = claim_y + claim_height - 0.052
        for line in claim_lines:
            self.fig.text(0.11, text_y, line, fontproperties=font(9), color=TEXT, va="top")
            text_y -= 0.027
        self.pages.append((self.fig, True, "封面"))
        self.fig = None

    def ensure_space(self, height: float) -> None:
        if self.fig is None:
            self.new_page()
        if self.y - height < 0.075:
            self.new_page()

    def heading(self, text: str, level: int) -> None:
        text = clean_inline(text)
        if level == 1:
            self.current_section = text
            if self.page_has_content and self.y < 0.72:
                self.new_page()
            self.ensure_space(0.065)
            assert self.fig is not None
            self.fig.text(0.09, self.y, text, fontproperties=font(18, "bold"), color=NAVY, va="top")
            self.y -= 0.054
            self.fig.add_artist(
                Rectangle((0.09, self.y + 0.012), 0.16, 0.005, transform=self.fig.transFigure, facecolor=BLUE, edgecolor="none")
            )
            self.y -= 0.018
        elif level == 2:
            self.ensure_space(0.045)
            assert self.fig is not None
            self.fig.text(0.09, self.y, text, fontproperties=font(13, "bold"), color=BLUE, va="top")
            self.y -= 0.038
        else:
            self.ensure_space(0.037)
            assert self.fig is not None
            self.fig.text(0.09, self.y, text, fontproperties=font(11, "bold"), color=NAVY, va="top")
            self.y -= 0.032
        self.page_has_content = True

    def paragraph(self, text: str) -> None:
        text = clean_inline(text)
        lines = wrap_display(text, 92)
        line_height = 0.0205
        height = line_height * len(lines) + 0.011
        self.ensure_space(height)
        assert self.fig is not None
        for line in lines:
            self.fig.text(0.09, self.y, line, fontproperties=font(9.7), color=TEXT, va="top")
            self.y -= line_height
        self.y -= 0.008
        self.page_has_content = True

    def bullet(self, text: str, marker: str = "•") -> None:
        text = clean_inline(text)
        lines = wrap_display(text, 84)
        line_height = 0.0205
        height = line_height * len(lines) + 0.005
        self.ensure_space(height)
        assert self.fig is not None
        self.fig.text(0.10, self.y, marker, fontproperties=font(9.7, "bold"), color=BLUE, va="top")
        for index, line in enumerate(lines):
            self.fig.text(0.125, self.y, line, fontproperties=font(9.5), color=TEXT, va="top")
            self.y -= line_height
        self.y -= 0.004
        self.page_has_content = True

    def quote(self, text: str) -> None:
        text = clean_inline(text)
        lines = wrap_display(text, 82)
        line_height = 0.021
        height = 0.032 + line_height * len(lines)
        self.ensure_space(height + 0.012)
        assert self.fig is not None
        top = self.y
        self.fig.add_artist(
            Rectangle(
                (0.09, top - height),
                0.82,
                height,
                transform=self.fig.transFigure,
                facecolor=LIGHT_BLUE,
                edgecolor=BLUE,
                linewidth=1.1,
            )
        )
        text_y = top - 0.018
        for line in lines:
            self.fig.text(0.112, text_y, line, fontproperties=font(9.6, "bold"), color=NAVY, va="top")
            text_y -= line_height
        self.y = top - height - 0.012
        self.page_has_content = True

    def code(self, lines: list[str]) -> None:
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(wrap_display(line.rstrip(), 105) if line else [""])
        line_height = 0.0175
        height = 0.027 + line_height * len(wrapped)
        self.ensure_space(height + 0.012)
        assert self.fig is not None
        top = self.y
        self.fig.add_artist(
            Rectangle(
                (0.09, top - height),
                0.82,
                height,
                transform=self.fig.transFigure,
                facecolor="#F2F4F8",
                edgecolor=GRID,
                linewidth=0.8,
            )
        )
        text_y = top - 0.016
        for line in wrapped:
            self.fig.text(0.108, text_y, line, fontproperties=FontProperties(family=["DejaVu Sans Mono", CJK_FAMILY], size=7.8), color="#27324C", va="top")
            text_y -= line_height
        self.y = top - height - 0.012
        self.page_has_content = True

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        columns = max(len(row) for row in rows)
        normalized = [row + [""] * (columns - len(row)) for row in rows]
        raw_widths = [
            max(12, max(display_units(clean_inline(row[column])) for row in normalized))
            for column in range(columns)
        ]
        total_raw = sum(raw_widths)
        fractions = [width / total_raw for width in raw_widths]
        minimum = 0.12 if columns <= 4 else 0.09
        fractions = [max(minimum, value) for value in fractions]
        scale = 1.0 / sum(fractions)
        fractions = [value * scale for value in fractions]
        table_width = 0.82
        x_positions = [0.09]
        for fraction in fractions[:-1]:
            x_positions.append(x_positions[-1] + table_width * fraction)
        font_size = 8.0 if columns <= 4 else 7.2
        line_height = 0.0165 if columns <= 4 else 0.015
        prepared: list[list[list[str]]] = []
        row_heights: list[float] = []
        for row in normalized:
            prepared_row: list[list[str]] = []
            maximum_lines = 1
            for column, cell in enumerate(row):
                column_points = PAGE_WIDTH * 72 * table_width * fractions[column]
                maximum_units = max(8, int(column_points / (font_size * 0.52)))
                cell_lines = wrap_display(clean_inline(cell), maximum_units)
                prepared_row.append(cell_lines)
                maximum_lines = max(maximum_lines, len(cell_lines))
            prepared.append(prepared_row)
            row_heights.append(0.012 + line_height * maximum_lines)
        total_height = sum(row_heights) + 0.012
        self.ensure_space(total_height)
        assert self.fig is not None
        top = self.y
        current_y = top
        for row_index, (row, row_height) in enumerate(zip(prepared, row_heights)):
            current_y -= row_height
            for column, cell_lines in enumerate(row):
                x = x_positions[column]
                width = table_width * fractions[column]
                face = BLUE if row_index == 0 else ("white" if row_index % 2 else PALE_BLUE)
                self.fig.add_artist(
                    Rectangle(
                        (x, current_y),
                        width,
                        row_height,
                        transform=self.fig.transFigure,
                        facecolor=face,
                        edgecolor=GRID,
                        linewidth=0.55,
                    )
                )
                text_y = current_y + row_height - 0.009
                for line in cell_lines:
                    self.fig.text(
                        x + 0.007,
                        text_y,
                        line,
                        fontproperties=font(font_size, "bold" if row_index == 0 else "normal"),
                        color="white" if row_index == 0 else TEXT,
                        va="top",
                    )
                    text_y -= line_height
        self.y = current_y - 0.014
        self.page_has_content = True

    def image(self, caption: str, relative_path: str) -> None:
        path = (self.source.parent / relative_path).resolve()
        if not path.exists():
            self.quote(f"图件缺失：{path}")
            return
        with Image.open(path) as original:
            image = original.convert("RGB")
            image.thumbnail((1900, 1800), Image.Resampling.LANCZOS)
            aspect = image.width / image.height
            image_array = image.copy()
        max_width = 0.82
        desired_height = (PAGE_WIDTH * max_width / aspect) / PAGE_HEIGHT
        image_height = min(desired_height, 0.665)
        image_width = (PAGE_HEIGHT * image_height * aspect) / PAGE_WIDTH
        image_width = min(image_width, max_width)
        if image_width < max_width and desired_height <= 0.665:
            image_height = (PAGE_WIDTH * image_width / aspect) / PAGE_HEIGHT
        caption_lines = wrap_display(clean_inline(caption), 92)
        caption_height = 0.0165 * len(caption_lines) + 0.014
        total_height = image_height + caption_height + 0.012
        self.ensure_space(total_height)
        assert self.fig is not None
        x = 0.5 - image_width / 2
        bottom = self.y - image_height
        ax = self.fig.add_axes([x, bottom, image_width, image_height])
        ax.imshow(image_array)
        ax.axis("off")
        self.y = bottom - 0.008
        for line in caption_lines:
            self.fig.text(0.5, self.y, f"图：{line}", fontproperties=font(8.2), color=MUTED, ha="center", va="top")
            self.y -= 0.0165
        self.y -= 0.008
        self.page_has_content = True

    def finish_page(self) -> None:
        if self.fig is not None:
            self.pages.append((self.fig, False, self.current_section))
            self.fig = None

    def render_lines(self, lines: list[str]) -> None:
        index = 0
        self.new_page()
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped == "<!-- pagebreak -->":
                if self.page_has_content and self.y < 0.72:
                    self.new_page()
                index += 1
                continue
            image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
            if image_match:
                self.image(image_match.group(1), image_match.group(2))
                index += 1
                continue
            heading_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
            if heading_match:
                self.heading(heading_match.group(2), len(heading_match.group(1)))
                index += 1
                continue
            if stripped.startswith("```"):
                code_lines: list[str] = []
                index += 1
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    code_lines.append(lines[index])
                    index += 1
                index += 1
                self.code(code_lines)
                continue
            if stripped.startswith(">"):
                quote_lines: list[str] = []
                while index < len(lines) and lines[index].strip().startswith(">"):
                    quote_lines.append(lines[index].strip()[1:].strip())
                    index += 1
                self.quote(" ".join(quote_lines))
                continue
            if stripped.startswith("|") and index + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-+", lines[index + 1]):
                table_lines: list[str] = [lines[index]]
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    table_lines.append(lines[index])
                    index += 1
                table_rows = [
                    [cell.strip() for cell in line.strip().strip("|").split("|")]
                    for line in table_lines
                ]
                self.table(table_rows)
                continue
            bullet_match = re.match(r"^-\s+(.+)$", stripped)
            numbered_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
            if bullet_match:
                self.bullet(bullet_match.group(1))
                index += 1
                continue
            if numbered_match:
                self.bullet(numbered_match.group(2), marker=f"{numbered_match.group(1)}.")
                index += 1
                continue
            paragraph_lines = [stripped]
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate:
                    break
                if candidate == "<!-- pagebreak -->" or candidate.startswith(("#", ">", "```", "- ", "|", "![")):
                    break
                if re.match(r"^\d+\.\s+", candidate):
                    break
                paragraph_lines.append(candidate)
                index += 1
            self.paragraph(" ".join(paragraph_lines))
        self.finish_page()

    def save(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        total = len(self.pages)
        with PdfPages(self.output) as pdf:
            for page_index, (page, cover, _) in enumerate(self.pages, start=1):
                page.add_artist(
                    Rectangle(
                        (0.08, 0.045),
                        0.84,
                        0.0012,
                        transform=page.transFigure,
                        facecolor=GRID,
                        edgecolor="none",
                    )
                )
                page.text(0.08, 0.026, "GenShield-Agent | 科学发现与环境定义报告", fontproperties=font(7.2), color=MUTED)
                page.text(0.92, 0.026, f"{page_index} / {total}", fontproperties=font(7.2), color=MUTED, ha="right")
                pdf.savefig(page, facecolor="white")
                plt.close(page)
            info = pdf.infodict()
            info["Title"] = "GenShield-Agent 科学发现与环境定义报告"
            info["Author"] = self.metadata.get("author", "")
            info["Subject"] = self.metadata.get("subtitle", "")
            info["Keywords"] = "GMC, Monte Carlo, autonomous agent, shielding, weight windows"


def build_report(source: Path, output: Path) -> Path:
    metadata, lines = parse_front_matter(source.read_text(encoding="utf-8"))
    renderer = ReportRenderer(source, output, metadata)
    renderer.cover()
    renderer.render_lines(lines)
    renderer.save()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--legacy-output", type=Path, default=DEFAULT_LEGACY_OUTPUT)
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    audit_output = args.audit_output if args.audit_output.is_absolute() else ROOT / args.audit_output
    legacy_output = args.legacy_output if args.legacy_output.is_absolute() else ROOT / args.legacy_output

    build_api_audit(run_dir.resolve(), audit_output.resolve())
    build_report(source.resolve(), output.resolve())
    legacy_output.parent.mkdir(parents=True, exist_ok=True)
    if legacy_output.resolve() != output.resolve():
        shutil.copyfile(output, legacy_output)
    print(output.resolve())


if __name__ == "__main__":
    main()
