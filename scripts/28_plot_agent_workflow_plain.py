from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "submission" / "assets"
FONT_DIR = (
    ROOT
    / "GOAI2026决赛指南Finals Guide"
    / "世界人工智能开源大赛PPT模板16-9Global Open-source AI Challenge PPT Template"
    / "字体Font"
)


def _font(name: str, size: float) -> FontProperties:
    path = FONT_DIR / name
    return FontProperties(fname=path, size=size)


REGULAR = _font("OPPOSans-R.ttf", 12)
MEDIUM = _font("OPPOSans-M.ttf", 12)
HEAVY = _font("OPPOSans-H.ttf", 12)


COLORS = {
    "navy": "#12345A",
    "blue": "#2878B5",
    "light_blue": "#E9F3FB",
    "orange": "#E98B2A",
    "light_orange": "#FFF1DF",
    "purple": "#8055A5",
    "light_purple": "#F1EAF7",
    "green": "#4E9B66",
    "light_green": "#E9F5EC",
    "gray": "#65717E",
    "light_gray": "#F1F3F5",
    "red": "#C84B4B",
    "light_red": "#FBE9E9",
    "ink": "#142033",
}


def add_box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_size: float = 14,
    body_size: float = 10.5,
    linewidth: float = 2.0,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.09",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height - 0.22,
        title,
        ha="center",
        va="top",
        color=COLORS["ink"],
        fontproperties=_font("OPPOSans-H.ttf", title_size),
        zorder=3,
    )
    axis.text(
        x + width / 2,
        y + height - 0.60,
        body,
        ha="center",
        va="top",
        color=COLORS["ink"],
        fontproperties=_font("OPPOSans-R.ttf", body_size),
        linespacing=1.35,
        zorder=3,
    )
    return patch


def add_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#65717E",
    connectionstyle: str = "arc3",
    linewidth: float = 1.8,
    mutation_scale: float = 14,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle,
        shrinkA=3,
        shrinkB=3,
        zorder=1,
    )
    axis.add_patch(arrow)
    return arrow


def draw_system_workflow() -> None:
    figure, axis = plt.subplots(figsize=(16, 9), dpi=180)
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")

    axis.text(
        0.5,
        8.62,
        "GenShield-Agent 完整工作流程（通俗版）",
        ha="left",
        va="center",
        color=COLORS["navy"],
        fontproperties=_font("OPPOSans-H.ttf", 25),
    )
    axis.text(
        0.52,
        8.23,
        "Agent 决定下一次算什么；编译器、GMC、MC 和 Verifier 决定事实是否成立",
        ha="left",
        va="center",
        color=COLORS["gray"],
        fontproperties=_font("OPPOSans-R.ttf", 13),
    )

    top_y = 6.72
    top_h = 1.05
    top_w = 2.55
    top_x = [0.45, 3.55, 6.65, 9.75, 12.85]
    add_box(
        axis,
        top_x[0],
        top_y,
        top_w,
        top_h,
        "1. 冻结规则",
        "目标、材料、源、预算\n统计门槛均不能修改",
        COLORS["light_gray"],
        COLORS["gray"],
        14,
        10.2,
    )
    add_box(
        axis,
        top_x[1],
        top_y,
        top_w,
        top_h,
        "2. 整理观察",
        "候选、GMC场、排行榜\n预算和上一步反馈",
        COLORS["light_blue"],
        COLORS["blue"],
        14,
        10.2,
    )
    add_box(
        axis,
        top_x[2],
        top_y,
        top_w,
        top_h,
        "3. API 决策",
        "每步只选一个工具\n并说明假设和理由",
        COLORS["light_blue"],
        COLORS["blue"],
        14,
        10.2,
    )
    add_box(
        axis,
        top_x[3],
        top_y,
        top_w,
        top_h,
        "4. 协议检查",
        "检查 JSON、权限、参数\n不合规则要求重答",
        COLORS["light_gray"],
        COLORS["gray"],
        14,
        10.2,
    )
    add_box(
        axis,
        top_x[4],
        top_y,
        top_w,
        top_h,
        "5. 执行一个工具",
        "生成 / GMC / MC / 结束\n工具会强制预算和规则",
        COLORS["light_orange"],
        COLORS["orange"],
        14,
        10.2,
    )
    for left, right in zip(top_x[:-1], top_x[1:]):
        add_arrow(axis, (left + top_w, top_y + top_h / 2), (right, top_y + top_h / 2))

    axis.text(
        11.0,
        7.98,
        "不合规：记录原因并重试",
        color=COLORS["red"],
        fontproperties=_font("OPPOSans-M.ttf", 10),
        ha="center",
    )
    add_arrow(
        axis,
        (10.1, top_y + top_h),
        (7.95, top_y + top_h),
        color=COLORS["red"],
        connectionstyle="arc3,rad=0.35",
        linewidth=1.5,
    )

    tool_y = 4.15
    tool_h = 1.72
    tool_w = 3.35
    tool_x = [0.35, 4.27, 8.19, 12.11]
    add_box(
        axis,
        tool_x[0],
        tool_y,
        tool_w,
        tool_h,
        "A. 生成候选",
        "Agent 选父代和搜索策略\n编译器枚举合法材料交换\n物理排序 + 少量随机探索",
        COLORS["light_blue"],
        COLORS["blue"],
        14.5,
        10.3,
    )
    add_box(
        axis,
        tool_x[1],
        tool_y,
        tool_w,
        tool_h,
        "B. GMC 快速筛选",
        "计算完整通量场\n提取远场、热点和泄漏方向\n更新排行榜和下一代父代",
        COLORS["light_orange"],
        COLORS["orange"],
        14.5,
        10.3,
    )
    add_box(
        axis,
        tool_x[2],
        tool_y,
        tool_w,
        tool_h,
        "C. MC 精细认证",
        "基准和候选各自产生权窗\n各跑 Analog 与权窗 MC\n比较绝对通量和统计误差",
        COLORS["light_purple"],
        COLORS["purple"],
        14.5,
        10.3,
    )
    add_box(
        axis,
        tool_x[3],
        tool_y,
        tool_w,
        tool_h,
        "D. 请求结束",
        "Agent 提交成功 / 无改进 / 不确定\n独立 Verifier 逐项检查\nAgent 不能批准自己",
        COLORS["light_green"],
        COLORS["green"],
        14.5,
        10.3,
    )

    router_center_x = top_x[4] + top_w / 2
    branch_y = 6.18
    axis.plot(
        [tool_x[0] + tool_w / 2, router_center_x],
        [branch_y, branch_y],
        color=COLORS["orange"],
        linewidth=1.5,
        zorder=1,
    )
    add_arrow(
        axis,
        (router_center_x, top_y),
        (router_center_x, branch_y),
        color=COLORS["orange"],
        linewidth=1.5,
        mutation_scale=12,
    )
    for x in tool_x:
        center_x = x + tool_w / 2
        add_arrow(
            axis,
            (center_x, branch_y),
            (center_x, tool_y + tool_h),
            color=COLORS["orange"],
            linewidth=1.4,
            mutation_scale=12,
        )

    add_box(
        axis,
        3.1,
        2.25,
        7.05,
        0.95,
        "6. 写入证据并更新状态",
        "记录 API 响应、工具参数、计算结果、候选谱系和预算，并形成下一步观察",
        COLORS["light_gray"],
        COLORS["gray"],
        14,
        10.5,
    )
    merge_targets = [4.15, 6.62, 9.05]
    for x, target_x in zip(tool_x[:3], merge_targets):
        add_arrow(
            axis,
            (x + tool_w / 2, tool_y),
            (target_x, 3.20),
            linewidth=1.3,
        )

    add_arrow(
        axis,
        (3.1, 2.72),
        (4.82, top_y),
        color=COLORS["blue"],
        connectionstyle="arc3,rad=-0.42",
        linewidth=2.0,
    )
    axis.text(
        1.25,
        3.15,
        "循环：新结果成为下一步证据",
        color=COLORS["blue"],
        fontproperties=_font("OPPOSans-M.ttf", 11),
        ha="left",
    )

    add_box(
        axis,
        11.73,
        2.10,
        3.7,
        1.1,
        "Verifier 判定",
        "拒绝：原因返回 Agent，继续循环\n接受：输出最终结论并结束",
        COLORS["light_green"],
        COLORS["green"],
        14,
        10.3,
    )
    add_arrow(axis, (13.78, tool_y), (13.58, 3.20), color=COLORS["green"])
    add_arrow(
        axis,
        (11.73, 2.62),
        (4.83, top_y),
        color=COLORS["red"],
        connectionstyle="arc3,rad=-0.48",
        linewidth=1.5,
    )
    axis.text(
        9.95,
        1.86,
        "拒绝后继续",
        color=COLORS["red"],
        fontproperties=_font("OPPOSans-M.ttf", 10),
        ha="center",
    )
    add_box(
        axis,
        11.73,
        0.58,
        3.7,
        0.8,
        "最终输出",
        "verified / no improvement / inconclusive",
        COLORS["light_green"],
        COLORS["green"],
        13,
        9.7,
    )
    add_arrow(axis, (13.58, 2.10), (13.58, 1.38), color=COLORS["green"])

    axis.text(
        0.48,
        0.60,
        "关键边界：API 不计算物理、不直接填坐标、不修改阈值；它只管理实验顺序与有限预算。",
        ha="left",
        va="center",
        color=COLORS["navy"],
        fontproperties=_font("OPPOSans-M.ttf", 12),
    )

    figure.savefig(ASSET_DIR / "agent_workflow_plain.png", bbox_inches="tight", pad_inches=0.08)
    figure.savefig(ASSET_DIR / "agent_workflow_plain.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def draw_run_timeline() -> None:
    figure, axis = plt.subplots(figsize=(16, 9), dpi=180)
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")

    axis.text(
        0.5,
        8.62,
        "封版运行的 12 个真实决策步骤",
        ha="left",
        va="center",
        color=COLORS["navy"],
        fontproperties=_font("OPPOSans-H.ttf", 25),
    )
    axis.text(
        0.52,
        8.22,
        "蓝=生成结构　橙=GMC筛选　紫=MC认证　绿=结束请求　红框=被环境或Verifier拒绝",
        ha="left",
        va="center",
        color=COLORS["gray"],
        fontproperties=_font("OPPOSans-R.ttf", 12.5),
    )

    steps = [
        (1, "生成 12 个初代候选", "从基准 GMC 场出发", "blue", False),
        (2, "请求 GMC 批量筛选", "超过单次 4 个上限", "orange", True),
        (3, "GMC 筛选 4 个", "最佳风险 0.09842", "orange", False),
        (4, "生成 16 个第二代", "从两个最佳父代再生成", "blue", False),
        (5, "再次请求 GMC 批量", "仍超过单次上限", "orange", True),
        (6, "GMC 筛选 4 个", "发现冠军风险 0.07593", "orange", False),
        (7, "GMC 再筛选 4 个", "累计 12 个，MC 门槛通过", "orange", False),
        (8, "MC 认证冠军", "四组计算用完 80,000 历史", "purple", False),
        (9, "再生成 31 个候选", "有 GMC 预算，但无 MC 预算", "blue", False),
        (10, "请求宣布已验证", "一致性 z=-2.1477，被否决", "green", True),
        (11, "GMC 筛选最后 4 个", "累计 16/16，含失败候选", "orange", False),
        (12, "请求 inconclusive", "无可执行精算工具，获接受", "green", False),
    ]
    palette = {
        "blue": (COLORS["light_blue"], COLORS["blue"]),
        "orange": (COLORS["light_orange"], COLORS["orange"]),
        "purple": (COLORS["light_purple"], COLORS["purple"]),
        "green": (COLORS["light_green"], COLORS["green"]),
    }
    positions = []
    box_width = 3.45
    box_height = 1.35
    x_values = [0.45, 4.35, 8.25, 12.15]
    y_values = [6.25, 4.25, 2.25]
    for row, y in enumerate(y_values):
        order = x_values if row % 2 == 0 else list(reversed(x_values))
        for x in order:
            positions.append((x, y))

    for (number, title, body, kind, rejected), (x, y) in zip(steps, positions):
        face, edge = palette[kind]
        if rejected:
            edge = COLORS["red"]
        add_box(
            axis,
            x,
            y,
            box_width,
            box_height,
            f"第 {number} 步｜{title}",
            body + ("\n【被拒绝，原因写回下一步】" if rejected else ""),
            face,
            edge,
            13.2,
            10.0,
            linewidth=2.7 if rejected else 2.0,
        )

    centers = [(x + box_width / 2, y + box_height / 2) for x, y in positions]
    for index, (start, end) in enumerate(zip(centers[:-1], centers[1:])):
        if positions[index][1] == positions[index + 1][1]:
            add_arrow(axis, start, end, linewidth=1.5, mutation_scale=12)
        else:
            add_arrow(
                axis,
                (start[0], positions[index][1]),
                (end[0], positions[index + 1][1] + box_height),
                connectionstyle="arc3,rad=0.0",
                linewidth=1.5,
                mutation_scale=12,
            )

    add_box(
        axis,
        0.55,
        0.24,
        4.55,
        1.20,
        "搜索证据",
        "生成 59 个结构｜GMC 筛选 16 个\n12 个 GMC 引导后代完成复筛",
        COLORS["light_blue"],
        COLORS["blue"],
        13.5,
        9.2,
    )
    add_box(
        axis,
        5.72,
        0.24,
        4.55,
        1.20,
        "候选信号",
        "GMC 风险 0.07593｜MC 远场比 0.10999\n估计下降 89%",
        COLORS["light_purple"],
        COLORS["purple"],
        13.5,
        9.2,
    )
    add_box(
        axis,
        10.89,
        0.24,
        4.55,
        1.20,
        "最终结论",
        "闭环成立｜候选为强信号\n正式认证 inconclusive",
        COLORS["light_green"],
        COLORS["green"],
        13.5,
        9.3,
    )

    figure.savefig(ASSET_DIR / "agent_run_12_steps_plain.png", bbox_inches="tight", pad_inches=0.08)
    figure.savefig(ASSET_DIR / "agent_run_12_steps_plain.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    draw_system_workflow()
    draw_run_timeline()
    print(ASSET_DIR / "agent_workflow_plain.png")
    print(ASSET_DIR / "agent_run_12_steps_plain.png")


if __name__ == "__main__":
    main()
