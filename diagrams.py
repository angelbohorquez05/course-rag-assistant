"""Diagrams from the slides, redrawn in the style of the app.

When the slide that answers a question contains one of these diagrams, the app shows the
redrawn version instead of a photo of the slide. For now they are redrawn by hand;
generating them automatically with a multimodal model is a next step.
"""
BLUE = "#4A90E2"
TEXT = "#e8e8e8"
LINE = "#8a8f98"


def _box(x, y, w, h, lines, kind="plain", font=17):
    stroke = {"plain": LINE, "train": "#E07A7A", "valid": BLUE, "test": "#D4C04A", "model": BLUE}[kind]
    fill = BLUE if kind == "model" else "none"
    weight = 600 if kind == "model" else 400
    colour = "#ffffff" if kind == "model" else TEXT
    first_y = y + h / 2 - (len(lines) - 1) * (font * 0.62)
    text = "".join(
        f"<tspan x='{x + w / 2}' y='{first_y + k * font * 1.25}'>{line}</tspan>" for k, line in enumerate(lines))
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{stroke}' "
            f"stroke-width='1.6'/>"
            f"<text fill='{colour}' font-size='{font}' font-weight='{weight}' text-anchor='middle' "
            f"dominant-baseline='central' font-family='sans-serif'>{text}</text>")


def _arrow(points):
    path = " ".join(f"{'M' if k == 0 else 'L'} {x} {y}" for k, (x, y) in enumerate(points))
    return f"<path d='{path}' fill='none' stroke='{TEXT}' stroke-width='1.6' marker-end='url(#wf-arrow)'/>"


def supervised_workflow():
    top, mid, low, h = 30, 170, 330, 70
    parts = [
        f"<defs><marker id='wf-arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
        f"orient='auto-start-reverse'><path d='M1 1 L9 5 L1 9 z' fill='{TEXT}'/></marker></defs>",
        # training and validation (above the dashed line)
        _box(20, top, 150, h, ["Data", "collection"]),
        _box(220, top, 170, h, ["EDA &amp;", "Preprocessing"]),
        _box(440, top, 130, h, ["Training", "Set"], "train"),
        _box(620, top, 150, h, ["Train the", "model"], "model"),
        _box(440, mid, 130, h, ["Validation", "Set"], "valid"),
        _box(620, mid, 150, h, ["Evaluate", "the model"], "model"),
        _box(805, 98, 175, 64, ["Hyperparameter tuning,", "regularization, etc."], font=13),
        _arrow([(170, top + 35), (216, top + 35)]),
        f"<path d='M 390 {top + 35} L 410 {top + 35} L 410 {mid + 35}' fill='none' stroke='{TEXT}' stroke-width='1.6'/>",
        _arrow([(410, top + 35), (436, top + 35)]),
        _arrow([(410, mid + 35), (436, mid + 35)]),
        _arrow([(570, top + 35), (616, top + 35)]),
        _arrow([(570, mid + 35), (616, mid + 35)]),
        _arrow([(695, top + h), (695, mid - 4)]),
        _arrow([(770, mid + 35), (892, mid + 35), (892, 166)]),
        _arrow([(892, 98), (892, top + 35), (774, top + 35)]),
        # the dashed line separates training from real use
        f"<line x1='20' y1='280' x2='980' y2='280' stroke='{LINE}' stroke-width='1.4' stroke-dasharray='8 7'/>",
        _arrow([(695, mid + h), (695, low - 4)]),
        # prediction on new data (below the dashed line)
        _box(20, low, 150, h, ["New, unseen", "data"], font=15),
        _box(220, low, 170, h, ["Preprocessing"]),
        _box(440, low, 130, h, ["Test Set"], "test"),
        _box(620, low, 150, h, ["Prediction"], "model"),
        _box(820, low, 160, h, ["Evaluate", "the model"]),
        _arrow([(170, low + 35), (216, low + 35)]),
        _arrow([(390, low + 35), (436, low + 35)]),
        _arrow([(570, low + 35), (616, low + 35)]),
        _arrow([(770, low + 35), (816, low + 35)]),
    ]
    return f"<svg viewBox='0 0 1000 420' width='100%' role='img'>{''.join(parts)}</svg>"


# Which slides contain each diagram: {document: [pages]}.
DIAGRAMS = [
    {
        "title": "Supervised Learning Workflow (Summary)",
        "pages": {"lec5_slides": [30], "lec6_slides": list(range(8, 19)), "lec8_slides": [4, 5]},
        "svg": supervised_workflow,
    },
]


def find_diagram(document, pages):
    """Return the redrawn diagram for these pages of this document, or None."""
    for diagram in DIAGRAMS:
        if set(pages) & set(diagram["pages"].get(document, [])):
            return diagram
    return None
