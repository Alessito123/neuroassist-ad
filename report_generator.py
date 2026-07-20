"""Generación de reportes PDF reproducibles con ReportLab."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from models import EvaluationSuite
from preprocessing import dataset_summary


def _figure_image(
    figure: Figure, width: float = 23 * cm, max_height: float = 15.5 * cm
) -> Image:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    image = Image(buffer)
    draw_width = width
    draw_height = width * image.imageHeight / image.imageWidth
    if draw_height > max_height:
        scale = max_height / draw_height
        draw_width *= scale
        draw_height *= scale
    image.drawWidth = draw_width
    image.drawHeight = draw_height
    return image


def _dataframe_table(frame: pd.DataFrame, columns: list[str] | None = None) -> Table:
    display = frame[columns].copy() if columns else frame.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    data = [display.columns.tolist()] + display.astype(str).values.tolist()
    table = Table(data, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AAB7C4")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def generate_report(
    frame: pd.DataFrame,
    target: str,
    suite: EvaluationSuite | None = None,
    figures: Iterable[Figure] | None = None,
    patient_result: dict[str, Any] | None = None,
) -> bytes:
    """Devuelve un PDF en memoria, listo para st.download_button."""
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="Reporte NeuroAssist AD",
        author="NeuroAssist AD",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CenteredTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#173B57"),
        )
    )
    story: list[Any] = [
        Paragraph("NeuroAssist AD — Reporte analítico", styles["CenteredTitle"]),
        Paragraph(
            f"Generado el {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]
        ),
        Spacer(1, 0.35 * cm),
        Paragraph(
            "AVISO: herramienta educativa de apoyo. No sustituye evaluación, juicio ni diagnóstico médico.",
            styles["Heading3"],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("Resumen del dataset", styles["Heading2"]),
    ]
    summary = dataset_summary(frame, target)
    summary_frame = pd.DataFrame(
        [{"Indicador": key, "Valor": str(value)} for key, value in summary.items()]
    )
    story.append(_dataframe_table(summary_frame))

    if suite is not None:
        story.extend(
            [
                Spacer(1, 0.5 * cm),
                Paragraph("Comparación de modelos e intervalos de confianza", styles["Heading2"]),
                _dataframe_table(
                    suite.comparison,
                    [
                        "Modelo",
                        "AUC-ROC",
                        "IC95% AUC-ROC",
                        "AUC-PR",
                        "F1-Score",
                        "IC95% F1-Score",
                        "Recall / Sensibilidad",
                        "Puntaje clínico",
                    ],
                ),
                Spacer(1, 0.35 * cm),
                Paragraph(
                    f"Modelo seleccionado: <b>{suite.best_model_name}</b>. Criterio: promedio de AUC-ROC y F1.",
                    styles["Normal"],
                ),
                Paragraph(
                    "Friedman: " + suite.friedman_test["interpretacion"], styles["Normal"]
                ),
            ]
        )

    for index, figure in enumerate(figures or []):
        if index == 0:
            story.append(PageBreak())
            story.append(Paragraph("Visualizaciones", styles["Heading2"]))
        story.append(_figure_image(figure))
        story.append(Spacer(1, 0.3 * cm))

    if patient_result:
        story.extend(
            [
                PageBreak(),
                Paragraph("Resultado individual", styles["Heading2"]),
                Paragraph(
                    f"Paciente/código: {patient_result.get('patient_code', 'N/D')}", styles["Normal"]
                ),
                Paragraph(
                    f"Clase estimada: <b>{patient_result.get('predicted_class')}</b>", styles["Normal"]
                ),
                Paragraph(
                    f"Confianza del modelo: {patient_result.get('probability', 0):.1%}", styles["Normal"]
                ),
                Paragraph(
                    "Este resultado debe interpretarse junto con la historia clínica, pruebas cognitivas,"
                    " neuroimagen y criterio de un profesional calificado.",
                    styles["Normal"],
                ),
            ]
        )

    document.build(story)
    return output.getvalue()
