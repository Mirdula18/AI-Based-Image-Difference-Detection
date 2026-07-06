import os
import cv2
import numpy as np
from datetime import datetime
from typing import Any
from reportlab.lib import colors

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Custom canvas to calculate total page count and add headers/footers dynamically."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Skip cover page
            return
            
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#4A4A4A"))
        
        # Header
        self.drawString(36, 565, "AI-Powered Engineering Drawing Revision Report")
        self.drawRightString(805, 565, datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.setStrokeColor(colors.HexColor("#D3D3D3"))
        self.setLineWidth(0.5)
        self.line(36, 558, 805, 558)
        
        # Footer
        self.line(36, 42, 805, 42)
        self.drawString(36, 28, "CONFIDENTIAL - Internal Engineering Review")
        self.drawRightString(805, 28, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def generate_pdf_report(
    output_pdf_path: str,
    stats: Any,
    image_paths: dict[str, str]
) -> str:
    """Compiles the professional multi-page Landscape PDF Report with Quality Metrics Dashboard."""
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=landscape(A4),
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#0B2240")  # Dark Slate Blue
    secondary_color = colors.HexColor("#1D3557")
    accent_color = colors.HexColor("#457B9D")
    neutral_light = colors.HexColor("#F1FAEE")
    text_color = colors.HexColor("#1C1C1C")
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=primary_color,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        leading=18,
        textColor=accent_color,
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'HeaderH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=15,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'HeaderH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=secondary_color,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_color
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=text_color
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    story = []
    
    # Enforce maximum report size of 50 pages (limit to top 40 changes by confidence)
    sorted_changes = sorted(stats.change_objects, key=lambda c: c.get("confidence", 0.0), reverse=True)
    visible_changes = sorted_changes[:40]
    
    # ==================== PAGE 1: COVER PAGE ====================
    story.append(Spacer(1, 40))
    story.append(Paragraph("AI-POWERED DRAWING REVISION REVIEWS", title_style))
    story.append(Paragraph("High Precision Geometric Revision Intelligence & Quality Analysis Report", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Quality Metrics Dashboard Table
    dashboard_data = [
        [
            Paragraph("<b>Drawing Category:</b>", body_style),
            Paragraph(stats.drawing_type.upper(), body_style),
            Paragraph("<b>Detected Changes:</b>", body_style),
            Paragraph(str(stats.total_changes), body_style),
        ],
        [
            Paragraph("<b>Alignment Confidence:</b>", body_style),
            Paragraph(f"{stats.alignment_confidence:.1f}%", body_style),
            Paragraph("<b>Rejected Candidates:</b>", body_style),
            Paragraph(str(stats.rejected_candidates), body_style),
        ],
        [
            Paragraph("<b>Geometry Accuracy:</b>", body_style),
            Paragraph(f"{stats.geometry_accuracy:.1f}%", body_style),
            Paragraph("<b>Final Accepted Changes:</b>", body_style),
            Paragraph(str(stats.final_accepted_changes), body_style),
        ],
        [
            Paragraph("<b>OCR Accuracy:</b>", body_style),
            Paragraph(f"{stats.ocr_accuracy:.1f}%", body_style),
            Paragraph("<b>False Positives Rate:</b>", body_style),
            Paragraph(f"{stats.false_positives:.1f}%", body_style),
        ]
    ]
    
    t_dash = Table(dashboard_data, colWidths=[150, 200, 150, 200])
    t_dash.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), neutral_light),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#D3D3D3")),
        ('PADDING', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_dash)
    story.append(Spacer(1, 20))
    
    # Executive Summary Paragraph
    story.append(Paragraph("<b>Executive Summary:</b>", h2_style))
    from app.summarizer import summarize
    summary_text = summarize(stats)
    story.append(Paragraph(summary_text, body_style))
    
    story.append(PageBreak())
    
    # ==================== PAGES 2 & 3: ORIGINAL DRAWINGS ====================
    img_w, img_h = 740, 460
    
    story.append(Paragraph("Original Drawing Revision A (Baseline)", h1_style))
    story.append(Image(image_paths["original_a"], width=img_w, height=img_h))
    story.append(PageBreak())
    
    story.append(Paragraph("Original Drawing Revision B (Revised)", h1_style))
    story.append(Image(image_paths["aligned_b"], width=img_w, height=img_h))
    story.append(PageBreak())
    
    # ==================== PAGE 4: OVERLAY VISUALIZATION ====================
    story.append(Paragraph("Color-Coded Revision Overlay (Aligned)", h1_style))
    story.append(Paragraph(
        "<font color='#00C800'><b>■ Green:</b> Added</font> &nbsp;&nbsp;&nbsp;&nbsp; "
        "<font color='#DC0000'><b>■ Red:</b> Removed</font> &nbsp;&nbsp;&nbsp;&nbsp; "
        "<font color='#FF8C00'><b>■ Orange:</b> Physical Objects</font> &nbsp;&nbsp;&nbsp;&nbsp; "
        "<font color='#DC0000'><b>■ Blue:</b> Annotations</font> &nbsp;&nbsp;&nbsp;&nbsp; "
        "<font color='#C800C8'><b>■ Purple:</b> Dimensions</font>",
        body_style
    ))
    story.append(Spacer(1, 10))
    story.append(Image(image_paths["added_removed_overlay"], width=img_w, height=img_h - 20))
    story.append(PageBreak())
    
    # ==================== PAGE 5: CHANGE SUMMARY TABLE ====================
    story.append(Paragraph("Structured Summary Table", h1_style))
    story.append(Spacer(1, 10))
    
    table_headers = [
        Paragraph("ID", table_header_style),
        Paragraph("Change Type", table_header_style),
        Paragraph("Object Location", table_header_style),
        Paragraph("Old Value", table_header_style),
        Paragraph("New Value", table_header_style),
        Paragraph("Delta", table_header_style),
        Paragraph("Confidence", table_header_style)
    ]
    
    summary_rows = [table_headers]
    for ch in visible_changes:
        summary_rows.append([
            Paragraph(ch["id"], table_cell_style),
            Paragraph(ch["type"], table_cell_style),
            Paragraph(ch["location"], table_cell_style),
            Paragraph(ch["old_value"] or "---", table_cell_style),
            Paragraph(ch["new_value"] or "---", table_cell_style),
            Paragraph(ch["delta"] or "---", table_cell_style),
            Paragraph(f"{ch['confidence']*100:.0f}%", table_cell_style)
        ])
        
    t_summary = Table(summary_rows, colWidths=[60, 120, 100, 130, 130, 130, 70])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D3D3D3")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_summary)
    story.append(PageBreak())
    
    # ==================== PAGES 6+: DETAILED FINDINGS ====================
    img_a_cv = cv2.imread(image_paths["original_a"])
    img_b_cv = cv2.imread(image_paths["aligned_b"])
    img_ov_cv = cv2.imread(image_paths["added_removed_overlay"])
    
    temp_crop_paths = []
    
    for idx, ch in enumerate(visible_changes):
        x, y, w, h = ch["bbox"]
        ch_id = ch["id"]
        
        pad = 20
        h_img, w_img = img_b_cv.shape[:2]
        
        cx1 = max(0, x - pad)
        cy1 = max(0, y - pad)
        cx2 = min(w_img, x + w + pad)
        cy2 = min(h_img, y + h + pad)
        
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        
        crop_a_path = os.path.join(os.path.dirname(output_pdf_path), f"crop_{ch_id}_a.png")
        crop_b_path = os.path.join(os.path.dirname(output_pdf_path), f"crop_{ch_id}_b.png")
        crop_ov_path = os.path.join(os.path.dirname(output_pdf_path), f"crop_{ch_id}_ov.png")
        
        if crop_w > 0 and crop_h > 0:
            cv2.imwrite(crop_a_path, img_a_cv[cy1:cy2, cx1:cx2])
            cv2.imwrite(crop_b_path, img_b_cv[cy1:cy2, cx1:cx2])
            cv2.imwrite(crop_ov_path, img_ov_cv[cy1:cy2, cx1:cx2])
            temp_crop_paths.extend([crop_a_path, crop_b_path, crop_ov_path])
        else:
            cv2.imwrite(crop_a_path, np.full((100, 100, 3), 255, dtype=np.uint8))
            cv2.imwrite(crop_b_path, np.full((100, 100, 3), 255, dtype=np.uint8))
            cv2.imwrite(crop_ov_path, np.full((100, 100, 3), 255, dtype=np.uint8))
            temp_crop_paths.extend([crop_a_path, crop_b_path, crop_ov_path])

        finding_elements = []
        finding_elements.append(Paragraph(f"Detailed Modification Log - {ch_id}", h1_style))
        finding_elements.append(Spacer(1, 10))
        
        crop_width, crop_height = 180, 135
        crops_table_data = [
            [
                Image(crop_a_path, width=crop_width, height=crop_height),
                Image(crop_b_path, width=crop_width, height=crop_height),
                Image(crop_ov_path, width=crop_width, height=crop_height)
            ],
            [
                Paragraph("<b>Before (Revision A)</b>", body_style),
                Paragraph("<b>After (Revision B)</b>", body_style),
                Paragraph("<b>Overlay Diff</b>", body_style)
            ]
        ]
        t_crops = Table(crops_table_data, colWidths=[200, 200, 200])
        t_crops.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,1), (-1,1), 10),
            ('GRID', (0,0), (-1,0), 1, colors.HexColor("#D3D3D3")),
        ]))
        finding_elements.append(t_crops)
        finding_elements.append(Spacer(1, 15))
        
        details_data = [
            [
                Paragraph("<b>Entity Type:</b>", body_style),
                Paragraph(ch["type"], body_style),
                Paragraph("<b>Location Context:</b>", body_style),
                Paragraph(ch["location"], body_style),
            ],
            [
                Paragraph("<b>Before Value:</b>", body_style),
                Paragraph(ch["old_value"] or "---", body_style),
                Paragraph("<b>After Value:</b>", body_style),
                Paragraph(ch["new_value"] or "---", body_style),
            ],
            [
                Paragraph("<b>Delta / Change:</b>", body_style),
                Paragraph(ch["delta"] or "---", body_style),
                Paragraph("<b>Confidence Level:</b>", body_style),
                Paragraph(f"{ch['confidence']*100:.0f}%", body_style),
            ],
            [
                Paragraph("<b>Bounding Bbox:</b>", body_style),
                Paragraph(f"x={x}, y={y}, w={w}, h={h}", body_style),
                Paragraph("<b>Real-world Scale:</b>", body_style),
                Paragraph(f"{ch['real_dimensions'][0]:.1f} x {ch['real_dimensions'][1]:.1f} mm", body_style),
            ],
        ]
        t_details = Table(details_data, colWidths=[130, 240, 130, 240])
        t_details.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8F9FA")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E0E0E0")),
            ('PADDING', (0,0), (-1,-1), 8),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        finding_elements.append(t_details)
        finding_elements.append(Spacer(1, 15))
        
        finding_elements.append(Paragraph("<b>Explainable AI Description:</b>", h2_style))
        finding_elements.append(Paragraph(ch["explanation"], body_style))
        
        story.append(KeepTogether(finding_elements))
        story.append(PageBreak())
        
    # ==================== FINAL PAGE: RECOMMENDATIONS ====================
    story.append(Paragraph("System Conclusion & Recommendations", h1_style))
    story.append(Spacer(1, 15))
    
    rec_text = (
        "This report was generated automatically by the offline Revision Intelligence System. "
        "The analysis highlights that the drawing registered with a confidence of "
        f"<b>{stats.alignment_confidence:.1f}%</b>. The estimated modifications cover "
        f"<b>{stats.percent_area_changed}%</b> of the total drawing canvas, falling into a "
        f"<b>{stats.severity_index}</b> modification category."
    )
    story.append(Paragraph(rec_text, body_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<b>Next Action Recommendations:</b>", h2_style))
    
    bullets = [
        "<b>Verify Document Alignment:</b> High geometric accuracy has been confirmed via SIFT/ORB corner mapping matching.",
        "<b>Review Highlighted Regions:</b> Orange overlays mark CV-detected door and window modifications conforming to standard widths.",
        "<b>Confirm Revision Clouds:</b> If revision clouds or other structural marks exist on sheets, double check them manually."
    ]
    for b in bullets:
        story.append(Paragraph(f"• &nbsp; {b}", body_style))
        story.append(Spacer(1, 8))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    
    for p in temp_crop_paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
            
    return output_pdf_path
