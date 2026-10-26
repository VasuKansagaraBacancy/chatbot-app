"""Generate a synthetic annual report PDF for testing the RAG pipeline.

This exists so the pipeline can be validated end to end without the real
document: every fact below is invented, and the questions in
tests/fixture_questions.json assert against these exact values. If retrieval
or chunking regresses, those assertions fail.

    .venv\\Scripts\\python.exe tests\\make_fixture.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).parent / "fixtures" / "sample_annual_report.pdf"

BODY = """
GPO ANNUAL REPORT 2024

LETTER FROM THE DIRECTOR

This year the Office completed its transition to a digital-first publishing model.
We retired the last of the legacy composition systems in March 2024 and moved all
statutory publications onto the new platform. Staff across three facilities
contributed to the effort, and I am grateful for their persistence through a
demanding schedule.

Our priorities for the coming year are continuity of the statutory programme,
further automation of the deposit workflow, and a measured expansion of the
preservation programme described later in this report.

FINANCIAL HIGHLIGHTS

Total revenue for fiscal year 2024 was 1,284.6 million dollars, an increase of
6.2 per cent over the 1,209.3 million dollars recorded in fiscal year 2023.
Revenue from congressional publishing accounted for 412.8 million dollars, while
agency publishing services contributed 611.4 million dollars. The remaining
260.4 million dollars came from information dissemination and other sources.

Operating expenses totalled 1,197.2 million dollars against 1,151.8 million
dollars in the prior year. Personnel costs represented 64 per cent of operating
expenses. Net operating income was 87.4 million dollars, compared with 57.5
million dollars in fiscal year 2023.

Capital expenditure was 48.9 million dollars, directed principally at the
replacement of two web presses at the Washington facility and at the second
phase of the digital preservation repository.

WORKFORCE

Headcount at 30 September 2024 stood at 1,742 full-time equivalent employees,
down from 1,803 a year earlier. The reduction was achieved entirely through
attrition and voluntary separation; there were no involuntary separations during
the year. The median length of service was 11.4 years.

Forty-one apprentices were enrolled across the printing, binding and platemaking
trades, the largest intake since 2016. The apprenticeship programme runs for
four years and leads to journeyman certification.

Employee engagement, measured by the annual survey, rose to 71 per cent
favourable from 68 per cent in the previous year. The survey response rate was
83 per cent.
"""

BODY2 = """
DIGITAL PRESERVATION PROGRAMME

The preservation repository now holds 3.9 petabytes of content, comprising
2.1 million distinct publications. During the year the Office ingested 184,000
new items, of which 41,000 were retrospective digitisations of pre-1976
material.

Three preservation tiers are maintained. Tier one holds master files in
uncompressed form at two geographically separate sites. Tier two holds derivative
access copies. Tier three holds a dark archive copy written to tape and verified
on a six-month cycle. Fixity checking runs continuously across all three tiers;
during the year 11 checksum mismatches were detected and all 11 were repaired
from redundant copies.

The repository achieved certification against the ISO 16363 trustworthy digital
repository standard in July 2024, following an audit that began in October 2023.

FEDERAL DEPOSITORY LIBRARY PROGRAMME

The depository programme served 1,086 participating libraries at year end, a
net decrease of 28 from the prior year. Twenty-two libraries withdrew and six
were newly designated. Regional depositories numbered 44.

Distribution to depositories was 92 per cent electronic by title count. Tangible
distribution continued for 614 titles judged to require a physical copy under the
essential titles list.

The programme delivered 37 training webinars attended by 4,210 library staff,
and published the revised Depository Library Handbook in September 2024.

SUSTAINABILITY

Greenhouse gas emissions from facility operations were 24,100 metric tons of
carbon dioxide equivalent, a reduction of 9 per cent against the 2019 baseline.
Paper procurement reached 68 per cent recycled content by weight. Waste diverted
from landfill was 54 per cent.

The Washington facility completed a lighting retrofit in May 2024 that is
expected to reduce electricity consumption by 1.8 million kilowatt hours
annually.

GOVERNANCE AND RISK

The principal risks identified by the Office are concentration of specialised
printing capability at a single site, cyber security exposure arising from the
expanded digital estate, and the demographic profile of the skilled trades
workforce. Mitigation plans for each are reviewed quarterly by the Executive
Committee.

An independent audit of financial statements was conducted and issued an
unmodified opinion. There were no material weaknesses in internal control over
financial reporting.
"""


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Times-Roman",
                          fontSize=10.5, leading=15, spaceAfter=8)
    head = ParagraphStyle("head", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=11, spaceBefore=14, spaceAfter=6)

    story = []
    for page_text in (BODY, BODY2):
        for block in [b.strip() for b in page_text.strip().split("\n\n") if b.strip()]:
            flat = " ".join(block.split())
            is_heading = flat.isupper() and len(flat) < 70
            story.append(Paragraph(flat, head if is_heading else body))
        story.append(PageBreak())
    story.pop()

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="GPO Annual Report 2024 (synthetic test fixture)",
    )
    doc.build(story)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}  ({p.stat().st_size:,} bytes)")
