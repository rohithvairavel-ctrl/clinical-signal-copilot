from clinical_signal_copilot.report.copilot import CopilotReport, render_html, render_markdown


def test_report_contains_disclaimer():
    r = CopilotReport(
        detection={"f1": 0.5},
        conformal={"empirical_coverage": 0.9},
        fairness={"gaps": {"f1": 0.1}, "groups": {"A": {"sensitivity": 0.8, "ppv": 0.7, "f1": 0.75, "n": 10}}},
        sqi_summary={"mean_sqi": 0.6},
        notes=["synthetic only"],
    )
    md = render_markdown(r)
    html = render_html(r)
    assert "NOT A MEDICAL DEVICE" in md
    assert "NOT FOR CLINICAL USE" in md
    assert "NOT A MEDICAL DEVICE" in html
    assert "mean_sqi" in md
