"""Generate the original PDF document included in the sample corpus."""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape

CONTENT = """Community energy systems coordinate local generation, storage, and demand so that electricity can be used closer to where it is produced. A neighborhood may combine rooftop solar, a shared battery, efficient buildings, and a controller that schedules flexible loads. The aim is not to disconnect from the wider grid. Instead, local coordination can reduce peak demand, improve resilience, and help residents participate in the transition to cleaner electricity.

A community energy project begins with a practical assessment. Planners map the electricity use of homes and public buildings, identify available roof or land area, review connection capacity, and learn which residents are interested. A technically attractive design can still fail if it ignores building ownership, accessibility, maintenance responsibilities, or the needs of renters. Early engagement should explain costs, possible benefits, decision rights, and what happens when equipment reaches the end of its useful life.

Solar panels produce electricity when sunlight is available, while many homes need the most electricity in the morning and evening. A battery can shift some solar output to a later period, but storage is not automatically the best investment. Its value depends on the local tariff, export compensation, battery efficiency, degradation, installation cost, and how often stored energy is needed. Modeling several operating strategies is more informative than sizing storage from a single sunny day.

A local energy management system estimates generation and demand, then schedules controllable devices within agreed limits. Water heating, vehicle charging, and some commercial processes can often move in time without reducing the service people receive. The system should preserve user overrides and comfort constraints. It should also explain its decisions in language that residents can understand. Automation that is opaque or difficult to disable can undermine trust even when its energy forecasts are accurate.

Fair participation is a design requirement. Households without suitable roofs should not be excluded from every benefit. Shared ownership, subscription arrangements, community buildings, or bill credits may create alternatives, although each option has legal and administrative costs. A project should publish eligibility rules and monitor who joins, who receives savings, and who bears risk. Average savings can hide unequal outcomes between households with different schedules, incomes, or heating needs.

Resilience claims require careful definition. A grid-connected solar array will usually shut down during an outage unless equipment is configured to isolate safely and provide backup power. A battery sized for short interruptions may not support a long outage, especially during cold or cloudy weather. Project plans should identify critical loads, backup duration, islanding equipment, safe reconnection procedures, and who is responsible for operating the system during an emergency.

Cybersecurity and privacy belong in the initial design. Smart meters and controllers collect detailed operational data, so the project should minimize what is retained, restrict access, secure remote connections, and establish a process for software updates. Residents should know which data are collected and for what purpose. Equipment suppliers should describe support periods and incident response rather than treating connectivity as a substitute for long-term maintenance.

Economic evaluation should compare the project with a credible alternative, such as individual rooftop systems or grid electricity under current tariffs. Estimates should state assumptions about financing, replacement costs, energy prices, equipment lifetime, and participation. Sensitivity analysis can show which assumptions matter most. A pilot can reveal installation and coordination challenges, but a small pilot does not prove that a larger project will achieve the same costs or benefits.

Governance determines whether coordination remains legitimate over time. Residents and public partners need a clear process for setting priorities, reviewing performance, resolving complaints, and approving major changes. An independent operator may provide technical expertise, while a community board protects local accountability. These roles should be documented so that urgent maintenance does not become confused with authority over tariffs or membership.

Evaluation should use more than annual energy production. Useful measures include peak demand, self-consumption, outage performance, participation across household types, operating cost, equipment availability, and resident satisfaction. Baselines should be recorded before installation, and results should be reported with limitations. If the project changes its operating strategy, the reason and expected impact should be visible to participants.

A responsible deployment therefore treats technology as one part of a social and operational system. It combines realistic engineering, transparent economics, inclusive governance, privacy protections, and ongoing maintenance. Starting with local needs, testing assumptions, and reporting both benefits and shortfalls gives communities a stronger basis for deciding whether to expand, redesign, or stop a project."""


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "community_energy.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph("Community Energy Systems: Planning, Equity, and Resilience", styles["Title"]), Spacer(1, 14)]
    for paragraph in CONTENT.split("\n\n"):
        story.append(Paragraph(escape(paragraph), styles["BodyText"]))
        story.append(Spacer(1, 9))
    SimpleDocTemplate(str(output), pagesize=letter, title="Community Energy Systems").build(story)
    print(f"Generated {output}")


if __name__ == "__main__":
    main()
