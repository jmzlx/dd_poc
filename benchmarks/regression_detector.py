#!/usr/bin/env python3
"""
Performance Regression Detection for dd-poc

This module provides automated detection of performance regressions
in the dd-poc system by comparing benchmark results over time.

Features:
- Statistical comparison of benchmark runs
- Regression alerts based on configurable thresholds
- Historical performance trending
- Automated reporting of performance changes

Usage:
    python benchmarks/regression_detector.py --baseline-run baseline_20241201 --compare-run new_run_20241202
    python benchmarks/regression_detector.py --trend-analysis --days 30
    python benchmarks/regression_detector.py --alerts --email user@example.com
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class RegressionAlert:
    """Represents a performance regression alert"""
    metric: str
    baseline_value: float
    current_value: float
    change_percent: float
    threshold_percent: float
    severity: str  # "low", "medium", "high", "critical"
    description: str


@dataclass
class RegressionReport:
    """Complete regression analysis report"""
    baseline_run: str
    compare_run: str
    alerts: List[RegressionAlert]
    summary: Dict[str, Any]
    timestamp: str


class RegressionDetector:
    """Detects performance regressions in benchmark results"""

    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.alert_thresholds = {
            "accuracy": 0.05,  # 5% drop
            "precision": 0.05,
            "recall": 0.05,
            "f1_score": 0.05,
            "precision@10": 0.10,  # 10% drop for search metrics
            "recall@10": 0.10,
            "mrr": 0.10,
            "semantic_similarity": 0.05,
            "throughput": 0.15  # 15% drop for throughput
        }

    def detect_regression(self, baseline_run: str, compare_run: str,
                         confidence_level: float = 0.95) -> RegressionReport:
        """Detect regressions between two benchmark runs"""
        print(f"🔍 Detecting regressions: {baseline_run} vs {compare_run}")

        # Load benchmark results
        baseline_results = self._load_benchmark_results(baseline_run)
        compare_results = self._load_benchmark_results(compare_run)

        if not baseline_results or not compare_results:
            raise ValueError("Could not load benchmark results")

        # Analyze regressions
        alerts = []
        summary = {
            "total_metrics": 0,
            "regressions_detected": 0,
            "severity_breakdown": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "significant_improvements": 0
        }

        # Group results by task and metric
        baseline_metrics = self._group_results_by_metric(baseline_results)
        compare_metrics = self._group_results_by_metric(compare_results)

        # Compare each metric
        all_metrics = set(baseline_metrics.keys()) | set(compare_metrics.keys())

        for metric_key in all_metrics:
            if metric_key not in baseline_metrics or metric_key not in compare_metrics:
                continue

            baseline_values = baseline_metrics[metric_key]
            compare_values = compare_metrics[metric_key]

            if not baseline_values or not compare_values:
                continue

            # Calculate statistical comparison
            baseline_mean = statistics.mean(baseline_values)
            compare_mean = statistics.mean(compare_values)

            # Calculate change
            if baseline_mean != 0:
                change_percent = (compare_mean - baseline_mean) / abs(baseline_mean)
            else:
                change_percent = 0

            # Check for regression
            metric_name = metric_key.split('_', 1)[1] if '_' in metric_key else metric_key
            threshold = self.alert_thresholds.get(metric_name, 0.05)

            summary["total_metrics"] += 1

            if change_percent < -threshold:  # Negative change indicates regression
                severity = self._calculate_severity(abs(change_percent), metric_name)
                alert = RegressionAlert(
                    metric=metric_key,
                    baseline_value=baseline_mean,
                    current_value=compare_mean,
                    change_percent=change_percent * 100,
                    threshold_percent=threshold * 100,
                    severity=severity,
                    description=self._generate_alert_description(metric_key, change_percent)
                )
                alerts.append(alert)
                summary["regressions_detected"] += 1
                summary["severity_breakdown"][severity] += 1

            elif change_percent > threshold:  # Positive change indicates improvement
                summary["significant_improvements"] += 1

        # Sort alerts by severity
        alerts.sort(key=lambda x: ["critical", "high", "medium", "low"].index(x.severity))

        report = RegressionReport(
            baseline_run=baseline_run,
            compare_run=compare_run,
            alerts=alerts,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )

        return report

    def trend_analysis(self, days: int = 30, metric_filter: Optional[str] = None) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        print(f"📈 Analyzing performance trends over last {days} days")

        # Load all recent benchmark results
        recent_results = self._load_recent_results(days)

        if not recent_results:
            return {"error": "No recent benchmark results found"}

        # Group by date and metric
        trends = {}

        for result_file, results in recent_results.items():
            run_date = results.get("timestamp", "")[:10]  # Extract date

            for result in results.get("results", []):
                metric_key = f"{result['task']}_{result['metric']}"

                if metric_filter and metric_filter not in metric_key:
                    continue

                if metric_key not in trends:
                    trends[metric_key] = []

                trends[metric_key].append({
                    "date": run_date,
                    "value": result["value"],
                    "run_id": results.get("run_id", "")
                })

        # Sort trends by date
        for metric_key in trends:
            trends[metric_key].sort(key=lambda x: x["date"])

        # Calculate trend statistics
        trend_summary = {}
        for metric_key, data_points in trends.items():
            if len(data_points) < 2:
                continue

            values = [dp["value"] for dp in data_points]

            # Calculate trend slope (simple linear regression)
            x = list(range(len(values)))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

            trend_summary[metric_key] = {
                "slope": slope,
                "r_squared": r_value**2,
                "p_value": p_value,
                "significant_trend": p_value < 0.05,
                "direction": "improving" if slope > 0 else "degrading" if slope < 0 else "stable",
                "data_points": len(data_points),
                "latest_value": values[-1],
                "change_from_start": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            }

        return {
            "trends": trends,
            "summary": trend_summary,
            "analysis_period_days": days,
            "total_runs_analyzed": len(recent_results)
        }

    def send_alerts(self, report: RegressionReport, email_config: Dict[str, str]):
        """Send regression alerts via email"""
        if not report.alerts:
            print("✅ No regressions detected - no alerts to send")
            return

        print(f"📧 Sending {len(report.alerts)} regression alerts")

        # Create email content
        subject = f"🚨 dd-poc Performance Regression Alert - {len(report.alerts)} issues detected"

        body = f"""
Performance Regression Report
=============================

Baseline Run: {report.baseline_run}
Compare Run: {report.compare_run}
Generated: {report.timestamp}

Summary:
- Total metrics analyzed: {report.summary['total_metrics']}
- Regressions detected: {report.summary['regressions_detected']}
- Significant improvements: {report.summary['significant_improvements']}

Regression Details:
"""

        for alert in report.alerts:
            body += ".1f"".1f"

        # Group alerts by severity for email
        severity_groups = {}
        for alert in report.alerts:
            if alert.severity not in severity_groups:
                severity_groups[alert.severity] = []
            severity_groups[alert.severity].append(alert)

        # Send email
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(email_config['smtp_server'], int(email_config['smtp_port']))
            if email_config.get('use_tls', True):
                server.starttls()

            if 'username' in email_config:
                server.login(email_config['username'], email_config['password'])

            server.send_message(msg)
            server.quit()

            print("✅ Regression alerts sent successfully")

        except Exception as e:
            print(f"❌ Failed to send email alerts: {e}")

    def generate_trend_report(self, trend_data: Dict[str, Any], output_file: Optional[str] = None):
        """Generate trend analysis report with visualizations"""
        if not output_file:
            output_file = f"benchmarks/reports/trend_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Performance Trends", "Trend Significance",
                          "Regression Summary", "Metric Distribution"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Performance trends plot
        trend_summary = trend_data.get("summary", {})
        if trend_summary:
            metrics = list(trend_summary.keys())[:10]  # Top 10 metrics
            slopes = [trend_summary[m]["slope"] for m in metrics]
            p_values = [trend_summary[m]["p_value"] for m in metrics]

            fig.add_trace(
                go.Bar(name="Trend Slope", x=metrics, y=slopes, marker_color='lightblue'),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(name="P-Values", x=metrics, y=p_values, mode='lines+markers',
                          marker_color='red', line_color='red'),
                row=1, col=2
            )

            # Add significance threshold line
            fig.add_hline(y=0.05, line_dash="dot", line_color="red",
                         annotation_text="p=0.05 threshold", row=1, col=2)

        # Update layout
        fig.update_layout(
            title="Performance Trend Analysis Report",
            height=800,
            showlegend=True
        )

        # Add trend summary text
        summary_text = f"""
        <h2>Trend Analysis Summary</h2>
        <p><strong>Analysis Period:</strong> {trend_data.get('analysis_period_days', 'N/A')} days</p>
        <p><strong>Total Runs Analyzed:</strong> {trend_data.get('total_runs_analyzed', 0)}</p>

        <h3>Key Findings:</h3>
        <ul>
        """

        for metric, stats in trend_summary.items():
            if stats["significant_trend"]:
                summary_text += f"""
                <li><strong>{metric}:</strong> {stats['direction'].title()} trend
                (slope: {stats['slope']:.4f}, p-value: {stats['p_value']:.4f})</li>
                """

        summary_text += "</ul>"

        # Save as HTML with embedded plot
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Trend Analysis</title>
        </head>
        <body>
            <h1>dd-poc Performance Trend Analysis</h1>
            {summary_text}
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"📊 Trend analysis report saved to {output_path}")
        return str(output_path)

    def _load_benchmark_results(self, run_id: str) -> Optional[Dict]:
        """Load benchmark results for a specific run"""
        results_file = self.results_dir / f"{run_id}_results.json"

        if not results_file.exists():
            print(f"❌ Results file not found: {results_file}")
            return None

        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load results: {e}")
            return None

    def _load_recent_results(self, days: int) -> Dict[str, Dict]:
        """Load benchmark results from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = {}

        if not self.results_dir.exists():
            return recent_results

        for results_file in self.results_dir.glob("*_results.json"):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)

                run_timestamp = data.get("timestamp", "")
                if run_timestamp:
                    run_date = datetime.fromisoformat(run_timestamp.replace('Z', '+00:00'))
                    if run_date >= cutoff_date:
                        recent_results[results_file.stem] = data

            except Exception as e:
                print(f"⚠️ Failed to load {results_file}: {e}")
                continue

        return recent_results

    def _group_results_by_metric(self, results_data: Dict) -> Dict[str, List[float]]:
        """Group benchmark results by metric"""
        grouped = {}

        for result in results_data.get("results", []):
            metric_key = f"{result['task']}_{result['metric']}"
            if metric_key not in grouped:
                grouped[metric_key] = []
            grouped[metric_key].append(result["value"])

        return grouped

    def _calculate_severity(self, change_percent: float, metric_name: str) -> str:
        """Calculate severity level for a regression"""
        # Define severity thresholds
        if change_percent > 0.25:  # >25% drop
            return "critical"
        elif change_percent > 0.15:  # >15% drop
            return "high"
        elif change_percent > 0.08:  # >8% drop
            return "medium"
        else:
            return "low"

    def _generate_alert_description(self, metric_key: str, change_percent: float) -> str:
        """Generate human-readable description for regression alert"""
        task, metric = metric_key.split('_', 1)

        descriptions = {
            "accuracy": ".1f",
            "precision": ".1f",
            "recall": ".1f",
            "f1_score": ".1f",
            "precision@10": ".1f",
            "recall@10": ".1f",
            "mrr": ".1f",
            "semantic_similarity": ".1f",
            "throughput": ".1f"
        }

        return descriptions.get(metric, ".1f")


def main():
    """Main entry point for regression detection"""
    parser = argparse.ArgumentParser(description="Detect performance regressions in dd-poc")
    parser.add_argument("--baseline-run", help="Baseline benchmark run ID")
    parser.add_argument("--compare-run", help="Comparison benchmark run ID")
    parser.add_argument("--trend-analysis", action="store_true",
                       help="Perform trend analysis instead of direct comparison")
    parser.add_argument("--days", type=int, default=30,
                       help="Number of days for trend analysis (default: 30)")
    parser.add_argument("--metric-filter", help="Filter metrics for analysis")
    parser.add_argument("--alerts", action="store_true",
                       help="Send email alerts for regressions")
    parser.add_argument("--email-to", help="Email address for alerts")
    parser.add_argument("--smtp-server", default="smtp.gmail.com",
                       help="SMTP server for alerts")
    parser.add_argument("--smtp-port", type=int, default=587,
                       help="SMTP port for alerts")

    args = parser.parse_args()

    detector = RegressionDetector()

    try:
        if args.trend_analysis:
            # Perform trend analysis
            trend_data = detector.trend_analysis(args.days, args.metric_filter)

            # Generate trend report
            report_file = detector.generate_trend_report(trend_data)

            print("
📊 Trend Analysis Complete"            print(f"📁 Report saved to: {report_file}")

            # Print summary
            summary = trend_data.get("summary", {})
            significant_trends = [m for m, s in summary.items() if s["significant_trend"]]

            print(f"📈 Found {len(significant_trends)} significant trends:")
            for metric in significant_trends:
                stats = summary[metric]
                print(f"  • {metric}: {stats['direction']} ({stats['change_from_start']:+.1f}%)")

        elif args.baseline_run and args.compare_run:
            # Perform regression detection
            report = detector.detect_regression(args.baseline_run, args.compare_run)

            print("
🔍 Regression Detection Complete"            print(f"📊 Analyzed {report.summary['total_metrics']} metrics")
            print(f"🚨 Found {report.summary['regressions_detected']} regressions")

            if report.alerts:
                print("\nRegression Alerts:")
                for alert in report.alerts:
                    print(f"  {alert.severity.upper()}: {alert.metric}")
                    print(".1f"                    print()

                # Send alerts if requested
                if args.alerts and args.email_to:
                    email_config = {
                        'to_email': args.email_to,
                        'smtp_server': args.smtp_server,
                        'smtp_port': args.smtp_port,
                        'from_email': 'alerts@dd-poc.local',
                        'use_tls': True
                    }
                    detector.send_alerts(report, email_config)
            else:
                print("✅ No significant regressions detected")

        else:
            print("❌ Please specify either --baseline-run and --compare-run, or --trend-analysis")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Regression detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
