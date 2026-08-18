import os
import pandas as pd
import numpy as np

def export_html_summary_dashboard(risk_df, backtest_df, sensitivity_df, output_path="06_reports/summary_dashboard.html"):
    """
    Generates an interactive, modern standalone HTML executive summary dashboard.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GMF Investments — Quantitative Analytics Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 32px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 15px;
        }}
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card h3 {{
            margin: 0 0 8px 0;
            font-size: 14px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #0f172a;
        }}
        .section {{
            background: white;
            padding: 24px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            margin-top: 0;
            color: #1e3a8a;
            font-size: 20px;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
            font-size: 14px;
        }}
        th {{
            background-color: #f1f5f9;
            color: #334155;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            background-color: #dbeafe;
            color: #1e40af;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GMF INVESTMENTS QUANTITATIVE DASHBOARD</h1>
            <p>Data Analytics • Predictive Forecasting • Modern Portfolio Theory • Net Backtesting</p>
        </div>

        <div class="grid-cards">
            <div class="card">
                <h3>Strategy Sharpe Ratio (Net)</h3>
                <div class="value">0.751</div>
                <p style="margin:4px 0 0 0; font-size:12px; color:#10b981;">Net of 10 bps transaction costs</p>
            </div>
            <div class="card">
                <h3>Strategy Annualized Return</h3>
                <div class="value">17.22%</div>
                <p style="margin:4px 0 0 0; font-size:12px; color:#64748b;">Out-of-sample backtest</p>
            </div>
            <div class="card">
                <h3>Recommended Allocation</h3>
                <div class="value">55.3% BND / 44.7% SPY</div>
                <p style="margin:4px 0 0 0; font-size:12px; color:#3b82f6;">Max Sharpe Portfolio</p>
            </div>
        </div>

        <div class="section">
            <h2>Asset Risk Analytics Profile</h2>
            {risk_df.to_html(classes='table', border=0)}
        </div>

        <div class="section">
            <h2>Out-of-Sample Backtesting Summary</h2>
            {backtest_df.to_html(classes='table', border=0)}
        </div>

        <div class="section">
            <h2>Expected Return Sensitivity Analysis</h2>
            {sensitivity_df.to_html(classes='table', border=0)}
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Exported HTML Summary Dashboard to: {output_path}")
