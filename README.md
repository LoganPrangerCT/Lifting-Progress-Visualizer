Track your lifting progress. Parses your workout logs and generates an interactive HTML report showing your strength trends, personal records, and projections.

Setup:
pip install pandas matplotlib numpy

Usage:
Place your workout data in CSV files named `Lifts - Push.csv`, `Lifts - Pull.csv`, and `Lifts - Legs.csv`.

Format: first column is date (MM/DD/YYYY), subsequent columns are exercises. Each cell is 'reps x weight' (e.g., `6x225`).

## Data format example

| Date      | Bench | Incline DB |
|-----------|-------|------------|
| 3/19/2026 | 7x225 | 8x50       |
| 3/23/2026 | 6x225 | 6x50       |


Then run:
python lifts.py

This outputs `Lift_Progress_Report.html` — open it in your browser.

Output
- HTML dashboard with effort heatmap and progress charts
- Per-day exercise breakdown
- Personal records table
- 90-day strength projections

I included my own dashboard as an example
