# WC 2026 Penalty Story App

Streamlit app for exploring the World Cup 2026 player penalty dataset and the SofaScore-enriched penalty files.

Run from the repository root:

```powershell
pip install -r app\requirements.txt
python -m streamlit run app\app.py --server.port 8501
```

The app reads CSV files from the parent project folder and uses `image.png` as the goal background. It does not modify the CSV files.
