import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ids.config import LOG_FILE

st.set_page_config(page_title="IDS Dashboard", layout="wide")

st.title("Intrusion Detection Dashboard")
log_path = st.sidebar.text_input("Log file", str(LOG_FILE))
refresh_sec = st.sidebar.slider("Refresh (sec)", 0.5, 5.0, 1.0)

log_file = Path(log_path)
if not log_file.exists():
	st.warning(f"Log file not found: {log_file}")
	st.stop()

placeholder = st.empty()

while True:
	try:
		# Read log file with flexible parsing
		df = pd.read_csv(log_file, sep="\t", header=None, on_bad_lines='skip')
		
		# Handle different log formats
		if df.shape[1] >= 4:
			df.columns = ["timestamp", "level", "pred", "score"] + [f"extra_{i}" for i in range(df.shape[1] - 4)]
			
			# Parse pred and score if they contain "pred=X" format
			if df["pred"].dtype == object and df["pred"].str.contains("pred=", na=False).any():
				df["pred"] = df["pred"].str.extract(r'pred=(\d+)', expand=False).astype(float)
			if df["score"].dtype == object and df["score"].str.contains("score=", na=False).any():
				df["score"] = df["score"].str.extract(r'score=([\d.]+)', expand=False).astype(float)
		else:
			st.warning(f"Unexpected log format: {df.shape[1]} columns")
			time.sleep(refresh_sec)
			continue
			
		# Filter only alerts
		df_alerts = df[df["level"] == "ALERT"].copy()
		
		with placeholder.container():
			col1, col2 = st.columns(2)
			with col1:
				st.subheader("Recent Alerts")
				st.dataframe(df_alerts[["timestamp", "level", "pred", "score"]].tail(100))
			with col2:
				st.subheader("Alerts over Time")
				if not df_alerts.empty:
					df_alerts["ts"] = pd.to_datetime(df_alerts["timestamp"]) 
					df_counts = df_alerts.set_index("ts").resample("5s").size().rename("alerts").reset_index()
					st.line_chart(df_counts, x="ts", y="alerts", height=300)
				else:
					st.info("No alerts detected yet")
		time.sleep(refresh_sec)
	except st.runtime.scriptrunner.StopException:
		break
	except Exception as e:
		st.error(f"Error: {str(e)}")
		time.sleep(refresh_sec)
