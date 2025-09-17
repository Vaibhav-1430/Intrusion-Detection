import time
from pathlib import Path

import pandas as pd
import streamlit as st

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
		df = pd.read_csv(log_file, sep="\t", header=None, names=["timestamp", "level", "pred", "score"])
		with placeholder.container():
			col1, col2 = st.columns(2)
			with col1:
				st.subheader("Recent Alerts")
				st.dataframe(df.tail(100))
			with col2:
				st.subheader("Alerts over Time")
				df["ts"] = pd.to_datetime(df["timestamp"]) 
				df_counts = df.set_index("ts").resample("5s").size().rename("alerts").reset_index()
				st.line_chart(df_counts, x="ts", y="alerts", height=300)
		time.sleep(refresh_sec)
	except st.runtime.scriptrunner.StopException:
		break
	except Exception as e:
		st.error(str(e))
		time.sleep(refresh_sec)
