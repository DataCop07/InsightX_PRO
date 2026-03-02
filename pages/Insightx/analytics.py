# analytics.py
# Simple analytics module to log events
import csv
from datetime import datetime
import os

# CSV file to store analytics
ANALYTICS_FILE = "analytics_log.csv"

# Ensure the file exists with headers
if not os.path.exists(ANALYTICS_FILE):
    with open(ANALYTICS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "event_name", "details"])

def log_event(event_name, details=""):
    """
    Log an analytics event.
    
    Args:
        event_name (str): Name of the event (e.g., "button_click")
        details (str): Optional extra info about the event
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ANALYTICS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event_name, details])
    print(f"[ANALYTICS] {timestamp} | {event_name} | {details}")