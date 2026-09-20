import os, sys
sys.path.insert(0, os.path.abspath("."))
from streamlit.testing.v1 import AppTest

print("1. Running AppTest on dashboard/app.py (Default: Organizer Mode)...")
at = AppTest.from_file("dashboard/app.py", default_timeout=30)
at.run()
assert not at.exception, f"App crashed in Organizer mode: {at.exception}"
print(f"Organizer mode: {len(at.metric)} metrics, 0 exceptions!")

print("2. Switching AppTest to Demo Mode...")
if at.sidebar.radio:
    at.sidebar.radio[0].set_value("Demo").run()
    assert not at.exception, f"App crashed in Demo mode: {at.exception}"
    print(f"Demo mode: {len(at.metric)} metrics, 0 exceptions!")

print("ALL APP MODES VERIFIED WITH ZERO CRASHES!")
