"""
scratch/test_apptest_run.py
Run streamlit AppTest on dashboard/app.py in Demo, Organizer, and Upload modes.
"""
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def test_app_demo_mode():
    print("Testing AppTest: Demo mode...")
    at = AppTest.from_file("dashboard/app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Demo mode crashed with exception: {at.exception}"
    print("Demo mode passed!")

def test_app_organizer_mode():
    print("Testing AppTest: Organizer mode...")
    at = AppTest.from_file("dashboard/app.py", default_timeout=40)
    at.run()
    # Change radio to Organizer
    radio = at.sidebar.radio[0]
    radio.set_value("Organizer")
    at.run()
    assert not at.exception, f"Organizer mode crashed with exception: {at.exception}"
    print("Organizer mode passed!")

def test_app_upload_mode():
    print("Testing AppTest: Upload mode...")
    at = AppTest.from_file("dashboard/app.py", default_timeout=40)
    at.run()
    radio = at.sidebar.radio[0]
    radio.set_value("Upload")
    at.run()
    assert not at.exception, f"Upload mode crashed with exception: {at.exception}"
    print("Upload mode passed!")

if __name__ == "__main__":
    test_app_demo_mode()
    test_app_organizer_mode()
    test_app_upload_mode()
    print("ALL APPTESTS (DEMO, ORGANIZER, UPLOAD) PASSED SUCCESSFULLY!")
