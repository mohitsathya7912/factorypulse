"""
Verify dashboard/app.py executes without exception in:
1. Demo mode (default)
2. Organizer mode
3. Upload mode (empty state)
"""

import sys
from streamlit.testing.v1 import AppTest

def test_demo_mode():
    print("Testing Demo mode...")
    at = AppTest.from_file("dashboard/app.py", default_timeout=40)
    at.run()
    if at.exception:
        print("EXCEPTION IN DEMO MODE:")
        for exc in at.exception:
            print(exc)
        return False
    print("[PASS] Demo mode passed without exception.")
    return True

def test_organizer_mode():
    print("Testing Organizer mode...")
    at = AppTest.from_file("dashboard/app.py", default_timeout=60)
    at.run()
    # Switch to Organizer mode
    radio = at.sidebar.radio[0]
    radio.set_value("Organizer")
    at.run()
    if at.exception:
        print("EXCEPTION IN ORGANIZER MODE:")
        for exc in at.exception:
            print(exc)
        return False
    print("[PASS] Organizer mode passed without exception.")
    return True

def test_upload_mode():
    print("Testing Upload mode (empty state)...")
    at = AppTest.from_file("dashboard/app.py", default_timeout=40)
    at.run()
    # Switch to Upload mode
    radio = at.sidebar.radio[0]
    radio.set_value("Upload")
    at.run()
    if at.exception:
        print("EXCEPTION IN UPLOAD MODE:")
        for exc in at.exception:
            print(exc)
        return False
    print("[PASS] Upload mode passed without exception.")
    return True

if __name__ == "__main__":
    demo_ok = test_demo_mode()
    org_ok = test_organizer_mode()
    up_ok = test_upload_mode()

    if demo_ok and org_ok and up_ok:
        print("\nALL MODES PASSED WITH ZERO EXCEPTIONS!")
        sys.exit(0)
    else:
        print("\nSOME MODES FAILED!")
        sys.exit(1)
