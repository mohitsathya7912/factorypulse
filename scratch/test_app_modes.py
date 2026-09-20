import sys
from streamlit.testing.v1 import AppTest

def test_app():
    print("Testing Demo mode...")
    at = AppTest.from_file("dashboard/app.py", default_timeout=30)
    at.run()
    if at.exception:
        print("Exception in Demo mode:", at.exception)
        sys.exit(1)
    print("Demo mode passed with 0 exceptions!")

    print("Testing Organizer mode...")
    # Change radio to Organizer if present
    for r in at.sidebar.radio:
        if "Demo" in r.options and "Organizer" in r.options:
            r.set_value("Organizer")
            break
    at.run()
    if at.exception:
        print("Exception in Organizer mode:", at.exception)
        sys.exit(1)
    print("Organizer mode passed with 0 exceptions!")

    print("Testing Upload mode...")
    for r in at.sidebar.radio:
        if "Upload" in r.options:
            r.set_value("Upload")
            break
    at.run()
    if at.exception:
        print("Exception in Upload mode:", at.exception)
        sys.exit(1)
    print("Upload mode passed with 0 exceptions!")

if __name__ == "__main__":
    test_app()
