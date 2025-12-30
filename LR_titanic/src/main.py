"""
main.py
-------
Command-line runner for the FULL Titanic ML pipeline.

MENU:
1. Run Preprocessing
2. Train Model
3. Evaluate Model
4. Run ALL steps (Preprocess → Train → Evaluate → Launch Streamlit App)  [RECOMMENDED]
5. Exit
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Paths to scripts
PREPROCESS_PATH = os.path.join(BASE_DIR, "src", "preprocessing.py")
TRAIN_PATH = os.path.join(BASE_DIR, "src", "train_model.py")
EVAL_PATH = os.path.join(BASE_DIR, "src", "evaluate.py")

# Streamlit App path
STREAMLIT_APP_PATH = os.path.join(BASE_DIR, "streamlit_app", "app.py")


def run_script(script_path):
    """Run a python script and handle errors."""
    print(f"\n[RUNNING] {script_path}\n")

    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print("[ERROR] Script failed!")
        print(e)
        sys.exit(1)


def run_streamlit_app():
    """Runs Streamlit app WITHOUT blocking the menu."""
    print("\n[INFO] Launching Streamlit App...")
    print(f"[COMMAND] streamlit run {STREAMLIT_APP_PATH}\n")

    # Start Streamlit server in a separate non-blocking process
    process = subprocess.Popen(
        ["streamlit", "run", STREAMLIT_APP_PATH],
        shell=True
    )

    print("[INFO] Streamlit is now running.")
    print("[NOTE] Close the Streamlit BROWSER TAB when done.")
    print("[IMPORTANT] Streamlit continues running in BACKGROUND.")
    print("           To STOP it completely, go to the Streamlit terminal and press CTRL+C.\n")

    return process



if __name__ == "__main__":

    while True:
        print("\n===============================")
        print("   TITANIC ML PROJECT RUNNER")
        print("===============================")
        print("1. Run Preprocessing")
        print("2. Train Model")
        print("3. Evaluate Model")
        print("4. Run ALL steps (Preprocess → Train → Evaluate → Launch Streamlit App)  [RECOMMENDED]")
        print("5. Exit")
        print("===============================\n")

        choice = input("Choose an option (1-5): ").strip()

        if choice == "1":
            run_script(PREPROCESS_PATH)

        elif choice == "2":
            run_script(TRAIN_PATH)

        elif choice == "3":
            run_script(EVAL_PATH)

        elif choice == "4":
            print("\n[INFO] Running FULL Pipeline...")
            run_script(PREPROCESS_PATH)
            run_script(TRAIN_PATH)
            run_script(EVAL_PATH)

            # Launch Streamlit and do NOT return to menu
            streamlit_process = run_streamlit_app()

            # Do not return to menu; stop main.py execution here
            print("\n[INFO] main.py will now exit while Streamlit continues running.")
            sys.exit(0)

        elif choice == "5":
            print("Exiting...")
            sys.exit(0)

        else:
            print("Invalid option. Please choose 1-5.")
