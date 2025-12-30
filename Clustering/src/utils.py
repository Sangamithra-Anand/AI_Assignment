import os

def create_output_folders():
    """
    Creates all necessary folders for storing project outputs.

    Folders created:
        output/
        output/plots/
        output/labels/
        output/reports/

    Called at the start of main.py.
    """
    print("[INFO] Creating required output folders...")

    folders = [
        "output",
        "output/plots",
        "output/labels",
        "output/reports"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("[INFO] Output folders ready.")


def ensure_directory(path):
    """
    Ensures that the directory of a file path exists.

    Example:
        ensure_directory("output/reports/clustering_report.txt")
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_plot(fig, path):
    """
    Saves a matplotlib figure safely.

    Args:
        fig  -> matplotlib figure object
        path -> output file path
    """
    ensure_directory(path)
    fig.savefig(path)
    print(f"[INFO] Plot saved: {path}")
    fig.close()


def print_section(title):
    """
    Prints a visual section divider in console output.
    Useful for clean logs.

    Example:
        print_section("PREPROCESSING STARTED")
    """
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60 + "\n")
