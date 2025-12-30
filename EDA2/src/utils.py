import os

# -------------------------------------------------------------
# Function: ensure_directory
# Purpose : Safely create a folder if it does not exist.
#           This avoids errors when saving files.
# -------------------------------------------------------------
def ensure_directory(path):
    """
    Creates a directory if it does not exist.

    Example:
        ensure_directory("data/reports")
    """

    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Directory ensured: {path}")


# -------------------------------------------------------------
# Function: save_text_report
# Purpose : Save text-based reports such as EDA summary, notes, or logs.
# -------------------------------------------------------------
def save_text_report(text, save_path):
    """
    Saves a text report to the given file.

    Parameters:
        text (str): The report content
        save_path (str): File path for saving the report
    """

    # Create directory if needed
    folder = os.path.dirname(save_path)
    ensure_directory(folder)

    # Write content to file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[INFO] Report saved to: {save_path}")


# -------------------------------------------------------------
# Function: print_section
# Purpose : Display clean section headers in the terminal output.
#           Makes logs easier to read while running scripts.
# -------------------------------------------------------------
def print_section(title):
    """
    Prints a readable section divider in console logs.

    Example:
        print_section("DATA PREPROCESSING")
    """

    print("\n" + "="*60)
    print(f"{title}")
    print("="*60 + "\n")


# -------------------------------------------------------------
# Function: get_numeric_and_categorical_columns
# Purpose : A utility used by multiple files to quickly identify
#           numeric and categorical columns in the dataset.
# -------------------------------------------------------------
def get_numeric_and_categorical_columns(df):
    """
    Returns:
        numeric_cols (list)
        categorical_cols (list)
    """

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    return numeric_cols, categorical_cols


