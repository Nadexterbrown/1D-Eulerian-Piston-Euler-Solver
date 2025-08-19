import difflib

__all__ = [
    'load_data_file', 'find_closest_columns',

]

########################################################################################################################
# Pele Data Handling Functions
########################################################################################################################

def load_data_file(file_path):
    import re
    import pandas as pd

    # Read the file, extracting the second row as column names
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract the header row (second line) and clean column names
    raw_headers = lines[1].strip()

    # Use regex to split headers by multiple spaces while keeping words together
    headers = re.split(r'\s{2,}', raw_headers)  # Split on 2+ spaces
    headers[0] = headers[0].lstrip('#')  # Remove the '#' from the first column name

    # Read the data, skipping the first two rows to exclude the initial comment and headers
    return pd.read_csv(file_path, delim_whitespace=True, skiprows=2, names=headers)


def find_closest_columns(df, search_string, num_matches=1):
    """
    Finds the closest matching column names in a DataFrame to a given search string.

    Parameters:
    - df: pandas DataFrame.
    - search_string: string to match against column headers.
    - num_matches: number of closest matches to return.

    Returns:
    - List of column names that best match the search string.
    """
    columns = df.columns
    matches = difflib.get_close_matches(search_string, columns, n=num_matches, cutoff=0.2)
    return matches