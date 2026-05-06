import sys
import math
import numpy as np

def parse_numeric_arrays(file_path: str):
    """
    Read the file line by line, parse out numeric values (int or float),
    and store them in a 2D numpy array.
    Non-numeric tokens are skipped.

    Behavior:
    - Each line is split by whitespace.
    - Try to parse each token as int, if fail then as float, otherwise skip.
    - The resulting numeric values in each line are stored in a row of a list.
      We then create a 2D array of shape (num_lines, max_cols) filling missing slots with np.nan.
    """
    all_rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            row_values = []
            for token in tokens:
                # Try int
                try:
                    val_int = int(token)
                    row_values.append(val_int)
                    continue
                except ValueError:
                    pass
                # Try float
                try:
                    val_float = float(token)
                    row_values.append(val_float)
                    continue
                except ValueError:
                    pass
                # else skip non-numeric
            all_rows.append(row_values)

    # Create a 2D numpy array with np.nan as filler
    if not all_rows:
        return np.array([], dtype=float).reshape(0, 0)

    max_cols = max(len(r) for r in all_rows)
    arr = np.full((len(all_rows), max_cols), np.nan, dtype=float)

    for i, row in enumerate(all_rows):
        for j, val in enumerate(row):
            arr[i, j] = float(val)

    return arr


def compare_files_with_numpy(baseline_file: str, compare_file: str, tolerance: float):
    """
    Use numpy arrays to compare numeric data from baseline_file and compare_file.
    Steps:
    1) parse_numeric_arrays -> get 2D float arrays for both.
    2) If shapes differ, compare only the overlapping region (min rows x min cols)
       or optionally raise an error.
    3) For each cell:
       - If both are ints (no fractional part), must match exactly.
       - Otherwise, check np.isclose(...) with the given tolerance.
       - If any mismatch found, return fail.

    Returns (ok: bool, msg: str).
    """
    base_arr = parse_numeric_arrays(baseline_file)
    comp_arr = parse_numeric_arrays(compare_file)

    # Minimum overlap
    rows = min(base_arr.shape[0], comp_arr.shape[0])
    cols = min(base_arr.shape[1], comp_arr.shape[1])

    base_sub = base_arr[:rows, :cols]
    comp_sub = comp_arr[:rows, :cols]

    for i in range(rows):
        for j in range(cols):
            val_b = base_sub[i, j]
            val_c = comp_sub[i, j]
            # Both nan => skip
            if np.isnan(val_b) and np.isnan(val_c):
                continue

            # Distinguish int vs float.
            is_b_int = (not np.isnan(val_b)) and float(val_b).is_integer()
            is_c_int = (not np.isnan(val_c)) and float(val_c).is_integer()

            if is_b_int and is_c_int:
                # Both ints => must match exactly.
                if val_b != val_c:
                    return False, (f"Mismatch at row={i+1}, col={j+1}: int {val_b} != {val_c}")
            else:
                # Float comparison.
                if not np.isclose(val_b, val_c, atol=tolerance, rtol=0.0):
                    diff = abs(val_b - val_c)
                    return False, (f"Float mismatch at row={i+1}, col={j+1}: "
                                   f"{val_b} vs {val_c}, diff={diff} > tol={tolerance}")

    return True, "All numeric tokens match within tolerance."


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compare_with_tolerance_numpy.py <baseline_file> <compare_file> <tolerance>")
        sys.exit(1)

    baseline_path = sys.argv[1]
    compare_path = sys.argv[2]
    tol = float(sys.argv[3])

    ok, msg = compare_files_with_numpy(baseline_path, compare_path, tol)
    if ok:
        print(f"[INFO: PASS] {msg}")
        sys.exit(0)
    else:
        print(f"[INFO: FAIL] {msg}")
        sys.exit(1)
