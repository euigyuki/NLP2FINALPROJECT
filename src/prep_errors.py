"""Prepare test-set errors for hand-labeling.

Filters test_predictions.csv to errors only, adds an error_type column,
and writes a CSV ready for annotation.
"""

import pandas as pd

df = pd.read_csv("../outputs/test_predictions.csv")
errors = df[~df["correct"]].copy()
errors["error_type"] = errors["gold"] + "_predicted_as_" + errors["pred"]
errors["story_length"] = errors["story"].str.len()

# Sort by error type then story length (longer stories likely harder)
errors = errors.sort_values(["error_type", "story_length"], ascending=[True, False])
errors["taxonomy_category"] = ""  # for hand-labeling
errors["notes"] = ""               # for hand-labeling

# Print summary
print("Total errors:", len(errors))
print("\nError type distribution:")
print(errors["error_type"].value_counts())

# Save full errors
errors.to_csv("../outputs/test_errors.csv", index=False)
print(f"\nSaved {len(errors)} errors to ../outputs/test_errors.csv")

# Save a 100-error sample stratified by error type
sample = errors.groupby("error_type", group_keys=False).apply(
    lambda g: g.sample(min(len(g), max(5, int(100 * len(g) / len(errors)))), random_state=42)
)
sample.to_csv("../outputs/test_errors_sample100.csv", index=False)
print(f"Saved {len(sample)} stratified sample to ../outputs/test_errors_sample100.csv")
