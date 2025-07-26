import json


def load_json(file_path):
    """Load JSON file and return its content."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def compare_dicts(dict1, dict2, path=""):
    """Recursively compare two dictionaries and return differences."""
    differences = []

    # Check keys in dict1
    for key in dict1:
        if key not in dict2:
            differences.append(f"Key '{path + key}' is missing in second dictionary.")
        else:
            # If the value is another dictionary, recurse
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                differences.extend(compare_dicts(dict1[key], dict2[key], path + key + "."))
            else:
                val1 = dict1[key]
                val2 = dict2[key]
                if isinstance(val1, list) and isinstance(val2, list):
                    # Sort the lists before comparing
                    val1 = sorted(val1)
                    val2 = sorted(val2)
                vals_equal = val1 == val2
                if not vals_equal:
                    differences.append(
                        f"Value for key '{path + key}' differs: {dict1[key]} != {dict2[key]}"
                    )

    # Check keys in dict2 that are not present in dict1
    for key in dict2:
        if key not in dict1:
            differences.append(f"Key '{path + key}' is missing in first dictionary.")

    return differences


def compare_json_files(file1_path, file2_path):
    """Load two JSON files and compare them."""
    # Load the contents of the JSON files
    dict1 = load_json(file1_path)
    dict2 = load_json(file2_path)

    # Compare dictionaries
    differences = compare_dicts(dict1, dict2)

    if differences:
        print("Differences found between the files:")
        for diff in differences:
            print(diff)
    else:
        print("The files are identical.")


# Paths to the JSON files
json_file1_path = "generation/gen_results/queries.json"
json_file2_path = "generation/gen_results/queries_old.json"

# Compare the JSON files
compare_json_files(json_file1_path, json_file2_path)
