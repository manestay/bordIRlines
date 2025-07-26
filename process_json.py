import json
import uuid
import os

def process_json_files(base_directory, languages):
    # Directory where the JSON files are located
    directory = os.path.join(base_directory, "metadata")

    # Create a directory for each language within the base directory
    for lang in languages:
        lang_dir = os.path.join(base_directory, lang)
        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir)

    # Process each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            # Extract the language code from the filename (assuming format 'entity2ids.lang.json')
            file_lang = filename.split('.')[-2] if filename.count('.') >= 2 else 'unknown'
            if file_lang in languages:
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    updated_data = {}
                    
                    for entity, content in data.items():
                        if entity not in updated_data:
                            updated_data[entity] = {"urls": [], "countries": [], "snippets": [], "ids": []}
                            
                        for i, (url, country, snippet) in enumerate(zip(content['urls'], content['countries'], content['snippets'])):
                            # Create new JSON file in the corresponding language directory
                            new_id = uuid.uuid4().hex
                            new_filename = f"data_{new_id}.json"
                            new_filepath = os.path.join(base_directory, file_lang, new_filename)
                            new_data = {
                                "link": url,
                                "countries": country,
                                "content": snippet
                            }
                            with open(new_filepath, 'w') as new_file:
                                json.dump(new_data, new_file, indent=4)
                            
                            # Save ID back to the original data
                            updated_data[entity]["urls"].append(url)
                            updated_data[entity]["countries"].append(country)
                            updated_data[entity]["snippets"].append(snippet)
                            updated_data[entity]["ids"].append(new_id)
                            
                            print(f"File created: {new_filepath}")

                # Save updated data back to the original file
                with open(filepath, 'w') as file:
                    json.dump(updated_data, file, indent=4)

if __name__ == "__main__":
    base_directory_path = input("Enter the base directory path: ")
    languages = input("Enter the languages (comma-separated): ").split(',')
    process_json_files(base_directory_path, languages)
