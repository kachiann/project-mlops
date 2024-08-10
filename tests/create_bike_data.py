import base64
import json

from tests.utils import BIKE_DATA_TEMPLATE

# Convert the data to a JSON string
json_str = json.dumps(BIKE_DATA_TEMPLATE)

# Encode the JSON string to base64
base64_str = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")

# Write the base64 string to a file
with open("file_path", "r", encoding="utf-8") as f:
    f.write(base64_str)

print("Successfully created bike_data.b64")
