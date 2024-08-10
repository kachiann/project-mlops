import base64
import json

# Create the data you want to encode
data = {
    "ride": {
        "season": 1,
        "holiday": 0,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.3,
        "atemp": 0.3,
        "hum": 0.5,
        "windspeed": 0.2,
        "hr": 10,
        "mnth": 6,
        "yr": 1,
    }
}

# Convert the data to a JSON string
json_str = json.dumps(data)

# Encode the JSON string to base64
base64_str = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")

# Write the base64 string to a file
with open("bike_data.b64", "w") as f_out:
    f_out.write(base64_str)

print("Successfully created bike_data.b64")
