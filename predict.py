import json
import time

import mlflow
from mlflow.models import validate_serving_input


def main(model_uri: str):
    # The model is logged with an input example. MLflow converts
    # it into the serving payload format for the deployed model endpoint,
    # and saves it to 'serving_input_payload.json'
    serving_payload = """{
    "messages": [
      {
        "role": "user",
        "content": "What's the weather like in San Francisco and New York?"
      }
    ],
    "temperature": 1.0,
    "n": 1,
    "stream": false,
    "tools": [
      {
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather",
          "parameters": {
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and country, eg. San Francisco, USA"
              },
              "format": {
                "type": "string",
                "enum": [
                  "celsius",
                  "fahrenheit"
                ]
              }
            },
            "type": "object",
            "required": [
              "location",
              "format"
            ]
          },
          "strict": false
        },
        "type": "function"
      }
    ]
  }"""

    # Validate the serving payload works on the model
    output = validate_serving_input(model_uri, serving_payload)

    print("\nModel input validation passed")
    print("output:")
    print(output)
    print("\n")

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    print("\nModel loaded")

    # TODO: get user input y/n to continue
    input("\nPress Enter to continue...")

    start_time = time.time()

    result = loaded_model.predict(
        data=json.loads(serving_payload),
    )

    end_time = time.time()

    print(f"\nModel prediction completed in {end_time - start_time:.2f} seconds")

    print("\nModel prediction result:")
    print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict using the model")
    parser.add_argument(
        "--model-uri",
        type=str,
        help="The model URI",
        required=True,
    )

    args = parser.parse_args()

    main(args.model_uri)
