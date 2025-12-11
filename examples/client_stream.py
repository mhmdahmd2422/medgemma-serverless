# Example client using runpod Endpoint.stream to receive token-by-token output.
import runpod
import os

runpod.api_key = os.environ.get("RUNPOD_API_KEY", "")
endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "<YOUR_ENDPOINT_ID>")
ep = runpod.Endpoint(endpoint_id)

def run():
    for chunk in ep.stream({"prompt": "Describe this medical image.", "image": None}):
        print(chunk, end="", flush=True)

if __name__ == '__main__':
    run()
