"""Streaming sample: ensure <think> JSON does not trigger tool calls."""
from stream_handler import StreamHandler


def run_sample():
    handler = StreamHandler(model="sample-model")
    chunks = [
        "<think>Example JSON: {\"action\":\"current_time\",\"action_input\":{}}</think>",
        "\n```json\n{\"action\":\"current_time\",\"action_input\":{}}\n```\n"
    ]

    for chunk in chunks:
        for out in handler.process_chunk(chunk):
            print(out)

    for out in handler.finalize():
        print(out)


if __name__ == "__main__":
    run_sample()
