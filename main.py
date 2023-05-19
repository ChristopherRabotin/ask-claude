import anthropic
import argparse
import whisper
from os import environ

def transcribe(audio: str):
    """
    Only perform the transcription and save to a file
    """

    print("Describing audio...")
    model = whisper.load_model("base")
    result = model.transcribe(audio)
    print("Done!")

    with open(f"{audio}_transcript.txt", "w") as f:
        f.write(result["text"])

    return result["text"]

def main(max_tokens_to_sample: int = 100_000):
    """
    Given an audio file, transcribe it with whisper then provide it to Claude to ask it to write a summary.
    """

    parser = argparse.ArgumentParser(description='Summarize an SRT file.')
    parser.add_argument('--audio', type=str)
    parser.add_argument('--transcript', type=str)
    parser.add_argument('--ask', action='store_true')
    args = parser.parse_args()

    path = None
    if args.audio:
        transcription = transcribe(args.audio)
        path = args.audio
    elif args.transcript:
        with open(args.transcript, "r") as f:
            transcription = f.read()
        path = args.transcript
    else:
        raise ValueError("Either audio or transcript must be provided.")
    
    # Query Claude
    prompt = """The following is a conversation between several experts.
    I would like you to summarize their conversation. The summary shall start with an executive summary of no more than 1000 words.
    Then, create sections per topic discussed.
    Each section should be a markdown header, followed by a paragraph. Make sure to add a section for anecdotes at the end.
    Don't forget anything from the conversation, but don't add anything that wasn't discussed.
    Please answer only with the summary in markdown format without any introduction to the answer.
    Here is the full transcript of the conversation:\n\n"""

    claude = anthropic.Client(api_key=environ["ANTHROPIC_API_KEY"])

    print("Querying Claude...")

    prompt = f"{anthropic.HUMAN_PROMPT} {prompt} {transcription} {anthropic.AI_PROMPT}"

    resp = claude.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1-100k",
        max_tokens_to_sample=max_tokens_to_sample,
    )
    print(resp)

    with open(f"{path}_summary.md", "w") as f:
        f.write(resp["completion"])
    
    if args.ask:
        # Continue the conversation
        while True:
            prompt += resp["completion"] # Adds the summary on the first run.
            question = input("Question > ")
            prompt += f"{anthropic.HUMAN_PROMPT} {question} {anthropic.AI_PROMPT}"
            resp = claude.completion(
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-v1-100k",
                max_tokens_to_sample=max_tokens_to_sample,
            )
            print(resp["completion"])

if __name__ == "__main__":
    main()