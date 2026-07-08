"""Add a language without touching the library (no API key).

A language is one self-contained ``<code>.toml`` pack. In your own project you
would add ``ragas_lingua/prompts/<code>.toml``, or point ``register_prompts_dir``
at a directory of your own packs. This example writes a tiny pack to a temp dir
and registers it.

    python examples/05_add_a_language.py
"""

import tempfile
from pathlib import Path

from ragas_lingua import available_languages, get_profile, register_prompts_dir

PACK = """\
code = "xx"
name = "Example"
english_name = "Example"
reviewed = false
keep_language = "Write everything in the Example language, the same language as the answer."

[faithfulness]
extract_instruction = "Split the answer into simple, standalone statements."
verdict_instruction = "Mark each statement supported=true only if the context supports it directly."
question_label = "QUESTION"
answer_label = "ANSWER"
context_label = "CONTEXT"
statements_label = "STATEMENTS"
"""


def main() -> None:
    directory = Path(tempfile.mkdtemp())
    (directory / "xx.toml").write_text(PACK, encoding="utf-8")

    register_prompts_dir(directory)  # your packs take precedence over the built-ins

    print("available languages:", available_languages())
    profile = get_profile("xx")
    print(f"loaded pack: {profile.code} ({profile.english_name}), reviewed={profile.reviewed}")


if __name__ == "__main__":
    main()
