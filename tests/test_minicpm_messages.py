"""Message conversion for the MiniCPM vision bridge. No weights are loaded."""

import asyncio
import json
import pathlib
import tempfile
import unittest

from endpoints.OAI.types.chat_completion import (
    ChatCompletionImageUrl,
    ChatCompletionMessage,
    ChatCompletionMessagePart,
)
from backends.exllamav3.minicpm import (
    ARCHITECTURE,
    messages_have_image,
    official_messages,
    read_architecture,
)


class MiniCPMMessagesTest(unittest.TestCase):
    def test_read_architecture_ignores_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            (directory / "config.json").write_text(
                json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}),
                encoding="utf-8",
            )
            self.assertNotEqual(read_architecture(directory), ARCHITECTURE)

    def test_local_image_becomes_an_official_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = pathlib.Path(tmp) / "pixel.png"
            image.write_bytes(b"not-decoded")
            messages = [
                ChatCompletionMessage(
                    role="user",
                    content=[
                        ChatCompletionMessagePart(type="text", text="What is this?"),
                        ChatCompletionMessagePart(
                            type="image_url",
                            image_url=ChatCompletionImageUrl(url=str(image)),
                        ),
                    ],
                )
            ]
            self.assertTrue(messages_have_image(messages))
            official = asyncio.run(official_messages(messages, pathlib.Path(tmp)))

        self.assertEqual(official[0]["role"], "user")
        self.assertEqual(official[0]["content"][0], {"type": "text", "text": "What is this?"})
        self.assertEqual(official[0]["content"][1]["type"], "image")
        self.assertTrue(official[0]["content"][1]["url"].endswith("pixel.png"))


if __name__ == "__main__":
    unittest.main()
