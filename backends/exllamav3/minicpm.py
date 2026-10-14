"""MiniCPM-V 4.7 load hook for the ExLlamaV3 backend.

ExLlamaV3 does not ship this architecture. The checkpoint's adapter registers
it, runs the language stack as Qwen3.5-MoE, and keeps the vision tower in
Transformers. TabbyAPI has to call that registration before Config.from_directory.
"""

import json
import tempfile
from pathlib import Path

from common.image_util import get_image
from common.logger import xlogger

ARCHITECTURE = "MiniCPMV4_7ForConditionalGeneration"


class MiniCPMRequestError(Exception):
    """The chat request cannot be packed by the MiniCPM vision bridge."""


def read_architecture(directory) -> str | None:
    path = Path(directory) / "config.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        architectures = json.load(handle).get("architectures") or []
    if not architectures or not isinstance(architectures[0], str):
        return None
    return architectures[0]


def register_minicpm(directory) -> bool:
    """Register the adapter architecture. False for every other checkpoint."""

    if read_architecture(directory) != ARCHITECTURE:
        return False
    try:
        from minicpm47_exl3.exl3 import register
    except ImportError as exc:
        raise RuntimeError(
            "This checkpoint needs the minicpm47-exl3 adapter in the TabbyAPI "
            "environment. Install models/minicpm-4.7-35b-a3b/adapter/dist/"
            "minicpm47_exl3-0.1.3-py3-none-any.whl into tabbyAPI/.venv."
        ) from exc
    register()
    return True


def messages_have_image(messages) -> bool:
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, list) and any(
            getattr(part, "type", None) == "image_url" for part in content
        ):
            return True
    return False


async def _materialize_image(url: str, directory: Path, index: int) -> str:
    if url.startswith("file://"):
        url = url[7:]
    path = Path(url)
    if path.is_file():
        return str(path.resolve())
    image = await get_image(url)
    dest = directory / f"image-{index}.png"
    image.convert("RGB").save(dest)
    return str(dest)


async def official_messages(messages, directory: Path) -> list[dict]:
    """OpenAI chat messages to the processor's message list."""

    official = []
    image_index = 0
    for message in messages:
        item = {"role": message.role}
        if message.reasoning_content:
            item["reasoning_content"] = message.reasoning_content
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            calls = []
            for call in message.tool_calls:
                dumped = call.model_dump(exclude_none=True)
                function = dumped.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except (json.JSONDecodeError, ValueError):
                        pass
                calls.append(dumped)
            item["tool_calls"] = calls

        content = message.content
        if isinstance(content, str) or content is None:
            item["content"] = content or ""
            official.append(item)
            continue

        parts = []
        for part in content:
            if part.type == "text":
                parts.append({"type": "text", "text": part.text or ""})
            elif part.type == "image_url":
                url = part.image_url.url if part.image_url else ""
                if not url:
                    raise MiniCPMRequestError("image_url is missing a url")
                parts.append(
                    {
                        "type": "image",
                        "url": await _materialize_image(url, directory, image_index),
                    }
                )
                image_index += 1
            else:
                raise MiniCPMRequestError(
                    f"MiniCPM vision does not accept content type {part.type!r}"
                )
        item["content"] = parts
        official.append(item)
    return official


class MiniCPMVision:
    """Official processor plus the adapter's BF16 vision tower."""

    def __init__(self, directory: str, device: str = "cuda:0"):
        from minicpm47_exl3.bridge import VisionBridge
        from transformers import AutoProcessor

        self.directory = directory
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        self.bridge = VisionBridge.from_checkpoint(directory, device=device)

    def close(self):
        bridge = getattr(self, "bridge", None)
        self.bridge = None
        self.processor = None
        if bridge is not None:
            bridge.to("cpu")

    async def prompt(self, messages, template_vars: dict) -> tuple[str, list]:
        """Pack one request into the alias string and a single MMEmbedding.

        The canvas position check compares the encoded prompt with this
        embedding, so the alias must be the entire prompt and BOS must not be
        prepended.
        """

        if self.bridge is None or self.processor is None:
            raise MiniCPMRequestError("MiniCPM vision tower is not loaded")

        mode = template_vars.get("downsample_mode") or "16x"
        max_slices = int(template_vars.get("max_slice_nums") or 1)
        max_frames = int(template_vars.get("max_num_frames") or 3)
        thinking = template_vars.get("enable_thinking", True)
        thinking = True if thinking is None else bool(thinking)
        tools = template_vars.get("tools") or None

        with tempfile.TemporaryDirectory(prefix="minicpm-") as tmp:
            official = await official_messages(messages, Path(tmp))
            processed = self.processor.apply_chat_template(
                official,
                tokenize=True,
                add_generation_prompt=bool(template_vars.get("add_generation_prompt", True)),
                return_dict=True,
                return_tensors="pt",
                downsample_mode=mode,
                max_slice_nums=max_slices,
                max_num_frames=max_frames,
                stack_frames=1,
                enable_thinking=thinking,
                tools=tools,
            )
        processed["downsample_mode"] = mode
        _ids, embeddings = self.bridge.pack(processed)
        if len(embeddings) != 1:
            raise MiniCPMRequestError(
                "MiniCPM vision bridge did not return one prompt embedding"
            )
        embedding = embeddings[0]
        xlogger.debug(
            "Packed MiniCPM vision prompt",
            {
                "tokens": int(embedding.token_string.shape[-1]),
                "visual_tokens": int(embedding.mm_length),
                "downsample_mode": mode,
                "max_slice_nums": max_slices,
            },
        )
        return embedding.text_alias, [embedding]
