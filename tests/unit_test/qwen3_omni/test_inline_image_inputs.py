"""Candidate control-flow tests at processor boundary; no NPU/model dependency."""

import ast
import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[3] / "sglang_omni"


class Shell(SimpleNamespace):

    def __getattr__(self, key):
        return None

    def to_dict(self):
        return vars(self).copy()


def get_function(path, name, scope, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(
            (n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
        ).body
    node = next(
        (
            n
            for n in body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        )
    )
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), scope)
    return scope[name]


class BoundaryReached(Exception):
    pass


def image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def text(value):
    return {"type": "text", "text": value}


def run(inputs):
    scope = {"json": json, "asyncio": asyncio}
    pre = SRC / "models/qwen3_omni/components/preprocessor.py"
    get_function(pre, "extract_inline_image_messages", scope)
    get_function(pre, "is_pretokenized_prompt", scope)
    get_function(SRC / "preprocessing/text.py", "normalize_messages", scope)
    mm = get_function(pre, "build_multimodal_messages", scope, "Qwen3OmniPreprocessor")
    call = get_function(pre, "call_impl", scope, "Qwen3OmniPreprocessor")
    captured = {}

    async def images(raw, **kw):
        captured["loaded_urls"] = list(raw or [])
        # Fake media boundary, not an image-decoder validity test.
        return [object() for _ in (raw or [])]

    async def videos(*a, **kw):
        return [], None, []

    async def audios(*a, **kw):
        return []

    class Connection:
        async def close(self):
            pass

    class Processor:
        def apply_chat_template(self, messages, **kw):
            captured["messages"] = messages
            return json.dumps(messages)

        def __call__(self, **kw):
            captured["image_count"] = len(kw["images"] or [])
            captured["prompt"] = kw["text"]
            raise BoundaryReached()

    scope.update(
        ensure_image_list_async=images,
        ensure_video_list_async=videos,
        ensure_audio_list_async=audios,
        ResourceHTTPConnection=Connection,
        MultiModalResourceConnector=lambda **kw: None,
        compute_image_cache_key=lambda x: "key" if x else None,
        compute_video_cache_key=lambda x: None,
        compute_audio_cache_key=lambda x: None,
    )
    obj = Shell(processor=Processor())
    obj.build_multimodal_messages = lambda *a, **kw: mm(obj, *a, **kw)
    try:
        asyncio.run(call(obj, Shell(request=Shell(inputs=inputs))))
    except BoundaryReached:
        pass
    else:
        raise AssertionError("HF processor not reached")
    return captured


class Tests(unittest.TestCase):
    def test_missing_role_defaults_user_and_system_is_preserved(self):
        result = run(
            [
                {"role": "system", "content": "Be precise."},
                {"content": [image("A"), text("Describe.")]},
            ]
        )
        self.assertEqual(
            result["messages"][0], {"role": "system", "content": "Be precise."}
        )
        self.assertEqual(result["messages"][1]["role"], "user")
        self.assertEqual(result["image_count"], 1)

    def test_inline_data_uri_reaches_image_processor(self):
        req = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        text("Describe the image."),
                        image("data:image/png;base64,iVBORw0KGgo="),
                    ],
                }
            ]
        }
        result = run(req["messages"])
        self.assertEqual(result["image_count"], 1)
        self.assertNotIn("data:image/", result["prompt"])

    def test_multiple_images_keep_message_and_content_order(self):
        result = run(
            [
                {
                    "role": "user",
                    "content": [text("before"), image("A"), text("after")],
                },
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": [image("B"), text("question"), image("A")]},
            ]
        )
        self.assertEqual(result["loaded_urls"], ["A", "B", "A"])
        self.assertEqual(
            result["messages"][0]["content"],
            [text("before"), {"type": "image"}, text("after")],
        )
        self.assertEqual(
            result["messages"][2]["content"],
            [{"type": "image"}, text("question"), {"type": "image"}],
        )

    def test_top_level_legacy(self):
        result = run(
            {
                "messages": [{"role": "user", "content": "question"}],
                "images": ["A", "B"],
            }
        )
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["loaded_urls"], ["A", "B"])

    def test_plain_text(self):
        result = run([{"role": "user", "content": "hello"}])
        self.assertEqual(result["image_count"], 0)
        self.assertEqual(result["messages"], [{"role": "user", "content": "hello"}])

    def test_invalid_inline_reference_is_rejected(self):
        for item in ({"type": "image_url"}, image(""), image(123)):
            with self.subTest(item=item), self.assertRaises(ValueError):
                run([{"role": "user", "content": [item]}])

    def test_identical_dual_declaration_is_loaded_once(self):
        result = run(
            {
                "messages": [{"role": "user", "content": [image("A"), image("B")]}],
                "images": ["A", "B"],
            }
        )
        self.assertEqual(result["image_count"], 2)

    def test_ambiguous_dual_declaration_rejected(self):
        with self.assertRaises(ValueError):
            run(
                {
                    "messages": [{"role": "user", "content": [image("A"), image("B")]}],
                    "images": ["B", "A"],
                }
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
