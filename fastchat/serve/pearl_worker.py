import argparse
import asyncio
import json
import multiprocessing
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from fastchat.conversation import IMAGE_PLACEHOLDER_STR
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length, is_partial_stop
from transformers import AutoTokenizer, AutoConfig
try:
    from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams
except ImportError:
    logger.error("Warning: nano_pearl not found.")
    PEARLConfig = PEARLEngine = SamplingParams = None
import traceback
import time
from collections import deque

BATCH_MAX_SIZE = 32
BATCH_MAX_WAIT = 1.0  # seconds

_batch_buffer = deque()
_batch_lock = asyncio.Lock()
_batch_start_ts = None

_inflight_sem = asyncio.Semaphore(BATCH_MAX_SIZE)

class BatchItem:
    def __init__(self, prompt, sampling_params):
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.future = asyncio.get_running_loop().create_future()

def _run_batch_generate(batch_items):
    for item in batch_items:
        worker.engine.add_request(item.prompt, item.sampling_params)

    text_list, num_tokens_list, _, _ = worker.engine.generate()

    results = []
    for text, tokens in zip(text_list, num_tokens_list):
        results.append((text, tokens))
    return results

app = FastAPI()


class PearlWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        tokenizer_path: str,
        draft_model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: str,
        trust_remote_code: bool,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: pearl worker..."
        )

        config = PEARLConfig(
            draft_model_path=draft_model_path,
            target_model_path=model_path,
            draft_tensor_parallel_size=1,
            target_tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        self.engine = PEARLEngine(config)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        config = AutoConfig.from_pretrained(model_path)
        self.context_len = get_context_length(config)

        if not no_register:
            self.init_heart_beat()


    async def generate_stream(self, params):
        self.call_ct += 1
        await _inflight_sem.acquire()
        try:
            # 🔑 heartbeat
            yield {
                "text": "",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "error_code": 0,
            }

            prompt = params.pop("prompt")
            if isinstance(prompt, list):
                prompt = "\n".join(
                    p.get("content", "") if isinstance(p, dict) else str(p)
                    for p in prompt
                )

            sampling_params = SamplingParams(
                temperature=float(params.get("temperature", 1.0)),
                max_tokens=params.get("max_new_tokens", 256),
                ignore_eos=False,
            )

            item = BatchItem(prompt, sampling_params)


            global _batch_start_ts

            async with _batch_lock:
                _batch_buffer.append(item)

                if len(_batch_buffer) == 1:
                    # 👑 我是 leader
                    is_leader = True
                    _batch_start_ts = time.time()
                else:
                    # 🙋 我是 follower
                    is_leader = False

            if is_leader:
                while True:
                    async with _batch_lock:
                        elapsed = time.time() - _batch_start_ts
                        if (
                            len(_batch_buffer) >= BATCH_MAX_SIZE
                            or elapsed >= BATCH_MAX_WAIT
                        ):
                            batch_items = list(_batch_buffer)
                            _batch_buffer.clear()
                            _batch_start_ts = None
                            break
                    await asyncio.sleep(0.01)

                # 真正的 batch generate（只會跑一次）
                results = await asyncio.to_thread(
                    _run_batch_generate, batch_items
                )

                # 發結果給所有人
                for bi, (text, tokens) in zip(batch_items, results):
                    if not bi.future.done():
                        bi.future.set_result((text, tokens))

            text, num_tokens = await item.future

            # fake streaming
            entire_output = ""
            for i in range(0, len(text), 20):
                entire_output += text[i:i+20]
                yield {
                    "text": entire_output,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": num_tokens,
                        "total_tokens": num_tokens,
                    },
                    "error_code": 0,
                }
                await asyncio.sleep(0)

        finally:
            _inflight_sem.release()


    async def generate_stream_gate(self, params):
        try:
            async for ret in self.generate_stream(params):
                yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("🔥 generate_stream_gate exception:\n%s", tb)

            ret = {
                "text": f"generate_stream_gate\n\n{tb}",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    async def generate_gate(self, params):
        x = None
        async for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = await worker.generate_gate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--draft-model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to"
        "reserve for the model weights, activations, and KV cache. Higher"
        "values will increase the KV cache size and thus improve the model's"
        "throughput. However, if the value is too high, it may cause out-of-"
        "memory (OOM) errors.",
    )

    args = parser.parse_args()

    args.tp_size = args.num_gpus if args.num_gpus > 1 else 1
    args.tokenizer_path = (
        args.model_path if args.tokenizer_path == "" else args.tokenizer_path
    )

    worker = PearlWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.tokenizer_path,
        args.draft_model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        args.trust_remote_code,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")