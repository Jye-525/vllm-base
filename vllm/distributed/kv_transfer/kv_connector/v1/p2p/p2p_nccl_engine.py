# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import math
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import msgpack
import torch
import zmq

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_timing import (
    CudaEventSpan,
    P2pRequestTimingTracker,
    parse_timing_options,
    timing_log_record,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (  # noqa: E501
    TensorMemoryPool,
)
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import current_stream

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32


@contextmanager
def set_p2p_nccl_context(num_channels: str):
    original_values: dict[str, Any] = {}
    env_vars = [
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CUMEM_ENABLE",
        "NCCL_BUFFSIZE",
        "NCCL_PROTO",  # LL,LL128,SIMPLE
        "NCCL_ALGO",  # RING,TREE
    ]

    for var in env_vars:
        original_values[var] = os.environ.get(var)

    logger.info("set_p2p_nccl_context, original_values: %s", original_values)

    try:
        os.environ["NCCL_MAX_NCHANNELS"] = num_channels
        os.environ["NCCL_MIN_NCHANNELS"] = num_channels
        os.environ["NCCL_CUMEM_ENABLE"] = "1"
        yield
    finally:
        for var in env_vars:
            if original_values[var] is not None:
                os.environ[var] = original_values[var]
            else:
                os.environ.pop(var, None)


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor


class P2pNcclEngine:
    def __init__(
        self,
        local_rank: int,
        config: KVTransferConfig,
        hostname: str = "",
        port_offset: int = 0,
        library_path: str | None = None,
        tp_rank: int | None = None,
        tp_size: int = 1,
    ) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        self.send_type = self.config.get_from_extra_config("send_type", "PUT_ASYNC")
        self.enable_timing, self.timing_detail = parse_timing_options(self.config)

        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        # Each card corresponds to a ZMQ address.
        self.zmq_address = f"{self._hostname}:{self._port}"

        # If `proxy_ip` or `proxy_port` is `""`,
        # then the ping thread will not be enabled.
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
            self.http_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port
            # the `http_port` must be consistent with the port of OpenAI.
            http_port = self.config.get_from_extra_config("http_port", None)
            if http_port is None:
                example_cfg = {
                    "kv_connector": "P2pNcclConnector",
                    "kv_connector_extra_config": {"http_port": 8000},
                }
                example = (
                    f"--port=8000 --kv-transfer-config='{json.dumps(example_cfg)}'"
                )
                raise ValueError(
                    "kv_connector_extra_config.http_port is required. "
                    f"Example: {example}"
                )
            self.http_address = f"{self._hostname}:{http_port}"

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        if self.enable_timing:
            self._timing_anchor = torch.cuda.Event(enable_timing=True)
            self._timing_anchor.record(torch.cuda.current_stream(self.device))
            resolved_tp_rank = self.rank if tp_rank is None else int(tp_rank)
            role = "producer" if self.config.is_kv_producer else "consumer"
            self._timing_tracker = P2pRequestTimingTracker(
                side="send" if role == "producer" else "recv",
                detail=self.timing_detail,  # type: ignore[arg-type]
                anchor=self._timing_anchor,
                common={
                    "schema_version": 1,
                    "connector": "P2pNcclConnector",
                    "role": role,
                    "send_type": self.send_type,
                    "tp_rank": resolved_tp_rank,
                    "tp_size": int(tp_size),
                    "local_rank": self.local_rank,
                    "host": self._hostname,
                    "device": str(self.device),
                },
                emit=lambda record: timing_log_record(logger, record),
            )

        mem_pool_size_gb = float(
            self.config.get_from_extra_config(
                "mem_pool_size_gb", DEFAULT_MEM_POOL_SIZE_GB
            )
        )
        self.pool = TensorMemoryPool(
            max_block_size=int(mem_pool_size_gb * 1024**3)
        )  # GB

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC.
        if self.send_type == "GET":
            # tensor_id: torch.Tensor
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            # PUT or PUT_ASYNC
            # tensor_id: torch.Tensor
            self.send_queue: deque[SendQueueItem] = deque()
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(
                    target=self.send_async, daemon=True
                )
                self._send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.comms: dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)

        self.nccl_num_channels = self.config.get_from_extra_config(
            "nccl_num_channels", "8"
        )

        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True
        )
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()

        logger.info(
            "💯P2pNcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, nccl_num_channels:%s",
            self.rank,
            self.local_rank,
            self.http_address,
            self.zmq_address,
            self.proxy_address,
            self.send_type,
            self.buffer_size_threshold,
            self.nccl_num_channels,
        )

    def create_connect(self, remote_address: str | None = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                logger.info(
                    "👋comm exists, remote_address:%s, comms:%s",
                    remote_address,
                    self.comms,
                )
                return sock, self.comms[remote_address]

            unique_id = self.nccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            with torch.accelerator.device_index(self.device.index):
                rank = 0
                with set_p2p_nccl_context(self.nccl_num_channels):
                    comm: ncclComm_t = self.nccl.ncclCommInitRank(2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info(
                    "🤝ncclCommInitRank Success, %s👉%s, MyRank:%s",
                    self.zmq_address,
                    remote_address,
                    rank,
                )

        return self.socks[remote_address], self.comms[remote_address]

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str | None = None,
        batch_id: str | None = None,
    ) -> bool:
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True

        item = SendQueueItem(
            tensor_id=tensor_id,
            remote_address=remote_address,
            tensor=tensor,
        )

        if self.enable_timing:
            queued_perf_ns = time.perf_counter_ns()
            queued_wall_ns = time.time_ns()
            self._safe_timing_call(
                self._timing_tracker.enqueue,
                tensor_id=tensor_id,
                payload_bytes=tensor.element_size() * tensor.numel(),
                batch_id=batch_id,
                queued_perf_ns=queued_perf_ns,
                queued_wall_ns=queued_wall_ns,
            )

        if self.send_type == "PUT":
            return self.send_sync(item)

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                self.send_queue_cv.notify()
            return True

        # GET
        with self.send_store_cv:
            tensor_size = tensor.element_size() * tensor.numel()
            if tensor_size > self.buffer_size_threshold:
                logger.warning(
                    "❗[GET]tensor_id:%s, tensor_size:%d, is greater than"
                    "buffer size threshold :%d, skip send to %s, rank:%d",
                    tensor_id,
                    tensor_size,
                    self.buffer_size_threshold,
                    remote_address,
                    self.rank,
                )
                return False
            while self.buffer_size + tensor_size > self.buffer_size_threshold:
                assert len(self.send_store) > 0
                oldest_tensor_id = next(iter(self.send_store))
                oldest_tensor = self.send_store.pop(oldest_tensor_id)
                oldest_tensor_size = (
                    oldest_tensor.element_size() * oldest_tensor.numel()
                )
                self.buffer_size -= oldest_tensor_size
                logger.debug(
                    "⛔[GET]Send to %s, tensor_id:%s, tensor_size:%d,"
                    " buffer_size:%d, oldest_tensor_size:%d, rank:%d",
                    remote_address,
                    tensor_id,
                    tensor_size,
                    self.buffer_size,
                    oldest_tensor_size,
                    self.rank,
                )

            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
            logger.debug(
                "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                "shape:%s, rank:%d, buffer_size:%d(%.2f%%)",
                remote_address,
                tensor_id,
                tensor_size,
                tensor.shape,
                self.rank,
                self.buffer_size,
                self.buffer_size / self.buffer_size_threshold * 100,
            )
        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: str | None = None,
    ) -> torch.Tensor:
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    self.recv_store_cv.wait()
                tensor = self.recv_store[tensor_id]

            if tensor is not None:
                if isinstance(tensor, tuple):
                    addr, dtype, shape = tensor
                    tensor = self.pool.load_tensor(addr, dtype, shape, self.device)
                else:
                    self.buffer_size -= tensor.element_size() * tensor.numel()
            else:
                duration = time.time() - start_time
                logger.warning(
                    "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, rank:%d",
                    remote_address,
                    tensor_id,
                    duration * 1000,
                    self.rank,
                )
            return tensor

        # GET
        if remote_address is None:
            return None

        if remote_address not in self.socks:
            self.create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]

        data = {"cmd": "GET", "tensor_id": tensor_id}
        sock.send(msgpack.dumps(data))

        message = sock.recv()
        data = msgpack.loads(message)
        if data["ret"] != 0:
            logger.warning(
                "🔴[GET]Recv From %s, tensor_id: %s, ret: %d",
                remote_address,
                tensor_id,
                data["ret"],
            )
            return None

        with torch.cuda.stream(self.recv_stream):
            tensor = torch.empty(
                data["shape"], dtype=getattr(torch, data["dtype"]), device=self.device
            )

        self.recv(comm, tensor, rank ^ 1, self.recv_stream)

        return tensor

    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue

            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["cmd"] == "NEW":
                unique_id = self.nccl.unique_id_from_bytes(bytes(data["unique_id"]))
                with torch.accelerator.device_index(self.device.index):
                    rank = 1
                    with set_p2p_nccl_context(self.nccl_num_channels):
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank
                        )
                    self.comms[remote_address.decode()] = (comm, rank)
                    logger.info(
                        "🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        rank,
                    )
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                timing_queued_perf_ns = (
                    time.perf_counter_ns() if self.enable_timing else 0
                )
                timing_queued_wall_ns = time.time_ns() if self.enable_timing else 0
                dtype = getattr(torch, data["dtype"])
                if self.enable_timing:
                    payload_bytes = (
                        math.prod(data["shape"])
                        * torch.empty((), dtype=dtype).element_size()
                    )
                    self._safe_timing_call(
                        self._timing_tracker.enqueue,
                        tensor_id=tensor_id,
                        payload_bytes=payload_bytes,
                        batch_id=None,
                        queued_perf_ns=timing_queued_perf_ns,
                        queued_wall_ns=timing_queued_wall_ns,
                    )
                try:
                    allocation_start_perf_ns = (
                        time.perf_counter_ns() if self.enable_timing else 0
                    )
                    allocation_start_wall_ns = (
                        time.time_ns() if self.enable_timing else 0
                    )
                    with torch.cuda.stream(self.recv_stream):
                        tensor = torch.empty(
                            data["shape"],
                            dtype=dtype,
                            device=self.device,
                        )
                    if self.enable_timing:
                        self._safe_timing_call(
                            self._timing_tracker.update_tensor,
                            tensor_id,
                            allocation_start_perf_ns=allocation_start_perf_ns,
                            allocation_end_perf_ns=time.perf_counter_ns(),
                            allocation_start_wall_ns=allocation_start_wall_ns,
                            allocation_end_wall_ns=time.time_ns(),
                        )
                    control_start_perf_ns = (
                        time.perf_counter_ns() if self.enable_timing else 0
                    )
                    control_start_wall_ns = (
                        time.time_ns() if self.enable_timing else 0
                    )
                    self.router_socket.send_multipart([remote_address, b"0"])
                    if self.enable_timing:
                        self._safe_timing_call(
                            self._timing_tracker.update_tensor,
                            tensor_id,
                            control_start_perf_ns=control_start_perf_ns,
                            control_end_perf_ns=time.perf_counter_ns(),
                            control_start_wall_ns=control_start_wall_ns,
                            control_end_wall_ns=time.time_ns(),
                        )
                    comm, rank = self.comms[remote_address.decode()]
                    nccl_span = self._new_cuda_span(record_start=False)
                    nccl_start_perf_ns = (
                        time.perf_counter_ns() if self.enable_timing else 0
                    )
                    nccl_start_wall_ns = time.time_ns() if self.enable_timing else 0
                    try:
                        self.recv(
                            comm,
                            tensor,
                            rank ^ 1,
                            self.recv_stream,
                            timing_span=nccl_span,
                        )
                    except BaseException as exc:
                        if self.enable_timing:
                            self._safe_timing_call(
                                self._timing_tracker.complete_tensor,
                                tensor_id,
                                completed_perf_ns=time.perf_counter_ns(),
                                completed_wall_ns=time.time_ns(),
                                status="failed",
                                error=f"nccl_receive_failed:{type(exc).__name__}",
                            )
                            self._safe_timing_call(
                                self._timing_tracker.fail_request,
                                tensor_id.split("#", 1)[0],
                                f"nccl_receive_failed:{type(exc).__name__}",
                            )
                        raise
                    if self.enable_timing:
                        self._safe_timing_call(
                            self._timing_tracker.update_tensor,
                            tensor_id,
                            nccl_start_perf_ns=nccl_start_perf_ns,
                            nccl_end_perf_ns=time.perf_counter_ns(),
                            nccl_start_wall_ns=nccl_start_wall_ns,
                            nccl_end_wall_ns=time.time_ns(),
                            nccl_span=nccl_span,
                        )
                    tensor_size = tensor.element_size() * tensor.numel()
                    if self.buffer_size + tensor_size > self.buffer_size_threshold:
                        # Store Tensor in memory pool
                        addr = self.pool.store_tensor(tensor)
                        tensor = (addr, tensor.dtype, tensor.shape)
                        logger.warning(
                            "🔴[PUT]Recv Tensor, Out Of Threshold, "
                            "%s👈%s, data:%s, addr:%d",
                            self.zmq_address,
                            remote_address.decode(),
                            data,
                            addr,
                        )
                    else:
                        self.buffer_size += tensor_size

                except torch.cuda.OutOfMemoryError:
                    self.router_socket.send_multipart([remote_address, b"1"])
                    tensor = None
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, data:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                    )

                if self.enable_timing:
                    self._safe_timing_call(
                        self._timing_tracker.complete_tensor,
                        tensor_id,
                        completed_perf_ns=time.perf_counter_ns(),
                        completed_wall_ns=time.time_ns(),
                        status="ok" if tensor is not None else "failed",
                        error=None if tensor is not None else "receiver_out_of_memory",
                    )

                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()

            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                with self.send_store_cv:
                    tensor = self.send_store.pop(tensor_id, None)
                    if tensor is not None:
                        data = {
                            "ret": 0,
                            "shape": tensor.shape,
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                        # LRU
                        self.send_store[tensor_id] = tensor
                        self.have_sent_tensor_id(tensor_id)
                    else:
                        data = {"ret": 1}

                self.router_socket.send_multipart([remote_address, msgpack.dumps(data)])

                if data["ret"] == 0:
                    comm, rank = self.comms[remote_address.decode()]
                    self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)
            else:
                logger.warning(
                    "🚧Unexpected, Received message from %s, data:%s",
                    remote_address,
                    data,
                )

    def have_sent_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        if request_id not in self.send_request_id_to_tensor_ids:
            self.send_request_id_to_tensor_ids[request_id] = set()
        self.send_request_id_to_tensor_ids[request_id].add(tensor_id)

    def have_received_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)

    def send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                if not self.send_queue:
                    self.send_queue_cv.notify()
            self.send_sync(item)

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.send_queue_cv:
                while self.send_queue:
                    self.send_queue_cv.wait()
            duration = time.time() - start_time
            logger.debug(
                "🚧[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d",
                duration * 1000,
                self.rank,
            )

    def send_sync(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        if item.remote_address not in self.socks:
            self.create_connect(item.remote_address)

        tensor = item.tensor

        sock = self.socks[item.remote_address]
        comm, rank = self.comms[item.remote_address]
        data = {
            "cmd": "PUT",
            "tensor_id": item.tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        control_start_perf_ns = time.perf_counter_ns() if self.enable_timing else 0
        control_start_wall_ns = time.time_ns() if self.enable_timing else 0
        sock.send(msgpack.dumps(data))

        response = sock.recv()
        if self.enable_timing:
            self._safe_timing_call(
                self._timing_tracker.update_tensor,
                item.tensor_id,
                control_start_perf_ns=control_start_perf_ns,
                control_end_perf_ns=time.perf_counter_ns(),
                control_start_wall_ns=control_start_wall_ns,
                control_end_wall_ns=time.time_ns(),
            )
        if response != b"0":
            logger.error(
                "🔴Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "MyRank:%s, data:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address,
                item.remote_address,
                rank,
                data,
                tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode(),
            )
            if self.enable_timing:
                self._safe_timing_call(
                    self._timing_tracker.complete_tensor,
                    item.tensor_id,
                    completed_perf_ns=time.perf_counter_ns(),
                    completed_wall_ns=time.time_ns(),
                    status="failed",
                    error=f"peer_rejected_transfer:{response.decode()}",
                )
            return False

        nccl_span = self._new_cuda_span(record_start=False)
        nccl_start_perf_ns = time.perf_counter_ns() if self.enable_timing else 0
        nccl_start_wall_ns = time.time_ns() if self.enable_timing else 0
        try:
            self.send(
                comm,
                tensor.to(self.device),
                rank ^ 1,
                self.send_stream,
                timing_span=nccl_span,
            )
        except BaseException as exc:
            if self.enable_timing:
                self._safe_timing_call(
                    self._timing_tracker.complete_tensor,
                    item.tensor_id,
                    completed_perf_ns=time.perf_counter_ns(),
                    completed_wall_ns=time.time_ns(),
                    status="failed",
                    error=f"nccl_send_failed:{type(exc).__name__}",
                )
            raise
        if self.enable_timing:
            completed_perf_ns = time.perf_counter_ns()
            completed_wall_ns = time.time_ns()
            self._safe_timing_call(
                self._timing_tracker.update_tensor,
                item.tensor_id,
                nccl_start_perf_ns=nccl_start_perf_ns,
                nccl_end_perf_ns=completed_perf_ns,
                nccl_start_wall_ns=nccl_start_wall_ns,
                nccl_end_wall_ns=completed_wall_ns,
                nccl_span=nccl_span,
            )
            # In PUT_ASYNC this call runs on the existing send thread. A sealed
            # request is therefore published only after the real in-flight
            # transfer has completed, independently of wait_for_sent().
            self._safe_timing_call(
                self._timing_tracker.complete_tensor,
                item.tensor_id,
                completed_perf_ns=completed_perf_ns,
                completed_wall_ns=completed_wall_ns,
                status="ok",
            )

        if self.send_type == "PUT_ASYNC":
            self.have_sent_tensor_id(item.tensor_id)

        return True

    def get_finished(
        self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        # Clear the buffer upon request completion.
        for request_id in finished_req_ids:
            for layer_name in no_compile_layers:
                tensor_id = request_id + "#" + layer_name
                if tensor_id in self.recv_store:
                    with self.recv_store_cv:
                        tensor = self.recv_store.pop(tensor_id, None)
                        self.send_request_id_to_tensor_ids.pop(request_id, None)
                        self.recv_request_id_to_tensor_ids.pop(request_id, None)
                    if isinstance(tensor, tuple):
                        addr, _, _ = tensor
                        self.pool.free(addr)

        # TODO:Retrieve requests that have already sent the KV cache.
        finished_sending: set[str] = set()

        # TODO:Retrieve requests that have already received the KV cache.
        finished_recving: set[str] = set()

        return finished_sending or None, finished_recving or None

    def ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address,
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def send(
        self,
        comm,
        tensor: torch.Tensor,
        dst: int,
        stream=None,
        timing_span: CudaEventSpan | None = None,
    ):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            if timing_span is not None:
                timing_span.start.record(stream)
            self.nccl.ncclSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                dst,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
            if timing_span is not None:
                timing_span.end.record(stream)
        stream.synchronize()

    def recv(
        self,
        comm,
        tensor: torch.Tensor,
        src: int,
        stream=None,
        timing_span: CudaEventSpan | None = None,
    ):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            if timing_span is not None:
                timing_span.start.record(stream)
            self.nccl.ncclRecv(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                src,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
            if timing_span is not None:
                timing_span.end.record(stream)
        stream.synchronize()

    # ==============================
    # Optional timing methods
    # ==============================

    def begin_request_timing(self, request_id: str, batch_id: str | None) -> None:
        if self.enable_timing:
            self._safe_timing_call(
                self._timing_tracker.bind_request, request_id, batch_id
            )

    def start_cuda_timing_span(self) -> CudaEventSpan | None:
        return self._new_cuda_span(record_start=True)

    def finish_cuda_timing_span(
        self,
        request_id: str,
        stage: str,
        span: CudaEventSpan | None,
    ) -> None:
        if not self.enable_timing or span is None:
            return
        try:
            span.end.record()
            self._timing_tracker.add_stage_span(
                request_id,
                stage,  # type: ignore[arg-type]
                span,
            )
        except Exception:
            self._safe_timing_log_exception(
                "Failed to record P2P NCCL %s timing for %s", stage, request_id
            )

    def seal_request_timing(self, request_id: str, batch_id: str | None) -> None:
        if self.enable_timing:
            self._safe_timing_call(
                self._timing_tracker.seal_request, request_id, batch_id
            )

    def fail_request_timing(self, request_id: str, message: str) -> None:
        if self.enable_timing:
            self._safe_timing_call(
                self._timing_tracker.fail_request, request_id, message
            )

    def publish_ready_timing(self) -> None:
        if self.enable_timing:
            self._safe_timing_call(self._timing_tracker.publish_ready)

    def discard_request_timing(self, request_id: str) -> None:
        if self.enable_timing:
            self._safe_timing_call(
                self._timing_tracker.discard_request, request_id
            )

    @property
    def timing_anchor(self) -> Any | None:
        return getattr(self, "_timing_anchor", None)

    def _new_cuda_span(self, *, record_start: bool) -> CudaEventSpan | None:
        if not self.enable_timing:
            return None
        try:
            span = CudaEventSpan(
                start=torch.cuda.Event(enable_timing=True),
                end=torch.cuda.Event(enable_timing=True),
            )
            if record_start:
                span.start.record()
            return span
        except Exception:
            self._safe_timing_log_exception("Failed to create P2P NCCL CUDA events")
            return None

    def _safe_timing_call(self, callback, *args, **kwargs) -> None:
        try:
            callback(*args, **kwargs)
        except Exception:
            self._safe_timing_log_exception("P2P NCCL timing operation failed")

    @staticmethod
    def _safe_timing_log_exception(message: str, *args: Any) -> None:
        try:
            logger.exception(message, *args)
        except Exception:
            pass

    def close(self) -> None:
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
