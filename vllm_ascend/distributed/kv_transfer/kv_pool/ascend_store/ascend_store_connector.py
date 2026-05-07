import threading
from collections.abc import Iterable
from typing import Any

import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
    get_zmq_rpc_path_lookup,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class AscendStoreKVEvents(KVConnectorKVEvents):
    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "AscendStoreKVEvents":
        """
        Aggregate KV events and retain only common events.
        """
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<AscendStoreKVEvents events={self.get_all_events()}>"


class AscendStoreConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )

        connector_name = vllm_config.kv_transfer_config.kv_connector
        if connector_name == "MooncakeConnectorStoreV1":
            logger.warning(
                "It is recommended to use the AscendStoreConnector, "
                "as the MoonCakeStoreConnector will be removed in the future."
            )

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._kv_cache_events: AscendStoreKVEvents | None = None

        self.sended_but_unfinished_reqs: set[str] = set()

        self.num_write_buffers = vllm_config.kv_transfer_config.kv_connector_extra_config.get("kv_offload_num_buffers", 0)
        self.num_read_buffers = vllm_config.kv_transfer_config.kv_connector_extra_config.get("kv_offload_num_read_buffers", 2)
        
        self._layer_name_to_idx: dict[str, int] = {}
        self._layer_names: list[str] = []
        self._has_continuation_chunks = False
        self._original_kv_caches: dict[str, torch.Tensor] = {}
        
        self._offload_active = (
            self.use_layerwise
            and self.num_write_buffers > 0
            and self.kv_role in ["kv_producer", "kv_both"]
            and role == KVConnectorRole.WORKER
        )

        if self._offload_active:
            # Inline buffer state tracking
            self._transfer_stream: torch.npu.Stream | None = None
            self._read_stream: torch.npu.Stream | None = None
            self._layer_to_storage_id: dict[str, int] = {}
            
            self._write_buffers: dict[int, list[torch.Tensor]] = {}
            self._read_buffers: dict[int, list[torch.Tensor]] = {}
            
            self._write_events: dict[int, list[torch.npu.Event | None]] = {}
            self._read_events: dict[int, list[torch.npu.Event | None]] = {}
            self._read_consumed_events: dict[int, list[torch.npu.Event | None]] = {}

            logger.info(
                "AscendStoreConnector: offload mode ACTIVE, "
                "num_write_buffers=%d, num_read_buffers=%d",
                self.num_write_buffers, self.num_read_buffers
            )

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = KVPoolScheduler(vllm_config, self.use_layerwise)
        else:
            self.connector_worker = KVPoolWorker(
                vllm_config,
                self.use_layerwise,
            )

            assert self.connector_worker is not None
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = LookupKeyServer(self.connector_worker, vllm_config, self.use_layerwise)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side connectors output.
        """
        # Get the KV events
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, AscendStoreKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(kv_cache_events.get_number_of_workers())
        return

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)
        
        if not self._offload_active:
            return
            
        self._layer_names = sorted(kv_caches.keys())
        self._layer_name_to_idx = {name: idx for idx, name in enumerate(self._layer_names)}
        self._original_kv_caches = {name: tensor for name, tensor in kv_caches.items()}
        
        sample_cache = list(kv_caches.values())[0]
        device = sample_cache[0].device if isinstance(sample_cache, (list, tuple)) else sample_cache.device
        
        self._transfer_stream = torch.npu.Stream()
        self._read_stream = torch.npu.Stream()
        
        unique_storages = {}
        for layer_name, tensor in kv_caches.items():
            storage = tensor[0].untyped_storage() if isinstance(tensor, (list, tuple)) else tensor.untyped_storage()
            storage_id = storage.data_ptr()
            self._layer_to_storage_id[layer_name] = storage_id
            
            if storage_id not in unique_storages:
                proto = torch.empty(storage.size(), dtype=torch.uint8, device=device)
                proto.set_(storage)
                unique_storages[storage_id] = proto

        ptrs = []
        lengths = []
        for storage_id, proto in unique_storages.items():
            w_bufs = [proto] + [torch.zeros_like(proto) for _ in range(self.num_write_buffers - 1)]
            r_bufs = [torch.zeros_like(proto) for _ in range(self.num_read_buffers)]
            
            self._write_buffers[storage_id] = w_bufs
            self._read_buffers[storage_id] = r_bufs
            
            self._write_events[storage_id] = [None] * self.num_write_buffers
            self._read_events[storage_id] = [None] * self.num_read_buffers
            self._read_consumed_events[storage_id] = [None] * self.num_read_buffers
            
            for buf in w_bufs + r_bufs:
                ptrs.append(buf.data_ptr())
                lengths.append(buf.nelement() * buf.element_size())

        if hasattr(self.connector_worker, 'm_store'):
            try:
                self.connector_worker.m_store.register_buffer(ptrs, lengths)
                logger.info("Registered %d offload buffers with transfer backend", len(ptrs))
            except Exception as e:
                logger.warning("Failed to register offload buffers with backend: %s", e)

    def _check_continuation_chunks(self) -> None:
        self._has_continuation_chunks = False
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is not None and load_spec.can_load:
                self._has_continuation_chunks = True
                break

    def _remap_layer_to_buffer(self, layer_name: str, buffer: torch.Tensor) -> None:
        if layer_name not in self._original_kv_caches:
            return
        original = self._original_kv_caches[layer_name]
        if isinstance(original, (list, tuple)):
            for orig_tensor in original:
                orig_tensor.set_(buffer.untyped_storage(), orig_tensor.storage_offset(), orig_tensor.shape)
        else:
            original.set_(buffer.untyped_storage(), original.storage_offset(), original.shape)

    def _get_buffer_addrs(self, layer_name: str, buffer: torch.Tensor) -> list[int]:
        original = self._original_kv_caches[layer_name]
        base_ptr = buffer.data_ptr()
        if isinstance(original, (list, tuple)):
            return [base_ptr + orig.storage_offset() * orig.element_size() for orig in original]
        else:
            return [base_ptr + original.storage_offset() * original.element_size()]

    def _start_next_layer_prefetch(self, next_layer_name: str, next_layer_idx: int) -> None:
        storage_id = self._layer_to_storage_id.get(next_layer_name)
        if storage_id is None:
            return
            
        r_idx = next_layer_idx % self.num_read_buffers
        if self.num_read_buffers <= 0:
            return
            
        read_buf = self._read_buffers[storage_id][r_idx]
        
        consume_event = self._read_consumed_events[storage_id][r_idx]
        if consume_event is not None:
            self._read_stream.wait_event(consume_event)
            
        if hasattr(self.connector_worker, "fetch_layer_from_pool"):
            buffer_addrs = self._get_buffer_addrs(next_layer_name, read_buf)
            with torch.npu.stream(self._read_stream):
                self.connector_worker.fetch_layer_from_pool(
                    buffer_base_addrs=buffer_addrs, layer_idx=next_layer_idx, connector_metadata=self._get_connector_metadata()
                )
                fetch_ev = torch.npu.Event()
                fetch_ev.record(self._read_stream)
                self._read_events[storage_id][r_idx] = fetch_ev

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        if self._offload_active:
            self._check_continuation_chunks()
            if self._has_continuation_chunks and len(self._layer_names) > 0:
                for i in range(min(self.num_read_buffers, len(self._layer_names))):
                    self._start_next_layer_prefetch(self._layer_names[i], i)
                    
        self.connector_worker.start_load_kv(self._get_connector_metadata())

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.use_layerwise:
            return
        if self._offload_active:
            layer_idx = self._layer_name_to_idx.get(layer_name, -1)
            if layer_idx < 0:
                return
                
            storage_id = self._layer_to_storage_id[layer_name]
            w_idx = layer_idx % self.num_write_buffers
            write_buf = self._write_buffers[storage_id][w_idx]
            
            # Wait for previous transfer
            if self._write_events[storage_id][w_idx] is not None:
                self._write_events[storage_id][w_idx].synchronize()
                
            if self._has_continuation_chunks and self.num_read_buffers > 0:
                r_idx = layer_idx % self.num_read_buffers
                read_buf = self._read_buffers[storage_id][r_idx]
                
                # Wait for prefetch
                if self._read_events[storage_id][r_idx] is not None:
                    self._read_events[storage_id][r_idx].synchronize()
                    
                write_buf.copy_(read_buf, non_blocking=True)
                
                consume_ev = torch.npu.Event()
                consume_ev.record()
                self._read_consumed_events[storage_id][r_idx] = consume_ev
                
            self._remap_layer_to_buffer(layer_name, write_buf)
            return
            
        self.connector_worker.wait_for_layer_load()

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._offload_active:
            layer_idx = self._layer_name_to_idx.get(layer_name, -1)
            if layer_idx < 0:
                return
                
            storage_id = self._layer_to_storage_id[layer_name]
            w_idx = layer_idx % self.num_write_buffers
            write_buf = self._write_buffers[storage_id][w_idx]
            
            compute_ev = torch.npu.Event()
            compute_ev.record()
            self._transfer_stream.wait_event(compute_ev)
            
            with torch.npu.stream(self._transfer_stream):
                if hasattr(self.connector_worker, "transfer_buffer_to_pool"):
                    buffer_addrs = self._get_buffer_addrs(layer_name, write_buf)
                    self.connector_worker.transfer_buffer_to_pool(
                        buffer_base_addrs=buffer_addrs, layer_idx=layer_idx, 
                        connector_metadata=self._get_connector_metadata(), compute_event=compute_ev
                    )
                else:
                    self.connector_worker.save_kv_layer(self._get_connector_metadata())
                    
                transfer_ev = torch.npu.Event()
                transfer_ev.record(self._transfer_stream)
                self._write_events[storage_id][w_idx] = transfer_ev
                
            if self._has_continuation_chunks and self.num_read_buffers > 0:
                next_idx = layer_idx + self.num_read_buffers
                if next_idx < len(self._layer_names):
                    self._start_next_layer_prefetch(self._layer_names[next_idx], next_idx)
            return
            
        self.connector_worker.save_kv_layer(self._get_connector_metadata())

    def wait_for_save(self):
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            if self._offload_active:
                for events in self._write_events.values():
                    for ev in events:
                        if ev is not None:
                            ev.synchronize()
            return

        self.connector_worker.wait_for_save(self._get_connector_metadata())

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        done_sending, done_recving = self.connector_worker.get_finished(
            finished_req_ids, self._get_connector_metadata()
        )
        return done_sending, done_recving

    def get_kv_connector_kv_cache_events(self) -> AscendStoreKVEvents | None:
        """
        Get the KV connector kv cache events collected during the last interval.
        """
        events = self.connector_worker.get_kv_events()
        if not events:
            return None

        ascend_store_kv_events = AscendStoreKVEvents(num_workers=1)
        ascend_store_kv_events.add_events(events)
        return ascend_store_kv_events




class LookupKeyServer:
    def __init__(
        self,
        pool_worker: KVPoolWorker,
        vllm_config: "VllmConfig",
        use_layerwise: bool,
    ):
        self.decoder = MsgpackDecoder()
        self.decoder_tensor = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.pool_worker = pool_worker
        self.running = True
        self.use_layerwise = use_layerwise

        def process_request():
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                token_len = int.from_bytes(all_frames[0], byteorder="big")
                hash_frames = all_frames[1:]
                hashes_str = self.decoder.decode(hash_frames)
                result = self.pool_worker.lookup_scheduler(token_len, hashes_str, self.use_layerwise)
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
