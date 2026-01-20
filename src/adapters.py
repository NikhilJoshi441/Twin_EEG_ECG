"""Streaming adapter skeletons for OpenBCI / Muse / ESP32.

These are placeholders showing expected methods and where to plug hardware-specific code.
"""
from typing import Iterator, Dict, Any, Optional
import time


class BaseAdapter:
    def connect(self) -> None:
        raise NotImplementedError()

    def read_stream(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError()


class ESP32Adapter(BaseAdapter):
    def __init__(self, host: str, port: int = 80):
        self.host = host
        self.port = port

    def connect(self):
        # Implement socket or HTTP connection to ESP32 streaming endpoint
        pass

    def read_stream(self):
        # yield dicts like {"ecg": [..], "eeg": [..]}
        while False:
            yield {}


class OpenBCIAdapter(BaseAdapter):
    def __init__(self, port: str, board_id: Optional[int] = None):
        self.port = port
        self.board_id = board_id
        self.board = None
        self._BoardShim = None

    def connect(self, retry: int = 3, retry_delay: float = 2.0):
        # Initialize BrainFlow BoardShim if available.
        try:
            # Import here to keep brainflow optional for users without the library/hardware
            from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
            self._BoardShim = BoardShim
            self._BoardIds = BoardIds
            self._BrainFlowInputParams = BrainFlowInputParams
        except ImportError:
            self._BoardShim = None
            raise ImportError(
                "brainflow package is required for OpenBCIAdapter. Install it with `pip install brainflow` and ensure board drivers are available.`"
            )
        except Exception as e:
            self._BoardShim = None
            raise

        params = self._BrainFlowInputParams()
        params.serial_port = self.port
        # Default to CYTON if board_id not provided
        board_id = self.board_id if self.board_id is not None else self._BoardIds.CYTON_BOARD.value

        last_exc = None
        for attempt in range(retry):
            try:
                self.board = self._BoardShim(board_id, params)
                self.board.prepare_session()
                self.board.start_stream()
                return
            except Exception as e:
                last_exc = e
                time.sleep(retry_delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to initialize board connection")

    def read_stream(self, chunk_size: int = 256):
        """Yield mapped frames as dicts like { 'ecg': [...], 'eeg': [...] } when possible.

        This method tries to map common channel indices to ECG/EEG based on board type.
        Returns early (generator ends) if board is not ready.
        """
        if self._BoardShim is None or self.board is None:
            return
        try:
            while True:
                data = self.board.get_current_board_data(chunk_size)
                if data is None:
                    # board not ready
                    continue
                # data shape: (n_channels, n_samples)
                # Attempt to map channels to known signals. This mapping may need user adjustment.
                out = {"raw": data.tolist() if hasattr(data, 'tolist') else []}
                # Common boards place ECG on channel index 0 or a dedicated channel; naive mapping follows:
                try:
                    n_channels = data.shape[0]
                    # heuristics: if many channels (>8) assume EEG channels, pick first channel as ECG proxy
                    if n_channels >= 8:
                        eeg = data[:min(8, n_channels), :].tolist() if hasattr(data, 'tolist') else []
                        out["eeg"] = eeg
                    # ECG proxy: if there is a dedicated ECG channel (often last), attempt to use it
                    if n_channels >= 1:
                        ecg_proxy = data[0, :].tolist() if hasattr(data, 'tolist') else []
                        out["ecg"] = ecg_proxy
                except Exception:
                    pass
                yield out
        except Exception:
            return

    def close(self):
        try:
            if self.board is not None:
                self.board.stop_stream()
                self.board.release_session()
        except Exception:
            pass


class MuseAdapter(BaseAdapter):
    def __init__(self):
        pass

    def connect(self):
        # Use muselsl or BLE APIs
        pass

    def read_stream(self):
        while False:
            yield {}
