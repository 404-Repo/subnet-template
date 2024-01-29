import typing

import bittensor as bt


class Task404(bt.Synapse):
    # For backward compatibility with the dummy neurons
    dummy_input: int = 0
    dummy_output: typing.Optional[int] = None

    prompt: str = ""

    def deserialize(self) -> int:
        return self.dummy_output or 0
