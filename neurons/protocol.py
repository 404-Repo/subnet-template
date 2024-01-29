import typing

import bittensor as bt


class TextTo3D(bt.Synapse):
    prompt_in: str = ""
    mesh_out: bytes | None = None

    def deserialize(self) -> bytes | None:
        return self.mesh_out or None
