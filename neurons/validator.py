import time
from typing import List

import bittensor as bt
import torch
from config import read_config
from validating import Validate3DModels, load_models, score_responses
from protocol import TextTo3D

from neurons.base_validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):
    models: Validate3DModels

    def __init__(self, config: bt.config):
        super(Validator, self).__init__(config)

        self.models = load_models(self.device, config.neuron.full_path)

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        miner_uids = self.get_random_miners_uids(self.config.neuron.sample_size)

        # TODO: get prompts from the dataset
        prompt = "A Golden Poison Dart Frog"

        responses = await self.dendrite.forward(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=TextTo3D(prompt_in=prompt),
            deserialize=False,
            timeout=30,
        )

        bt.logging.info(
            f"Received {len([r for r in responses if r.mesh_out is not None])} responses"
        )

        scores = score_responses(prompt, responses, self.device, self.models)

        bt.logging.info(f"Scored responses: {scores}")

        self.update_scores(scores, miner_uids)


def main():
    config = read_config(Validator)
    bt.logging.info(f"Starting with config: {config}")

    with Validator(config) as validator:
        while True:
            bt.logging.debug("Validator running...", time.time())
            time.sleep(60)

            if validator.should_exit.is_set():
                bt.logging.debug("Stopping the validator")
                break


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    main()
