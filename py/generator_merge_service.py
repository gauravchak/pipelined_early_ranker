"""Service that initiates candidate generators and is called when they
    have candidates ready to be ranked"""
from dataclasses import dataclass
from typing import List
from numpy import log2


@dataclass
class GeneratorResult:
    """Result from a candidate generator"""

    item_id: str
    rank: int
    score: float


class CandidateGeneratorService:
    """Service that initiates candidate generators and is called when they
    have candidates ready to be ranked"""

    def __init__(
        self,
        max_num_lsr: int,
        lsr_sufficiency_threshold: float,
        max_num_esr: int,
        lsr_batch_size: int,
        weights: dict,
    ) -> None:
        self.max_num_lsr = max_num_lsr
        self.lsr_sufficiency_threshold = lsr_sufficiency_threshold
        self.max_num_esr = max_num_esr
        self.lsr_batch_size = lsr_batch_size
        self.uv_dict = {}
        self.num_lsr_sent = 0
        self.items_waiting_for_lsr = (
            []
        )  # noqa List to store items waiting to be sent to late stage ranker
        self.already_sent = (
            {}
        )  # noqa Dictionary to track items that have already been sent
        self.weights = weights  # noqa Dictionary to store weights for each generator

    def OnGeneratorCompletion(
        self,
        generator_id: int,
        results: List[GeneratorResult],
    ):
        for result in results:
            item_id, rank, score = result.item_id, result.rank, result.score
            uv_estimate = self.calculate_user_value_estimate(generator_id, rank, score)
            self.uv_dict[item_id] = max(self.uv_dict.get(item_id, 0), uv_estimate)

            if (
                uv_estimate > self.lsr_sufficiency_threshold
                and item_id not in self.already_sent
            ):
                self.enqueue_for_late_stage_ranker(item_id)

    def enqueue_for_late_stage_ranker(self, item_id):
        if (
            self.num_lsr_sent < self.max_num_lsr
            and item_id not in self.items_waiting_for_lsr
        ):
            self.items_waiting_for_lsr.append(item_id)

            if len(self.items_waiting_for_lsr) >= self.lsr_batch_size:
                self.send_to_late_stage_ranker()

    def send_to_late_stage_ranker(self):
        items_to_send = []
        for item_id in self.items_waiting_for_lsr:
            if item_id not in self.already_sent:
                items_to_send.append(item_id)
                self.already_sent[item_id] = True
        # noqa Implement the logic to send items to the late stage ranker (not implemented here)
        print(f"Sending to late stage ranker: {items_to_send}")
        self.num_lsr_sent += len(items_to_send)
        self.items_waiting_for_lsr = []

    def calculate_user_value_estimate(self, generator_id, rank, score):
        # Adjust these weights based on your model
        w0, w1, w2 = self.weights.get(generator_id, (1.0, 1.0, 1.0))
        return w0 * 1 / log2(max(1 + rank, 2)) + w1 * score + w2

    def OnTimeOut(self):
        # self.send_to_late_stage_ranker()
        top_K_items = sorted(self.uv_dict.items(), key=lambda x: x[1], reverse=True)
        top_K_items = [
            (item_id, value)
            for item_id, value in top_K_items
            if item_id not in self.already_sent
        ]
        top_K_items = top_K_items[: self.max_num_esr]
        self.send_to_early_stage_ranker(top_K_items)

    def send_to_early_stage_ranker(self, top_K_items):
        print(f"Sending to early stage ranker: {top_K_items}")
        # noqa Implement the logic to send items to the early stage ranker (not implemented here)
        pass


# Example usage:
max_num_lsr: int = 10
lsr_sufficiency_threshold: float = 1.1
max_num_esr: int = 5
lsr_batch_size: int = 3

# Dictionary to store weights for each generator
weights = {
    1: (0.5, 0.8, 0.2),
    2: (0.7, 0.5, 0.3),
    # Add more weights as needed
}

candidate_generator_service = CandidateGeneratorService(
    max_num_lsr, lsr_sufficiency_threshold, max_num_esr, lsr_batch_size, weights
)

# Example 1
generator_id_1 = 1
results_1 = [
    GeneratorResult("item_1", 1, 0.9),
    GeneratorResult("item_2", 2, 0.8),
    GeneratorResult("item_3", 3, 0.7),
]
candidate_generator_service.OnGeneratorCompletion(generator_id_1, results_1)

# Example 2
generator_id_2 = 2
results_2 = [
    GeneratorResult("item_4", 1, 0.85),
    GeneratorResult("item_5", 2, 0.78),
    GeneratorResult("item_6", 3, 0.72),
]
candidate_generator_service.OnGeneratorCompletion(generator_id_2, results_2)

# Assume OnTimeOut is called after some time
candidate_generator_service.OnTimeOut()
