"""Batch native generation while retaining the audited candidate adapter."""

from itertools import islice

from .adapters import generate_grid


def generate_grid_batched(model, requests, item_to_sid, item_groups, metadata, *, device="cpu", batch_size=16):
    import torch
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Generation batch size must be a positive integer")
    if batch_size == 1:
        return generate_grid(model, requests, item_to_sid, item_groups, metadata, device=device)

    class BufferedModel:
        current = None

        def __getattr__(self, name):
            return getattr(model, name)

        def generate(self, history, users):
            return self.current

    buffered = BufferedModel()

    def batched_requests():
        iterator = iter(requests)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                return
            length = max(len(r.history_ids) for r in batch)
            histories = torch.tensor([r.history_ids + [-1] * (length - len(r.history_ids)) for r in batch],
                                     dtype=torch.long, device=device)
            if any(r.user_id is None for r in batch) and not all(r.user_id is None for r in batch):
                raise ValueError("Batched requests must consistently supply user IDs")
            users = (None if batch[0].user_id is None else
                     torch.tensor([r.user_id for r in batch], dtype=torch.long, device=device))
            ids, probabilities = model.generate(histories, users)
            if ids.ndim != 3 or probabilities.ndim != 2 or ids.shape[:2] != probabilities.shape or ids.shape[0] != len(batch):
                raise ValueError("Unexpected batched GRID output shapes")
            ids, probabilities = ids.cpu(), probabilities.cpu()
            for i, request in enumerate(batch):
                buffered.current = ids[i:i+1], probabilities[i:i+1]
                yield request

    result = generate_grid(buffered, batched_requests(), item_to_sid, item_groups, metadata, device=device)
    result.metadata["generation_adapter"]["evaluation_batch_size"] = batch_size
    return result
