from collections import OrderedDict

import torch
from torch.func import functional_call


class MAMLInnerLoop:
    """
    Inner-loop adaptation: plain SGD on the support set.

    second_order=False -> first-order MAML (fast, low memory)
    second_order=True  -> full second-order MAML (slow, high memory)
    """

    def __init__(
        self,
        model,
        loss_function,
        inner_lr=1e-2,
        update_steps=3,
        second_order=False,
        grad_clip=5.0,
    ):
        self.model = model
        self.loss_function = loss_function
        self.inner_lr = inner_lr
        self.update_steps = update_steps
        self.second_order = second_order
        self.grad_clip = grad_clip

    def adapt(self, support_images, support_masks, update_steps=None):
        steps = self.update_steps if update_steps is None else update_steps

        params = OrderedDict(self.model.named_parameters())
        support_loss = None

        for _ in range(steps):

            outputs = functional_call(self.model, params, (support_images,))

            losses = self.loss_function(
                outputs["segmentation_logits"],
                outputs["coarse_map"],
                outputs["refined_map"],
                support_masks,
            )
            support_loss = losses["total_loss"]

            grads = torch.autograd.grad(
                support_loss,
                list(params.values()),
                create_graph=self.second_order,
                allow_unused=True,
            )

            # gradient-norm clipping inside the inner loop (stability)
            if self.grad_clip is not None:
                sq = sum(g.pow(2).sum() for g in grads if g is not None)
                norm = torch.sqrt(sq + 1e-12)
                coef = torch.clamp(self.grad_clip / norm, max=1.0)
            else:
                coef = 1.0

            params = OrderedDict(
                (
                    name,
                    p - self.inner_lr * coef * g if g is not None else p,
                )
                for (name, p), g in zip(params.items(), grads)
            )

        return params, support_loss
