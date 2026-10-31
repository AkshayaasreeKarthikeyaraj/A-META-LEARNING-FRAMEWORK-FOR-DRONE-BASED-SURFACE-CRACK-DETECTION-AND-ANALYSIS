
import torch
import torch.optim as optim
import importlib


class MAMLOuterLoop:

    def __init__(
        self,
        model,
        loss_function,
        inner_lr=1e-3,
        update_steps_dict=None,
        meta_lr=1e-5,
        weight_decay=1e-8,
        momentum=0.999
    ):

        self.model = model
        self.loss_function = loss_function
        self.inner_lr = inner_lr

        self.update_steps_dict = update_steps_dict or {
            "Volker": 2,
            "Crack500": 3,
            "OUC": 2
        }

        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=meta_lr,
            weight_decay=weight_decay,
            momentum=momentum
        )

        inner_module = importlib.import_module(
            "15_maml_inner_loop"
        )

        self.MAMLInnerLoop = inner_module.MAMLInnerLoop

    def meta_train_task(
        self,
        task,
        dataset_name
    ):

        support_images = task["support_images"]
        support_masks = task["support_masks"]

        query_images = task["query_images"]
        query_masks = task["query_masks"]

        update_steps = self.update_steps_dict[dataset_name]

        inner_loop = self.MAMLInnerLoop(
            model=self.model,
            loss_function=self.loss_function,
            inner_lr=self.inner_lr,
            update_steps=update_steps
        )

        adapted_parameters, support_loss = inner_loop.adapt(
            support_images,
            support_masks
        )

        query_outputs = torch.func.functional_call(
            self.model,
            adapted_parameters,
            (query_images,)
        )

        query_losses = self.loss_function(
            query_outputs["segmentation_logits"],
            query_outputs["coarse_map"],
            query_outputs["refined_map"],
            query_masks
        )

        query_loss = query_losses["total_loss"]

        return {
            "support_loss": support_loss,
            "query_loss": query_loss,
            "segmentation_loss": query_losses["segmentation_loss"],
            "coarse_loss": query_losses["coarse_loss"],
            "refinement_loss": query_losses["refinement_loss"]
        }

    def meta_update(self, meta_loss):

        self.optimizer.zero_grad()

        meta_loss.backward()

        self.optimizer.step()
