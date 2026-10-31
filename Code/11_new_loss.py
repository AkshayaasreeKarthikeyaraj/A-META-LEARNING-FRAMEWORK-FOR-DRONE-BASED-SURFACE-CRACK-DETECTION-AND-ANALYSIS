
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.contiguous().view(
            probabilities.size(0), -1
        )

        targets = targets.contiguous().view(
            targets.size(0), -1
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return (1.0 - dice).mean()


class HybridLoss(nn.Module):
    def __init__(
        self,
        pos_weight=1.0,
        lambda_coarse=1.0,
        lambda_refine=1.0
    ):
        super().__init__()

        self.pos_weight = torch.tensor(float(pos_weight))

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight
        )

        self.dice = DiceLoss()

        self.lambda_coarse = lambda_coarse
        self.lambda_refine = lambda_refine

    def segmentation_loss(self, prediction, target):

        bce_loss = self.bce(
            prediction,
            target
        )

        dice_loss = self.dice(
            prediction,
            target
        )

        return bce_loss + dice_loss

    def forward(
        self,
        segmentation_logits,
        coarse_map,
        refined_map,
        target
    ):

        segmentation_loss = self.segmentation_loss(
            segmentation_logits,
            target
        )

        coarse_loss = self.segmentation_loss(
            coarse_map,
            target
        )

        refinement_loss = self.segmentation_loss(
            refined_map,
            target
        )

        total_loss = (
            segmentation_loss
            + self.lambda_coarse * coarse_loss
            + self.lambda_refine * refinement_loss
        )

        return {
            "total_loss": total_loss,
            "segmentation_loss": segmentation_loss,
            "coarse_loss": coarse_loss,
            "refinement_loss": refinement_loss
        }
