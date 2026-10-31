
import torch
import torch.nn as nn
import importlib

FeatureExtractor = importlib.import_module("04_feature_extractor").FeatureExtractor
DFEB = importlib.import_module("05_dfeb").DFEB
CoarseLocalization = importlib.import_module("06_coarse_localization").CoarseLocalization
LocalizationRefinement = importlib.import_module("07_localization_refinement").LocalizationRefinement
GuidedFusion = importlib.import_module("08_guided_fusion").GuidedFusion

adcb_dalfm = importlib.import_module("09_adcb_dalfm")
ADCB = adcb_dalfm.ADCB
DALFM = adcb_dalfm.DALFM

SegmentationDecoder = importlib.import_module("10_segmentation_decoder").SegmentationDecoder


class CrackAnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        self.dfeb = DFEB(in_channels=256, out_channels=128)
        self.coarse_localization = CoarseLocalization(in_channels=128)
        self.refinement = LocalizationRefinement(feature_channels=128)
        self.guided_fusion = GuidedFusion(feature_channels=128)
        self.adcb = ADCB(channels=128)
        self.dalfm = DALFM(channels=128)
        self.decoder = SegmentationDecoder(in_channels=128)

    def forward(self, x):
        output_size = x.shape[-2:]

        features = self.feature_extractor(x)

        f3 = features["f3"]
        f4 = features["f4"]

        localization_features = self.dfeb(f4)

        coarse_map = self.coarse_localization(
            localization_features,
            output_size
        )

        refined_map = self.refinement(
            localization_features,
            coarse_map,
            output_size
        )

        guided_features = self.guided_fusion(
            f3,
            refined_map
        )

        adcb_output = self.adcb(guided_features)

        dalfm_output = self.dalfm(adcb_output)

        segmentation_logits = self.decoder(
            dalfm_output,
            output_size
        )

        return {
            "features": features,
            "localization_features": localization_features,
            "coarse_map": coarse_map,
            "refined_map": refined_map,
            "guided_features": guided_features,
            "adcb_output": adcb_output,
            "dalfm_output": dalfm_output,
            "segmentation_logits": segmentation_logits
        }
