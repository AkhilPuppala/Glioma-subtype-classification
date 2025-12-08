import os
from functools import partial
import timm
import torch
from .timm_wrapper import TimmCNNEncoder
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms

# import your custom ResNet50
from models.resnet_custom_dep import resnet50_baseline    # <-- corrected import path


def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH


def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    try:
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH



# ============================================================
# MAIN ENCODER FUNCTION (NOW WITH RESNET50 SUPPORT)
# ============================================================
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')

    # -----------------------------------------------
    # 1) ResNet50 (YOUR feature extractor)
    # -----------------------------------------------
    if model_name == "resnet50":
        model = resnet50_baseline(pretrained=True)   # outputs 2048-dim vectors
        model.eval()
        print("✅ ResNet50 model loaded successfully.")

    # ------------------------------------------------
    # 2) Timm resnet50 trunc (existing)
    # ------------------------------------------------
    elif model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()

    # ------------------------------------------------
    # 3) UNI Vision Transformer
    # ------------------------------------------------
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'

        model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True
        )
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)

    # ------------------------------------------------
    # 4) CONCH v1
    # ------------------------------------------------
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'

        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)

    # ------------------------------------------------
    # 5) CONCH v1.5 (TITAN)
    # ------------------------------------------------
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Install transformers for CONCH v1.5: pip install transformers")

        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN expects 448x448 image size'

    else:
        raise NotImplementedError("model {} not implemented".format(model_name))

    # ------------------------------------------------
    # LOAD NORMALIZATION CONSTANTS
    # ------------------------------------------------
    if model_name not in MODEL2CONSTANTS:
        raise KeyError(f"Model constants for '{model_name}' not found in MODEL2CONSTANTS. Please add them.")

    constants = MODEL2CONSTANTS[model_name]

    img_transforms = get_eval_transforms(
        mean=constants['mean'],
        std=constants['std'],
        target_img_size=target_img_size
    )

    print(model)
    return model, img_transforms
