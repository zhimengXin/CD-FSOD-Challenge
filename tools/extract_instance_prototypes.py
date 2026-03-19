import json

import numpy as np
import torch
import fire
torch.set_grad_enabled(False)
from torchvision import transforms
from typing import Sequence
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), ".."))
import os
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
import torch.nn.functional as F

from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, DatasetCatalog, MetadataCatalog
from detectron2.data import transforms as T
from tqdm.auto import tqdm
from torchvision.transforms import functional as tvF
import torchvision as tv
from detectron2.data.dataset_mapper import DatasetMapper

from lib.detr_mapper import DetrDatasetMapper
from fast_pytorch_kmeans import KMeans
from lib.categories import ALL_CLS_DICT
import lib.data.fewshot
import lib.data.ovdshot
import lib.data.lvis

import matplotlib.pyplot as plt
from skimage.filters import gaussian
import cv2



pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
normalize_image = lambda x: (x - pixel_mean) / pixel_std
denormalize_image = lambda x: (x * pixel_std) + pixel_mean


def compress(tensor, n_clst=5):
    if len(tensor) <= n_clst:
        # may be normalize this
        # the raw tokens are not normalized
        return tensor
    else:
        kmeans = KMeans(n_clusters=n_clst, verbose=False, mode='cosine')
        kmeans.fit(tensor)
        return kmeans.centroids
    
    
def save_img(img, path):
    tv.utils.save_image(denormalize_image(img) / 255, path)

def iround(x): return int(round(x))

def crop(img, box, enlarge=0.2):
    h,w = img.shape[1:]

    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2  
    lx = (box[2] - box[0]) * (1 + enlarge)
    ly = (box[3] - box[1]) * (1 + enlarge)

    x0 = max(int(round(cx - lx / 2)), 0)
    x1 = min(int(round(cx + lx / 2)), w)
    y0 = max(int(round(cy - ly / 2)), 0)
    y1 = min(int(round(cy + ly / 2)), h)

    return img[:, y0:y1, x0:x1]


def resize_with_largest_edge(img, size=224):
    h, w = img.shape[1:]
    if h >= w:
        ratio = w / h
        h = size
        w = iround(h * ratio)
    else:
        ratio = h / w
        w = size
        h = iround(ratio * w)
    return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)

def resize_to_closest_14x(img):
    h, w = img.shape[1:]
    h, w = max(iround(h / 14), 1) * 14, max(iround(w / 14), 1) * 14
    return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)


def to_mask(boxes, height, width):
    result = torch.zeros(len(boxes), height, width)
    boxes = torch.round(boxes).long()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        result[i, y1:y2, x1:x2] = 1
    return result.bool()
    
    
def get_dataloader(dname, aug=False, split=0, idx=0):
    if aug:
        print("ENABLE AUGMENTATION")
        augmentation = [
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),
            T.RandomFlip(),
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ]
        augmentation_with_crop=[
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),
            T.RandomFlip(),
            T.ResizeShortestEdge(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            T.RandomCrop(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ]
    else:
        augmentation=[
            T.ResizeShortestEdge(
                short_edge_length=800,
                max_size=1333,
            ),
        ]
        augmentation_with_crop=[]

    dataset_dicts = get_detection_dataset_dicts(dname)
    if split > 0:
        size = len(dataset_dicts) // split
        if idx < split - 1:
            dataset_dicts = dataset_dicts[size * idx: size * (idx + 1)]
        else:
            dataset_dicts = dataset_dicts[size * idx:]
    
    return build_detection_test_loader(dataset=dataset_dicts,
                            mapper=DetrDatasetMapper(
            augmentation=augmentation,
            augmentation_with_crop=augmentation_with_crop,
            is_train=True,
            mask_on=True,
            mask_format='bitmask',
            img_format="RGB"
        ) if 'stuff' not in dname else DatasetMapper(
                    augmentations=augmentation,
                    is_train=True,
                    image_format="RGB"
            ), num_workers=4)

def shrink_boxes(boxes, scale=0.7071, height=None, width=None):
    boxes = boxes.clone().float()

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    bw = (boxes[:, 2] - boxes[:, 0]) * scale
    bh = (boxes[:, 3] - boxes[:, 1]) * scale

    boxes[:, 0] = cx - bw / 2
    boxes[:, 2] = cx + bw / 2
    boxes[:, 1] = cy - bh / 2
    boxes[:, 3] = cy + bh / 2

    if width is not None:
        boxes[:, 0].clamp_(0, width)
        boxes[:, 2].clamp_(0, width)
    if height is not None:
        boxes[:, 1].clamp_(0, height)
        boxes[:, 3].clamp_(0, height)

    return boxes

def visualize_mask_overlay(image, mask, bbox=None, alpha=0.4, save_path=None):
    """
    image: tensor (3,H,W)
    mask: tensor (1,H,W) or (H,W)
    bbox: tensor [x1,y1,x2,y2] optional
    """

    img = image.permute(1,2,0).cpu().numpy().astype(np.uint8)
    mask = mask.squeeze().cpu().numpy()

    # 创建红色 overlay
    color = np.zeros_like(img)
    color[:,:,0] = 255  # red

    # mask boolean
    mask_bool = mask > 0.5

    # alpha blending
    img_overlay = img.copy()
    img_overlay[mask_bool] = (
        img[mask_bool] * (1-alpha) + color[mask_bool] * alpha
    )

    # 画 bbox
    if bbox is not None:
        x1,y1,x2,y2 = [int(v) for v in bbox]
        cv2.rectangle(img_overlay, (x1,y1), (x2,y2), (0,255,0), 2)

    img_overlay = img_overlay.astype(np.uint8)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
    else:
        plt.imshow(img_overlay)
        plt.axis("off")
        plt.show()

def main(model='vitl14', dataset='fs_coco17_support_novel_30shot', use_bbox='yes',
            epochs=1, device=0, n_clst=5, split=0, idx=0, out_dir=None, without_mask=False, box_scale=0.6):

    use_bbox = use_bbox == 'yes'
    dataset_name = dataset
    model_name = model
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_' + model)
    model = torch.hub.load(
    '/root/.cache/torch/hub/facebookresearch_dinov2_main',
    'dinov2_' + model,
    source='local'
    )
    dataloader = get_dataloader(dataset_name, split=split, idx=idx)

    D = DatasetCatalog.get(dataset_name)
    thing_cats = {b['category_id'] for a in D for b in a.get('annotations', [])}
    print(f'Found thing categories: {len(thing_cats)}')
    
    if device != 'cpu':
        device = int(device)

    model = model.to(device)

    dataset = {
        'labels': [],
        # 'class_tokens': [],
        'patch_tokens': [],
        'avg_patch_tokens': [], 
        'image_id': [],
        'boxes': [],
        'areas': [],
        'skip': 0
    }

    if 'stuff' in dataset_name: # for background 
        meta = MetadataCatalog.get(dataset_name)
        stuff_id_set = set(meta.stuff_dataset_id_to_contiguous_id.values())
    class2tokens = {}
    class2filename = {}
    class2imgs = {}
    class2bbox = {}
    # class2blur = {}
    with tqdm(total=epochs * len(dataloader)) as bar:
        for _ in range(epochs):
            for item_i, item in enumerate(dataloader):
                item = item[0]
                if 'instances' in item:
                    instances = item['instances'].to(device)
                image = item['image'].clone()
                image14x = resize_to_closest_14x(normalize_image(image)).to(device)
                target_mask_size = image14x.shape[1] // 14, image14x.shape[2] // 14

                r = model.get_intermediate_layers(image14x[None, ...], 
                                        return_class_token=True, reshape=True)    
                patch_tokens = r[0][0][0] # c, h, w

                if 'stuff' in dataset_name:
                    patch_tokens = patch_tokens.flatten(1).permute(1, 0) # h*w, c
                    smask = tvF.resize(item['sem_seg'].float()[None, ...], target_mask_size, interpolation=tvF.InterpolationMode.NEAREST)
                    smask = smask.flatten().long()
                    for sem_id in smask.unique().tolist():
                        if sem_id != 255 and sem_id != 0 and sem_id in stuff_id_set:
                            mask = smask == sem_id
                            if mask.sum() <= 0.5:
                                continue
                            stuff_tokens = patch_tokens[mask]
                            stuff_tokens = compress(stuff_tokens, n_clst)
                            dataset['patch_tokens'].append(stuff_tokens.cpu())
                            dataset['labels'].append(sem_id)
                else:
                    if not without_mask:
                        masks_used = instances.gt_masks.tensor
                        box_h, box_w = masks_used.shape[1], masks_used.shape[2]
                    else:
                        box_h, box_w = instances.image_size
                    if use_bbox:
                        # bbox_masks = to_mask(instances.gt_boxes.tensor,
                        #                     box_h, box_w).to(instances.gt_boxes.tensor.device)

                        # print("yes")
                        shrunk_boxes = shrink_boxes(  # crop
                            instances.gt_boxes.tensor,
                            scale=box_scale,
                            height=box_h,
                            width=box_w
                        )
                        bbox_masks = to_mask(shrunk_boxes, box_h, box_w).to(instances.gt_boxes.tensor.device)

                        masks_used = bbox_masks
                    
                    for bmask, label, bbox in zip(masks_used, instances.gt_classes, 
                                                instances.gt_boxes.tensor):
                        # if item_i < 5:
                        #     visualize_mask_overlay(image, bmask, bbox, save_path=f"mask_base/dataset3_1shot/img_{item_i}.jpg")


                        bmask = bmask.float()[None, ...]
                        bmask = tvF.resize(bmask, target_mask_size)
                        if bmask.sum() <= 0.5:
                            dataset['skip'] += 1
                            continue

                        avg_patch_token = (bmask * patch_tokens).flatten(1).sum(1) / bmask.sum()

                        # mask shape -> [H, W]
                        mask = bmask.squeeze()  # [H, W]

                        # patch_tokens flatten -> [H*W, C]
                        tokens_flat = patch_tokens.flatten(1).permute(1,0)  # [H*W, C]

                        # 只选 mask 内的 token
                        mask_flat = mask.flatten() > 0.5
                        tokens_masked = tokens_flat[mask_flat]  # [num_mask_tokens, C]

                        # topk = 2
                        

                        # num_mask_tokens = tokens_masked.shape[0]
                        # # topk = max(1, int(0.05 * num_mask_tokens))  # 选择 patch 数的 5%
                        # topk = min(2, num_mask_tokens)
                        # # 计算 token norm
                        # token_norm = tokens_masked.norm(dim=-1)  # [num_mask_tokens]
                        # # print(token_norm.shape)
                        # topk_idx = torch.topk(token_norm, topk).indices
                        # prototypes = tokens_masked[topk_idx]  # [topk, C]
                        # prototypes = F.normalize(prototypes, dim=-1)


                        # n_clusters = 5  # 你希望聚成几个 prototype
                        # if tokens_masked.shape[0] <= n_clusters:
                        #     prototypes = tokens_masked  # token 太少就直接用原来的
                        # else:
                        #     # KMeans 聚类
                        #     kmeans = KMeans(n_clusters=n_clusters, verbose=False, mode='cosine')
                        #     kmeans.fit(tokens_masked)
                        #     prototypes = kmeans.centroids  # [n_clusters, C]

                        # # 归一化
                        # prototypes = F.normalize(prototypes, dim=-1)


                        dataset['avg_patch_tokens'].append(avg_patch_token.cpu())
                        dataset['labels'].append(label.cpu().item())
                        bbox = bbox.cpu()
                        dataset['image_id'].append(item['image_id'])
                        if label.cpu().item() not in class2tokens:
                            class2tokens[label.cpu().item()] = []
                            class2filename[label.cpu().item()] = []
                            class2imgs[label.cpu().item()] = []
                            class2bbox[label.cpu().item()] = []

                        class2tokens[label.cpu().item()].append(avg_patch_token)
                        
                        
                        # for proto in prototypes:
                        #     class2tokens[label.item()].append(proto.cpu())

                        class2filename[label.cpu().item()].append(item['file_name'])
                        class2imgs[label.cpu().item()].append(image)

                        class2bbox[label.cpu().item()].append(bbox)


                        # num_proto = prototypes.shape[0]
                        # class2bbox[label.item()].extend([bbox] * num_proto)
                bar.update()
    shot = 0
    for cls in class2tokens:
        if shot != 0 and len(class2tokens[cls]) != shot:
            print(f"################### please check the datasets: shot{shot} with input {len(class2tokens[cls])} ###################")
        shot = max(shot, len(class2tokens[cls]))
    print("shot:", shot)
    for cls in class2tokens:
        class2tokens[cls] = torch.stack(class2tokens[cls])
        if len(class2tokens[cls]) < shot:
            add_num = shot - len(class2tokens[cls])
            add_tensor = class2tokens[cls][-1]
            class2tokens[cls] = torch.cat([class2tokens[cls], add_tensor.unsqueeze(0).expand(add_num, -1)], dim=0)
            for i in range(add_num):
                class2filename[cls].append(class2filename[cls][-1])
                class2imgs[cls].append(class2imgs[cls][-1])
                class2bbox[cls].append(class2bbox[cls][-1])

    classes = sorted(class2tokens.keys())

    prototypes = F.normalize(torch.stack([class2tokens[c] for c in classes]), dim=-1)
    file_names = [class2filename[c] for c in classes]
    images = [class2imgs[c] for c in classes]
    for cls in class2bbox:
        class2bbox[cls] = torch.stack(class2bbox[cls])
    support_bbox = torch.stack([class2bbox[c] for c in classes])
    if dataset_name == "clipart1k_test":
        dataset_name = "clipart1k_TEST"
    if dataset_name == "clipart1k_train":
        dataset_name = "clipart1k_TRAIN"
    label_names = [ALL_CLS_DICT[dataset_name][c] for c in classes]
    if dataset_name == "clipart1k_TEST":
        dataset_name = "clipart1k_test"
    if dataset_name == "clipart1k_TRAIN":
        dataset_name = "clipart1k_train"
    category_dict = {
        'prototypes': prototypes,
        'label_names': label_names,
        'file_names': file_names,
        'images': images,
        'support_bbox': support_bbox
    }

    name = dataset_name + '.' + model_name

    if 'stuff' in dataset_name:
        name += f'.c{n_clst}'

    if split > 0:
        name += f'.s{split}i{idx}'
    
    if use_bbox:
        name += '.bbox'
    ori_proto_name = name + "_ori.pkl"
    name += '.pkl'
    if out_dir is not None:
        name = osp.join(out_dir, name)
        ori_proto_name = osp.join(out_dir, ori_proto_name)
    print(f'Saving to {name} and {ori_proto_name}')
    torch.save(dataset, name)
    torch.save(category_dict, ori_proto_name)



if __name__ == "__main__":
    fire.Fire(main)
