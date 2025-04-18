from groundingdino.util import box_ops
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
import torch


def find_coco_id(coco_cat: dict, name: str):
    for _, cat_info in coco_cat.items():
        if cat_info["name"].lower() == name.lower():
            return cat_info["id"]
    return 0


def create_caption_from_labels(id2name, labels):
    cat_names = [id2name[l] for l in labels]
    cat_list = sorted(list(set(cat_names)))  # 중복 제거 및 정렬
    return " . ".join(cat_list) + " .", cat_list


class PostProcessCoco(torch.nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, cat_lists=None, cats2id_dict=None, tokenlizer=None, train_mode=True) -> None:
        super().__init__()
        self.num_select = num_select
        self.train_mode = train_mode

        assert cat_lists is not None
        new_pos_map_list = []
        for cat_list in cat_lists:
            captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
            tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
            positive_map = create_positive_map_from_span(
                tokenlizer(captions), tokenspanlist)  # 80, 256. normed

            # id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
            #           41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
            #           , 80:91}
            id_map = {}
            for i, c in enumerate(cat_list):
                id_map[i] = find_coco_id(cats2id_dict, c)

            # build a mapping from label_id to pos_map
            new_pos_map = torch.zeros((92, 256))
            for k, v in id_map.items():
                pos_org = positive_map[k]
                new_pos_map[v] = pos_org
            new_pos_map_list.append(new_pos_map)

        self.positive_maps = torch.stack(new_pos_map_list, dim=0)

    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 900, 256
        pos_maps = self.positive_maps.to(prob_to_token.device)

        # (bs, 900, 256) @ (cls, 256).T -> (bs, 900, cls)
        prob_to_label = prob_to_token @ pos_maps.transpose(1, 2)

        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), 300, dim=1) # top 300 in (900 * cls = ex: 81900)
        scores = topk_values

        topk_boxes = topk_indexes // prob.shape[2] # 0~899
        labels = topk_indexes % prob.shape[2] # 0~cls == argmax
        # 300 개 뽑는데, 같은 박스 내에서 logit 큰게 여러개 있으면 중복선정도 가능
        # boxes 는 복사해서 생성되므로 별개의 prediction 으로 여김
        # 따라서 topk_prob 의 argmax != labels
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        topk_prob = torch.gather(
            prob, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 92))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]


        results = {'scores': scores, 'labels': labels, 'boxes': boxes, 'prob': topk_prob}

        return results