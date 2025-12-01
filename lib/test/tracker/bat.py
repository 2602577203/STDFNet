# import math
# from lib.models.bat import build_batrack
# from lib.test.tracker.basetracker import BaseTracker
# import torch
# from lib.test.tracker.vis_utils import gen_visualization
# from lib.test.utils.hann import hann2d
# from lib.train.data.processing_utils import sample_target
# # for debug
# import cv2
# import os
# import vot
# from lib.test.tracker.data_utils import PreprocessorMM
# from lib.utils.box_ops import clip_box
# from lib.utils.ce_utils import generate_mask_cond


# class BATTrack(BaseTracker):
#     def __init__(self, params):
#         super(BATTrack, self).__init__(params)
#         network = build_batrack(params.cfg, training=False)
#         network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)  
#         self.cfg = params.cfg
#         self.network = network.cuda()
#         self.network.eval()
#         self.preprocessor = PreprocessorMM()
#         self.state = None

#         self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
#         # motion constrain
#         self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

#         # for debug
#         if getattr(params, 'debug', None) is None:
#             setattr(params, 'debug', 0)
#         self.use_visdom = True #params.debug   
#         #self._init_visdom(None, 1)
#         self.debug = params.debug
#         self.frame_id = 0
#         # for save boxes from all queries
#         self.save_all_boxes = params.save_all_boxes

#     def initialize(self, image, info: dict):
#         # forward the template once
#         z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
#                                                     output_sz=self.params.template_size)
#         self.z_patch_arr = z_patch_arr
#         template = self.preprocessor.process(z_patch_arr)
#         with torch.no_grad():
#             self.z_tensor = template

#         self.box_mask_z = None
#         if self.cfg.MODEL.BACKBONE.CE_LOC:
#             template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
#                                                         template.device).squeeze(1)
#             self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)
            
#         self.template_bboxes = template_bbox
#         H,W,_ = z_patch_arr.shape
#         self.template_mask = self.get_bboxes_masks(self.template_bboxes,1,H,W)
#         self.template_bboxes_masks = self.template_mask.view(1, -1)
#         self.template_masks = torch.where(self.template_bboxes_masks.to('cuda:0'), torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0')).to('cuda:0')

#         #初始化in_dict
#         self.in_dict = {
#             "tokens_target_rgb": None,
#             "tokens_target_tir": None,
#             "tokens_background_rgb": None,
#             "tokens_background_tir": None,
#             "dynamic_template": None,
#             "dynamic_template_mask": None,
#         }
#         self.in_dict_new = {
#             "tokens_target_rgb": None,
#             "tokens_target_tir": None,
#             "tokens_background_rgb": None,
#             "tokens_background_tir": None,
#             "dynamic_template": None,
#             "dynamic_template_mask": None,
#         }

#         # save states
#         self.state = info['init_bbox']
#         self.frame_id = 0
#         if self.save_all_boxes:
#             '''save all predicted boxes'''
#             all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
#             return {"all_boxes": all_boxes_save}

#     def track(self, image, info: dict = None):
#         H, W, _ = image.shape
#         self.frame_id += 1
#         x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
#                                                                 output_sz=self.params.search_size)  # (x1, y1, w, h)
#         search = self.preprocessor.process(x_patch_arr)

#         with torch.no_grad():
#             x_tensor = search
#             # merge the template and the search
#             # run the transformer
#             out_dict = self.network.forward(
#                 template=self.z_tensor, search=x_tensor,in_dict=self.in_dict, ce_template_mask=self.template_masks)
#             # #使用新的in_dict
#             # out_dict_new = self.network.forward(
#             #     template=self.z_tensor, search=x_tensor,in_dict=self.in_dict_new, ce_template_mask=self.template_masks)

#         # add hann windows
#         pred_score_map = out_dict['score_map']
#         response = self.output_window * pred_score_map
#         pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
#         max_score = best_score[0][0].item()
#         pred_boxes = pred_boxes.view(-1, 4)
#         # Baseline: Take the mean of all pred boxes as the final result
#         pred_box = (pred_boxes.mean(
#             dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
#         # get the final box result
#         self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
#         #计算out_dict_new的best_score和bbox
#         pred_score_map = out_dict_new['score_map']
#         response = self.output_window * pred_score_map
#         pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict_new['size_map'], out_dict_new['offset_map'], return_score=True)
#         max_score_new = best_score[0][0].item()
#         pred_boxes = pred_boxes.view(-1, 4)
#         # Baseline: Take the mean of all pred boxes as the final result
#         pred_box = (pred_boxes.mean(
#             dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
#         # get the final box result
#         bbox_new = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
#         #对比并更新
#         if max_score_new > max_score:
#             self.in_dict = self.in_dict_new
#             self.state = bbox_new
#             max_score = max_score_new
#             out_dict = out_dict_new

#         #保存更新的in_dict
#         #获取搜索区域mask
#         search_bbox = self.transform_bbox_to_crop(self.state, resize_factor, search.device).squeeze(1)
#         H1,W1,_ = x_patch_arr.shape
#         mask_search = self.get_bboxes_masks(search_bbox,1,H1,W1)
#         mask_search = mask_search.view(1, -1)
#         mask_search = torch.where(mask_search.to('cuda:0'), torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0')).to('cuda:0')
#         #提取前景背景token
#         self.in_dict_new["tokens_target_rgb"] = out_dict["search_tokens_rgb"][0][mask_search[0].unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#         self.in_dict_new["tokens_target_tir"] = out_dict["search_tokens_tir"][0][mask_search[0].unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#         self.in_dict_new["tokens_background_rgb"] = out_dict["search_tokens_rgb"][0][(1-mask_search)[0].unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#         self.in_dict_new["tokens_background_tir"] = out_dict["search_tokens_tir"][0][(1-mask_search)[0].unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#         #获取动态模板
#         z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
#         self.z_patch_arr = z_patch_arr
#         self.in_dict_new["dynamic_template"] = self.preprocessor.process(z_patch_arr)
#         #获取动态模板mask
#         H2,W2,_ = z_patch_arr.shape
#         mask_template = self.get_bboxes_masks(search_bbox,1,H2,W2)
#         mask_template = mask_template.view(1, -1)
#         mask_template = torch.where(mask_template.to('cuda:0'), torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0')).to('cuda:0')
#         self.in_dict_new["dynamic_template_mask"] = mask_template

#         #self.debug = 1
        
#         # for debug
#         if self.debug == 1:
#             x1, y1, w, h = self.state
#             image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
#             cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
#             cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1,
#                             (0, 255, 255), 2)
#             cv2.imshow('debug_vis', image_BGR)
#             cv2.waitKey(1)


#         if self.save_all_boxes:
#             '''save all predictions'''
#             all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
#             all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
#             return {"target_bbox": self.state,
#                     "all_boxes": all_boxes_save,
#                     "best_score": max_score}
#         else:
#             return {"target_bbox": self.state,
#                     "best_score": max_score}

#     def map_box_back(self, pred_box: list, resize_factor: float):
#         cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
#         cx, cy, w, h = pred_box
#         half_side = 0.5 * self.params.search_size / resize_factor
#         cx_real = cx + (cx_prev - half_side)
#         cy_real = cy + (cy_prev - half_side)
#         return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

#     def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
#         cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
#         cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
#         half_side = 0.5 * self.params.search_size / resize_factor
#         cx_real = cx + (cx_prev - half_side)
#         cy_real = cy + (cy_prev - half_side)
#         return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    
#     def get_bboxes_masks(self, bboxes, B, H, W, patch_size=16):
#         # 计算grid大小
#         grid_size = H // patch_size

#         # 初始化mask，大小为[B, grid_size, grid_size]
#         bboxes_masks = torch.zeros(B, grid_size, grid_size, dtype=torch.bool, device=bboxes.device)

#         # 获取所有bbox的归一化坐标
#         x1, y1, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

#         # 将归一化坐标转换为像素坐标
#         x1_pixel = x1 * W
#         y1_pixel = y1 * H
#         w_pixel = w * W
#         h_pixel = h * H
#         x2_pixel = x1_pixel + w_pixel
#         y2_pixel = y1_pixel + h_pixel

#         # 计算patch的索引
#         patch_x1 = (x1_pixel // patch_size).long()
#         patch_y1 = (y1_pixel // patch_size).long()
#         patch_x2 = (x2_pixel // patch_size).long()
#         patch_y2 = (y2_pixel // patch_size).long()

#         # 手动限制索引在有效范围内 (0 - grid_size - 1)
#         patch_x1 = torch.clamp(patch_x1, 0, grid_size - 1)
#         patch_y1 = torch.clamp(patch_y1, 0, grid_size - 1)
#         patch_x2 = torch.clamp(patch_x2, 0, grid_size - 1)
#         patch_y2 = torch.clamp(patch_y2, 0, grid_size - 1)

#         # 使用广播将bbox位置标记在mask中
#         for b in range(B):
#             bboxes_masks[b, patch_y1[b]:patch_y2[b] + 1, patch_x1[b]:patch_x2[b] + 1] = True

#         return bboxes_masks 


# def get_tracker_class():
#     return BATTrack




import math
from lib.models.bat import build_batrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import vot
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

from lib.utils.box_ops import giou_loss


class BATTrack(BaseTracker):
    def __init__(self, params):
        super(BATTrack, self).__init__(params)
        network = build_batrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)  
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = True #params.debug   
        #self._init_visdom(None, 1)
        self.debug = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)
            
        self.template_bboxes = template_bbox
        H,W,_ = z_patch_arr.shape
        self.template_mask = self.get_bboxes_masks(self.template_bboxes,1,H,W)
        self.template_bboxes_masks = self.template_mask.view(1, -1)
        self.template_masks = torch.where(self.template_bboxes_masks.to('cuda:0'), torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0')).to('cuda:0')

        #初始化in_dict
        self.in_dict = {
            "tokens_target_rgb": None,
            "tokens_target_tir": None,
            "tokens_background_rgb": None,
            "tokens_background_tir": None,
            "dynamic_template": None,
            "dynamic_template_mask": None,
        }

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None, vis_feat = False):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        # from sklearn.decomposition import PCA
        # import numpy as np
        
        # h, w, c = self.z_patch_arr.shape
        # # rgb_feature = np.mean(self.z_patch_arr[:,:,:3], axis=-1)  # RGB特征（均值）：(256,256)
        # # tir_feature = np.mean(self.z_patch_arr[:,:,3:], axis=-1)  # TIR特征（均值）：(256,256)
        # concatenated = np.column_stack((self.z_patch_arr[:,:,:3].reshape(-1), self.z_patch_arr[:,:,3:].reshape(-1)))
        
        
        

        # # 2.3 构建样本-特征矩阵：(65536, 2)
        # # X = self.z_patch_arr.reshape(-1, 6)

        # # 3. 应用PCA（降维到2维，若需更低维度可调整n_components）
        # pca = PCA(n_components=1)  # 此处保持2维，若需压缩可设为1
        # X_pca = pca.fit_transform(concatenated)  # 降维后：(65536, 2)

        # # 4. 重塑回图像形状（可视化或后续处理）
        # pca_image = X_pca.reshape(h, w, -1)  # 形状：(256,256,2)
        
        
        # # concat = self.z_patch_arr.reshape(-1, 3).reshape(h, w, 3, 2).reshape(-1, 2)
        # # pca = PCA(n_components=2)
        # # reduced = pca.fit_transform(concat).reshape(h, w, 3, 2).reshape(h, w, 6)
        # from torchvision.transforms import ToPILImage
        # import numpy as np
        # # 转换为PIL图像
        # pil_image = ToPILImage()(self.z_patch_arr[:,:,:3])
        # pil_image.save('rgb.jpg')
        # pil_image = ToPILImage()(self.z_patch_arr[:,:,3:])
        # pil_image.save('tir.jpg')
        # pil_image = ToPILImage()(pca_image[:,:,:1].astype(np.uint8))
        # pil_image.save('reduced_rgb.jpg')
        # pil_image = ToPILImage()(pca_image[:,:,1:].astype(np.uint8))
        # pil_image.save('reduced_tir.jpg')
        
        #保存初始bbox和搜索区域
        if self.frame_id == 1:
            self.init_bbox = self.state
            self.init_s = search
            self.init_in_dict = self.in_dict

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor,in_dict=self.in_dict, ce_template_mask=self.template_masks)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        #获取候选模板
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        candidate_template = self.preprocessor.process(z_patch_arr)
        #与第一帧搜索区域进行一次跟踪以获取iou
        out_iou_compute = self.network.forward(
            template=candidate_template, search=self.init_s,in_dict=self.init_in_dict, ce_template_mask=self.template_masks)

        # add hann windows
        pred_score_map1 = out_iou_compute['score_map']
        response1 = self.output_window * pred_score_map1
        pred_boxes1, best_score1 = self.network.box_head.cal_bbox(response1, out_iou_compute['size_map'], out_iou_compute['offset_map'], return_score=True)
        max_score1 = best_score1[0][0].item()
        pred_boxes1 = pred_boxes1.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box1 = (pred_boxes1.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        bbox = clip_box(self.map_box_back(pred_box1, resize_factor), H, W, margin=10)

        iou = self.compute_iou_xywh(bbox, self.init_bbox)
        if max_score >= 0.7:
            #获取搜索区域mask
            search_bbox = self.transform_bbox_to_crop(self.state, resize_factor, search.device).squeeze(1)
            H1,W1,_ = x_patch_arr.shape
            mask_search = self.get_bboxes_masks(search_bbox,1,H1,W1)
            mask_search = mask_search.view(1, -1)
            mask_search = torch.where(mask_search.to('cuda:0'), torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0')).to('cuda:0')
            #提取前景背景token
            self.in_dict["tokens_target_rgb"] = out_dict["search_tokens_rgb"][0][mask_search[0].unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
            self.in_dict["tokens_target_tir"] = out_dict["search_tokens_tir"][0][mask_search[0].unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
            self.in_dict["tokens_background_rgb"] = out_dict["search_tokens_rgb"][0][(1-mask_search)[0].unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
            self.in_dict["tokens_background_tir"] = out_dict["search_tokens_tir"][0][(1-mask_search)[0].unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
            #获取动态模板
            z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
            self.z_patch_arr = z_patch_arr
            self.in_dict["dynamic_template"] = self.preprocessor.process(z_patch_arr)
            
            
            # from torchvision.transforms import ToPILImage
            # import numpy as np
            # # 转换为PIL图像
            # pil_image = ToPILImage()(z_patch_arr[:,:,:3])
            # pil_image.save('output.jpg')
            
            #获取动态模板mask
            H2,W2,_ = z_patch_arr.shape
            mask_template = self.get_bboxes_masks(search_bbox,1,H2,W2)
            mask_template = mask_template.view(1, -1)
            mask_template = torch.where(mask_template.to('cuda:0'), torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0')).to('cuda:0')
            self.in_dict["dynamic_template_mask"] = mask_template

        #self.debug = 1
        
        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)


        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        elif vis_feat:
            return [search[:, :3, :, :], search[:, 3:, :, :]], out_dict
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    
    def get_bboxes_masks(self, bboxes, B, H, W, patch_size=16):
        # 计算grid大小
        grid_size = H // patch_size

        # 初始化mask，大小为[B, grid_size, grid_size]
        bboxes_masks = torch.zeros(B, grid_size, grid_size, dtype=torch.bool, device=bboxes.device)

        # 获取所有bbox的归一化坐标
        x1, y1, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        # 将归一化坐标转换为像素坐标
        x1_pixel = x1 * W
        y1_pixel = y1 * H
        w_pixel = w * W
        h_pixel = h * H
        x2_pixel = x1_pixel + w_pixel
        y2_pixel = y1_pixel + h_pixel

        # 计算patch的索引
        patch_x1 = (x1_pixel // patch_size).long()
        patch_y1 = (y1_pixel // patch_size).long()
        patch_x2 = (x2_pixel // patch_size).long()
        patch_y2 = (y2_pixel // patch_size).long()

        # 手动限制索引在有效范围内 (0 - grid_size - 1)
        patch_x1 = torch.clamp(patch_x1, 0, grid_size - 1)
        patch_y1 = torch.clamp(patch_y1, 0, grid_size - 1)
        patch_x2 = torch.clamp(patch_x2, 0, grid_size - 1)
        patch_y2 = torch.clamp(patch_y2, 0, grid_size - 1)

        # 使用广播将bbox位置标记在mask中
        for b in range(B):
            bboxes_masks[b, patch_y1[b]:patch_y2[b] + 1, patch_x1[b]:patch_x2[b] + 1] = True

        return bboxes_masks 
    
    #用大模型写的计算iou代码
    def xywh_to_xyxy(self,bbox):
        """
        将 (x, y, w, h) 格式的边界框转换为 (x1, y1, x2, y2) 格式。
        参数:
            bbox: [x, y, w, h]
        返回:
            [x1, y1, x2, y2]
        """
        x, y, w, h = bbox
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return [x1, y1, x2, y2]


    def compute_iou(self,bbox1, bbox2):
        """
        计算两个边界框的 IoU。
        参数:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
        返回:
            IoU 值
        """
        # 计算交集的坐标
        xx1 = max(bbox1[0], bbox2[0])
        yy1 = max(bbox1[1], bbox2[1])
        xx2 = min(bbox1[2], bbox2[2])
        yy2 = min(bbox1[3], bbox2[3])

        # 计算交集的宽度和高度
        inter_width = max(0, xx2 - xx1)
        inter_height = max(0, yy2 - yy1)

        # 计算交集面积
        inter_area = inter_width * inter_height

        # 计算两个边界框的面积
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # 计算并集面积
        union_area = area1 + area2 - inter_area

        # 计算 IoU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou


    def compute_iou_xywh(self,bbox1_xywh, bbox2_xywh):
        """
        计算两个以 (x, y, w, h) 格式表示的边界框的 IoU。
        参数:
            bbox1_xywh: [x, y, w, h]
            bbox2_xywh: [x, y, w, h]
        返回:
            IoU 值
        """
        # 转换为 (x1, y1, x2, y2) 格式
        bbox1_xyxy = self.xywh_to_xyxy(bbox1_xywh)
        bbox2_xyxy = self.xywh_to_xyxy(bbox2_xywh)

        # 计算 IoU
        iou = self.compute_iou(bbox1_xyxy, bbox2_xyxy)
        return iou


    # # 示例
    # bbox1_xywh = [50, 50, 40, 30]  # (x, y, w, h)
    # bbox2_xywh = [60, 60, 50, 40]  # (x, y, w, h)

    # iou = compute_iou_xywh(bbox1_xywh, bbox2_xywh)
    # print("IoU:", iou)


def get_tracker_class():
    return BATTrack
