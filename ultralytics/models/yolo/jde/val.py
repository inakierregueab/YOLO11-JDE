# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path, PosixPath

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import DetMetrics, box_iou, ReIDMetrics
from ultralytics.utils.torch_utils import smart_inference_mode


class JDEValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a joint detection and embedding model.

    Example:
        ```python
        from ultralytics.models.yolo.jde import JDEValidator

        args = dict(model="yolov8n-jde.pt", data="coco8-seg.yaml")
        validator = JDEValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "jde"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.reid_metrics = ReIDMetrics()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Performs validation on the model and sets the epoch and best attributes."""
        if trainer is not None:
            self.epoch = trainer.epoch + 1
            self.best = trainer.best
        if model is not None:
            self.model_path = model if (isinstance(model, str) or isinstance(model, PosixPath)) else model.pt_path
        stats = super().__call__(trainer, model)
        return stats

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        tags = batch["tags"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "tags": tags}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"])  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        batch_matched_tags = []     # List to store matched tags for each batch
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            matched_tags = torch.zeros(npr, dtype=torch.int, device=self.device)    # Initialize matched tags tensor
            pbatch = self._prepare_batch(si, batch)
            cls, bbox, tags = pbatch.pop("cls"), pbatch.pop("bbox"), pbatch.pop("tags")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"], matched_tags = self._process_batch(predn, bbox, cls, tags)  # correct, matched_tags
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])
            batch_matched_tags.append(matched_tags)   # Append matched tags to list

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )
        # Process batch for reid metrics
        self.reid_metrics.process_batch(preds, batch_matched_tags)

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
            reid_metrics = self.reid_metrics.get_metrics()
        detector_results = self.metrics.results_dict
        detector_results.update(reid_metrics)
        return detector_results

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["tags"] = batch["tags"].to(self.device).float()
        return batch

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )

    def _process_batch(self, detections, gt_bboxes, gt_cls, gt_tags):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.
            gt_tags (torch.Tensor): Tensor of shape (M,) representing target tags.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, gt_tags, iou)

    def match_predictions(self, pred_classes, true_classes, true_tags, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            true_tags (torch.Tensor): Target tags of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Initialize the list for storing matched tags using IoU threshold of 0.5
        matched_tags = [False] * pred_classes.shape[0]  # Default to None if no match

        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
                        # Assign tags to matched predictions
                        if threshold == 0.5:
                            for gt_idx, pred_idx in zip(labels_idx[valid], detections_idx[valid]):
                                matched_tags[pred_idx] = true_tags[gt_idx].item()
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
                    # Assign tags to matched predictions
                    if threshold == 0.5:
                        for gt_idx, pred_idx in matches:
                            matched_tags[pred_idx] = true_tags[gt_idx].item()
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device), torch.tensor(matched_tags, dtype=torch.int, device=pred_classes.device)