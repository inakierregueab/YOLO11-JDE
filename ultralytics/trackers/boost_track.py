"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import os
from copy import deepcopy
from typing import Optional, List

import cv2
import numpy as np

from .utils.boost_track.assoc import associate, iou_batch, MhDist_similarity, shape_similarity, soft_biou_batch
from .utils.boost_track.ecc import ECC
from .utils.boost_track.kalmanfilter import KalmanFilter


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,h,r] where x,y is the centre of the box and h is the height and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0

    r = w / float(h + 1e-6)

    return np.array([x, y, h, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,h,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """

    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h

    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, emb: Optional[np.ndarray] = None, idx: Optional[int] = None):
        """
        Initialises a tracker using initial bounding box.
        """

        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.emb = emb
        self.hit_streak = 0
        self.age = 0

        self.idx = idx

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7

        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update-1)

    def update(self, bbox: np.ndarray, score: float = 0, idx: Optional[int] = None):
        """
        Updates the state vector with observed bbox.
        """

        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)
        self.idx = idx

    def camera_update(self, transform: np.ndarray):
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = transform @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = transform @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h,  w / h]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb


class BoostTrack(object):
    def __init__(self, args, frame_rate=30):
        # TODO: inherit frame rate from the video or config
        self.frame_count = 0
        self.trackers: List[KalmanBoxTracker] = []

        # General tracking parameters
        self.max_age = max(int(frame_rate * 2), args.max_age)
        self.iou_threshold = args.iou_thresh
        self.det_thresh = args.det_thresh
        self.min_hits = args.min_hits

        # BoostTrack parameters
        self.lambda_iou = args.lambda_iou
        self.lambda_mhd = args.lambda_mhd
        self.lambda_shape = args.lambda_shape
        self.use_dlo_boost = args.use_dlo_boost
        self.use_duo_boost = args.use_duo_boost
        self.dlo_boost_coef = args.dlo_boost_coef
        self.s_sim_corr = args.s_sim_corr

        # BoostTrack++ parameters
        self.use_rich_s = args.use_rich_s
        self.use_sb = args.use_sb
        self.use_vt = args.use_vt

        self.with_reid = args.with_reid
        self.ecc = ECC(scale=350, video_name=None, use_cache=True) if args.use_ecc else None

    def update(self, boxes, img_numpy, features=None, tag=None):
        """
        Args:
            boxes: torch.Tensor, shape=(N, 6), where N is the number of detected objects.
            img_numpy: np.ndarray, shape=(H, W, 3), where H and W are the height and width of the image.
            tag: str, the tag of the image.
            features: np.ndarray, shape=(N, D), where D is the dimension of the feature.
        Returns:
            np.ndarray, shape=(M, 5), where M is the number of tracked objects.
        """
        self.frame_count += 1

        # Extract detections and generate idxs
        dets = deepcopy(boxes.data[:, :5])  # x1, y1, x2, y2, score
        idxs = np.arange(len(dets)).reshape(-1, 1)

        # Apply ECC
        if self.ecc is not None and img_numpy is not None:
            transform = self.ecc(img_numpy, self.frame_count, tag)
            for trk in self.trackers:
                trk.camera_update(transform)

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        confs = np.zeros((len(self.trackers), 1))

        for t in range(len(trks)):
            pos = self.trackers[t].predict()[0]
            confs[t] = self.trackers[t].get_confidence()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], confs[t, 0]]

        # Apply boosters
        if self.use_dlo_boost:
            dets = self.dlo_confidence_boost(dets, self.use_rich_s, self.use_sb, self.use_vt)

        if self.use_duo_boost:
            dets = self.duo_confidence_boost(dets)

        # Filter out low confidence detections
        remain_inds = dets[:, 4] >= self.det_thresh
        dets = dets[remain_inds]
        scores = dets[:, 4]
        idxs = idxs[remain_inds]

        # Generate embeddings
        emb_cost = None
        if self.with_reid:
            dets_embs = self.normalize(features[remain_inds])
            trk_embs = []
            for t in range(len(self.trackers)):
                trk_embs.append(self.trackers[t].get_emb())
            trk_embs = np.array(trk_embs)
            if trk_embs.size > 0 and dets.size > 0:
                emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ trk_embs.reshape((trk_embs.shape[0], -1)).T

        # Data association
        matched, unmatched_dets, unmatched_trks, sym_matrix = associate(
            dets,
            trks,
            self.iou_threshold,
            mahalanobis_distance=self.get_mh_dist_matrix(dets),
            track_confidence=confs,
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape,
            s_sim_corr=self.s_sim_corr
        )

        # Update embeddings' alpha factor
        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = 0.95
        dets_alpha = af + (1 - af) * (1 - trust)

        # Update trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], scores[m[0]], idxs[m[0]])
            if self.with_reid:
                self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            if dets[i, 4] >= self.det_thresh:
                if self.with_reid:
                    self.trackers.append(KalmanBoxTracker(dets[i, :], emb=dets_embs[i], idx=idxs[i]))
                else:
                    self.trackers.append(KalmanBoxTracker(dets[i, :], idx=idxs[i]))

        # Build the results and remove dead tracklets
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                # TODO: instead of trk.get_confidence(), use the confidence from the detection
                ret.append(np.concatenate((d, [trk.id + 1], [boxes.conf[trk.idx[0]]], [0], [trk.idx[0]])).reshape(1, -1))
            i -= 1
            # remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(ret) if len(ret) > 0 else np.empty((0, 5))

    @staticmethod
    def normalize(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()

    def get_iou_matrix(self, detections: np.ndarray, buffered: bool = False) -> np.ndarray:
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].get_confidence()]

        return iou_batch(detections, trackers) if not buffered else soft_biou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray:
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = self.trackers[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1, ))[:n_dims]
        for i in range(len(self.trackers)):
            x[i] = self.trackers[i].kf.x[:n_dims]
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(np.diag(self.trackers[i].kf.covariance[:n_dims, :n_dims]))

        return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        n_dims = 4
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections, n_dims)

        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)

            mask = (min_mh_dists > limit) & (detections[:, 4] < self.det_thresh)
            boost_detections = detections[mask]
            boost_detections_args = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections, boost_detections) - np.eye(len(boost_detections))
                bdiou_max = bdiou.max(axis=1)

                remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
                args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for i in range(len(args)):
                    boxi = args[i]
                    tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                    args_tmp = np.append(np.intersect1d(boost_detections_args[args], boost_detections_args[tmp]), boost_detections_args[boxi])

                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_detections_args[boxi], 4] == conf_max:
                        remaining_boxes = np.array(remaining_boxes.tolist() + [boost_detections_args[boxi]])

                mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                mask[remaining_boxes] = True

            detections[:, 4] = np.where(mask, self.det_thresh + 1e-4, detections[:, 4])

        return detections

    def dlo_confidence_boost(self, detections: np.ndarray, use_rich_sim: bool, use_soft_boost: bool, use_varying_th: bool) -> np.ndarray:
        sbiou_matrix = self.get_iou_matrix(detections, True)
        if sbiou_matrix.size == 0:
            return detections
        trackers = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, self.trackers[t].time_since_update - 1]

        if use_rich_sim:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections), 1)
            shape_sim = shape_similarity(detections, trackers, self.s_sim_corr)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False)

        if not use_soft_boost and not use_varying_th:
            max_s = S.max(1)
            coef = self.dlo_boost_coef
            detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

        else:
            if use_soft_boost:
                max_s = S.max(1)
                alpha = 0.65
                detections[:, 4] = np.maximum(detections[:, 4], alpha*detections[:, 4] + (1-alpha)*max_s**(1.5))
            if use_varying_th:
                threshold_s = 0.95
                threshold_e = 0.8
                n_steps = 20
                alpha = (threshold_s - threshold_e) / n_steps
                tmp = (S > np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)).max(1)
                scores = deepcopy(detections[:, 4])
                scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)

                detections[:, 4] = scores

        return detections