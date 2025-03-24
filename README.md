# YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID

We introduce YOLO11-JDE, a fast and accurate multi-object tracking (MOT) solution that combines real-time object detection with self-supervised Re-Identification (Re-ID). By incorporating a dedicated Re-ID branch into YOLO11s, our model performs Joint Detection and Embedding (JDE), generating appearance features for each detection. The Re-ID branch is trained in a fully self-supervised setting while simultaneously training for detection, eliminating the need for costly identity-labeled datasets. The triplet loss, with hard positive and semi-hard negative mining strategies, is used for learning discriminative embeddings. Data association is enhanced with a custom tracking implementation that successfully integrates motion, appearance, and location cues. YOLO11-JDE achieves competitive results on MOT17 and MOT20 benchmarks, surpassing existing JDE methods in terms of FPS and using up to ten times fewer parameters. Thus, making our method a highly attractive solution for real-world applications.

**Note:** This paper has been accepted for presentation at the 5th Real-World Surveillance: Applications and Challenges workshop at WACV 2025. Read the full paper on [arXiv](https://arxiv.org/abs/2501.13710v1).

---

## Key Features

- **Real-Time Performance**: Achieves competitive FPS rates while maintaining high tracking accuracy on MOT17 and MOT20 benchmarks.
- **Self-Supervised Re-ID Training**: Eliminates the need for costly identity-labeled datasets through Mosaic data augmentation and triplet loss with hard and semi-hard mining strategies.
- **Custom Data Association**: Integrates motion, appearance, and location cues for enhanced object tracking, including robust handling of occlusions.
- **Lightweight Architecture**: Uses up to 10x fewer parameters than other JDE methods, making it efficient and scalable for diverse applications.

---

## Dataset Information

The datasets used for training YOLO11-JDE are:

1. **CrowdHuman**  
   - **Description**: Contains a wide range of crowded scenes with rich annotations.
   - **Download**: Please download from the official website: [https://www.crowdhuman.org](https://www.crowdhuman.org).
   - **Comments**: The original training and validation splits are preserved.

2. **MOT17**  
   - **Description**: Provides sequences for multiple object tracking. Only bounding box annotations are used for training (track IDs are only used for validation).  
   - **Download**: Please download from the official website: [https://motchallenge.net/data/MOT17/](https://motchallenge.net/data/MOT17/).
   - **Comments**: Following previous work (e.g., *Towards Real-Time Multi-Object Tracking* and *Boost-track: boosting the similarity measure and detection confidence for improved multiple object tracking*), we construct a validation set by using the second half of each training sequence and removing videos in ETH that overlap with the MOT16 benchmark.
  
Both datasets must be converted to [YOLO format](https://docs.ultralytics.com/datasets/detect/). For validation and testing in the re-identification task, an extra column containing the original MOT17 track ID must be added to the dataset files. This column is unnecessary during training unless weak supervision with a few re-ID labels is desired. The format follows the original YOLO structure, with the extra track ID appended only for MOT17 validation and test sets, as CrowdHuman lacks tracking IDs.

The config file used is `crowdhuman.yaml` which should be mapped to a folder containing both datasets merged.

---

## Download Model Weights

Pre-trained model weights for YOLO11s-JDE are available for [download](https://drive.google.com/drive/folders/16btXRPikwXOsaItn06p4A8cBhTiIJdoZ?usp=share_link).

---

## Results

### Benchmarks
**MOT17** and **MOT20** results under private detection protocols:

| Metric   | MOT17 | MOT20 |
|----------|-------|-------|
| HOTA     | 56.6  | 53.1  |
| MOTA     | 65.8  | 70.9  |
| IDF1     | 70.3  | 66.4  |
| FPS      | 35.9  | 18.9  |

Compared to state-of-the-art methods, YOLO11-JDE offers superior FPS and competitive tracking accuracy with significantly fewer parameters.

---

## Enhancing Performance in Custom Scenarios

To improve performance in custom scenarios, consider the following approaches:

1. **Hyperparameter tuning** – Adjust the tracker settings in `ultralytics/cfg/trackers/yolojdetracker.yaml`, keeping in mind that the default values are optimized for MOT17 but generalize to MOT20 as well.  
2. **Reducing domain drift** – Integrate reference images from your dataset into the training set using only bounding box annotations to improve adaptation.  
3. **Weakly supervised learning** – If track information is available in your dataset, include both bounding box and track ID annotations in the training set.  
4. **Fine-tuning the tracker** – If track information is available in your dataset, optimize the tracker's hyperparameters using `tracker/finetune/evolve.py` for a better fit to your camara setup.

---

## Acknowledgements

This work was partially supported by:
- The Spanish project PID2022-136436NB-I00.
- ICREA under the ICREA Academia programme.
- The Milestone Research Program at the University of Barcelona.

The code for YOLO11-JDE is based on the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) repository, which provides a robust foundation for real-time object detection models.


---

## Citation

If you find YOLO11-JDE useful in your research or applications, please cite our paper:

```bibtex
@InProceedings{Erregue_2025_WACV,
    author    = {Erregue, I\~naki and Nasrollahi, Kamal and Escalera, Sergio},
    title     = {YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {February},
    year      = {2025},
    pages     = {824-833}
}
```
