# [YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID](https://arxiv.org/abs/2501.13710v1)

We introduce YOLO11-JDE, a fast and accurate multi-object tracking (MOT) solution that combines real-time object detection with self-supervised Re-Identification (Re-ID). By incorporating a dedicated Re-ID branch into YOLO11s, our model performs Joint Detection and Embedding (JDE), generating appearance features for each detection. The Re-ID branch is trained in a fully self-supervised setting while simultaneously training for detection, eliminating the need for costly identity-labeled datasets. The triplet loss, with hard positive and semi-hard negative mining strategies, is used for learning discriminative embeddings. Data association is enhanced with a custom tracking implementation that successfully integrates motion, appearance, and location cues. YOLO11-JDE achieves competitive results on MOT17 and MOT20 benchmarks, surpassing existing JDE methods in terms of FPS and using up to ten times fewer parameters. Thus, making our method a highly attractive solution for real-world applications.

**Note:** This paper has been accepted for presentation at the 5th Real-World Surveillance: Applications and Challenges workshop at WACV 2025.

---

## Key Features

- **Real-Time Performance**: Achieves competitive FPS rates while maintaining high tracking accuracy on MOT17 and MOT20 benchmarks.
- **Self-Supervised Re-ID Training**: Eliminates the need for costly identity-labeled datasets through Mosaic data augmentation and triplet loss with hard and semi-hard mining strategies.
- **Custom Data Association**: Integrates motion, appearance, and location cues for enhanced object tracking, including robust handling of occlusions.
- **Lightweight Architecture**: Uses up to 10x fewer parameters than other JDE methods, making it efficient and scalable for diverse applications.

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
@misc{erregue2025yolo11jdefastaccuratemultiobject,
      title={YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID}, 
      author={Iñaki Erregue and Kamal Nasrollahi and Sergio Escalera},
      year={2025},
      eprint={2501.13710},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.13710}, 
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
