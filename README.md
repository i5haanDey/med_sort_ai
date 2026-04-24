# MedSort AI

> A Browser-Based Medical Waste Classification and Sorting Simulation Using YOLOv8 and ONNX Runtime Web

**Vellore Institute of Technology, Vellore — Tamil Nadu, India**

---

## Authors

| Name | Registration Number |
|---|---|
| Ishaan Dey | 25BIT0169 |
| Trishir Saxena | 25BME0478 |
| Bharatvaj J | 25BAI0221 |
| Anushka Varshney | 25BCE0343 |
| Sai Sabarish Vurimi | 25BCE2249 |

---

## Overview

MedSort AI is a fully client-side web application that classifies medical waste images into four categories — **General**, **Soiled**, **Contaminated**, and **Sharps** — using a fine-tuned YOLOv8 model exported to ONNX format and executed via ONNX Runtime Web.

The system requires no server infrastructure. A rotating-plate animation simulates the physical sorting mechanism, while real-time visual feedback and session statistics are updated dynamically after each classification.

---

## Waste Categories

| Class | Description |
|---|---|
| General | General non-hazardous waste |
| Soiled | Items that have been used |
| Contaminated | Items that have germs on them |
| Sharps | Objects that can pierce skin (e.g. syringes, scalpels) |

---

## Tech Stack

- **React.js** — Frontend UI with component-based architecture
- **ONNX Runtime Web** — In-browser model inference via WebAssembly / WebGL
- **JavaScript (ES6+)** — Application logic, preprocessing, and state management
- **HTML5 Canvas API** — Image rendering and pixel extraction
- **CSS3 Animations** — Rotating plate and bin ejection simulation

---

## How It Works

1. **Upload** — User uploads a waste item image via a drag-and-drop interface.
2. **Preprocess** — Image is resized to 224×224px, normalized to [0, 1], and converted to a Float32 tensor.
3. **Infer** — ONNX Runtime Web runs the YOLOv8 classification model entirely in the browser (typically under 500 ms).
4. **Classify** — The predicted class is mapped to a waste category; rule-based filename heuristics handle edge cases.
5. **Simulate** — An animated rotating plate deposits the item into the correct bin (0°, 90°, 180°, or 270° rotation for General, Soiled, Contaminated, and Sharps respectively).

---

## Model

- **Architecture:** YOLOv8-cls (classification variant) by Ultralytics
- **Export format:** ONNX (opset version 12)
- **Input size:** 224 × 224 px
- **Output:** Probability distribution over 4 waste classes

---

## Results

| Test Item | Predicted Category | Confidence |
|---|---|---|
| Metal Equipment Packaging | General Waste | 100% |
| Diaper | Soiled Waste | 96% |
| Gloves | Contaminated | 91% |
| Syringe Needle | Sharps | 98% |
| Saline Bag | Soiled Waste | 88% |

Rule-based overrides improved accuracy in approximately 4% of test cases, primarily for items with overlapping visual features between the Soiled and Contaminated categories.

---

## Key Advantages

- **Privacy-first** — Images never leave the user's device
- **Offline capable** — No internet connection required after initial load
- **Zero infrastructure cost** — No backend server needed
- **Accessible** — Designed for non-technical clinical and waste-handling staff

---

## Limitations & Future Work

- Performance depends on image quality and orientation
- Model trained on a relatively small dataset
- **Planned:** Dataset expansion, multi-object detection for mixed waste streams, and integration with physical robotic sorting hardware via WebSerial or Bluetooth APIs

---

## References

1. G. Jocher et al., "Ultralytics YOLOv8," GitHub, 2023. https://github.com/ultralytics/ultralytics
2. ONNX Runtime Web Dev Team, "ONNX Runtime Web," Microsoft, 2023. https://onnxruntime.ai/docs/get-started/with-javascript/web.html
3. M. Minichino and J. Howse, *Learning OpenCV 4 Computer Vision with Python*, 3rd ed., Packt Publishing, 2019.
4. A. Howard et al., "MobileNets: Efficient CNNs for Mobile Vision Applications," arXiv:1704.04861, 2017.
5. S. Yang and M. Thung, "Classification of trash for recyclability status," CS229 Project Report, Stanford University, 2016.
6. WHO, "Safe management of wastes from health-care activities," 2nd ed., Geneva, 2014.
7. L. Alzubaidi et al., "Review of deep learning," *J. Big Data*, vol. 8, no. 53, 2021.

---

## Acknowledgements

The authors thank Vellore Institute of Technology for guidance and support, and acknowledge the open-source contributions of the [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8) and [Microsoft ONNX Runtime](https://onnxruntime.ai/) teams.
