import supervision as sv
import numpy as np


def plot_results(results, image):
    # 결과 전처리
    # 박스, score, 라벨( = "cat", "cat", "dog" ... )
    boxes = results[0]['boxes'].cpu().numpy()
    confidence = results[0]['scores'].cpu().numpy()
    text_labels = results[0]['text_labels']

    unique_classes = sorted(set(text_labels))
    class_name_to_id = {name: idx for idx, name in enumerate(unique_classes)}
    class_ids = np.array([class_name_to_id[name] for name in text_labels])
    # 주석 커스텀
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(text_labels, confidence)
    ]

    # 디텍션 객체 생성
    detections = sv.Detections(
        xyxy=boxes,
        confidence=confidence,
        class_id=class_ids,
    )

    # BoxAnnotator 는 box 만 생성한다.
    box_annotator = sv.BoxAnnotator(thickness=1)
    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)

    # LabelAnnotator 는 주석만 생성한다.
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=4)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    # Plot
    sv.plot_image(annotated_image)