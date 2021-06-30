from neurot2tfrecord import TFRecordManager
from detection_loss import DetectionLoss
from label_manager import DecodePredictions
from efficientdet import EfficientDet
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
from image_utils import neurot_resize_and_pad_image
from retinanet import RetinaNet

def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    import numpy as np
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax

def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    with tf.device('/GPU:0'):
        manager = TFRecordManager("./data/VOLVO_OCRSET", "volvo_train.json")
        num_classes = 36
        batch_size = 8
        dataset, dataset_sz = manager.get_tfdataset(batch_size, True, None, (384, 512))
        loss_fn = DetectionLoss(num_classes)
        # model = EfficientDet(num_classes)
        model = RetinaNet(num_classes)

        optimizer = tf.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss=loss_fn, optimizer=optimizer)

        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("weights/retinanet", "weights" + "_epoch_{epoch}"),
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
        ]
        
        is_train = True
        if is_train:
            model.fit(
                dataset,
                epochs=150,
                verbose=1,
                callbacks=callbacks_list,
                steps_per_epoch=dataset_sz // batch_size
            )
        else:
            # Change this to `model_dir` when not using the downloaded weights
            weights_dir = "weights/effdet"

            latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
            model.load_weights(latest_checkpoint)
            image = tf.keras.Input(shape=[None, None, 3], name="image")
            predictions = model(image, training=False)
            detections = DecodePredictions(num_classes, confidence_threshold=0.5)(image, predictions)
            inference_model = tf.keras.Model(inputs=image, outputs=detections)

            def prepare_image(image):
                image, _, ratio = neurot_resize_and_pad_image(image)
                image = tf.keras.applications.efficientnet.preprocess_input(image)
                return tf.expand_dims(image, axis=0), ratio


            image = cv2.imread("data/VOLVO_OCRSET/train/rotated/20201125_145030_crop_resize_512x384.jpg")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image, ratio = prepare_image(image)
            detections = inference_model.predict(input_image)
            num_detections = detections.valid_detections[0]
            print(detections)
            class_names = [
                "CHOCO" for x in detections.nmsed_classes[0][:num_detections]
            ]
            visualize_detections(
                image,
                detections.nmsed_boxes[0][:num_detections] / ratio,
                class_names,
                detections.nmsed_scores[0][:num_detections],
            )

if __name__ == "__main__":
    main()