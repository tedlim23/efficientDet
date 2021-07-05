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
import gc
import time
import record
import numpy as np
import string

def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
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

def get_det_score(dataset, preds, image_shape):
    for i, lblset in enumerate(dataset.as_numpy_iterator()):
        bboxes = lblset[1][0]
        idx = list(range(4, len(bboxes), 4))
        for i in idx:
            bbox = bboxes[i-4:i]
            
        pass
    iou = 0.0
    accuracy = 0.0
    return iou, accuracy

def train_step():
    @tf.function
    def default_grad(x, y, model, optimizer, loss_fn, train_acc_metric):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # train_acc_metric.update_state(y, logits)
        return loss_value

    return default_grad

def test_step():
    @tf.function
    def default_test(x, y, model, loss_fn, val_acc_metric):    
        val_logits = model(x, training=False)
        loss_value = loss_fn(y, val_logits)
        # val_acc_metric.update_state(y, val_logits)
        return loss_value

    return default_test


def main(model_name):
    gc.collect()
    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    with tf.device('/GPU:0'):
        train_manager = TFRecordManager("./data/VOLVO_OCRSET", "volvo_train.json", middle_path="train")
        val_manager = TFRecordManager("./data/VOLVO_OCRSET", "volvo_test.json", middle_path="test")
        num_classes = 36
        batch_size = 8
        train_dataset, train_sz = train_manager.get_tfdataset(batch_size, True, None, (384, 512))
        val_dataset, val_sz = val_manager.get_tfdataset(batch_size, False, None, (384, 512))
        loss_fn = DetectionLoss(num_classes)
        if model_name == "effdet":
            model = EfficientDet(num_classes)
        else:
            model = RetinaNet(num_classes)

        optimizer = tf.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss=loss_fn, optimizer=optimizer)

        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(f"weights/{model_name}", "weights" + "_epoch_{epoch}"),
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ]
        
        is_train = False
        save_path = os.path.join("weights", model_name, model_name)
        if is_train:
            epochs = 400
            train_steps_per_epoch = train_sz // batch_size
            test_steps_per_epoch = val_sz // batch_size
            train_acc_metric = tf.keras.metrics.Precision()
            val_acc_metric = tf.keras.metrics.Precision()
            ckpt = tf.train.Checkpoint(model)

            print("Train Size: {}, Validation Size: {}".format(train_sz, val_sz))

            train_fn = train_step()
            test_fn = test_step()
            early_stopper = record.EarlyStopping(ckpt, save_path)
            plotter = record.PlotLib(epochs, save_path)
            
            for epoch in range(epochs):
                print("\nStart of epoch %d" % (epoch+1))
                start_time = time.time()

                train_loss = 0.0
                batch_count = 0
                # Iterate over the batches of the dataset.
                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    if step == train_steps_per_epoch:
                        break
                    train_loss += train_fn(x_batch_train, y_batch_train, model, optimizer, loss_fn, train_acc_metric)
                    batch_count += 1

                print("Train loss: %.4f" % (float(train_loss/batch_count),))
                # metrics at the end of each epoch.
                # train_acc = train_acc_metric.result()

                # Reset training metrics at the end of each epoch
                # train_acc_metric.reset_states()

                # Run a validation loop at the end of each epoch.
                val_loss = 0.0
                batch_count_val = 0
                for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                    if step == test_steps_per_epoch:
                        break
                    val_loss += test_fn(x_batch_val, y_batch_val, model, loss_fn, val_acc_metric)
                    batch_count_val += 1

                # val_acc = val_acc_metric.result()
                # val_acc_metric.reset_states()

                print("Validation loss: %.4f" % (float(val_loss/batch_count_val),))
                print("Time taken: %.2fs" % (time.time() - start_time))

                # plotter.add_to_list(train_loss, val_loss, train_acc, val_acc)

                if early_stopper.is_stoppable(train_loss):
                    # plotter.notice_early_stop(epoch+1)
                    break
            
            # plotter.draw_plot()

        else:
            # Change this to `model_dir` when not using the downloaded weights
            weights_dir = f"weights/{model_name}"

            latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
            model.load_weights(latest_checkpoint)
            image = tf.keras.Input(shape=[None, None, 3], name="image")
            predictions = model(image, training=False)
            detections = DecodePredictions(num_classes, confidence_threshold=0.5)(image, predictions)
            inference_model = tf.keras.Model(inputs=image, outputs=detections)

            def prepare_image(image, batched = False):
                image, _, ratio = neurot_resize_and_pad_image(image)
                image = tf.keras.applications.efficientnet.preprocess_input(image)
                if not batched:
                    image = tf.expand_dims(image, axis=0)
                return image, ratio


            image = cv2.imread("data/VOLVO_OCRSET/test/Image__2021-05-12__09-19-27.jpg")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image, ratio = prepare_image(image)
            detections = inference_model.predict(input_image)
            num_detections = detections.valid_detections[0]
            print(detections)
            name_to_class_id = {
                case: i for i, case in enumerate(string.ascii_uppercase)
            }
            cnt = len(string.ascii_uppercase)
            for i in range(10):
                name_to_class_id[str(i)] = cnt
                cnt += 1

            class_names = [
                name_to_class_id[int(x)] for x in detections.nmsed_classes[0][:num_detections]
            ]
            visualize_detections(
                image,
                detections.nmsed_boxes[0][:num_detections] / ratio,
                class_names,
                detections.nmsed_scores[0][:num_detections],
            )
            # image_shape = (384, 512)
            # val_manager = TFRecordManager("./data/VOLVO_OCRSET", "volvo_test.json", middle_path="val")
            # val_dataset, val_sz = val_manager.get_tfdataset(batch_size, False, None, image_shape)
            # image_batch, ratio = prepare_image(val_dataset, batched = True)
            # detections = inference_model.predict(image_batch)
            # iou, accuracy = get_det_score(val_dataset, detections, image_shape)
            """
            nmsed_boxes = array([[[ 43.289345, 104.8941  ,  74.90653 , 141.55518 ],
                                    [158.06544 , 104.830025, 188.86772 , 139.17584 ],
                                    [ 91.495964, 107.45092 , 120.3758  , 141.7847  ],
            """

if __name__ == "__main__":
    for m in ["effdet", "retina"]:
        main(m)