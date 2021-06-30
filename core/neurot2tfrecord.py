import enum
import json
import tensorflow as tf
import os
import glob
import uuid
import cv2
import numpy as np
import functools
from image_utils import swap_xy, convert_to_xywh, convert_to_corners, random_flip_horizontal, neurot_resize_and_pad_image
import string

def debug_dataset(dataset):
    ex = dataset.take(10)
    for feature in ex:
        imageb = feature[0] # [2, 768, 768, 3]
        labelb = feature[1]
        firstimage = imageb.numpy()[0] # [768, 768, 3]
        firstlabel = labelb.numpy()[0] # [768, 768, 3]
        firstimage = firstimage.astype(np.uint8)
        for coord in firstlabel:
            img = cv2.rectangle(firstimage, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (0,255,0), 2)

        cv2.imshow('win',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def rotate_img(img, rotation_angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), float(-rotation_angle), 1.0)
    transformed_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return transformed_img


class DetectionData:
    def __init__(self, image_path, bboxes, labels):
        self.image = image_path
        self.bboxes = np.array(bboxes).flatten().astype(np.float32)# [(xmin, ymin, xmax, ymax), ]
        self.labels = labels

class TFRecordManager:
    def __init__(self, data_path, json_file_name):
        self.data_path = data_path
        tfrecords_basedir = os.path.join(data_path, "tfr")

        if os.path.exists(tfrecords_basedir):
            import shutil
            shutil.rmtree(tfrecords_basedir)
        os.makedirs(tfrecords_basedir)
        
        self.bg_tfrecord_basefile = os.path.join(tfrecords_basedir, str(uuid.uuid4()))
        self.fg_tfrecord_basefile = os.path.join(tfrecords_basedir, str(uuid.uuid4()))
        self.nrt_json_path = os.path.join(self.data_path, json_file_name)
        self.raw_images_path = os.path.join(self.data_path, "train")
        self.bg_detection_data_list = []
        self.fg_detection_data_list = []
        self._parse_nrt_json(ocr = True)

    def _parse_nrt_json(self, ocr = False):
        """
        parse and fill self.detection_data_list
        """
        name_to_class_id = {
            "Choco": 0
        }
        if ocr:
            name_to_class_id = {
                case: i for i, case in enumerate(string.ascii_uppercase)
            }
            cnt = len(string.ascii_uppercase)
            for i in range(10):
                name_to_class_id[str(i)] = cnt
                cnt += 1

        with open(self.nrt_json_path, "r") as f:
            json_data = json.load(f)

        for data in json_data["data"]:
            file_name = os.path.join(self.raw_images_path, data["fileName"])
            _img = cv2.imread(file_name)
            if _img is None:
                continue

            if data.get("rotation_angle") and data.get("rotation_angle") != 0:
                _img = rotate_img(_img, data.get("rotation_angle"))
                file_name = os.path.join(self.raw_images_path, "rotated", data["fileName"])
                cv2.imwrite(file_name, _img)
            
            h, w, c = _img.shape

            region_labels = data["regionLabel"]
            bboxes = []
            labels = []

            for label in region_labels:
                # assert label["type"] == "Rect"
                try:
                    class_id = name_to_class_id[label["className"]]
                except:
                    print("label exception: ",label["className"])
                    continue
                xmin = label["x"]
                ymin = label["y"]
                xmax = xmin + label["width"]
                ymax = ymin + label["height"]

                bboxes.append((ymin / h, xmin / w, ymax / h, xmax / w))
                labels.append(class_id)

            if len(labels) > 0:
                self.fg_detection_data_list.append(DetectionData(file_name, bboxes, labels))
            else:
                self.bg_detection_data_list.append(DetectionData(file_name, bboxes, labels))

    def write_tfrecord_to_disk(self):
        with tf.io.TFRecordWriter(self.bg_tfrecord_basefile) as f:
            for data in self.bg_detection_data_list:
                image_string = open(data.image, 'rb').read()
                image_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": _bytes_feature(image_string),
                            "bboxes": _float_list_feature(data.bboxes),
                            "labels": _int64_list_feature(data.labels)
                        }))
                f.write(image_example.SerializeToString())
            
        with tf.io.TFRecordWriter(self.fg_tfrecord_basefile) as f:
            for data in self.fg_detection_data_list:
                image_string = open(data.image, 'rb').read()
                image_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": _bytes_feature(image_string),
                            "bboxes": _float_list_feature(data.bboxes),
                            "labels": _int64_list_feature(data.labels)
                        }))
                f.write(image_example.SerializeToString())
            
    def _parse_function(self, example_proto, _is_train, _aug_flag, _imsz, _model_name):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "bboxes": tf.io.VarLenFeature(tf.float32),
            "labels": tf.io.VarLenFeature(tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)

        image_decoded = tf.io.decode_png(parsed_example["image"], channels=3)
        if _is_train:
            image_decoded = tf.image.random_jpeg_quality(image_decoded, 75, 100)
        image_decoded = tf.cast(image_decoded, dtype = tf.float32)

        # keras's efficientnet doesnt use 
        assert _model_name == "EfficientNet", "only efficientnet implemented"

        bboxes = parsed_example["bboxes"]
        bboxes = tf.sparse.to_dense(bboxes)
        bboxes = tf.reshape(bboxes, [-1, 4])
        
        class_id = parsed_example["labels"]
        class_id = tf.cast(tf.sparse.to_dense(class_id), tf.float32)

        # bbox = bboxes
        bbox = swap_xy(bboxes)

        # image_decoded, bbox = random_flip_horizontal(image_decoded, bbox)
        image_decoded, image_shape, _ = neurot_resize_and_pad_image(image_decoded, min_side = float(_imsz[0]), max_side = float(_imsz[1]))

        bbox = tf.stack(
            [
                bbox[:, 0] * image_shape[1],
                bbox[:, 1] * image_shape[0],
                bbox[:, 2] * image_shape[1],
                bbox[:, 3] * image_shape[0],
            ],
            axis=-1,
        )
        bbox = convert_to_xywh(bbox)

        return image_decoded, bbox, class_id


    def get_tfdataset(self, batch_size, is_train, aug_flag, imsz):
        assert batch_size % 2 == 0, "batch size must be even number."
        self.write_tfrecord_to_disk()
        dataset_sz = len(self.bg_detection_data_list) + len(self.fg_detection_data_list)
        fg_ds = tf.data.TFRecordDataset([self.fg_tfrecord_basefile]).shuffle(len(self.fg_tfrecord_basefile)).repeat()
        bg_ds = tf.data.TFRecordDataset([self.bg_tfrecord_basefile]).shuffle(len(self.bg_tfrecord_basefile)).repeat()
        
        _parse_function = functools.partial(
            self._parse_function,
            _is_train=is_train,
            _aug_flag=aug_flag,
            _imsz=imsz,
            _model_name="EfficientNet"
        )

        from label_manager import LabelEncoder
        self.label_encoder = LabelEncoder()


        choice_dataset = tf.data.Dataset.range(2).repeat()
        merged_ds = tf.data.experimental.choose_from_datasets([fg_ds, bg_ds], choice_dataset)
        merged_ds = merged_ds.map(_parse_function)
        merged_ds = merged_ds.padded_batch(
            batch_size=batch_size, padding_values=(0.0, 1e-8, -2.0), drop_remainder=True
        )
        merged_ds = merged_ds.map(
            self.label_encoder.encode_batch
        ).prefetch(1)


        
        return merged_ds, dataset_sz

if __name__ == "__main__":
    # manager = TFRecordManager("./data/VOLVO_OCRSET", "volvo_train.json")
    manager = TFRecordManager("./data/ocrs", "neuro_ocr.json")
    dataset, dataset_sz = manager.get_tfdataset(10, True, None, (192, 256))
    print("Done.")
    debug_dataset(dataset)