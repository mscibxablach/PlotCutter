import argparse
import os
from plot_recognizer import PlotRecognizer
from plot_cutter import PlotCutter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='Path to model')
    parser.add_argument('--labels-path', type=str, help='Path to label map')
    parser.add_argument('--num-classes', type=int, help='Number of classes')
    parser.add_argument('--input-path', type=str, help='Path to directory with files')
    parser.add_argument('--output-path', type=str, help='Directory for output')

    return parser.parse_args()


def main():
    args = parse_arguments()

    plot_recognizer = PlotRecognizer(args.model_path, args.labels_path, args.num_classes)
    plot_cutter = PlotCutter(plot_recognizer)

    files = os.listdir(args.input_path)

    counter = 0
    for file in files:
        plot_cutter.cut_plot(os.path.join(args.input_path, file), os.path.join(args.output_path, str(counter) + '.jpg'))
        counter += 1


if __name__ == "__main__":
    main()















#
# plot_recognizer = PlotRecognizer(MODEL_PATH, LABELS_PATH,  NUM_CLASSES)
# plot = plot_recognizer.get_plot_coordinates(IMAGE_PATH)
# print(plot)


# label_map = label_map_util.load_labelmap(LABELS_PATH)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map,
#     max_num_classes=
#     NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
#
#
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name="")
#
# session = tf.Session(graph=detection_graph)
#
# image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#
# detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#
# detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
# detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#
# num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#
# image = cv2.imread(IMAGE_PATH)
# image_expanded = np.expand_dims(image, axis=0)
# height, width, channels = image.shape
#
# (boxes, scores, classes, num) = session.run(
#     [detection_boxes, detection_scores, detection_classes, num_detections],
#     feed_dict={image_tensor: image_expanded})
#
# coordinates = get_boxes_coordinates(boxes, scores, width, height)
# print(coordinates)

# plot_to_cut = image[coordinates[0][0]:coordinates[0][2], coordinates[0][1]:coordinates[0][3]].copy()
# cv2.imwrite("plot.jpg", plot_to_cut)

# vis_util.visualize_boxes_and_labels_on_image_array(
#     image,
#     np.squeeze(boxes),
#     np.squeeze(classes).astype(np.int32),
#     np.squeeze(scores),
#     category_index,
#     use_normalized_coordinates=True,
#     line_thickness=8,
#     min_score_thresh=0.60)
#
# cv2.imshow('Object detector', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



