def generate_labels(fruit_names):
    labels = []

    for fruit_name in fruit_names:
        labels.append("fresh_" + fruit_name)
        labels.append("rotten_" + fruit_name)

    return labels


def get_label_from_file_name(file_name):
    splits = file_name.split("_")
    # file name example: fresh_orange_1.jpg
    # the label is the first two tokens joined by _
    label = "_".join(s for s in splits[:-1])
    return label
