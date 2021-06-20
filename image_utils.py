def generate_labels(fruit_names):
    labels = []

    for fruit_name in fruit_names:
        labels.append("fresh_" + fruit_name)
        labels.append("rotten_" + fruit_name)

    return labels