
def select_a_dominant_attribute(v_attribute="Male"):
    # Path to the attributes file : it has all pictures so initial 25000 are matched with image.
    attributes_file = "list_attr_celeba.txt"

    # Initialize a dictionary to store the "Young" attribute for each image
    current_attribute = {}
    y_att = list()

    # Read the file
    with open(attributes_file, 'r') as file:
        # Read the number of images (first line)
        num_images = int(file.readline().strip())

        # Read the attributes names (second line) and find the index for "Young"
        attributes = file.readline().strip().split()
        print(attributes)

        current_index = attributes.index(v_attribute)

        # Read each line, split it and extract the "Young" attribute
        for line in file:
            parts = line.strip().split()
            image_name = parts[0]
            current_value = int(parts[current_index + 1])  # +1 because of the image name
            current_attribute[image_name] = current_value
            y_att.append(current_value)

    y_att_onehot = [[1, 0] if y == -1 else [0, 1] for y in y_att]
    # print(y_att_onehot)
    return y_att_onehot


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    q = select_a_dominant_attribute(v_attribute="Male")
    print(q)
    pass
