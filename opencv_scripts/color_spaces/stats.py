from opencv_scripts import str_min_max_bincount, test_images


def print_out_properties(method_name, input_data):
    print(method_name)
    for k, v in input_data.items():
        print(f"{k}:\n{str_min_max_bincount(v, method_name)}")


print_out_properties("BGR", {'light green': test_images['light green'], 'dark green': test_images['dark green']})
print("")
print_out_properties("HSV", {'light green': test_images['light green'], 'dark green': test_images['dark green']})
