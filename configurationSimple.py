class ConfigSimple(object):
    """
    This class has some of our config values.
    Num_episodes is purely for testing purposes. For field use, there should only be one episode
    total_rows and total_cols describes the size of the larger map.
    num_images is how many images we train on. For field use, anything having to do with the image should be replaced
    max_steps search and trace is how long we give it to complete each episode. In most cases it does not need that long
    num_targets is the total amount of regions the row x col map is divided into.
    """
    num_images = 1
    #image = 'env_images/image2.jpg'
    image = 'env_images/new_map.png'
    total_rows = 180
    total_cols = 180

    num_episodes = 100
    max_steps_search = 300
    max_steps_trace = 500 #Change to 500 for testing

    num_targets = 9
