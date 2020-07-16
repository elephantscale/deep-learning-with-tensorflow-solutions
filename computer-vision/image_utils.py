## Utility functions 

# %matplotlib inline

## Gets number of files in the directory and total size in MB
def get_dir_stats(adir):
    import os, glob
    
    listing = glob.glob(os.path.join(adir, "**/*.*"), recursive=True)
    files = [f for f in listing if os.path.isfile(f)]
    file_sizes = [os.path.getsize(f) for f in files]
    total_file_size_MB = round(sum(file_sizes) / (1024*1024), 2)
    return (len(files), total_file_size_MB)
    

def print_dir_stats (a_dir_name, adir, a_class_labels):
    import os 
    
    dir_stats = get_dir_stats(adir)
    print ('--- {} ({}):  files={},  size={} MB'.format(a_dir_name, adir, dir_stats[0], dir_stats[1]))
    for class_label in a_class_labels:
        class_dir = os.path.join(adir, class_label)
        dir_stats = get_dir_stats (class_dir)
        print ('       +-- {} :  files={},  size={} MB'.format(class_label, dir_stats[0], dir_stats[1]))


def get_class_labels(a_training_dir):
    import os 
    return [d for d in os.listdir(a_training_dir) if os.path.isdir(os.path.join(a_training_dir,d))]
        
def print_training_validation_stats (a_training_dir, a_validation_dir):
    class_labels = get_class_labels(a_training_dir)
    print ('Found class lables:', class_labels)
    print ()

    print_dir_stats('training_data', a_training_dir, class_labels)
    print()
    if a_validation_dir:
        print_dir_stats('validation_data', a_validation_dir, class_labels)
        


def display_images_from_dir (a_train_dir, num_images_per_label):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import random
    import os
    
    class_labels = get_class_labels(train_dir)
    
    fig_rows = len(class_labels)
    fig_cols = num_images_per_label + 1  # adding 1 to columns, for text labels
    
    fig = plt.gcf()
    fig.set_size_inches(fig_cols * 3, fig_rows * 3)
    #     fig.subplots_adjust(hspace=0.4, wspace=0.4)

    row = 0
    index = 0
    for label in class_labels:

        class_dir = os.path.join(train_dir, label)
        class_file_listing = os.listdir(class_dir)

        random_class_images = random.sample(class_file_listing, num_images_per_label )
       
        row = row + 1
        index = index + 1
        sp = plt.subplot(fig_rows, fig_cols, index)
        sp.text (0.5,0.5, label, fontsize=18, ha='center')
        sp.axis('Off') # Don't show axes (or gridlines)
   
        for img_file in random_class_images:
            index = index + 1
            sp = plt.subplot(fig_rows, fig_cols, index)
            sp.axis('Off') # Don't show axes (or gridlines)
            
            img_file_path = os.path.join(class_dir, img_file)
            img = mpimg.imread(img_file_path)
            plt.imshow(img)
            
            # this will print image file name
            # sp.text(0,0, img_file)
        
        
    plt.show()
# ---------- end : display_images_from_dir

def calculate_prediction_table(a_model, a_test_data_gen):
    from math import ceil
    import numpy as np
    
    ground_truth = tests_labels = a_test_data_gen.classes
    a_test_data_gen.reset()  # revert back to batch 1
    predictions = a_model.predict(a_test_data_gen, batch_size=a_test_data_gen.batch_size, 
                                      steps=ceil(a_test_data_gen.n / a_test_data_gen.batch_size) )
    ## Ensure all predictions match
    assert(len(predictions) == len(ground_truth) )
    
    prediction_table = {}
    for index, val in enumerate(predictions):
        #get argmax index
        index_of_highest_probability = np.argmax(val)
        value_of_highest_probability = val[index_of_highest_probability]
#         prediction_table[index] = [value_of_highest_probability, 
#                                    index_of_highest_probability, 
#                                    ground_truth[index]]
        prediction_table[index] = { 'softmax_output' : val,
                                    'max_probability' : value_of_highest_probability, 
                                    'predicted_class' : index_of_highest_probability, 
                                    'expected_class' : ground_truth[index]}
    
    return prediction_table
#  ------   end : calculate_prediction_table



def sample_from_dict(d, sample=10):
    import random
    
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))
## -- end: sample_from_dict()



## ----------------------------
# Helper function that finds images that are closest
# Input parameters:
#   prediction_table: dictionary from the image index to the prediction
#                      and ground truth for that image
#   get_highest_probability: boolean flag to indicate if the results
#                            need to be highest (True) or lowest (False) probabilities
#   label: id of category
#   number_of_items: num of results to return
#   only_false_predictions: boolean flag to indicate if results
#                           should only contain incorrect predictions
def get_images_with_sorted_probabilities(prediction_table, get_highest_probability,
                                         label, number_of_items, only_false_predictions=False):
    import pprint
    
    # sorted_prediction_table = [ (k, a_prediction_table[k]) for k in sorted(a_prediction_table.items(), 
                # key=lambda x: x[1].get('max_probability'), reverse = a_get_highest_probability)]
    sorted_prediction_table = sorted(prediction_table.items(), 
                                     key=lambda x: x[1].get('max_probability'), 
                                     reverse = get_highest_probability)
    # pprint.pprint (sorted_prediction_table)
    result = []
    for index, (k,v) in enumerate(sorted_prediction_table):
        #  image_index, [probability, predicted_index, gt] = key
        
        image_index = k
        probability = v.get('max_probability')
        predicted_index = v.get('predicted_class')
        expected_index = v.get('expected_class')
        
        if predicted_index == label:
            if only_false_predictions == True:
                if predicted_index != expected_index:
                    # result.append([image_index, [probability, predicted_index, expected_index] ])
                    result.append([image_index, v ])
            else:
                # result.append([image_index, [probability, predicted_index, expected_index] ])
                result.append([image_index, v ])
        if len(result) >= number_of_items:
            return result
## --- end: get_images_with_sorted_probabilities()




## ----------------------------
# Helper functions to plot the nearest images given a query image
def display_image_predictions_helper(filenames, predicted_index, distances, message):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import os
    
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    plt.figure(figsize=(20,15))
    columns = 5
    for i, image in enumerate(images):
        image_name = os.path.basename(filenames[i])
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        ax.set_title("\n\n{}\nPredicted:{}\nProbability:{:.2f}".format(image_name, predicted_index, distances[i]))
#         ax.set_title( "\n\n"+  image_name +"\n"+"\nProbability: " +
#             str(float("{0:.2f}".format(distances[i]))))
        plt.suptitle( message, fontsize=20, fontweight='bold')
        plt.axis('off')
        plt.imshow(image)
## ------------- end: display_image_predictions_helper()

        
## ----------------------------
def display_image_predictions(validation_dir, sorted_indicies, message, fnames):
    import os
    
    similar_image_paths = []
    distances = []
    for name, value in sorted_indicies:
        # [probability, predicted_index, gt] = value
        probability = value.get('max_probability')
        predicted_index = value.get('predicted_class')
        expected_index = value.get('expected_class')
        similar_image_paths.append(os.path.join(validation_dir,  fnames[name]))
        distances.append(probability)
        
    display_image_predictions_helper(similar_image_paths, predicted_index, distances, message)
## ------------- end: display_image_predictions()


def predict_on_images(model, files, image_width, image_height):
    import random
    import numpy as np
    from tensorflow.keras.preprocessing import image
    import pprint

    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    prediction_results = []
    
    for file in files:
        img = image.load_img(file, target_size = (image_width, image_height))
        # print (image_file)
        img_data = image.img_to_array(img) / 255.
        # option 1 reshape
        #img_data = np.expand_dims(img_data, axis = 0)
        #prediction = model.predict (image_data)
        # or option 2: no reshape and predict
        prediction = model.predict(img_data[None]) # this is softmax
        
        index_of_highest_probability = np.argmax(prediction[0])
        value_of_highest_probability = prediction[0][index_of_highest_probability]
        # print (prediction)

        x = {
            'img_file': file,
            'softmax_output' : prediction, 
            'max_probability' : value_of_highest_probability, 
            'predicted_class' : index_of_highest_probability, 
            }
        # pprint.pprint (x)
        prediction_results.append (x)
    # end for
    return prediction_results
## ---- end : predict_on_random_images()
    

## ----------------------
def plot_image_predictions (prediction_results, message):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import os
    
    columns = 5
    plt.figure(figsize=(20,15))
    for i, result in enumerate(prediction_results):
        image_name = os.path.basename(result['img_file'])
        img = mpimg.imread(result['img_file'])
        ax = plt.subplot(len(prediction_results) / columns + 1, columns, i + 1)
        ax.set_title("\n\n{}\nPredicted:{}\nProbability:{:.2f}".format(image_name, result['predicted_class'], result['max_probability']))
        plt.suptitle( message, fontsize=20, fontweight='bold')
        plt.axis('off')
        plt.imshow(img)
    
## --- end: plot_image_predictions