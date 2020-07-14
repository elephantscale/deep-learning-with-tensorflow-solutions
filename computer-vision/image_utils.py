## Utility functions 

%matplotlib inline

import glob
import os

import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## Gets number of files in the directory and total size in MB
def get_dir_stats(adir):
    listing = glob.glob(os.path.join(adir, "**/*.*"), recursive=True)
    files = [f for f in listing if os.path.isfile(f)]
    file_sizes = [os.path.getsize(f) for f in files]
    total_file_size_MB = round(sum(file_sizes) / (1024*1024), 2)
    return (len(files), total_file_size_MB)
    

def print_dir_stats (dir_name, adir, class_labels):
    dir_stats = get_dir_stats(adir)
    print ('--- {} ({}):  files={},  size={} MB'.format(dir_name, adir, dir_stats[0], dir_stats[1]))
    for class_label in class_labels:
        class_dir = os.path.join(adir, class_label)
        dir_stats = get_dir_stats (class_dir)
        print ('       +-- {} :  files={},  size={} MB'.format(class_label, dir_stats[0], dir_stats[1]))


def get_class_labels(a_training_dir):
    return [d for d in os.listdir(a_training_dir) if os.path.isdir(os.path.join(a_training_dir,d))]
        
def print_training_validation_stats (a_training_dir, a_validation_dir):
    class_labels = get_class_labels(a_training_dir)
    print ('Found class lables:', class_labels)
    print ()

    print_dir_stats('training_data', a_training_dir, class_labels)
    print()
    if a_validation_dir:
        print_dir_stats('validation_data', a_validation_dir, class_labels)
        


def display_images (a_train_dir, num_images_per_label):
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
        
