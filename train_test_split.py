#source = https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = 'D:\\Documents\\Academics\\semester VI\\ES 654- ML\\assi 3\\assignment-3-alianas23\\dataset\\train\\'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['apes/', 'apples/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		makedirs(newdir, exist_ok=True)
# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = 'D:\\Documents\\Academics\\semester VI\\ES 654- ML\\assi 3\\assignment-3-alianas23\\dataset\\train\\'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'test/'
	if file.startswith('apes'):
		dst = dataset_home + dst_dir + 'apes/'  + file
		copyfile(src, dst)
	elif file.startswith('apples'):
		dst = dataset_home + dst_dir + 'apples/'  + file
		copyfile(src, dst)