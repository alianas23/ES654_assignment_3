import os
path = "D:\\Documents\\Academics\\semester VI\\ES 654- ML\\assi 3\\assignment-3-alianas23\\dataset\\test\\apes\\"

def main():

	for count, filename in enumerate(os.listdir(path)):
		dst ='apes' + str(count+30) + ".jpg"
		src =path+ filename
		dst =path+ dst
		os.rename(src, dst)


if __name__ == '__main__':
	main()
