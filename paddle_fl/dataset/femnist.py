import requests
import os
import json
import tarfile
import random
url = "https://paddlefl.bj.bcebos.com/leaf/"
target_path = "femnist_data"
tar_path = target_path+".tar.gz"
print(tar_path)

def download(url):
	r = requests.get(url)
	with open(tar_path,'wb') as f:
		f.write(r.content)

def extract(tar_path):
	tar = tarfile.open(tar_path, "r:gz")
	file_names = tar.getnames()
	for file_name in file_names:
		tar.extract(file_name)

	tar.close()

def train(trainer_id,inner_step,batch_size,count_by_step):
	if not os.path.exists(target_path):
		print("Preparing data...")
		if not os.path.exists(tar_path):
			download(url+tar_path)
		extract(tar_path)
	def train_data():
		train_file = open("./femnist_data/train/all_data_%d_niid_0_keep_0_train_9.json" % trainer_id,'r')
		json_train = json.load(train_file)
		users = json_train["users"]
		rand = random.randrange(0,len(users)) # random choose a user from each trainer
		cur_user = users[rand]
		print('training using '+cur_user)
                train_images = json_train["user_data"][cur_user]['x']
                train_labels = json_train["user_data"][cur_user]['y']
                if count_by_step:
                        for i in xrange(inner_step*batch_size):
                                yield train_images[i%(len(train_images))], train_labels[i%(len(train_images))]
                else:
                        for i in xrange(len(train_images)):
                                yield train_images[i], train_labels[i]

		train_file.close()

	return train_data

def test(trainer_id,inner_step,batch_size,count_by_step):
	if not os.path.exists(target_path):
                print("Preparing data...")
                if not os.path.exists(tar_path):
                        download(url+tar_path)
                extract(tar_path)
	def test_data():
		test_file = open("./femnist_data/test/all_data_%d_niid_0_keep_0_test_9.json" % trainer_id, 'r')
		json_test = json.load(test_file)
		users = json_test["users"]
		for user in users:
                        test_images = json_test['user_data'][user]['x']
                        test_labels = json_test['user_data'][user]['y']
                        for i in xrange(len(test_images)):
                                yield test_images[i], test_labels[i]

		test_file.close()

	return test_data

	
