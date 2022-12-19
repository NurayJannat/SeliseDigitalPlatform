## SELISE DIGITAL PLATFORM ASSESSMENT

All solutions of problems have been organized under respective folders.
1. Problem 1:
	A pdf file has been shared inside **Problem 1** describing the solution.
	
2. Problem 2:
	Inside **Problem 2** folder, there is a Docker file, folder weight, folder image.
	- Download weights from this link 
	"https://drive.google.com/drive/folders/1_NeiWqbzmJX0FGGSkhkHVrgpQZYo6yY5?usp=share_link"
	Here, 
			i. weight.hdf5 - weight for trained model with VGG16
			ii. weights.hdf5 - weight for trained model with VGG19
	Keep this weight file inside **weight** folder of the repository.
	- Place all test images you want to infer inside **image** folder of the repository.
	
	- To build the image, run this
		```
		docker image build .
		```
	- To Run inference
		```
		docker run -it -v $(pwd)/weight:/weight -v $(pwd)/image:/image <image_id> python3 inference.py 
		```
	- To train from pretrained(weights from imagenet) VGG16
	```
	docker run -it -v <path-to-dataset>:/dataset <image_id> python3 trainig.py
	```
3. Problem 3:
	Run: python3 solution3.py




