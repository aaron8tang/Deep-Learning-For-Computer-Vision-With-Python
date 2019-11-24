# import the necessary packages
import cv2

class SimplePreprocessor:
	"""
	主要是resize image
	"""
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		"""
		store the target image width, height, and interpolation method used when resizing
		:param width:目标图像的width
		:param height:目标图像的height
		:param inter:默认使用INTER_AREA（像素区域关系进行重采样）。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。
		"""
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		"""
		resize the image to a fixed size, ignoring the aspect ratio
		:param image:
		:return:
		"""
		return cv2.resize(image, (self.width, self.height),
						  interpolation=self.inter)
