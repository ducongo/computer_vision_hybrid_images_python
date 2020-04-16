
import numpy as np
import cv2
import math
import threading

class Filters:

    def __init__(self):
        pass

    def gaussian_kernel(self, sigma=1):
            size = 2 * math.ceil(3 * sigma) + 1 
            normal_distribution = 1 / (2.0 * np.pi * sigma**2)
            x, y = np.ogrid[-size : size + 1, -size :size + 1]
            h = np.exp(-(((x**2) + (y**2)) / (2.0*(sigma**2))))
            g_kernel =  h * normal_distribution
            return g_kernel
    
    def sobel_kernels(self):
        #(vertical kernel, horizontal kernel)
        return (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

class MYCV:
    '''
    Class containing all my computer vision operation
    '''
    def __init__(self):
        pass

    def convolution2d(self, orig_img, kernel):
        # https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
        m, n = kernel.shape
        #print(f'{title} max{np.max(orig_img)} min:{np.min(orig_img)}')
        
        if (m == n):
            height, width = orig_img.shape

            height, width = orig_img.shape
            offset = math.floor(m/2) #offset of the image that will be convoluted
            padded = np.zeros(((offset * 2) + height, (offset * 2) + width)) # padd the new image using the offset
            padded[offset:-offset, offset:-offset] = orig_img # center the original img in the center of the padded image
            new_image = np.zeros(orig_img.shape)
            
            # duplicate the edges of the rigimal matrix into the offseted area of the padded matrix
            for i in range(offset - 1,-1, -1):
                padded[i:i+1,offset:orig_img.shape[1] + offset] = orig_img[0]
                padded[padded.shape[0] - 1 - i:padded.shape[0] + i,offset:orig_img.shape[1] + offset] = orig_img[0]
            for j in range(offset - 1,-1, -1):
                padded[0:padded.shape[0],j:j+1] = padded[0:padded.shape[0],offset:offset+1]
                padded[0:padded.shape[0],padded.shape[1] -1 - j:padded.shape[1] + j] = padded[0:padded.shape[0],offset:offset+1]

            for i in range(height):
                for j in range(width):
                    new_image[i][j] = np.sum(padded[i:i+m, j:j+m] * kernel)
            #print(f'{title} max{np.max(new_image)} min:{np.min(new_image)}')            
            return new_image
        return None


    def correlation2d(self, orig_img, kernel):
        # https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
        m, n = kernel.shape
        #print(f'{title} max{np.max(orig_img)} min:{np.min(orig_img)}')
        
        if (m == n):
            height, width = orig_img.shape
            offset = math.floor(m/2) #offset of the image that will be convoluted
            padded = np.zeros(((offset * 2) + height, (offset * 2) + width)) # padd the new image using the offset
            padded[offset:-offset, offset:-offset] = orig_img # center the original img in the center of the padded image
            new_image = np.zeros(orig_img.shape)
            
            # duplicate the edges of the rigimal matrix into the offseted area of the padded matrix
            for i in range(offset - 1,-1, -1):
                padded[i:i+1,offset:orig_img.shape[1] + offset] = orig_img[0]
                padded[padded.shape[0] - 1 - i:padded.shape[0] + i,offset:orig_img.shape[1] + offset] = orig_img[0]
            for j in range(offset - 1,-1, -1):
                padded[0:padded.shape[0],j:j+1] = padded[0:padded.shape[0],offset:offset+1]
                padded[0:padded.shape[0],padded.shape[1] -1 - j:padded.shape[1] + j] = padded[0:padded.shape[0],offset:offset+1]

            #convolve
            for i in range(height):
                for j in range(width):
                    region = padded[i:i+m, j:j+m]
                    center  = math.floor(m/2)
                    region[center][center] = 0 # exclude the center for the sum
                    new_image[i][j] = np.sum(padded[i:i+m, j:j+m] * kernel)
            #print(f'{title} max{np.max(new_image)} min:{np.min(new_image)}')            
            return new_image
        return None

    
    def sobel_filtering(self, img, kernels):
        vertical_kernel, horizontal_kernel = kernels

        # apply sobels X and Y kernels to the image
        ImgX = self.convolution2d(img, vertical_kernel)
        ImgY = self.convolution2d(img, horizontal_kernel)
        # print(ImgX.dtype)
        # calculate the gradient magnitude and normalize 
        gradient_mag = np.sqrt(np.square(ImgX) + np.square(ImgY))
        gradient_mag *= 255.0 / gradient_mag.max()

        #compute the slope
        slope = np.arctan2(ImgY, ImgX)
        # print(slope)

        return (gradient_mag, slope)



    def non_max_suppression(self, img, direction):
        '''
        essentially this function iterates over every pixels and the direction angles and
        checks the direction of the current pixel. If the direction is at a 0 degree 
        then it checks the neighbouring pixels intensity. If the neighbouring have a higher intensity then the
        current pixel gets set to 0
        '''
        M, N = img.shape
        
        angle = direction * 180 / np.pi
        angle[angle < 0] += 180 # if the angle is less than 0 add 180 to normalize

        m, n = img.shape
        non_maxima_img = np.zeros((M,N), dtype=np.int32)

        for i in range(1,m-1):
            for j in range(1,n-1):
                try:
                    # pixel intensity set to maximum for neightbouring pixels based on agle
                    a = 255 
                    b = 255

                    #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        a = img[i, j+1]
                        b = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        a = img[i+1, j-1]
                        b = img[i-1, j+1]
                    #angle 90
                    elif (67 <= angle[i,j] < 112):
                        a = img[i+1, j]
                        b = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        a = img[i-1, j-1]
                        b = img[i+1, j+1]

                    #if the local pixel is more intense than the neightbour pixel keep intensity else reduco to 0 to thin edge
                    if (a <= img[i,j]) and (b <= img[i,j]):
                        non_maxima_img[i,j] = img[i,j]
                    else:
                        non_maxima_img[i,j] = 0

                except:
                    pass
        return non_maxima_img   




import threading

fnames = ['left', 'right']
exts = ['png','png',]
left, right = 0, 1


my_kernels = Filters()
my_cv = MYCV()

original_img_color1 = cv2.imread(f'./input_images/{fnames[left]}.{exts[left]}', cv2.IMREAD_COLOR )
original_img_color2 = cv2.imread(f'./input_images/{fnames[right]}.{exts[right]}', cv2.IMREAD_COLOR)

original_img_color1 = cv2.cvtColor(original_img_color1, cv2.COLOR_BGR2RGB)
original_img_color2 = cv2.cvtColor(original_img_color2, cv2.COLOR_BGR2RGB)

img1_channels = []
img2_channels = []

img2_channels_high = []

for img1_channel, img2_channel in zip(cv2.split(original_img_color1),cv2.split(original_img_color2)):
    img1_channels.append(img1_channel)
    img2_channels.append(img2_channel)
    img2_channels_high.append(img2_channel)

sigma1 = 3
sigma2 = 5

def applyBlur(arr, index, sigma):
    arr[index] = my_cv.convolution2d(arr[index], my_kernels.gaussian_kernel(sigma))
    
    
def highpass(orig_channels, blurred_channels, index):
    # to get low pass I simply subtract the orginal img from the blured image for each channels
    orig_channels[index] = cv2.subtract(orig_channels[index], blurred_channels[index], dtype = cv2.CV_64F)

    

#using threading to speed up the process of applying the blurring
threads = []
for i in range(0,len(img1_channels)):
    new_thread_img1 = threading.Thread(target=applyBlur, args=(img1_channels, i, sigma1))
    new_thread_img1.start()

    new_thread_img2 = threading.Thread(target=applyBlur, args=(img2_channels, i, sigma2))
    new_thread_img2.start()

    threads.append(new_thread_img1)
    threads.append(new_thread_img2)

#join the thread in order to wait for all current threads to terminate before continuing execution
for t in threads:
    t.join()


threads = []
#using threading to speed up the process of applying the highpass filter
for i in range(0,len(img1_channels)):
    new_thread = threading.Thread(target=highpass, args=(img2_channels_high, img2_channels, i))
    new_thread.start()
    threads.append(new_thread)

#join the thread in order to wait for all current threads to terminate before continuing execution
for t in threads:
    t.join()


# merge all 3 channels for the image 1 and scale it
merged_img1 = cv2.merge((img1_channels[2], img1_channels[1], img1_channels[0]))
merged_img1 = cv2.convertScaleAbs(merged_img1)

# merge all 3 channels for the image 2
merged_img2 = cv2.merge((img2_channels_high[2], img2_channels_high[1], img2_channels_high[0]))
# merged_img2 = cv2.convertScaleAbs(merged_img2)

hybrid_image = cv2.addWeighted(merged_img1, 0.5 , merged_img2, 0.8, 35,dtype = cv2.CV_8U)

cv2.imwrite('./output_images/lowpass.jpg', merged_img1)
cv2.imwrite('./output_images/highpass.jpg', merged_img2)

cv2.imwrite('./output_images/hybrid.jpg', hybrid_image)





