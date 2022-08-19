from PIL import Image
import numpy as np
import pickle
for num in xrange(1, 3001):
    im = Image.open(r'./image/'+str(num)+'.jpg')
    
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
    l=np.array(pixels)
    l=l/255.0
    l=l.flatten()
    print l.shape
    l=l.reshape(16*8*3,1)
    pickle.dump(l, open(r'./pickles/'+str(num)+'.pkl', 'wb'))
