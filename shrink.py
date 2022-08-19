from PIL import Image
for i in xrange(1,3001):
    foo = Image.open("./images/"+str(i)+".jpg")
    print i,foo.size
    foo=foo.resize((16,8),Image.ANTIALIAS)
    foo.save("./image/"+str(i)+".jpg",quality=95)
