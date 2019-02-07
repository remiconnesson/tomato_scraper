# tomato_scraper
A toy project to illustrate the use of google images to download a dataset of tomato & not tomato. Use Fast.AI for preprocessing.

...


In this notebook we are going to download a list of images we previously scraped on google images. 

The urls of the images are stored in a .txt file, separated by '\n' characters.

You can learn how to do this by following this notebook here : 
- https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb

Or by following this article in French there:
- https://medium.com/france-school-of-ai/scraper-un-dataset-depuis-google-images-342f670b9bad

# Importing the .txt files containing the urls


```python
from pathlib import Path
import os

# Get current directory
p = Path('.')

# Provide path to the .txt files with the url inside
tomato_urls_txt = p/"tomato_urls.txt"
pepper_urls_txt = p/"pepper_urls.txt" 
```

# Downloading images with fast.ai library


```python
# Create the directories to store the images ..
# .. we split images into two folders
# .. because there is two classes.
os.makedirs(p/"tomatoes", exist_ok=True)
os.makedirs(p/"peppers", exist_ok=True)

# Import of the helper function which will help us download the images.
from fastai.vision import download_images
```

## Downloading the tomatoes


```python
print("  - Downloading Tomato Images -   ")
download_images(urls=p/"tomato_urls.txt",
                dest=p/'tomatoes')
```

## Downloading the peppers


```python
print("  - Downloading Pepper Images -   ")
download_images(urls=p/"pepper_urls.txt",
                dest=p/'peppers')
```
# Verifying each images 
We want images we can read and are in RGB format, so we verify each image one by one and discard the unsuitable ones.


```python
from fastai.vision import verify_images

classes = ['tomatoes', 'peppers']
for c in classes:
    print(c)  
    path_to_class_folder = p/c
    
    # verify images have correct properties for training
    verify_images(path_to_class_folder,
                  delete=True, img_format=f'{c} %d')
```

# Bravo, here are your images stored into two folders

Now, you are ready to do whatever you want with it.

Like preprocessing them and feeding them in a neural network like done here:

- https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb
