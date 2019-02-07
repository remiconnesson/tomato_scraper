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

      - Downloading Tomato Images -   




    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='200' class='' max='200', style='width:300px; height:20px; vertical-align: middle;'></progress>
      100.00% [200/200 00:19<00:00]
    </div>
    


    Error https://greenstalkgarden.com/wp-content/uploads/2016/03/tomato-garden-tower.jpg HTTPSConnectionPool(host='greenstalkgarden.com', port=443): Max retries exceeded with url: /wp-content/uploads/2016/03/tomato-garden-tower.jpg (Caused by SSLError(SSLError("bad handshake: SysCallError(104, 'ECONNRESET')")))
    Error http://www.tomatodirt.com/images/mulching-tomatoes-straw-gardenweb-forum.jpg HTTPConnectionPool(host='www.tomatodirt.com', port=80): Max retries exceeded with url: /images/mulching-tomatoes-straw-gardenweb-forum.jpg (Caused by ConnectTimeoutError(<urllib3.connection.HTTPConnection object at 0x7f96e0e4d3c8>, 'Connection to www.tomatodirt.com timed out. (connect timeout=4)'))
    Error https://ottan.me/wp-content/uploads/in-tomato-gardening-ideas-5aef2b58caecf.jpg HTTPSConnectionPool(host='ottan.me', port=443): Max retries exceeded with url: /wp-content/uploads/in-tomato-gardening-ideas-5aef2b58caecf.jpg (Caused by SSLError(SSLCertVerificationError("hostname 'ottan.me' doesn't match either of '*.web-hosting.com', 'web-hosting.com'")))
    Error https://www.veggiegardener.com/wp-content/uploads/sites/3/2010/05/Growing-tomatoes-620x264.jpg HTTPConnectionPool(host='127.0.0.1', port=80): Max retries exceeded with url: / (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f96e0b9f080>: Failed to establish a new connection: [Errno 111] Connection refused'))


## Downloading the peppers


```python
print("  - Downloading Pepper Images -   ")
download_images(urls=p/"pepper_urls.txt",
                dest=p/'peppers')
```

      - Downloading Pepper Images -   




    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='100' class='' max='100', style='width:300px; height:20px; vertical-align: middle;'></progress>
      100.00% [100/100 00:11<00:00]
    </div>
    


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

    tomatoes




    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='185' class='' max='185', style='width:300px; height:20px; vertical-align: middle;'></progress>
      100.00% [185/185 00:01<00:00]
    </div>
    


    peppers




    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='95' class='' max='95', style='width:300px; height:20px; vertical-align: middle;'></progress>
      100.00% [95/95 00:00<00:00]
    </div>
    


# Bravo, here are your images stored into two folders

Now, you are ready to do whatever you want with it.

Like preprocessing them and feeding them in a neural network like done here:

- https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb
