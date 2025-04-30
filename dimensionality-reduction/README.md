# PCA and Eigenfaces Implementation

This project implements dimensionality reduction via Principal Component Analysis (PCA) and generates eigenfaces as part of a computer vision assignment.

## Project Setup

Ensure the `yalefaces` dataset folder is located in the same directory as the Python file. This is why the `image_directory` variable is set to `'yalefaces'`.

> ⚠️ **Note**: The original `yalefaces` folder contained a `README` text file, which caused issues when looping through the image directory. I removed that file to avoid processing errors. If you'd prefer not to delete it, you can modify the image-loading logic to skip non-image files.

## Running the Code

- Executing the script will output a visualization of the PCA (Part 2).
- After closing the graph, the script will generate a video file (`eigenfaces.avi`) as part of Part 3.
- You’ll see the following confirmation in the terminal: "_The video was saved as 'eigenfaces.avi'._"
- The video can be viewed using most video players. On macOS, I used **VLC** for playback.

## Notes

- Please refer to the in-code comments for a detailed explanation of the implementation.
- External resources and references used for inspiration are cited in the comments.

