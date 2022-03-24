# A Variant of The Eigenfaces Algorithm for Arabic Text Calligraphy Recognition

## How to Use

### Download the dataset

- download the dataset from [here](https://drive.google.com/file/d/1WNFYiAHcCemTZtdsjbTWVh9_WgZSzlbx/view?usp=sharing).
- copy the dataset into a folder in your local repo and make sure the folder is named 'dataset'.

### Run the Provided Notebook

After setting up the dataset you can reproduce the results by running all the cell in `run.ipynb` notebook.

## Explaining The Algorithm

### Training

- For each class of text calligraphy we construct a PCA matrix.
- And that's it :)

### Testing

- We reconstruct the given input using the PCA matrix for each class (So, for each input we have K reconstructed images with K = `number of classes`). We call these reconstructed images `Recovered Images`.
- Now, each class is represented by an image from the `recovered images`.
- We measure the similarity between each recovered image and the input image using [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance). Then we pick the closest image from the recovered images.
- If the class label of the closest image is equal to the true class label, then the algorithm has predicted correctly.
