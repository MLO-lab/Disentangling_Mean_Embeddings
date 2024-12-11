# Disentangling Cosine Similarity of Mean Embeddings

(Code appears here soon)

This is the code that accompanies the paper ["Disentangling Mean Embeddings for Better Diagnostics of Image Generators"](https://arxiv.org/abs/2409.01314), published at IAI Workshop @ NeurIPS 2024.

We propose a novel approach to disentangle the cosine similarity of mean embeddings into the product of cosine similarities for individual pixel clusters via central kernel alignment.
This allows quantifying the contribution of the cluster-wise performance to the overall image generation performance.


## To reproduce our results

First, download and unzip `img_align_celeba.zip into data/celeba/` from, e.g., `https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/`.
Second, run all cells in `preprocess_celeba.ipynb`.
Then, run the cells in `experiments.ipynb`.


## Reference
If you found this work or code useful, please cite:

```
@article{gruber2024disentangling,
  title={Disentangling Mean Embeddings for Better Diagnostics of Image Generators},
  author={Gruber, Sebastian G and Ziegler, Pascal Tobias and Buettner, Florian},
  journal={arXiv preprint arXiv:2409.01314},
  year={2024}
}
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).
