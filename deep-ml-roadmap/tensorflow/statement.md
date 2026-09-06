# tensorflow — Deep Learning

Stage 3 of the roadmap. Low-level graph operations, custom training, and production-ready models.

## Topics

- **Basics**: tensors, variables, GradientTape, eager execution
- **CNNs**: image classification, data augmentation, transfer learning
- **RNNs / LSTMs**: sequence modeling, text classification, time series
- **Advanced**: custom layers, custom training loops, mixed precision
- **Deployment**: SavedModel format, TensorFlow Lite, TF Serving

## Structure

```
tensorflow/
  basics/           # tensors, ops, autodiff
  cnns/             # convolutional networks
  rnns/             # recurrent networks, LSTMs, GRUs
  advanced/         # custom layers, training loops
  deployment/       # export and serve models
  exercises/        # practice problems
```

## Key Concepts to Lock In

1. Computational graph vs eager execution
2. GradientTape for custom autodiff
3. tf.data pipelines for efficient data loading
4. Transfer learning workflow (freeze → fine-tune)
