# Deep Learning Fundamentals
- Neural network: learnable transformations from inputs to outputs. 
- Neuron: `z = Wx + b`, followed by an activation.
- Core loop: forward pass -> loss -> gradients -> optimizer update.
- Activations: ReLU for hidden layers; sigmoid for binary output; softmax for multiclass.
- Losses: MSE for regression; binary cross-entropy for binary classification; cross-entropy for multiclass.
- Epoch = one pass over training data; batch = one group processed together.
- Overfitting: training improves while validation worsens; use regularization, augmentation, early stopping.
- 80/20: understand data -> model -> prediction -> loss -> gradient -> update.
