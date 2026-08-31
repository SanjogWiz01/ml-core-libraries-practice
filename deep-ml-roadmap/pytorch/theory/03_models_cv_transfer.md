# Models, CNNs, Sequences & Transfer Learning

### Tabular
A strong neural baseline is:
`Linear -> ReLU -> Dropout -> Linear -> output`.
Always benchmark against classical models such as logistic regression and gradient boosting.

### CNN
`Conv2d -> ReLU -> Pool -> Conv2d -> Adaptive/Global Pool -> Linear`
High-value APIs: `Conv2d`, `MaxPool2d`, `AdaptiveAvgPool2d`, `Flatten`, torchvision transforms.

### Sequences
Use `Embedding`, `LSTM`, `GRU`, and later Transformer components when appropriate.

### Transfer learning
1. Load pretrained backbone.
2. Freeze backbone.
3. Replace classifier.
4. Train head.
5. Optionally unfreeze selected layers.
6. Fine-tune with a small learning rate.

### Experiments
Record dataset/split, preprocessing, architecture, optimizer, learning rate, batch size, seed, validation metrics and model artifact.
