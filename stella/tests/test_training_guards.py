import pytest


def test_create_model_raises_without_dataset():
    from stella.neural_network import ConvNN

    cnn = ConvNN(output_dir='.')
    with pytest.raises(ValueError):
        cnn.create_model(seed=2)


def test_train_models_raises_without_dataset():
    from stella.neural_network import ConvNN

    cnn = ConvNN(output_dir='.')
    with pytest.raises(ValueError):
        cnn.train_models(seeds=[2], epochs=1, batch_size=4)
