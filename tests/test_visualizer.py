import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from visualizer import plot_losses, smooth, save_plot


class MockModel:
    def __init__(self):
        self.name = "mock"


def test_plot_losses():
    losses = [5, 4, 3, 2, 1]
    val_losses = [1, 2, 3, 4, 5]
    steps = [i for i in range(len(losses))]

    plot_losses(
        MockModel(),
        losses,
        val_losses,
        interval=1,
        show_plot=False,
        smooth_loss=False,
        smooth_val_loss=False,
        weight=1,
        save_data=False,
    )

    axes = plt.gca()
    plt_losses = axes.lines[0]
    plt_val_losses = axes.lines[1]
    plt.close()

    assert axes.get_title() == "Training and Validation Loss over Steps"
    assert axes.get_ylabel() == "Loss"
    assert axes.get_xlabel() == "Steps"
    assert np.array_equal(plt_losses.get_ydata(), losses)
    assert np.array_equal(plt_losses.get_xdata(), steps)
    assert np.array_equal(plt_val_losses.get_ydata(), val_losses)
    assert np.array_equal(plt_val_losses.get_xdata(), steps)


def test_smooth():
    losses = smooth([5, 4, 3, 2, 1], 0.9)
    expected = [5, 4.9, 4.71, 4.439, 4.0951]
    assert np.allclose(losses, expected)


def test_save_plot(tmp_path):
    model = MockModel()
    model.plot_dir = os.path.join(tmp_path, model.name)
    file_pattern = os.path.join(model.plot_dir, "mock_test_*.png")
    plt.plot([1, 2, 3])
    save_plot(model, plt, "test")
    assert os.path.exists(model.plot_dir)
    assert len(glob.glob(file_pattern)) == 1
