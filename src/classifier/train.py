"""Training loop and evaluation helpers with early stopping."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    """Evaluate the best saved model (or a passed-in model) on a dataloader.

    Returns
    -------
    (loss_test, loss_error) : tuple[float, float]
        Average NLL and classification error.
    """
    if model_best is None:
        model_best = torch.load(name + ".model", weights_only=False)

    model_best.eval()
    loss_test = 0.0
    loss_error = 0.0
    N = 0
    for batch, targets in test_loader:
        loss_test += model_best.forward(batch, targets, reduction="sum").item()
        y_pred = model_best.classify(batch)
        loss_error += (y_pred != targets).sum().item()
        N += batch.shape[0]

    loss_test /= N
    loss_error /= N

    if epoch is None:
        print(f"-> FINAL PERFORMANCE: nll={loss_test:.4f}, ce={loss_error:.4f}")
    elif epoch % 10 == 0:
        print(f"Epoch: {epoch}, val nll={loss_test:.4f}, val ce={loss_error:.4f}")

    return loss_test, loss_error


def training(name, max_patience, num_epochs, model, optimizer,
             training_loader, val_loader):
    """Train with early stopping based on validation NLL."""
    nll_val = []
    error_val = []
    best_nll = float("inf")
    patience = 0

    for e in range(num_epochs):
        model.train()
        for batch, targets in training_loader:
            loss = model.forward(batch, targets)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        loss_e, error_e = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_e)
        error_val.append(error_e)

        if e == 0 or loss_e < best_nll:
            torch.save(model, name + ".model")
            best_nll = loss_e
            patience = 0
        else:
            patience += 1

        if patience > max_patience:
            print(f"Early stopping at epoch {e}")
            break

    return np.asarray(nll_val), np.asarray(error_val)


def plot_curve(name, signal, file_name="curve.pdf", xlabel="epochs",
               ylabel="nll", color="b-", test_eval=None):
    """Plot a training curve and optionally annotate the final test value."""
    plt.figure()
    plt.plot(np.arange(len(signal)), signal, color, linewidth=3, label=f"{ylabel} val")
    if test_eval is not None:
        plt.hlines(test_eval, xmin=0, xmax=len(signal), linestyles="dashed",
                   label=f"{ylabel} test")
        plt.text(len(signal), test_eval, f"{test_eval:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(name + file_name, bbox_inches="tight")
    plt.show()
