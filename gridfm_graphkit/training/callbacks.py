import torch


class EarlyStopper:
    def __init__(
        self,
        saving_path,
        patience=5,
        tol=0,
        min_validation_loss=float("inf"),
    ):
        """
        Args:
            patience (int): number of epochs to wait before early stopping
                -1 means no early stopping
                0 means stop training the first time the validation loss increases
            tol (float): tolerance to consider validation loss as worse as the best one so far
        """

        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.min_validation_loss = min_validation_loss
        self.saving_path = saving_path

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = (
                validation_loss  # Update the best validation loss
            )
            self.counter = 0

            # Save the best model whenever a new minimum is found
            torch.save(model, self.saving_path)

        # check if the valid loss is worse than the best one so far, accounting for tolerance
        elif validation_loss > (self.min_validation_loss + self.tol):
            self.counter += 1

            if self.patience != -1 and self.counter > self.patience:
                print(
                    "Early stopping after {} epochs of no improvement.".format(
                        self.counter,
                    ),
                )
                return True
        return False
