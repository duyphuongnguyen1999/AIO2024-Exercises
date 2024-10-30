import numpy as np


class CustomLinearRegression:
    def __init__(self, x_data, y_target, learning_rate=0.01, num_epochs=1000):
        self.num_samples = x_data.shape[0]
        self.x_data = np.c_[np.ones((self.num_samples), 1), x_data]
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initial weights
        rng = np.random.default_rng(seed=1)  # Create a new random generator
        self.theta = rng.random(
            (self.x_data.shape[1], 1)
        )  # Use the generator to create random values
        self.losses = []

    def compute_loss(self, y_pred, y_target):
        loss = (y_pred - y_target) * (y_pred - y_target) / 2
        return loss

    def predict(self, x_data):
        y_pred = np.dot(x_data, self.theta)
        return y_pred

    def fit(self):
        for epoch in range(self.num_epochs):
            # Predict
            y_pred = self.predict(self.x_data)

            # Compute Loss
            loss = self.compute_loss(y_pred, self.y_target)
            self.losses.append(loss)

            # Compute Gradient
            loss_grad = 2 * (y_pred - self.y_target) / self.num_samples
            gradients = self.x_data.T.dot(loss_grad)

            # Update weight
            self.theta = self.theta - self.learning_rate * gradients

            if epoch % 50 == 0:
                print(f"Epoch: {epoch} - Loss: {loss}")
        return {"loss": sum(self.losses) / len(self.losses), "weight": self.theta}
