# --------------------------------------
# Import Packages
# --------------------------------------

import torch
import torch.optim

# --------------------------------------
# Optimizer that doesnt actually do anything other than reassign values to what you want them to be
# --------------------------------------


class Opt(torch.optim.Optimizer):
    '''

    Reassign values to model parameters

    '''

    def __init__(self, params, lr):

        # Verify inputs
        if lr is None or lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        # Make accessible dictionary for updating
        defaults = dict(lr=lr)

        # Make super class
        super(Opt, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, new_weights, device, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # I don't know why we have this
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Loop over the parameters
        for group in self.param_groups:

            # Update the gradients
            for ind, p in enumerate(group['params']):

                # Add the gradient
                p.data = new_weights[ind].to(device)

        return loss
