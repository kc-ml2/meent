from neuralop import LpLoss, H1Loss

from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
)

class GratingLoss:
    def __init__(self, loss_func, focal_coeff=0.7, marginal_coeff=0.3):
        super().__init__()
        self.loss_func = loss_func

    def __call__(self, out, **sample):
        grating_loss = self.loss_func(out[:,:,160:192,:], sample['y'][:,:,160:192,:])
        
        return grating_loss 

class SuperstrateLoss:
    def __init__(self, loss_func, focal_coeff=0.7, marginal_coeff=0.3):
        super().__init__()
        self.loss_func = loss_func

    def __call__(self, out, **sample):

        superstrate_loss = self.loss_func(out[:,:,192:,:], sample['y'][:,:,192:,:])

        return superstrate_loss

class SubstrateLoss:
    def __init__(self, loss_func, focal_coeff=0.7, marginal_coeff=0.3):
        super().__init__()
        self.loss_func = loss_func

    def __call__(self, out, **sample):

        substrate_loss = self.loss_func(out[:,:,:160,:], sample['y'][:,:,:160,:])

        return substrate_loss    

    
class RangeAwareLoss:
    def __init__(self, loss_func, focal_coeff=0.7, marginal_coeff=0.3):
        super().__init__()
        self.loss_func = loss_func
        
        self.focal_coeff = focal_coeff
        self.marginal_coeff = marginal_coeff

    def __call__(self, out, **sample):
        grating_loss = self.loss_func(out[:,:,160:192,:], sample['y'][:,:,160:192,:])
        
        substrate_loss = self.loss_func(out[:,:,:160,:], sample['y'][:,:,:160,:])
        superstrate_loss = self.loss_func(out[:,:,192:,:], sample['y'][:,:,192:,:])

        return (
            self.focal_coeff * grating_loss + \
            self.marginal_coeff * (substrate_loss + superstrate_loss)
        )


class LossFuncWrapper:
    def __init__(self, loss_func):
        super().__init__()

        self.loss_func = loss_func
    
    def __call__(self, out, **sample):
        return self.loss_func(out, sample['y'])
    

class CarstenLossFuncWrapper:
    def __init__(self, loss_func):
        super().__init__()

        self.loss_func = loss_func
    
    def __call__(self, out, **sample):
        return self.loss_func(out, **sample).mean()
    
    
class L2pMSSSIM:
    def __init__(self, device) -> None:
        super().__init__()

        self.l2 = LpLoss(d=2, p=2)
        self.mssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)

    def __call__(self, out, **sample):
        return 0.7 * self.l2(out, sample['y']) + 0.3 * self.mssim(out, sample['y'])