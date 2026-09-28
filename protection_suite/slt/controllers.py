from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression

class FixedController:
    def __init__(self,q=0.5): self.q=float(q)
    def __call__(self,ctx): return self.q

class SafetyController:
    def __call__(self,ctx): return float(ctx.get('qstar',0.0))

class StaticSignalController:
    def __init__(self,q0:float): self.q0=float(q0)
    def __call__(self,ctx): return self.q0

class SLTController:
    def __init__(self,q0:float): self.q0=float(q0)
    def __call__(self,ctx): return max(self.q0,float(ctx.get('qstar',0.0)))

class ReplayCalibrator:
    """Monotone map from a risk-oriented scalar to preferred replay mixture q.

    ``fit`` preserves the original omega=1-rho API.  ``fit_signal`` is used for
    alternative pre-task signals after the caller explicitly declares whether
    larger values mean more compatibility or more risk.  Orientation is never
    guessed from outcomes.
    """
    def __init__(self): self.m=None
    def fit(self,omega,best_q):
        return self.fit_signal(omega,best_q,direction='risk')
    def fit_signal(self,signal,best_q,direction='risk'):
        x=np.asarray(signal,float); y=np.asarray(best_q,float)
        if direction not in ('risk','compatibility'):
            raise ValueError("direction must be 'risk' or 'compatibility'")
        risk=x if direction=='risk' else -x
        self.m=IsotonicRegression(y_min=0,y_max=1,increasing=True,out_of_bounds='clip').fit(risk,y)
        self.direction=direction
        return self
    def predict_signal(self,signal):
        if self.m is None: raise RuntimeError('fit first')
        x=float(signal); risk=x if self.direction=='risk' else -x
        return float(self.m.predict([risk])[0])
    def predict_rho(self,rho):
        if self.m is None: raise RuntimeError('fit first')
        # Backward-compatible rho path: larger rho means more compatibility.
        if getattr(self,'direction','risk')=='risk':
            return float(self.m.predict([1-float(rho)])[0])
        return self.predict_signal(rho)
