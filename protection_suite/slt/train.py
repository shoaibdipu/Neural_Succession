from __future__ import annotations
from typing import Callable, Optional
from copy import deepcopy
import random
import numpy as np
import torch
import torch.nn.functional as F
from .metrics import accuracy


def masked_ce(logits, y, task_classes=None):
    if task_classes is not None:
        mask=torch.full((logits.shape[1],),False,dtype=torch.bool,device=logits.device)
        mask[list(task_classes)]=True
        logits=logits.masked_fill(~mask[None,:],-1e9)
    return F.cross_entropy(logits,y)


def train_task(model, dl, device, epochs=10, lr=0.05, weight_decay=5e-4, momentum=0.9,
               task_classes=None, val_dl=None, target_acc=None, max_epochs=None,
               target_tolerance=None, restore_closest=False, log_every=0):
    """Train one task, optionally restoring the checkpoint closest to a target validation accuracy.

    ``restore_closest`` is used only for the P1 resident-quality counterfactual.
    Selection uses the deterministic training-validation split, never benchmark test accuracy.
    """
    model.train(); opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    total_epochs=max_epochs if max_epochs is not None else epochs
    hist=[]; best_gap=float('inf'); best_state=None; best_val=None
    # For resident matching, epoch 0 is a valid candidate checkpoint. This
    # matters for pretrained residents that may begin near the A-only target.
    if restore_closest and target_acc is not None and val_dl is not None:
        va0=accuracy(model,val_dl,device,task_classes)
        gap0=abs(float(va0)-float(target_acc))
        best_gap=gap0; best_val=float(va0)
        best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        hist.append({"epoch":0,"loss":None,"val_acc":float(va0),"initial_candidate":True})
        if target_tolerance is not None and gap0 <= target_tolerance:
            model.load_state_dict(best_state)
            hist[-1]['restored_closest']=True; hist[-1]['closest_val_acc']=best_val; hist[-1]['closest_gap']=best_gap
            return hist
    for ep in range(total_epochs):
        model.train(); total=0.0; n=0
        for x,y in dl:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            loss=masked_ce(model(x),y,task_classes); loss.backward(); opt.step()
            total += loss.item()*len(y); n += len(y)
        va=None
        if val_dl is not None:
            va=accuracy(model,val_dl,device,task_classes)
        hist.append({"epoch":ep+1,"loss":total/max(n,1),"val_acc":va})
        if target_acc is not None and va is not None:
            gap=abs(float(va)-float(target_acc))
            if restore_closest and gap < best_gap:
                best_gap=gap; best_val=float(va)
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if restore_closest:
                if target_tolerance is not None and gap <= target_tolerance:
                    break
            elif va >= target_acc:
                break
    if restore_closest and best_state is not None:
        model.load_state_dict(best_state)
        for r in hist: r['restored_closest']=False
        hist[-1]['restored_closest']=True; hist[-1]['closest_val_acc']=best_val; hist[-1]['closest_gap']=best_gap
    return hist


class ReservoirBuffer:
    def __init__(self, capacity:int, device_store='cpu'):
        self.capacity=capacity; self.n_seen=0; self.x=[]; self.y=[]; self.logits=[]; self.task_ids=[]
        self.device_store=device_store
    def __len__(self): return len(self.y)
    def add_batch(self,x,y,logits,task_id=None):
        x=x.detach().cpu(); y=y.detach().cpu(); logits=logits.detach().cpu()
        if task_id is None:
            tids=[-1]*len(y)
        elif torch.is_tensor(task_id):
            vals=task_id.detach().cpu().reshape(-1).tolist()
            if len(vals)==1: vals=vals*len(y)
            if len(vals)!=len(y): raise ValueError('task_id tensor length must equal batch size')
            tids=[int(v) for v in vals]
        elif isinstance(task_id,(list,tuple,np.ndarray)):
            vals=list(task_id)
            if len(vals)==1: vals=vals*len(y)
            if len(vals)!=len(y): raise ValueError('task_id list length must equal batch size')
            tids=[int(v) for v in vals]
        else:
            tids=[int(task_id)]*len(y)
        for i in range(len(y)):
            self.n_seen += 1
            if len(self.y)<self.capacity:
                self.x.append(x[i].clone()); self.y.append(y[i].clone()); self.logits.append(logits[i].clone()); self.task_ids.append(tids[i])
            else:
                j=random.randrange(self.n_seen)
                if j<self.capacity:
                    self.x[j]=x[i].clone(); self.y[j]=y[i].clone(); self.logits[j]=logits[i].clone(); self.task_ids[j]=tids[i]
    def sample(self,n:int,device,exclude_task_id=None,return_task_ids=False):
        if len(self)==0 or n<=0: return None
        eligible=np.arange(len(self),dtype=int)
        if exclude_task_id is not None:
            eligible=np.asarray([i for i,t in enumerate(self.task_ids) if int(t)!=int(exclude_task_id)],dtype=int)
        if len(eligible)==0: return None
        ids=np.random.choice(eligible,size=min(n,len(eligible)),replace=False)
        x=torch.stack([self.x[i] for i in ids]).to(device)
        y=torch.stack([self.y[i] for i in ids]).long().to(device)
        z=torch.stack([self.logits[i] for i in ids]).to(device)
        if return_task_ids:
            t=torch.tensor([self.task_ids[i] for i in ids],dtype=torch.long,device=device)
            return x,y,z,t
        return x,y,z


def _grad_list(loss, params, retain_graph=True):
    gs=torch.autograd.grad(loss,params,retain_graph=retain_graph,create_graph=False,allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p,g in zip(params,gs)]


def _dot(gs1,gs2):
    return sum((a*b).sum() for a,b in zip(gs1,gs2))


def _norm2(gs):
    return sum((g*g).sum() for g in gs)


def optimizer_adjusted_safety_q(model,opt,current_loss,resident_loss,protection_loss,gradient_scale=1.0,
                                return_endpoint_grads=False):
    """First-order resident-safety threshold for the actual next SGD update.

    The data-loss gradient family is
        g(q) = scale * ((1-q) g_B + q g_P),
    where g_P is the DER++ replay objective and g_A is the resident CE gradient
    used to assess local resident damage. PyTorch SGD momentum and L2 weight
    decay are added to the endpoint update directions before solving for q.

    Resident non-increase to first order requires g_A^T u(q) >= 0 because the
    parameter update is -lr * u(q). The routine returns the smallest feasible q
    in [0,1]. The manuscript's simple q* expression is the special case with no
    momentum/decay and g_P = g_A.

    If return_endpoint_grads=True, g_B and g_P are returned as detached tensors.
    derpp_train_task can then form the final mixture gradient directly instead of
    performing a fourth backward pass. This preserves the exact data-loss
    gradient while reducing the safety controller's net backward-equivalent
    overhead from three extra passes to two on evaluated steps.
    """
    params=[p for p in model.parameters() if p.requires_grad]
    gB=_grad_list(current_loss,params,retain_graph=True)
    gA=_grad_list(resident_loss,params,retain_graph=True)
    gP=_grad_list(protection_loss,params,retain_graph=False)
    group=opt.param_groups[0]
    mu=float(group.get('momentum',0.0)); wd=float(group.get('weight_decay',0.0))
    u0=[]; u1=[]
    for p,gb,gp in zip(params,gB,gP):
        prev=opt.state[p].get('momentum_buffer',None)
        base=torch.zeros_like(p) if prev is None else mu*prev.detach()
        decay=wd*p.detach() if wd else torch.zeros_like(p)
        u0.append(base + float(gradient_scale)*gb + decay)
        u1.append(base + float(gradient_scale)*gp + decay)
    h0=float(_dot(gA,u0).detach().item())
    h1=float(_dot(gA,u1).detach().item())
    if h0 >= 0:
        qstar=0.0; feasible=True
    elif h1 > h0 and h1 >= 0:
        qstar=float(np.clip(-h0/(h1-h0),0.0,1.0)); feasible=True
    else:
        qstar=1.0; feasible=False
    cos=float((_dot(gA,gB)/((_norm2(gA).sqrt()*_norm2(gB).sqrt())+1e-12)).detach().item())
    if return_endpoint_grads:
        return qstar,feasible,cos,h0,h1,[g.detach() for g in gB],[g.detach() for g in gP]
    return qstar,feasible,cos,h0,h1


def _assign_mixture_grads(model,gB,gP,q,gradient_scale):
    """Assign the exact gradient of scale*((1-q)L_B + q L_P)."""
    params=[p for p in model.parameters() if p.requires_grad]
    if len(params)!=len(gB) or len(params)!=len(gP):
        raise RuntimeError('endpoint gradient length mismatch')
    for p,gb,gp in zip(params,gB,gP):
        p.grad=float(gradient_scale)*((1.0-float(q))*gb + float(q)*gp)


def derpp_train_task(model, dl, device, buffer:ReservoirBuffer, epochs=10, lr=0.03,
                     weight_decay=5e-4, momentum=0.9, replay_batch_size=64, alpha=0.5, beta=0.5,
                     task_classes=None, controller:Optional[Callable]=None,
                     compute_safety_grad=False, record_steps=False, reference_q=0.5,
                     safety_every=1, current_task_id=None):
    """DER++ stream replay with a controller over the current/replay mixture q.

    Buffer semantics follow DER++ Algorithm 2: replay is sampled before the
    current examples are inserted into the reservoir. Stored logits are the
    pre-update logits computed for each current example.

    q is a loss-mixture weight, not a replay-sample fraction. The normalized
    objective is
        loss=((1-q)L_current + q L_DER++)/(1-reference_q),
    so reference_q=0.5 recovers standard DER++ exactly.

    ``safety_every`` controls how often the expensive local safety quantity is
    recomputed. safety_every=1 is the frozen primary setting because it preserves
    the per-step interpretation of the local controller. Larger values are an
    explicitly labeled efficiency ablation: the most recent q* is held between
    evaluations and therefore no per-step safety guarantee is claimed there.

    On safety-evaluated steps, the three endpoint/autopsy gradients are computed
    with autograd.grad and g_B/g_P are reused to form the final mixture gradient.
    This avoids a redundant fourth backward pass. The reported
    ``net_extra_backward_equiv`` is therefore two per safety evaluation.
    """
    if int(safety_every)<1:
        raise ValueError('safety_every must be >= 1')
    safety_every=int(safety_every)
    model.train()
    opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    rows=[]; total_replay=0; total_current=0; infeasible_steps=0
    safety_evaluations=0; safety_grad_calls=0; reused_mixture_backwards=0
    replay_steps=0; optimizer_steps=0; cached=None
    safety_resident_fallback_forwards=0; safety_resident_empty_steps=0
    for ep in range(epochs):
        model.train()
        for bi,(x,y) in enumerate(dl):
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            z=model(x); z_store=z.detach()
            cur=masked_ce(z,y,task_classes)
            rep=buffer.sample(replay_batch_size,device,return_task_ids=True)
            q=0.0; qstar=0.0; feasible=True; h0=np.nan; h1=np.nan; grad_cos=np.nan
            safety_evaluated=False; qstar_age=0
            endpoint_grads=None
            if rep is None:
                loss=cur
                loss.backward()
            else:
                xr,yr,zr_old,tr=rep; zr=model(xr)
                replay_ce=F.cross_entropy(zr,yr)
                rep_loss=alpha*F.mse_loss(zr,zr_old)+beta*replay_ce
                total_replay += len(yr)
                # The DER++ replay objective uses the normal reservoir sample,
                # including any current-task examples already admitted online.
                # The safety gradient g_A must be resident-only.  Prefer the old
                # examples already present in the replay sample; if that sample
                # happens to contain only the current task, draw an old-only
                # fallback batch.  fixed_compute_matched executes the same path.
                resident_ce=None; resident_safety_n=0; resident_fallback=0
                if current_task_id is None:
                    old_mask=torch.ones_like(tr,dtype=torch.bool)
                else:
                    old_mask=(tr != int(current_task_id))
                if bool(old_mask.any()):
                    resident_ce=F.cross_entropy(zr[old_mask],yr[old_mask])
                    resident_safety_n=int(old_mask.sum().item())
                elif compute_safety_grad and ((cached is None) or (replay_steps % safety_every == 0)):
                    old_rep=buffer.sample(replay_batch_size,device,exclude_task_id=current_task_id,return_task_ids=True)
                    if old_rep is not None:
                        xold,yold,_,_=old_rep
                        resident_ce=F.cross_entropy(model(xold),yold)
                        resident_safety_n=len(yold); resident_fallback=1
                        safety_resident_fallback_forwards += 1
                    else:
                        safety_resident_empty_steps += 1
                ctx={"epoch":ep,"batch":bi,"current_loss":float(cur.detach()),
                     "replay_loss":float(rep_loss.detach()),"qstar":0.0,
                     "resident_safety_n":resident_safety_n,
                     "resident_safety_fallback":resident_fallback}
                loss_scale=1.0/max(1.0-float(reference_q),1e-12)
                if compute_safety_grad:
                    must_eval=(cached is None) or (replay_steps % safety_every == 0)
                    if must_eval and resident_ce is not None:
                        qstar,feasible,grad_cos,h0,h1,gB,gP=optimizer_adjusted_safety_q(
                            model,opt,cur,resident_ce,rep_loss,gradient_scale=loss_scale,
                            return_endpoint_grads=True)
                        cached={"qstar":qstar,"feasible":feasible,"grad_cos":grad_cos,
                                "h0":h0,"h1":h1,"replay_step":replay_steps}
                        endpoint_grads=(gB,gP)
                        safety_evaluated=True; safety_evaluations += 1; safety_grad_calls += 3
                        # We reuse gB/gP for the actual optimizer step, eliminating one
                        # otherwise-required backward pass.
                        reused_mixture_backwards += 1
                    elif must_eval and resident_ce is None:
                        # No old-task example remains in the reservoir.  There is no
                        # valid resident gradient for this step, so use q0 only and
                        # record the missing safety evaluation rather than treating
                        # current-task replay as resident evidence.
                        qstar=0.0; feasible=False; grad_cos=np.nan; h0=np.nan; h1=np.nan
                        cached=None
                    else:
                        qstar=float(cached["qstar"]); feasible=bool(cached["feasible"])
                        grad_cos=float(cached["grad_cos"]); h0=float(cached["h0"]); h1=float(cached["h1"])
                    qstar_age=(int(replay_steps-int(cached["replay_step"])) if cached is not None else 0)
                    ctx.update({"qstar":qstar,"safety_feasible":feasible,
                                "grad_cos":grad_cos,"resident_dot_q0":h0,"resident_dot_q1":h1,
                                "safety_evaluated":int(safety_evaluated),"qstar_age":qstar_age})
                    if safety_evaluated and not feasible: infeasible_steps += 1
                q=float(controller(ctx) if controller is not None else 0.5)
                q=float(np.clip(q,0.0,1.0))
                loss=loss_scale*((1-q)*cur+q*rep_loss)
                if endpoint_grads is not None:
                    _assign_mixture_grads(model,*endpoint_grads,q=q,gradient_scale=loss_scale)
                else:
                    loss.backward()
                replay_steps += 1
            opt.step(); optimizer_steps += 1; total_current += len(y)
            buffer.add_batch(x,y,z_store,task_id=current_task_id)
            if record_steps:
                rows.append({"epoch":ep,"batch":bi,"q":q,"qstar":qstar,
                             "safety_feasible":int(feasible),"safety_evaluated":int(safety_evaluated),
                             "qstar_age":qstar_age,"resident_dot_q0":h0,
                             "resident_dot_q1":h1,"grad_cos":grad_cos,
                             "resident_safety_n":int(ctx.get("resident_safety_n",0)) if rep is not None else 0,
                             "resident_safety_fallback":int(ctx.get("resident_safety_fallback",0)) if rep is not None else 0,
                             "loss":float(loss.detach()),"buffer":len(buffer)})
    net_extra=max(0,safety_grad_calls-reused_mixture_backwards)
    return {"steps":rows,
            "extra_grad_calls":net_extra,  # legacy field: net extra backward-equivalent work
            "safety_grad_calls":safety_grad_calls,
            "reused_mixture_backwards":reused_mixture_backwards,
            "net_extra_backward_equiv":net_extra,
            "safety_evaluations":safety_evaluations,"safety_every":safety_every,
            "replay_examples":total_replay,"current_examples":total_current,
            "optimizer_steps":optimizer_steps,
            "safety_infeasible_steps":infeasible_steps,
            "safety_resident_fallback_forwards":safety_resident_fallback_forwards,
            "safety_resident_empty_steps":safety_resident_empty_steps}

def populate_buffer(model, dl, device, buffer:ReservoirBuffer, max_examples:int|None=None, task_id=None):
    model.eval(); seen=0
    with torch.no_grad():
        for x,y in dl:
            x=x.to(device); y=y.to(device); z=model(x)
            buffer.add_batch(x,y,z,task_id=task_id); seen += len(y)
            if max_examples is not None and seen>=max_examples: break
    return buffer


def estimate_fisher(model, dl, device, task_classes=None, max_batches=20):
    fisher={n:torch.zeros_like(p,device=device) for n,p in model.named_parameters() if p.requires_grad}
    model.eval(); nb=0
    for x,y in dl:
        x=x.to(device); y=y.to(device); model.zero_grad(set_to_none=True)
        loss=masked_ce(model(x),y,task_classes); loss.backward(); nb+=1
        for n,p in model.named_parameters():
            if p.requires_grad and p.grad is not None: fisher[n] += p.grad.detach().pow(2)
        if nb>=max_batches: break
    for n in fisher: fisher[n] /= max(nb,1)
    theta={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    return fisher,theta


def ewc_replay_train_task(model, dl, device, fisher, theta_ref, replay_buffer:ReservoirBuffer|None,
                          epochs=10,lr=0.03,weight_decay=5e-4,lam=10.0,replay_weight=0.3,
                          replay_batch_size=64,task_classes=None):
    model.train()
    opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for x,y in dl:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            loss=masked_ce(model(x),y,task_classes)
            pen=torch.tensor(0.0,device=device)
            for n,p in model.named_parameters():
                if n in fisher: pen = pen + (fisher[n]*(p-theta_ref[n]).pow(2)).sum()
            loss=loss+0.5*lam*pen
            if replay_buffer is not None and len(replay_buffer)>0:
                rep=replay_buffer.sample(replay_batch_size,device)
                xr,yr,_=rep; loss=loss+replay_weight*F.cross_entropy(model(xr),yr)
            loss.backward(); opt.step()
    return model
