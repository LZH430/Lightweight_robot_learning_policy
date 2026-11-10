"""Cache-only additive approach/closure supervision alongside the original full-task objective."""
import unittest
from types import SimpleNamespace
import torch
from lightweight_policy.gripper_finetune import left_grasp_loss, LEFT_GRASP_DIMS
from lightweight_policy.model_dynamics import DynamicsStudentPolicy
from scripts.losses_dynamics import loss_and_metrics


class GripperFinetuneTests(unittest.TestCase):
    def inputs(self, left, state=1.):
        p=torch.zeros(1,4,16,requires_grad=True)
        target=torch.full_like(p,.2);target[0,:,14]=torch.tensor(left);target[0,:,15]=-1
        mask=torch.ones_like(p);s=torch.zeros(1,16);s[0,14]=state;s[0,15]=-1
        return p,target,mask,s

    def test_first_step_closure_only_adds_left_side_gradients(self):
        p,t,m,s=self.inputs([-1,-1,-1,-1])
        loss,count=left_grasp_loss(p,t,m,s);self.assertEqual(count,8)
        loss.backward()
        self.assertGreater(p.grad[0,0,14],0)
        self.assertFalse(p.grad[...,7:14].any());self.assertFalse(p.grad[...,15].any())
        self.assertFalse(p.grad[:,1:].any())

    def test_preclosing_open_target_keeps_opening_gradient(self):
        p,t,m,s=self.inputs([1,1,-1,-1])
        loss,count=left_grasp_loss(p,t,m,s);self.assertEqual(count,24)
        loss.backward();self.assertLess(p.grad[0,1,14],0)
        self.assertGreater(p.grad[0,2,14],0);self.assertLess(p.grad[0,0,14],0)

    def test_opening_only_does_not_add_closure_loss(self):
        p,t,m,s=self.inputs([-1,-1,1,1],state=-1)
        loss,count=left_grasp_loss(p,t,m,s);self.assertEqual(count,0);self.assertEqual(loss.item(),0)

    def test_holding_and_lifting_have_no_additional_weight(self):
        p,t,m,s=self.inputs([-1,-1,-1,-1],state=-.2)
        self.assertEqual(left_grasp_loss(p,t,m,s)[1],0)

    def test_empty_fully_closed_hand_is_not_a_holding_proxy(self):
        p,t,m,s=self.inputs([-1,-1,-1,-1],state=-1)
        self.assertEqual(left_grasp_loss(p,t,m,s)[1],0)

    def test_release_stops_extra_weight_but_keeps_original_targets(self):
        p,t,m,s=self.inputs([1,-1,1,1],state=1)
        before=m.clone();loss,count=left_grasp_loss(p,t,m,s)
        self.assertEqual(count,16);loss.backward()
        self.assertFalse(p.grad[:,2:].any());self.assertTrue(torch.equal(m,before))

    def test_handoff_before_left_closure_is_not_approach_supervision(self):
        p,t,m,s=self.inputs([1,1,-1,-1],state=1)
        s[0,15]=1;t[0,:,15]=torch.tensor([1,-1,-1,-1])
        self.assertEqual(left_grasp_loss(p,t,m,s)[1],0)

    def test_right_hand_already_holding_excludes_handoff(self):
        p,t,m,s=self.inputs([1,1,-1,-1],state=1);s[0,15]=-.2
        self.assertEqual(left_grasp_loss(p,t,m,s)[1],0)

    def test_invalid_targets_are_excluded(self):
        p,t,m,s=self.inputs([-1,-1,-1,-1]);m[:,1:]=0;t[:,1:]=float('nan')
        loss,count=left_grasp_loss(p,t,m,s);self.assertEqual(count,8);self.assertTrue(torch.isfinite(loss))

    def test_original_and_extra_losses_train_the_whole_model(self):
        torch.manual_seed(42)
        spec=dict(visual_dim=8,text_dim=6,state_dim=16,action_dim=16,horizon=4,
                  tokens=2,obs_steps=2,width=8,layers=1,heads=2,gated_future=True)
        model=DynamicsStudentPolicy(**spec)
        cfg=dict(model=spec,initial_action_weight=.5,dynamics_teacher_weight=.1,
                 first_action_weight=.5,gripper_action_weight=.25,gripper_transition_weight=.5)
        batch=dict(visual=torch.randn(1,2,2,8),text=torch.randn(1,6),state=torch.zeros(1,16),
                   action=torch.randn(1,4,16),action_mask=torch.ones(1,4,16),
                   dynamics_action=torch.randn(1,4,16),teacher_future=torch.randn(1,2,8))
        batch['state'][0,14]=1;batch['state'][0,15]=-1;batch['action'][0,:,14:16]=-1
        result=model(batch['visual'],batch['text'],batch['state'])
        original,old_metrics=loss_and_metrics(model,batch,cfg,True,result)
        total,metrics=loss_and_metrics(model,batch,dict(cfg,left_grasp_action_weight=.5),True,result)
        extra,count=left_grasp_loss(result['action'],batch['action'],batch['action_mask'],batch['state'])
        torch.testing.assert_close(total,original+.5*extra,rtol=0,atol=0)
        self.assertEqual(metrics['left_grasp_action_mse'][1],count)
        for key,value in old_metrics.items():self.assertEqual(metrics[key],value)
        total.backward();self.assertTrue(all(p.requires_grad for p in model.parameters()))
        for grad in (model.visual.weight.grad,model.dynamics_out.weight.grad,
                     model.refine_out.weight.grad[7:14],model.refine_out.weight.grad[15]):
            self.assertGreater(float(grad.abs().sum()),0)

    def test_zero_extra_weight_is_the_original_objective(self):
        p,t,m,s=self.inputs([1,1,1,1])
        model=SimpleNamespace(future=False,gated_future=False,predict_dynamics=lambda *args:torch.zeros(1,2,8))
        batch=dict(action=t,action_mask=m,state=s,visual=torch.zeros(1,2,2,8),text=torch.zeros(1,6),
                   dynamics_action=t,teacher_future=torch.zeros(1,2,8))
        cfg=dict(initial_action_weight=.5,dynamics_teacher_weight=.1,first_action_weight=.5,
                 gripper_action_weight=.25,gripper_transition_weight=.5)
        result=dict(action=p,initial_action=p)
        a,am=loss_and_metrics(model,batch,cfg,True,result)
        b,bm=loss_and_metrics(model,batch,dict(cfg,left_grasp_action_weight=0),True,result)
        torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertEqual(am,bm)


if __name__=='__main__':unittest.main()
