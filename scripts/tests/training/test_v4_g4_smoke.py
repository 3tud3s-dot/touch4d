import tempfile
from pathlib import Path
import unittest
import torch
from scripts.training.v4_model import V4Stage1Predictor
from scripts.training.v4_runtime import save_checkpoint, load_checkpoint
from scripts.experiments.v4_g4_global_motion.model import G4GlobalMotionPredictor
from scripts.experiments.v4_g4_global_motion.smoke import transfer_shared, save_g4, restore_g4, same_tree


class G4SmokeTests(unittest.TestCase):
    def test_transfer_checks_config_and_zero_projection(self):
        g0=V4Stage1Predictor(horizon=5);g4=G4GlobalMotionPredictor(horizon=5)
        extra=transfer_shared(g0,g4)
        self.assertTrue(extra)
        self.assertTrue(all(k.startswith('global_motion.') for k in extra))
        for k,v in g0.state_dict().items(): self.assertTrue(torch.equal(v,g4.state_dict()[k]))
        with self.assertRaises(ValueError): transfer_shared(g0,G4GlobalMotionPredictor(horizon=15))
        with torch.no_grad():g4.global_motion.mapping[-1].bias.fill_(1)
        with self.assertRaises(ValueError):transfer_shared(g0,g4)

    def test_checkpoint_independent_strict_restore_and_no_clobber(self):
        model=G4GlobalMotionPredictor(horizon=5)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        # Exercise nonempty Adam state using a tiny synthetic diagnostic update.
        model.global_motion.mapping[-1].weight.sum().backward();optimizer.step()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'g4.pt'
            save_g4(path,model,optimizer,1,{'scale':1},'test-sha')
            other=G4GlobalMotionPredictor(horizon=5)
            opt2=torch.optim.AdamW(other.parameters(),lr=1e-4)
            state=restore_g4(path,other,opt2,{'scale':1},'test-sha')
            self.assertEqual(state['successful_steps'],1)
            self.assertTrue(same_tree(optimizer.state_dict(),opt2.state_dict()))
            self.assertTrue(same_tree(model.state_dict(),other.state_dict()))
            with self.assertRaises(FileExistsError):save_g4(path,model,optimizer,1,{'scale':1},'test-sha')
            with self.assertRaises(ValueError):restore_g4(path,other,None,{'scale':2},'test-sha')
            with self.assertRaises(ValueError):restore_g4(path,other,None,{'scale':1},'wrong-sha')
            with self.assertRaises(ValueError):restore_g4(path,G4GlobalMotionPredictor(horizon=15),None,{'scale':1},'test-sha')
            with self.assertRaises(ValueError):load_checkpoint(path,V4Stage1Predictor(horizon=5))
            old=Path(d)/'g0.pt';base=V4Stage1Predictor(horizon=5)
            save_checkpoint(old,base,torch.optim.AdamW(base.parameters()),0,source_sha='old',processor_state={'scale':1})
            with self.assertRaises(ValueError):restore_g4(old,other,None,{'scale':1},'old')


if __name__=='__main__':unittest.main()
