"""CPU contracts for separate depth-v1 metadata and depth target preparation."""
from contextlib import ExitStack, contextmanager
import ast
import copy
import hashlib
from io import BytesIO, StringIO
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.dataloader.dataset import DepthInfo, read_depth
from scripts.rebuild_dataset import depth
from scripts.tests.core_ast_audit import ast_sha256, canonical_ast


class DepthMetadataFixture:
    """A complete small extension exists in memory; no filesystem writes."""

    def __init__(self):
        self.path=Path('__memory_only_depth__/008-pink-cloth/episode_0').resolve()
        self.core_path=Path('__memory_only_core__/008-pink-cloth/episode_0').resolve()
        self.timestamps=np.array([100000,132000],dtype=np.int64)
        self.cameras=['cam0']
        self.binding={'metadata_sha256':'a'*64,'timestamps_sha256':'b'*64}
        self.core={'schema_version':'touch4d-v1','scene':'008-pink-cloth','episode':0,
                   'frames':2,'observation_cameras':self.cameras}
        self.intrinsics=np.array([[[100.,0.,1.],[0.,101.,1.],[0.,0.,1.]]],dtype=np.float64)
        self.camera_to_world=np.eye(4,dtype=np.float64)[None]
        self.camera_to_world[0,:3,3]=[.2,-.1,.3]
        self.depth_bytes=b'opaque original uint16 mm H5 payload; no HDF5 I/O in metadata fixture'
        self.metadata={'schema_version':'touch4d-depth-v1','script_version':'1.0.0',
                       'scene':'008-pink-cloth','episode':0,'frames':2,'cameras':self.cameras,
                       'height':2,'width':2,'depth_unit':'mm','invalid_value':0,
                       'depth_kind':'camera_z_expected','extrinsics_convention':'camera_to_world',
                       'coordinate_system':'native_world_metres','core_binding':self.binding.copy()}

    @contextmanager
    def mounted(self):
        stream=BytesIO()
        np.savez(stream,intrinsics=self.intrinsics,camera_to_world=self.camera_to_world,
                 observation_timestamp_us=self.timestamps)
        parameters=stream.getvalue()
        self.metadata['output_files']={'camera_parameters.npz':hashlib.sha256(parameters).hexdigest(),
                                      'cam0/rendered_depth.h5':hashlib.sha256(self.depth_bytes).hexdigest()}
        files={self.path/'metadata.json':json.dumps(self.metadata).encode(),
               self.path/'camera_parameters.npz':parameters,
               self.path/'cam0/rendered_depth.h5':self.depth_bytes}
        original_load=np.load

        def load(path,**kwargs):
            return original_load(BytesIO(files[Path(path)]),**kwargs)

        def open_file(path,*args,**kwargs):
            if path not in files:raise FileNotFoundError(str(path))
            return StringIO(files[path].decode())

        with ExitStack() as stack:
            stack.enter_context(patch.object(depth,'read_core_depth_binding',return_value=(self.core,self.timestamps,self.binding)))
            stack.enter_context(patch.object(Path,'open',open_file))
            stack.enter_context(patch.object(Path,'is_file',lambda p:p in files))
            stack.enter_context(patch.object(Path,'rglob',lambda p,pattern:iter(files) if p==self.path else iter(())))
            stack.enter_context(patch.object(depth,'hash_file',lambda p:hashlib.sha256(files[Path(p)]).hexdigest()))
            stack.enter_context(patch.object(depth.np,'load',load))
            yield self


class CoreDepthTests(unittest.TestCase):
    def test_metadata_preserves_mm_camera_z_and_camera_to_world(self):
        fixture=DepthMetadataFixture()
        with fixture.mounted():
            metadata=depth.read_depth_metadata(fixture.path,fixture.core_path)
        self.assertEqual(metadata['depth_unit'],'mm')
        self.assertEqual(metadata['depth_kind'],'camera_z_expected')
        self.assertEqual(metadata['extrinsics_convention'],'camera_to_world')
        self.assertEqual(metadata['core_binding'],fixture.binding)

    def test_unit_convention_and_stale_binding_fail_closed(self):
        mutations=[('depth_unit','m'),('depth_kind','ray_distance'),
                   ('extrinsics_convention','world_to_camera'),('invalid_value',-1),
                   ('core_binding',{'metadata_sha256':'c'*64,'timestamps_sha256':'b'*64})]
        for key,value in mutations:
            with self.subTest(key=key):
                fixture=DepthMetadataFixture();fixture.metadata[key]=value
                with fixture.mounted(),self.assertRaises(ValueError):
                    depth.read_depth_metadata(fixture.path,fixture.core_path)

    def test_invalid_camera_rotation_is_rejected(self):
        fixture=DepthMetadataFixture()
        fixture.camera_to_world[0,0,0]=-1
        with fixture.mounted(),self.assertRaises(ValueError):
            depth.read_depth_metadata(fixture.path,fixture.core_path)

    def test_reader_converts_only_target_mm_to_metres_preserves_zero_mask_and_camera_order(self):
        """Exercise target preparation with a storage shim, not a fake HDF5 validation claim."""
        class HardLink:pass
        class Dataset:
            is_virtual=False
            external=False
            def __init__(self,values):self.values=values;self.shape=values.shape;self.dtype=values.dtype
            def __getitem__(self,selection):return self.values[selection]
        class Archive:
            def __init__(self,values):self.dataset=Dataset(values)
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def keys(self):return ['data']
            def get(self,name,getlink=False):return HardLink() if getlink else self.dataset
            def __getitem__(self,name):return self.dataset
        values={name:np.full((3,2,2),fill,dtype=np.uint16) for name,fill in [('cam0',1000),('cam1',2000)]}
        values['cam0'][2,0,0]=0
        h5=SimpleNamespace(HardLink=HardLink,Dataset=Dataset,
                           File=lambda p,mode:Archive(values[Path(p).parent.name]))
        intrinsics=np.tile(np.eye(3,dtype=np.float64),(2,1,1));intrinsics[1,0,0]=2
        c2w=np.tile(np.eye(4,dtype=np.float64),(2,1,1));c2w[1,0,3]=.25
        info=DepthInfo(Path('__memory_depth_reader__'),3,('cam0','cam1'),2,2,intrinsics,c2w)
        with patch.dict(sys.modules,{'h5py':h5}):
            output=read_depth(info,0,('cam1','cam0'),horizon=2)
        self.assertEqual(tuple(output['target_depth'].shape),(2,2,2,2))
        self.assertTrue((output['target_depth'][:,0]==2).all())
        self.assertEqual(output['target_depth'][0,1,0,0].item(),1)
        self.assertEqual(output['target_depth'][1,1,0,0].item(),0)
        self.assertFalse(output['target_depth_valid'][1,1,0,0])
        self.assertEqual(output['target_depth_valid'].sum().item(),15)
        self.assertEqual(output['camera_to_world'][0,0,3].item(),.25)
        self.assertEqual(output['camera_intrinsics'][0,0,0].item(),2)

    def test_retained_production_scripts_and_protocol_match_real_metadata(self):
        root=Path(__file__).resolve().parents[2]
        migration=json.loads((root/'assets/current/data_migration.json').read_text(encoding='utf-8'))
        class CanonicalImports(ast.NodeTransformer):
            def __init__(self,allowed_cli):
                self.allowed_cli=allowed_cli
            def visit_FunctionDef(self,node):
                if node.name in self.allowed_cli:
                    return None
                return self.generic_visit(node)
            def visit_If(self,node):
                self.generic_visit(node)
                if (isinstance(node.test,ast.Name) and node.test.id=='__package__'
                    and len(node.body)==len(node.orelse)==1
                    and isinstance(node.body[0],ast.ImportFrom) and isinstance(node.orelse[0],ast.ImportFrom)):
                    relative,direct=node.body[0],node.orelse[0]
                    assert relative.level==1 and direct.level==0 and relative.module==direct.module
                    assert canonical_ast(ast.ImportFrom(module=relative.module,names=relative.names,level=0))==canonical_ast(direct)
                    return direct
                return node
        for episode in [0,1]:
            v1=json.loads((root/f'assets/current/source_metadata/v1_ep{episode}.json').read_text(encoding='utf-8'))
            extension=json.loads((root/f'assets/current/source_metadata/depth_ep{episode}.json').read_text(encoding='utf-8'))
            self.assertEqual(v1['script_version'],'1.3.0')
            for name,digest in v1['scripts_sha256'].items():
                current=root/'scripts/rebuild_dataset'/name
                adaptation=migration['production_converter']['namespace_adaptation'][name]
                self.assertEqual(adaptation['production_source_sha256'],digest)
                self.assertEqual(hashlib.sha256(current.read_bytes()).hexdigest(),adaptation['migrated_sha256'])
                tree=ast.parse(current.read_text(encoding='utf-8'))
                adapters={node.name:ast_sha256(node)
                          for node in tree.body if isinstance(node,ast.FunctionDef)
                          and node.name in adaptation['excluded_cli_functions']}
                self.assertEqual(adapters,adaptation['cli_adapters_ast_sha256'])
                canonical=CanonicalImports(adaptation['excluded_cli_functions']).visit(tree)
                self.assertEqual(ast_sha256(canonical),adaptation['canonical_core_ast_sha256'])
            self.assertEqual(hashlib.sha256((root/'assets/current/dataset_protocol_source.txt').read_bytes()).hexdigest(),v1['protocol_sha256'])
            self.assertEqual(hashlib.sha256((root/'docs/DATASET_PROTOCOL.md').read_bytes()).hexdigest(),v1['protocol_sha256'])
            self.assertEqual(extension['core_binding']['metadata_sha256'],hashlib.sha256((root/f'assets/current/source_metadata/v1_ep{episode}.json').read_bytes()).hexdigest())
            self.assertEqual(extension['core_binding']['timestamps_sha256'],v1['output_files']['timestamps.npz'])
            depth.check_depth_sources_schema(extension)

    def test_ast_audit_normalizes_only_empty_type_parameters(self):
        node=ast.parse('def function(value): return value').body[0]
        empty=copy.deepcopy(node)
        empty.type_params=[]
        no_field=copy.deepcopy(node)
        if hasattr(no_field,'type_params'):
            del no_field.type_params
        self.assertEqual(ast_sha256(empty),ast_sha256(no_field))
        empty.type_params=[ast.Name(id='T',ctx=ast.Load())]
        with self.assertRaisesRegex(ValueError,'Nonempty AST type_params'):
            ast_sha256(empty)
        changed=ast.parse('def function(value): return value + 1').body[0]
        self.assertNotEqual(ast_sha256(node),ast_sha256(changed))

    @unittest.skipUnless(importlib.util.find_spec('h5py') is not None,'h5py absent: real HDF5 I/O remains unverified locally')
    def test_real_hdf5_memory_reader(self):
        import h5py
        stream=BytesIO()
        with h5py.File(stream,'w') as file:
            file.create_dataset('data',data=np.array([[[0]],[[1234]],[[0]]],dtype=np.uint16))
        payload=stream.getvalue();original=h5py.File
        info=DepthInfo(Path('__memory_real_h5__'),3,('cam0',),1,1,np.eye(3)[None],np.eye(4)[None])
        with patch.object(h5py,'File',lambda path,mode:original(BytesIO(payload),mode)):
            output=read_depth(info,0,('cam0',),horizon=2)
        torch.testing.assert_close(output['target_depth'][:,0,0,0],torch.tensor([1.234,0.]))
        self.assertEqual(output['target_depth_valid'][:,0,0,0].tolist(),[True,False])


if __name__=='__main__':
    unittest.main()
