"""Test cross-mesh resharding."""
import unittest
from alpa.pipeline_parallel.runtime_emitter import PipelineInstEmitter

import jax
from jax import xla
from jax.core import Var
from jax._src.abstract_arrays import ShapedArray
from jax.interpreters.pxla import (Chunked, NoSharding, Replicated, ShardedAxis,
                                   ShardingSpec, spec_to_indices)
import jax.numpy as jnp
import numpy as np

from alpa import init
from alpa.device_mesh import (DistributedArray, create_remote_array_refs,
                              get_global_virtual_physical_mesh)
from alpa.mesh_executable import next_mesh_executable_uuid
from alpa.global_env import global_config
from alpa.pipeline_parallel.cross_mesh_resharding import (
    CollectiveGroup, ReshardingTaskSpec, CrossMeshCommunicator,
    SymbolicReshardingTask, SymbolicBroadcastReshardingTask)
from alpa.pipeline_parallel.pipeshard_executable import (
    AllocateZeroWorkerExecutableConfig, PipelineInstruction,
    PipeshardMeshWorkerExecuable)
from alpa.pipeline_parallel.resharding_tensor import VirtualDistributedArray
from alpa.testing import assert_allclose
from alpa.util import get_shard_shape, OrderedSet
from alpa.timer import timers

def work(mesh_shape_list, host_indices, device_indices, 
         use_local_allgather, resharding_mode, tensor_shapes, 
         mesh_ids, src_specs, dst_specs):
    init(cluster="ray")

    global_config.use_local_allgather = use_local_allgather
    global_config.resharding_mode = resharding_mode

    virtual_mesh = get_global_virtual_physical_mesh()
    n_meshes = len(mesh_shape_list)
    
    meshes = []
    for i in range(n_meshes):
        src_host_indices = host_indices[i]
        src_device_indices = device_indices[i]
        src_mesh = virtual_mesh.slice_2d(src_host_indices, src_device_indices).get_physical_mesh()
        meshes.append(src_mesh)


    mesh_loads = [{m: 0 for m in mesh.device_strs} for mesh in meshes]

    instruction_lists = {}
    executable_config_lists = {}
    for mesh in meshes:
        for worker in mesh.workers:
            instruction_lists[worker] = []
            executable_config_lists[worker] = []

    collective_groups = [[None for _ in range(n_meshes)] for _ in range(n_meshes)]
    for i in range(n_meshes):
        for j in range(i+1, n_meshes):
            src_mesh = meshes[i]
            dst_mesh = meshes[j]
            device_strs = OrderedSet(src_mesh.device_strs + dst_mesh.device_strs)
            cg = CollectiveGroup(device_strs, src_mesh, dst_mesh)
            if global_config.eagerly_create_communicators:
                cg.instantiate_now()
            else:
                cg.instantiate()
            collective_groups[i][j] = cg
            collective_groups[j][i] = cg

    input_uuids_all = [[] for _ in range(n_meshes)]
    output_uuids_all = [[] for _ in range(n_meshes)]
    src_uuid_all = [[] for _ in range(n_meshes)]
    dst_uuid_all = [[] for _ in range(n_meshes)]
    uuid_counter = 21474

    for tensor_shape, (src, dst), src_sharding_spec, dst_sharding_spec in zip(tensor_shapes,
                                                                              mesh_ids,
                                                                              src_specs,
                                                                              dst_specs):
        tensor_dtype = jnp.int32
        var = Var(0, "", ShapedArray(tensor_shape, tensor_dtype))

        src_mesh = meshes[src]
        dst_mesh = meshes[dst]
        src_loads = mesh_loads[src]
        dst_loads = mesh_loads[dst]

        if resharding_mode == "send_recv":
            rewrite_dst_sharding_spec = CrossMeshCommunicator._rewrite_allgather_spec(
                dst_sharding_spec, dst_mesh, var.aval.shape)
        else:
            rewrite_dst_sharding_spec = dst_sharding_spec

        src_array = VirtualDistributedArray(device_mesh=src_mesh,
                                            aval=var.aval,
                                            sharding_spec=src_sharding_spec)
        dst_array = VirtualDistributedArray(device_mesh=dst_mesh,
                                            aval=var.aval,
                                            sharding_spec=rewrite_dst_sharding_spec)
        task_spec = ReshardingTaskSpec(src_array, dst_array, dst_sharding_spec)
        if resharding_mode == "send_recv":
            strategy = CrossMeshCommunicator._generate_send_recv_resharding_strategy_by_loads(
                task_spec, src_loads, dst_loads)
        else:
            strategy = CrossMeshCommunicator._generate_broadcast_resharding_strategy_by_loads(
                task_spec, src_loads, dst_loads)
        task_spec.set_resharding_strategy(strategy)

        collective_group = collective_groups[src][dst]
        if resharding_mode == "send_recv":
            task = SymbolicReshardingTask(task_spec, collective_group, src_mesh,
                                          dst_mesh)
        else:
            task = SymbolicBroadcastReshardingTask(task_spec, collective_group,
                                                   src_mesh, dst_mesh)

        src_uuid = uuid_counter
        dst_uuid = uuid_counter + 1
        uuid_counter = uuid_counter + 2

        # allocate the buffer
        exec_uuid = next_mesh_executable_uuid()
        config = AllocateZeroWorkerExecutableConfig(
            exec_uuid, [get_shard_shape(var.aval, rewrite_dst_sharding_spec)],
            [var.aval.dtype])
        output_uuids = [dst_uuid]
        for worker in dst_mesh.workers:
            executable_config_lists[worker].append(config)
            in_uuids = []
            out_uuids = output_uuids
            instruction_lists[worker].append(
                PipelineInstruction.run(config.exec_uuid,
                                        in_uuids,
                                        out_uuids, {
                                            "sync_before": False,
                                            "sync_after": False
                                        },
                                        info="allocate zero for recv")
                )

        # Create resharding task
        if resharding_mode == "send_recv":
            PipelineInstEmitter._compile_resharding_task(src_uuid, task, dst_uuid,
                                                         instruction_lists)
        else:
            PipelineInstEmitter._compile_broadcast_resharding_task(
                src_mesh, src_uuid, task, dst_uuid, instruction_lists)
        src_uuid_all[src].append(src_uuid)
        dst_uuid_all[dst].append(dst_uuid)

        # Prepare array and shard args
        test_array = np.arange(np.prod(var.aval.shape),
                               dtype=var.aval.dtype).reshape(var.aval.shape)
        indices = spec_to_indices(var.aval.shape, src_sharding_spec)
        test_array = xla.canonicalize_dtype(test_array)
        input_refs = src_mesh.shard_args_to_bufs([indices], (False,), (False,),
                                                  None, [test_array])
        input_refs = np.array(input_refs)
        input_uuids = [ref.uuid for ref in input_refs]
        output_refs, output_uuids = create_remote_array_refs(dst_mesh)
        
        input_uuids_all[src] += list(input_uuids)
        output_uuids_all[dst] += list(output_uuids)

        # for i, mesh in enumerate(meshes):
        #     print(src_uuid_all[i], dst_uuid_all[i], "v.s.", input_uuids_all[i], output_uuids_all[i])
    
    exec_uuids = {}
    # Compile Pipeline Executable
    for i, mesh in enumerate(meshes):
        # print(src_uuid_all[i], dst_uuid_all[i], "v.s.", input_uuids_all[i], output_uuids_all[i])
        for worker in mesh.workers:
            exec_uuid = next_mesh_executable_uuid()
            worker.put_executable.remote(exec_uuid, PipeshardMeshWorkerExecuable,
                                         instruction_lists[worker], src_uuid_all[i], dst_uuid_all[i],
                                         executable_config_lists[worker], [], [],
                                         [False] * src_mesh.num_devices_per_host)
            exec_uuids[worker] = exec_uuid

    # Run executables
    for _ in range(3):
        timers("overall_resharding_time").start()
        for i, mesh in enumerate(meshes):
            for worker in mesh.workers:
                worker.run_executable.remote(exec_uuids[worker],
                                             input_uuids_all[i], output_uuids_all[i],
                                             sync_for_timer=True)
        for mesh in meshes:
            mesh.sync_workers()
        timers("overall_resharding_time").stop()
        timers("overall_resharding_time").log()
        timers("overall_resharding_time").reset()

    for mesh in meshes:
        mesh.shutdown()



def task1():
    mesh_shape_list = [(1, 2), (1, 2), (1, 4)]
    host_indices = [[0], [0], [0]]
    device_indices = [[[0,1]], [[2,3]], [[4,5,6,7]]]
    use_local_allgather = False
    resharding_mode = "send_recv"
    tensor_shapes = [(128, 1024, 1024), (128, 1024, 1024), (128, 1024, 1024)]
    mesh_ids = [(0, 1), (0, 2), (1, 2)]
    src_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                 ShardingSpec([NoSharding(), Chunked([2]), NoSharding()], [ShardedAxis(0)]),
                 ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0)])]
    dst_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                 ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)]),
                 ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)])]
    
    work(mesh_shape_list, host_indices, device_indices, 
         use_local_allgather, resharding_mode, tensor_shapes, 
         mesh_ids, src_specs, dst_specs)
# default stream: 385
# multistream: 324
# send+recv 2 stream: 229

def task2():
    mesh_shape_list = [(1, 2), (1, 2), (1, 4)]
    host_indices = [[0], [0], [0]]
    device_indices = [[[0,1]], [[2,3]], [[4,5,6,7]]]
    use_local_allgather = False
    resharding_mode = "send_recv"
    k = 2
    tensor_shapes = [(128, 1024, 1024)]*3*k
    mesh_ids = [(0, 1), (0, 2), (1, 2)]*k
    tmp_src_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), Chunked([2]), NoSharding()], [ShardedAxis(0)]),
                     ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0)])]
    tmp_dst_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)])]

    src_specs = tmp_src_specs*k
    dst_specs = tmp_dst_specs*k
    
    work(mesh_shape_list, host_indices, device_indices, 
         use_local_allgather, resharding_mode, tensor_shapes, 
         mesh_ids, src_specs, dst_specs)
# default stream: 766
# multistream: 641
# send+recv 2 stream: 420

def task3():
    mesh_shape_list = [(1, 2), (1, 2), (1, 4)]
    host_indices = [[0], [0], [0]]
    device_indices = [[[0,1]], [[2,3]], [[4,5,6,7]]]
    use_local_allgather = False
    resharding_mode = "send_recv"
    k = 4
    tensor_shapes = [(16, 1024, 1024)]*3*k
    mesh_ids = [(0, 1), (0, 2), (1, 2)]*k
    tmp_src_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                    ShardingSpec([NoSharding(), Chunked([2]), NoSharding()], [ShardedAxis(0)]),
                    ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0)])]
    tmp_dst_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                    ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)]),
                    ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(4)])]
    
    src_specs = tmp_src_specs*k
    dst_specs = tmp_dst_specs*k
    
    work(mesh_shape_list, host_indices, device_indices, 
         use_local_allgather, resharding_mode, tensor_shapes, 
         mesh_ids, src_specs, dst_specs)
# default stream: 209
# multistream: 177
# send+recv 2 stream: 122

def task4():
    mesh_shape_list = [(1, 2), (1, 2), (1, 2), (1, 2)]
    host_indices = [[0], [0], [0], [0]]
    device_indices = [[[0,1]], [[2,3]], [[4,5]], [[6,7]]]
    use_local_allgather = False
    resharding_mode = "send_recv"
    k=2
    tensor_shapes = [(16, 1024, 1024)]*4*k
    mesh_ids = [(0, 1), (1, 2), (2, 3), (3, 0)]*k
    tmp_src_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), Chunked([2]), NoSharding()], [ShardedAxis(0)]),
                     ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0)]),
                     ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0)])]
    src_specs = tmp_src_specs*k
    tmp_dst_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)])]
    dst_specs = tmp_dst_specs*k
    
    work(mesh_shape_list, host_indices, device_indices, 
         use_local_allgather, resharding_mode, tensor_shapes, 
         mesh_ids, src_specs, dst_specs)
# default stream: 70
# multistream: 68
# send+recv 2 stream: 31


def task5():
    mesh_shape_list = [(1, 2), (1, 2), (1, 2), (1, 2)]
    host_indices = [[0], [0], [0], [0]]
    device_indices = [[[0,1]], [[2,3]], [[4,5]], [[6,7]]]
    use_local_allgather = False
    resharding_mode = "send_recv"
    k=4
    tensor_shapes = [(16, 1024, 1024)]*4*k
    mesh_ids = [(0, 1), (1, 2), (2, 3), (3, 0)]*k
    tmp_src_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), Chunked([2]), NoSharding()], [ShardedAxis(0)]),
                     ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0)]),
                     ShardingSpec([Chunked([2]), NoSharding(), NoSharding()], [ShardedAxis(0)])]
    src_specs = tmp_src_specs*k
    tmp_dst_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(2)])]
    dst_specs = tmp_dst_specs*k
    
    work(mesh_shape_list, host_indices, device_indices, 
         use_local_allgather, resharding_mode, tensor_shapes, 
         mesh_ids, src_specs, dst_specs)
# default stream: 132
# multistream: 127
# send+recv 2 stream: 55

def task6():
    mesh_shape_list = [(1, 1)]*8
    host_indices = [[0] for _ in range(8)]
    device_indices = [[[i]] for i in range(8)]
    use_local_allgather = False
    resharding_mode = "send_recv"
    k=1
    tensor_shapes = [(16, 1024, 1024)]*8*k
    mesh_ids = [(0, 1), (0, 2), (0, 3), (0, 4), (2, 5), (3, 6), (4, 7), (7, 0)]*k
    tmp_src_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)])]
    src_specs = tmp_src_specs*k
    tmp_dst_specs = [ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)]),
                     ShardingSpec([NoSharding(), NoSharding(), NoSharding()], [Replicated(1)])]
    dst_specs = tmp_dst_specs*k
    
    work(mesh_shape_list, host_indices, device_indices, 
         use_local_allgather, resharding_mode, tensor_shapes, 
         mesh_ids, src_specs, dst_specs)
# default stream: 40
# multistream: 39
# send+recv 2 stream: 24

if __name__ == '__main__':
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()