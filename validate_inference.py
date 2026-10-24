"""Compare the adapter with pinned author conversion and inference functions."""
import argparse
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from adapter import HumanoidGPT
from motion_challenge.physics import World
from motion_challenge.sdk import ReferenceFrame
from motion_challenge.validation import external_output_path


def selected(source, path, names, namespace):
    nodes = []
    for node in ast.parse((source / path).read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            methods = names[node.name]
            if methods is not None:
                node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
                assert {n.name for n in node.body} == methods
            nodes.append(node)
    assert {n.name for n in nodes} == set(names)
    future = ast.parse('from __future__ import annotations').body
    exec(compile(ast.Module(future + nodes, []), str(source / path), 'exec'), namespace)


def load_author(source):
    constants = {'np': np, 'os': NS(environ={}), 'Path': Path,
                 'PATH_ASSET': source / 'storage/assets', 'HAND': 'Default'}
    nodes = [n for n in ast.parse((source / 'tracking/constants.py').read_text()).body
             if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(nodes, []), str(source / 'tracking/constants.py'), 'exec'), constants)
    consts = NS(**{k: v for k, v in constants.items() if not k.startswith('__')})
    namespace = {'np': np, 'mujoco': mujoco, 'R': R, 'Slerp': Slerp, 'consts': consts,
                 'EMA_ALPHA': .8, 'NUM_ACTION': 29, 'NUM_STATE': 136}
    for name in ('ACTION_JOINT_NAMES', 'OBS_JOINT_NAMES', 'KPT_NAMES'):
        namespace[name] = getattr(consts, name)
    selected(source, 'utils/transforms_np.py', dict.fromkeys([
        'WarpPi', 'quat2yaw', 'quat2mat', 'wxyz2xyzw', 'xyzw2wxyz', 'batch_base2navi']), namespace)
    selected(source, 'utils/sim_mj.py', dict.fromkeys([
        'get_qpos_ids', 'get_dof_ids', 'get_sensor_data', 'get_collision_info', 'geoms_colliding']), namespace)
    namespace['mj_sensor'] = namespace['get_sensor_data']
    namespace['mj_coll'] = namespace['geoms_colliding']
    selected(source, 'tracking/convert_qpos2kpt.py', dict.fromkeys([
        'resample_state', 'recompute_qvel', 'mj_body_pose', 'extract_kpt', 'qpos2kpt']), namespace)
    selected(source, 'scripts/inference.py', {'_convert_traj_to_kpt': None}, namespace)
    selected(source, 'tracking/infer_utils.py', {
        'apply_ema_qpos': None,
        'G1TrackInferFn': {'__init__', 'infer_onnx', 'update_state', 'update_coord_cmd',
                          'get_nn_state', 'nn2motor_action', 'get_mj_feet_contact'},
    }, namespace)
    selected(source, "projects/transformer/infer_utils.py", {"G1TrackTransformerInferFn": None}, namespace)
    return namespace


def native_model(source, expected):
    root = ET.parse(source / 'storage/assets/unitree_g1_5010/g1_mjx_track.xml').getroot()
    assert len(root.findall('.//body')) == len(root.findall('.//inertial')) == 30
    for parent in root.iter():
        for node in list(parent):
            if node.tag == 'mesh' or node.tag == 'geom' and node.get('mesh') is not None:
                parent.remove(node)
    ET.indent(root)
    assert ET.tostring(root, encoding='unicode') + '\n' == expected.read_text()
    # The author's scene adds this floor. It is used only by ignored contact
    # diagnostics, never by stepping this FK model or by the policy input.
    ET.SubElement(root.find('worldbody'), 'geom', name='floor', type='plane', size='0 0 .1')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root / 'native-sources.json').read_text())
    for item in pins['files']:
        assert hashlib.sha256((args.source / item['path']).read_bytes()).hexdigest() == item['sha256']
    native = load_author(args.source)
    consts = native['consts']
    runtime = json.loads((root / 'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    policy = HumanoidGPT(root, runtime)
    for actual, expected in ((policy.default, consts.DEFAULT_QPOS[7:]),
                             (policy.kp, consts.KPs), (policy.kd, consts.KDs),
                             (policy.scale, consts.ACTION_SCALE * .25)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    model = native_model(args.source, root / runtime['kinematics'])
    data = mujoco.MjData(model)
    qadr = [model.joint(n).qposadr[0] for n in policy.names]
    vadr = [model.joint(n).dofadr[0] for n in policy.names]
    world = World(root / 'MotionChallenge-Core/robots/g1-29dof-cpu-v0.json')
    order = [world.joint_names.index(n) for n in policy.names]
    rng = np.random.default_rng(241024)
    permutation = rng.permutation(29)
    shuffled_names = tuple(np.asarray(world.joint_names)[permutation])
    inverse = [shuffled_names.index(n) for n in policy.names]
    def shuffle(packet):
        fields = dict(joint_position_rad=tuple(np.asarray(packet.joint_position_rad)[permutation]),
                      joint_velocity_rad_s=tuple(np.asarray(packet.joint_velocity_rad_s)[permutation]))
        if hasattr(packet, 'joint_names'):
            fields['joint_names'] = shuffled_names
        return replace(packet, **fields)
    errors = dict(reference_qpos=0., reference_cvel=0., observation=0., action=0.,
                  motor_target=0., history=0., same_action_motor_target=0.)
    def compare(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a) - b))))
    endpoints = 0
    captured = {}
    original = policy.infer
    def capture(inputs):
        captured['adapter_obs'] = np.asarray(inputs['obs'], np.float32).copy()
        result = original(inputs)
        captured['adapter_action'] = result['continuous_actions'].copy()
        return result
    policy.infer = capture
    # Use the author's unmodified policy session configuration and infer method.
    policy_namespace = {'__name__': 'author_policy'}
    exec(compile((args.source / 'tracking/policy.py').read_text(), 'author_policy', 'exec'), policy_namespace)
    author_policy = policy_namespace['Transformer_Policy_ONNX'](NS(device='cpu', onnx_track=str(args.weights)))
    author_infer = author_policy.infer
    def native_infer(obs):
        captured['native_obs'] = obs.copy()
        action = author_infer(obs)
        captured['native_action'] = action.copy()
        return action
    author_policy.infer = native_infer
    for reset in range(4):
        count = 34
        t = np.arange(count) * .02
        qpos = np.zeros((count, 36), np.float32)
        qpos[:, :3] = np.c_[.2*t + rng.uniform(-1, 1), .1*np.sin(t), .8+.05*np.cos(t)]
        angles = np.c_[.2*np.sin(t*2), .15*np.cos(t*3), .4*t + reset*.7]
        qpos[:, 3:7] = R.from_euler('xyz', angles).as_quat(scalar_first=True)
        qpos[7::9, 3:7] *= -1  # Quaternion sign continuity in the reference filter.
        qpos[:, 7:] = consts.DEFAULT_QPOS[7:] + .15*np.sin(3*t[:, None] + np.arange(29)*.1)
        smooth = native['apply_ema_qpos'](qpos)
        features = native['_convert_traj_to_kpt']({'qpos': smooth, 'frequency': 50}, model, 50)
        refs = []
        for i, row in enumerate(qpos):
            canonical_q = np.empty(29); canonical_q[order] = row[7:]
            refs.append(ReferenceFrame(i*.02, i*.02, tuple(canonical_q), tuple(row[:3]),
                                       tuple(row[3:7][[1, 2, 3, 0]]), (0.,)*29))
        fn = native['G1TrackTransformerInferFn'](NS(ctrl_dt=.02, action_scale=.25, soft_joint_pos_limit_factor=.9),
                                    model, author_policy, privileged=False, history_len=16)
        for step in range(40):
            index = min(step, count-1); future = min(step+1, count-1)
            endpoints += int(future == count-1)
            row = qpos[index].astype(np.float64)
            row[:3] += rng.normal(0, .03, 3)
            row[3:7] = R.from_euler('xyz', angles[index] + rng.normal(0, .03, 3)).as_quat(scalar_first=True)
            row[7:] += rng.normal(0, .02, 29)
            world.data.qpos[:7] = row[:7]
            world.data.qpos[world.qadr] = row[7:][np.argsort(order)]
            world.data.qvel[:] = rng.normal(0, .3, 35)
            world.data.time = step*.02
            mujoco.mj_forward(world.model, world.data)
            state = world.state()
            data.qpos[:7] = world.data.qpos[:7]
            data.qpos[qadr] = np.asarray(state.joint_position_rad)[order]
            data.qvel[:6] = world.data.qvel[:6]
            data.qvel[vadr] = np.asarray(state.joint_velocity_rad_s)[order]
            mujoco.mj_forward(model, data)
            packet = shuffle(state)
            window = tuple(shuffle(replace(refs[i], requested_time_s=when))
                           for i, when in ((index, step*.02), (future, (step+1)*.02)))
            if step == 0:
                policy.reset(packet, window, reset)
            target = policy.act(packet, window)
            ref_state = {'ref_curr': {k: v[index:index+1] for k, v in features.items()},
                         'ref_next': {k: v[future:future+1] for k, v in features.items()}}
            native_target = fn.infer_onnx(NS(mj_data=data), ref_state)
            compare('reference_qpos', policy.reference.cache[round(refs[future].source_time_s, 9)]['qpos'], features['qpos'][future])
            compare('reference_cvel', policy.reference.cache[round(refs[future].source_time_s, 9)]['cvel'], features['kpt_cvel_in_gv'][future, 0])
            compare('observation', captured['adapter_obs'], captured['native_obs'])
            compare('action', captured['adapter_action'], captured['native_action'])
            compare('motor_target', np.asarray(target.joint_position_target_rad)[inverse], native_target[0])
            compare('history', policy.last, fn.info['last_action'][0])
            same = fn.nn2motor_action(captured['adapter_action'])[0]
            compare('same_action_motor_target', np.asarray(target.joint_position_target_rad)[inverse], same)
    tolerance = 2e-5
    result = dict(status='passed' if max(errors.values()) <= tolerance else 'failed',
                  cases=160, resets=4, endpoint_cases=endpoints, shuffled_joint_order=True,
                  quaternion_sign_flips=True, max_abs_errors=errors, tolerance=tolerance,
                  source_commit=pins['commit'], checkpoint_sha256=runtime['weights_sha256'], history_len=16,
                  comparison='Unmodified author EMA, qpos conversion, observation, state update, CPU ONNX inference and motor mapping; identical actual states and independently maintained action histories.',
                  adaptations='Canonical reference arrives at50Hz; EMA is policy-input-only. Fresh inference instance per clip; hold the final feature at endpoint. Author FK inertials retained, benchmark physics unchanged.')
    output = external_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
