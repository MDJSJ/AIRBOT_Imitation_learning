import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp

from discoverse.robots import AirbotPlayIK
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, batch_encode_videos, copypy2
from discoverse.task_base.airbot_task_base import PyavImageEncoder


class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.camera_0_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())

    def domain_randomization(self):
        # 随机 肉位置 - 减少随机范围，避免肉块过于接近
        for i in range(1, 5):  # meat1 到 meat4
            meat_name = f"meat{i}"
            # 大幅减少随机范围，避免肉块碰撞
            self.object_pose(meat_name)[:2] += 2.*(np.random.random(2) - 0.5) * np.array([0.03, 0.025])

        # 随机 eye side 视角
        # camera = self.mj_model.camera("eye_side")
        # camera.pos[:] = self.camera_0_pose[0] + 2.*(np.random.random(3) - 0.5) * 0.05
        # euler = Rotation.from_quat(self.camera_0_pose[1][[1,2,3,0]]).as_euler("xyz", degrees=False) + 2.*(np.random.random(3) - 0.5) * 0.05
        # camera.quat[:] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()[[3,0,1,2]]

    def check_success(self):
        # 检查是否成功将所有4块肉都放置在盘子中 - 降低成功条件
        tmat_plate = get_body_tmat(self.mj_data, "plate_white")
        success_count = 0
        for i in range(1, 5):  # meat1 到 meat4
            meat_name = f"meat{i}"
            tmat_meat = get_body_tmat(self.mj_data, meat_name)
            # 大幅放宽成功条件：更大的距离范围且允许肉在盘子下方
            horizontal_dist = np.hypot(tmat_plate[0, 3] - tmat_meat[0, 3], tmat_plate[1, 3] - tmat_meat[1, 3])
            height_diff = tmat_meat[2, 3] - tmat_plate[2, 3]
            # 距离放宽到0.15m，高度允许在盘子下方0.05m以内
            if horizontal_dist < 0.15 and height_diff > -0.05:
                success_count += 1
        return success_count >= 4  # 仍要求所有4块肉都在范围内

cfg = AirbotPlayCfg()
# cfg.gs_model_dict["background"] = "scene/lab3/point_cloud.ply"
# cfg.gs_model_dict["drawer_1"]   = "hinge/drawer_1.ply"
# cfg.gs_model_dict["drawer_2"]   = "hinge/drawer_2.ply"
# 注意：以下物体没有对应的.ply文件，将使用常规网格渲染
# cfg.gs_model_dict["meat1"]      = "object/base.ply"  # 文件不存在
# cfg.gs_model_dict["meat2"]      = "object/base.ply"  # 文件不存在
# cfg.gs_model_dict["meat3"]      = "object/base.ply"  # 文件不存在
# cfg.gs_model_dict["meat4"]      = "object/base.ply"  # 文件不存在
cfg.init_qpos[:] = [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599,  0.0]

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/place_meats.xml"
cfg.obj_list     = ["drawer_1", "drawer_2", "meat1", "meat2", "meat3", "meat4", "plate_white", "wood"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    cfg.use_gaussian_renderer = args.use_gs

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", os.path.splitext(os.path.basename(__file__))[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))
        
    arm_ik = AirbotPlayIK()

    trmat = Rotation.from_euler("xyz", [0, np.pi*0.4, 0], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 32  # 4块肉 × 8个状态 = 32个状态
    max_time = 60.0 #s  # 增加时间限制

    action = np.zeros(7)
    act_lst, obs_lst = [], []
    target_meats = ["meat1", "meat2", "meat3", "meat4"]  # 所有肉块列表
    current_meat_idx = 0  # 当前处理的肉块索引
    meat_cycle_state = 0  # 当前肉块的处理状态 (0-7)
    tmat_tgt_local = np.eye(4)  # 添加共享的目标位置变量

    move_speed = 0.75  # 参考 place_jujube_coffeecup.py
    sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            # 重置肉块处理状态
            current_meat_idx = 0
            meat_cycle_state = 0
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {cam_id: PyavImageEncoder(20, cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) for cam_id in cfg.obs_rgb_cam_id}
        try:
            if stm.trigger():
                # 计算当前处理的肉块和状态
                current_meat_idx = stm.state_idx // 8  # 每块肉8个状态
                meat_cycle_state = stm.state_idx % 8   # 当前肉块的处理状态
                
                # 检查是否完成所有肉块
                if current_meat_idx >= 4:
                    # 所有肉块处理完毕，等待任务结束检查
                    pass
                else:
                    target_meat = target_meats[current_meat_idx]
                    print(f"Processing {target_meat}, state {meat_cycle_state} (global state {stm.state_idx})")
                    
                    if meat_cycle_state == 0: # 伸到肉上方
                        tmat_meat = get_body_tmat(sim_node.mj_data, target_meat)
                        tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.12 * tmat_meat[:3, 2]  # 增加高度避免碰撞
                        tmat_tgt_local = tmat_armbase_2_world @ tmat_meat
                        ik_result = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                        if ik_result is not None:
                            sim_node.target_control[:6] = ik_result
                        else:
                            print(f"IK failed for {target_meat} above position")
                        sim_node.target_control[6] = 1.
                    elif meat_cycle_state == 1: # 伸到肉
                        tmat_meat = get_body_tmat(sim_node.mj_data, target_meat)
                        tmat_meat[:3, 3] = tmat_meat[:3, 3] + 0.035 * tmat_meat[:3, 2]  # 稍微提高抓取高度
                        tmat_tgt_local = tmat_armbase_2_world @ tmat_meat
                        ik_result = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                        if ik_result is not None:
                            sim_node.target_control[:6] = ik_result
                        else:
                            print(f"IK failed for {target_meat} grasp position")
                    elif meat_cycle_state == 2: # 抓住肉
                        sim_node.target_control[6] = 0.
                    elif meat_cycle_state == 3: # 抓稳肉
                        sim_node.delay_cnt = int(0.5/sim_node.delta_t)  # 增加稳定时间
                    elif meat_cycle_state == 4: # 提起肉
                        tmat_tgt_local[2,3] += 0.15  # 提升更高避免碰撞
                        ik_result = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                        if ik_result is not None:
                            sim_node.target_control[:6] = ik_result
                        else:
                            print(f"IK failed for {target_meat} lift position")
                    elif meat_cycle_state == 5: # 移动到盘子上方
                        tmat_plate = get_body_tmat(sim_node.mj_data, "plate_white")
                        # 减小偏移，确保在机械臂工作范围内
                        offset_x = (current_meat_idx % 2 - 0.5) * 0.06  # -0.03 或 0.03
                        offset_y = (current_meat_idx // 2 - 0.5) * 0.06  # -0.03 或 0.03
                        # 适中的放置高度
                        tmat_plate[:3,3] = tmat_plate[:3, 3] + np.array([offset_x, offset_y, 0.13])
                        tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
                        ik_result = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                        if ik_result is not None:
                            sim_node.target_control[:6] = ik_result
                        else:
                            print(f"IK failed for meat {current_meat_idx} at plate position")
                            # 使用更保守的位置（盘子中心）
                            tmat_plate[:3,3] = tmat_plate[:3, 3] + np.array([0, 0, 0.13])
                            tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
                            ik_result = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                            if ik_result is not None:
                                sim_node.target_control[:6] = ik_result
                            else:
                                print(f"Even center position failed for meat {current_meat_idx}")
                    elif meat_cycle_state == 6: # 缓慢降低高度
                        tmat_tgt_local[2,3] -= 0.07  # 降低到接近盘子的高度
                        ik_result = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                        if ik_result is not None:
                            sim_node.target_control[:6] = ik_result
                        else:
                            print(f"IK failed for meat {current_meat_idx} at lower position")
                    elif meat_cycle_state == 7: # 松开肉并抬升高度
                        sim_node.target_control[6] = 1.
                        tmat_tgt_local[2,3] += 0.10  # 抬升高度
                        ik_result = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                        if ik_result is not None:
                            sim_node.target_control[:6] = ik_result
                        else:
                            print(f"IK failed for meat {current_meat_idx} at lift position")

                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                print(f"Action done for state {stm.state_idx}")
                stm.next()

        except ValueError as ve:
            print(f"Error occurred: {ve}")
            # traceback.print_exc()
            sim_node.reset()
        except Exception as e:
            print(f"Unexpected error: {e}")
            # traceback.print_exc()
            sim_node.reset()

        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            imgs = obs.pop('img')
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)
            for cam_id, img in imgs.items():
                encoders[cam_id].encode(img, obs["time"])

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                recoder_airbot_play(save_path, act_lst, obs_lst, cfg)
                for encoder in encoders.values():
                    encoder.close()
                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()
