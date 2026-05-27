import time

import numpy as np


class CoordinateMixin:
    """RGBD 像素→3D/世界坐标转换、点击处理与坐标记录管理。"""

    def pixel_to_3d_coordinate(self, camera_name, pixel_x, pixel_y, depth_value):
        """将像素坐标和深度值转换为3D空间坐标"""
        try:
            # 获取相机内参
            intrinsics = self.camera_intrinsics.get(camera_name)
            if not intrinsics:
                print(f"无法获取 {camera_name} 的相机内参")
                return [None, None, None]

            # 提取内参
            fx = intrinsics.get('fx', 615.0)
            fy = intrinsics.get('fy', 615.0)
            cx = intrinsics.get('cx', 320.0)
            cy = intrinsics.get('cy', 240.0)

            # 检查深度值是否有效
            if depth_value is None or depth_value <= 0:
                print(f"无效的深度值: {depth_value}")
                return [None, None, None]
            # 将深度值从毫米转换为米
            depth_meters = depth_value * self.depth_scale

            # 使用针孔相机模型计算3D坐标
            # X = (u - cx) * Z / fx
            # Y = (v - cy) * Z / fy
            # Z = depth
            x = (pixel_x - cx) * depth_meters / fx
            y = (pixel_y - cy) * depth_meters / fy
            z = depth_meters

            return [x, y, z]

        except Exception as e:
            print(f"像素到3D坐标转换失败: {e}")
            return [None, None, None]

    def get_depth_at_pixel(self, camera_name, pixel_x, pixel_y):
        """获取指定像素位置的深度值"""
        try:
            # 获取深度图像
            if camera_name == "head":
                # 如果是RGB相机，尝试获取对应的深度相机图像
                depth_image = self._latest_camera_frame("head_depth")
                if depth_image is None:
                    print("无法获取深度图像")
                    return None
            elif camera_name == "head_depth":
                depth_image = self._latest_camera_frame("head_depth")
            else:
                # 手部相机可能没有深度信息
                print(f"{camera_name} 没有深度信息")
                return None

            if depth_image is None:
                print("depth_image is None")
                return None

            # 确保图像是numpy数组
            if not isinstance(depth_image, np.ndarray):
                print("深度图像格式错误")
                return None

            # 获取图像尺寸
            height, width = depth_image.shape[:2]
            print(f"深度图像尺寸: {width}x{height}, 请求坐标: ({pixel_x}, {pixel_y})")

            # 确保坐标在图像范围内
            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                # 获取深度值
                depth_value = depth_image[pixel_y, pixel_x]

                # 处理不同类型的深度值
                if isinstance(depth_value, (int, float, np.number)):
                    print(f"深度信息 : {float(depth_value)}")
                    return float(depth_value)
                elif isinstance(depth_value, np.ndarray):
                    # 如果是数组，取第一个元素
                    if depth_value.size > 0:
                        print(f"深度信息是数组，取第一个元素 : {float(depth_value.flat[0])} 数值size: {depth_value.size}")
                        return float(depth_value.flat[0])
                    else:
                        print("深度数组为空")
                        return None
                else:
                    print(f"深度值类型错误: {type(depth_value)}, 值: {depth_value}")
                    return None
            else:
                print(f"像素坐标 ({pixel_x}, {pixel_y}) 超出图像范围 ({width}, {height})")
                return None

        except Exception as e:
            print(f"获取深度值失败: {e}")
            return None

    def handle_image_click(self, camera_name, event):
        """处理图像点击事件"""
        try:
            # 获取点击的像素坐标（基于显示尺寸320x240）
            display_x = event.x
            display_y = event.y

            # 根据相机类型使用已知的实际尺寸
            # 这些尺寸来自之前的日志信息
            if camera_name == "head" or camera_name == "head_depth":
                original_width, original_height = 1280, 800
            elif camera_name == "hand_left" or camera_name == "hand_right":
                original_width, original_height = 848, 480
            else:
                # 默认尺寸
                original_width, original_height = 640, 480

            if camera_name == "head":
                display_width, display_height = 800, 600
            else:
                display_width, display_height = 256, 195

            # 坐标映射：将显示坐标转换为原始图像坐标
            pixel_x = int(display_x * original_width / display_width)
            pixel_y = int(display_y * original_height / display_height)

            print(f"点击了 {camera_name} 图像，显示坐标: ({display_x}, {display_y}) -> 原始坐标: ({pixel_x}, {pixel_y}), 原始尺寸: {original_width}x{original_height}")


            # 获取深度值
            depth_value = self.get_depth_at_pixel(camera_name, pixel_x, pixel_y)
            if depth_value is None or depth_value <= 0:
                self.show_status("无法获取该位置的深度信息或深度值无效", "warning")
                return

            # 转换为3D坐标
            coordinates_3d = self.pixel_to_3d_coordinate(camera_name, pixel_x, pixel_y, depth_value)
            self.coordinates_3d = coordinates_3d
            if coordinates_3d and coordinates_3d[0] is not None and coordinates_3d[1] is not None and coordinates_3d[2] is not None:
                x, y, z = coordinates_3d

                # --- 新增：转换为世界坐标系 ---
                try:
                    # 获取当前机器人关节状态
                    arm_states, _ = self.robot.arm_joint_states()
                    head_states, _ = self.robot.head_joint_states()
                    waist_states, _ = self.robot.waist_joint_states()

                    # 提取关键关节角度 [腰部升降(m), 腰部俯仰(rad), 头部偏航(rad), 头部俯仰(rad)]
                    # 注意：waist_states[1] 是高度(cm)，转换为米
                    joint_angles = [
                        (waist_states[1] * 0.01 if waist_states and len(waist_states) >= 2 else 0.0),
                        (waist_states[0] if waist_states and len(waist_states) >= 1 else 0.0),
                        (head_states[0] if head_states and len(head_states) >= 1 else 0.0),
                        (head_states[1] if head_states and len(head_states) >= 2 else 0.0)
                    ]

                    world_coords = self.transformer.pixel_to_world(
                        pixel_x, pixel_y, depth_value,
                        self.camera_intrinsics[camera_name],
                        joint_angles
                    )
                    world_x, world_y, world_z = world_coords
                except Exception as world_e:
                    print(f"世界坐标转换失败: {world_e}")
                    world_x, world_y, world_z = None, None, None

                status_msg = (f"相机: {camera_name}, 像素: ({pixel_x}, {pixel_y}), "
                             f"深度: {depth_value:.1f}mm, 3D:({x:.3f}, {y:.3f}, {z:.3f})m, "
                             f"世界:({f'{world_x:.3f}' if world_x is not None else 'N/A'}, "
                             f"{f'{world_y:.3f}' if world_y is not None else 'N/A'}, "
                             f"{f'{world_z:.3f}' if world_z is not None else 'N/A'})m")
                self.show_status(status_msg, "info")
                print(f"点击了 {camera_name} 图像，像素: ({pixel_x}, {pixel_y}), 世界坐标: {world_coords}")
                # 存储点击坐标用于后续处理
                self.click_coordinates.append({
                    'camera': camera_name,
                    'pixel': (pixel_x, pixel_y),
                    'depth': depth_value,
                    '3d': coordinates_3d,
                    'world_3d': world_coords if 'world_coords' in locals() else None,
                    'timestamp': time.time()
                })
            else:
                self.show_status("3D坐标转换失败", "error")
                self.show_status(f"处理点击事件失败: {e}", "error")

        except Exception as e:
            print(f"处理图像点击事件失败: {e}")
            self.show_status(f"处理点击事件失败: {e}", "error")

    def transform_to_robot_base(self, camera_coords, camera_name):
        """将相机坐标系下的3D坐标转换到机器人基座坐标系"""
        try:
            # 这里需要实现相机到机器人基座的外参变换
            # 由于需要相机的外参矩阵，这里先返回原始坐标并给出提示
            print(f"需要将 {camera_name} 的相机坐标转换到机器人基座坐标系")
            print("这需要相机的外参矩阵（相机到机器人基座的变换矩阵）")
            return camera_coords
        except Exception as e:
            print(f"坐标系转换失败: {e}")
            return camera_coords

    def update_coordinate_display(self):
        """更新坐标显示"""
        try:
            if hasattr(self, 'click_coordinates') and self.click_coordinates:
                # 显示最近的几条记录
                recent_records = self.click_coordinates[-3:]  # 最近3条
                display_text = "最近获取的3D坐标:\n"

                for i, record in enumerate(reversed(recent_records)):
                    try:
                        camera = record.get('camera', 'unknown')
                        pixel = record.get('pixel', (0, 0))
                        depth = record.get('depth', 0)
                        coords = record.get('3d', [0, 0, 0])

                        # 安全地获取坐标值
                        x = coords[0] if coords and len(coords) > 0 else 0
                        y = coords[1] if coords and len(coords) > 1 else 0
                        z = coords[2] if coords and len(coords) > 2 else 0

                        display_text += f"{i+1}. {camera}: ({pixel[0]},{pixel[1]}) "
                        display_text += f"深度:{depth:.0f}mm -> "
                        display_text += f"3D:({x:.3f},{y:.3f},{z:.3f})\n"
                    except Exception as record_e:
                        print(f"处理坐标记录失败: {record_e}")
                        display_text += f"{i+1}. 记录格式错误\n"
                # 显示总记录数
                display_text += f"\n总计: {len(self.click_coordinates)} 条记录"
            else:
                display_text = "暂无坐标记录\n点击图像获取3D坐标"

            if hasattr(self, 'coordinate_display'):
                self.coordinate_display.config(text=display_text)

        except Exception as e:
            print(f"更新坐标显示失败: {e}")
            if hasattr(self, 'coordinate_display'):
                self.coordinate_display.config(text="显示更新失败")

    def clear_coordinate_records(self):
        """清除坐标记录"""
        self.click_coordinates.clear()
        self.show_status("已清除所有坐标记录", "info")

    def export_coordinates_to_file(self):
        """导出坐标记录到文件"""
        try:
            if not self.click_coordinates:
                self.show_status("没有坐标记录可以导出", "warning")
                return

            import json
            from datetime import datetime

            filename = f"coordinates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 准备导出数据
            export_data = []
            for record in self.click_coordinates:
                export_data.append({
                    'camera': record['camera'],
                    'pixel_x': record['pixel'][0],
                    'pixel_y': record['pixel'][1],
                    'depth_mm': record['depth'],
                    'x_m': record['3d'][0],
                    'y_m': record['3d'][1],
                    'z_m': record['3d'][2],
                    'timestamp': record['timestamp']
                })

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.show_status(f"坐标记录已导出到: {filename}", "success")

        except Exception as e:
            self.show_status(f"导出失败: {e}", "error")
