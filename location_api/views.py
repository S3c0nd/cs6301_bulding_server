# location_api/views.py

import json
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from .utils import upload_pdf_to_gemini, query_gemini_location
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import random
from io import BytesIO
import re

import math

def extract_between_markers(text):
    """
    提取 /*** 和 ***/ 之间的内容
    
    Args:
        text: 输入字符串
    
    Returns:
        str: 提取的内容，如果没有匹配则返回 None
    """
    pattern = r'/\*\*\*(.*?)\*\*\*/'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()  # .strip() 去除首尾空白
    return None



class MapMarker:
    def __init__(self, image_path, corners):
        """
        初始化地图标记器
        
        Args:
            image_path: 地图图片路径
            corners: 地图四个角的经纬度，格式：
                {
                    'top_left': (lat, lon),
                    'top_right': (lat, lon),
                    'bottom_left': (lat, lon),
                    'bottom_right': (lat, lon)
                }
        """
        self.image = Image.open(image_path)
        self.width, self.height = self.image.size
        self.corners = corners
        
    def latlon_to_pixel(self, lat, lon):
        """
        将经纬度转换为图片像素坐标
        
        使用简单的线性插值（适用于小范围地图）
        """
        # 获取四个角的坐标
        tl_lat, tl_lon = self.corners['top_left']
        tr_lat, tr_lon = self.corners['top_right']
        bl_lat, bl_lon = self.corners['bottom_left']
        br_lat, br_lon = self.corners['bottom_right']
        
        # 计算纬度范围
        lat_min = min(tl_lat, tr_lat, bl_lat, br_lat)
        lat_max = max(tl_lat, tr_lat, bl_lat, br_lat)
        
        # 计算经度范围
        lon_min = min(tl_lon, tr_lon, bl_lon, br_lon)
        lon_max = max(tl_lon, tr_lon, bl_lon, br_lon)
        
        # 线性插值到像素坐标
        # 注意：纬度是反向的（北纬越大，y 坐标越小）
        x = (lon - lon_min) / (lon_max - lon_min) * self.width
        y = (lat_max - lat) / (lat_max - lat_min) * self.height
        
        return int(x), int(y)
    
    def draw_arrow(self, lat, lon, direction, color='red', size=50):
        """
        在指定经纬度位置画箭头
        
        Args:
            lat: 纬度
            lon: 经度
            direction: 方向角度（0-360度，0表示正北）
            color: 箭头颜色
            size: 箭头大小
        """
        # 转换为像素坐标
        x, y = self.latlon_to_pixel(lat, lon)
        
        # 创建可绘制对象
        draw = ImageDraw.Draw(self.image)
        
        # 计算箭头的三个顶点
        # 将地理方向转换为图像坐标系（顺时针旋转90度）
        angle_rad = math.radians(direction - 90)
        
        # 箭头尖端
        tip_x = x + size * 0.3 * math.cos(angle_rad)
        tip_y = y + size * 0.3 * math.sin(angle_rad)
        
        # 箭头左翼
        left_angle = angle_rad + math.radians(150)
        left_x = x + (size * 0.5) * math.cos(left_angle)
        left_y = y + (size * 0.5) * math.sin(left_angle)
        
        # 箭头右翼
        right_angle = angle_rad - math.radians(150)
        right_x = x + (size * 0.5) * math.cos(right_angle)
        right_y = y + (size * 0.5) * math.sin(right_angle)
        
        # 绘制箭头（实心三角形）
        draw.polygon(
            [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)],
            fill=color,
            outline='black'
        )
        
        # 绘制箭头杆
        shaft_length = size * 2
        shaft_end_x = x - shaft_length * math.cos(angle_rad)
        shaft_end_y = y - shaft_length * math.sin(angle_rad)
        draw.line(
            [(x, y), (shaft_end_x, shaft_end_y)],
            fill=color,
            width=5
        )
        
        # 绘制中心点
        # draw.ellipse(
        #     [(x-5, y-5), (x+5, y+5)],
        #     fill='blue',
        #     outline='white'
        # )
        
        return self.image
    
    def save(self, output_path):
        """保存结果图片"""
        self.image.save(output_path)
    
    def show(self):
        """显示图片"""
        self.image.show()

def random_value_generator():
    """
    
    Yields:
        tuple: (value1, value2)
    """
    while True:
        value1 = random.uniform(0, 0.2)
        value2 = random.uniform(0.8, 1)
        yield value1, value2

gen = random_value_generator()

def annotate_building(image_path, label, output_path):
    """
    一键标注建筑物（最简版本）
    
    Args:
        image_path: 图片路径
        label: 建筑名称
        output_path: 输出路径（可选）
    
    Returns:
        str: 输出文件路径
    """
    # 自动生成输出路径
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_labeled{ext}"
    
    # 加载模型
    yolo = YOLO('yolov8n.pt')
    
    # 检测最大对象
    results = yolo(image_path, conf=0.2, verbose=False)
    
    max_area = 0
    best_box = None
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = [int(x1), int(y1), int(x2), int(y2)]
    
    # 如果没检测到，使用全图
    if best_box is None:
        img = Image.open(image_path)
        w, h = img.size
        v1, v2 = next(gen)
        best_box = [int(w*v1), int(h*v1), int(w*v2), int(h*v2)]
    
    # 加载图片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # 加载字体
    font = ImageFont.load_default(120)
    
    # 绘制边框
    draw.rectangle(best_box, outline='red', width=4)
    
    # 计算文字位置
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x1, y1 = best_box[0], best_box[1]
    
    # 文字背景
    bg_y1 = y1 - text_h - 20 if y1 - text_h - 20 > 0 else best_box[3]
    draw.rectangle(
        [x1, bg_y1, x1 + text_w + 20, bg_y1 + text_h + 20],
        fill='red'
    )
    
    # 绘制文字
    draw.text((x1 + 10, bg_y1 + 10), label, fill='white', font=font)
    

    buffer = BytesIO()
    # 保存
    img.save(buffer, format='JPEG')
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    annotated_data_uri = f"{base64_str}"
    return annotated_data_uri

@csrf_exempt
@require_http_methods(["POST"])
def identify_location(request):
    """
    API endpoint: 接收 GPS 和方向数据，返回位置识别结果
    
    Request Body (JSON):
    {
        "direction": "北",
        "gps": {
            "latitude": 39.9042,
            "longitude": 116.4074
        }
    }
    """
    try:
        # 解析 JSON 数据
        data = json.loads(request.body)
        
        # 验证必需字段
        if 'direction' not in data or 'gps' not in data:
            return JsonResponse({
                'success': False,
                'error': 'Missing required fields: direction and gps'
            }, status=400)
        
        direction = data['direction']
        gps = data['gps']
        base64_string = data['image_base64']

        corners = {
            'top_left': (32.99563626626291, -96.75615546459603),      # 左上角
            'top_right': (32.99562635860259, -96.74429935194813),     # 右上角
            'bottom_left': (32.9828203491122, -96.75615546459603),   # 左下角
            'bottom_right': (32.9828203491122, -96.74429935194813)   # 右下角
        }
        
        # 创建标记器
        marker = MapMarker('map_png.png', corners)
        

        marker.draw_arrow(
            lat=gps['latitude'],
            lon=gps['longitude'],
            direction=direction,
            color='purple',
            size=20
        )
        
        marker.save('output_map.png')
        
        # 验证 GPS 数据
        if 'latitude' not in gps or 'longitude' not in gps:
            return JsonResponse({
                'success': False,
                'error': 'GPS must contain latitude and longitude'
            }, status=400)
        

        
        # 读取地图文件
        map_pdf_path = settings.MAP_PDF_PATH
        
        map_file = upload_pdf_to_gemini(map_pdf_path)

        # map_context = map_file if map_file else map_text
        
        # 使用文本方式
        map_context = map_file
        
        # 查询 Gemini
        result = query_gemini_location(direction, gps, map_context)
        
        building_name = extract_between_markers(result['location'])

        image_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(image_data))
        rotated = img.rotate(-90, expand=True)
        rotated.save("./building.png", 'PNG')
        
        img_base64 = annotate_building("./building.png", building_name, '')
        # print(img_base64)


        if result['success']:
            return JsonResponse({
                'success': True,
                'direction': direction,
                'gps': gps,
                'location_info': building_name,
                'labeled_image': img_base64,
                'model': result['model']
            }, status=200)
        else:
            return JsonResponse({
                'success': False,
                'error': result['error']
            }, status=500)
    
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON format'
        }, status=400)
    
    # except Exception as e:
    #     return JsonResponse({
    #         'success': False,
    #         'error': str(e)
    #     }, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """健康检查端点"""
    return JsonResponse({
        'status': 'healthy',
        'service': 'location-api'
    })