# location_api/views.py

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from .utils import upload_pdf_to_gemini, query_gemini_location

from PIL import Image, ImageDraw
import math

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
        
        # 方法1：提取PDF文本
        
        # 方法2（可选）：上传PDF文件到Gemini（用于图像识别）
        map_file = upload_pdf_to_gemini(map_pdf_path)

        # map_context = map_file if map_file else map_text
        
        # 使用文本方式
        map_context = map_file
        
        # 查询 Gemini
        result = query_gemini_location(direction, gps, map_context)
        
        if result['success']:
            return JsonResponse({
                'success': True,
                'direction': direction,
                'gps': gps,
                'location_info': result['location'],
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