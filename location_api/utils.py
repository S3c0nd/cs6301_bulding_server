# location_api/utils.py

import google.generativeai as genai
from django.conf import settings
from pathlib import Path


def upload_pdf_to_gemini(pdf_path):
    """
    将PDF文件上传到Gemini（用于多模态处理）
    """
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        file = genai.upload_file(pdf_path)
        return file
    except Exception as e:
        print(f"Error uploading PDF: {str(e)}")
        return None

def query_gemini_location(direction, gps, map_context):
    """
    使用Gemini API查询当前位置
    
    Args:
        direction: 方向信息（如"北"、"East"等）
        gps: GPS坐标字典 {"latitude": xx, "longitude": xx}
        map_context: 地图文本内容或文件引用
    """
    # 配置 Gemini API
    genai.configure(api_key=settings.GEMINI_API_KEY)
    
    # 构建 prompt
    prompt = f"""
假设箭头是一个人, 他面朝的建筑的名字是什么? 
Only based on this image
{map_context}

Think it step by step, first describe the image, then find out where is the arraw, then find out which buiding it is point to, then give your answer, which is the buiding's name in this format: /*** BUIDING_NAME ***/

"""
    
    try:
        # 选择模型
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # 如果有上传的PDF文件
        if not isinstance(map_context, str):
            response = model.generate_content([prompt, map_context])
        else:
            response = model.generate_content(prompt)
        
        return {
            'success': True,
            'location': response.text,
            'model': 'gemini-2.5-pro'
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }