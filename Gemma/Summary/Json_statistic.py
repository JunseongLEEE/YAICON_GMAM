import json
import os
from pathlib import Path
from datetime import datetime

def analyze_and_save_json_structure(base_path):
    """JSON 파일들의 구조를 분석하고 TXT 파일로 저장하는 함수"""
    
    # 결과를 저장할 문자열
    output_content = []
    output_content.append(f"JSON 파일 구조 분석 결과")
    output_content.append(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_content.append("=" * 80)
    
    folders_to_check = ['Summary', 'Story']
    
    for main_folder in folders_to_check:
        folder_path = Path(base_path) / main_folder
        if not folder_path.exists():
            output_content.append(f"\n❌ {main_folder} 폴더가 존재하지 않습니다.")
            continue
            
        output_content.append(f"\n🔍 {main_folder} 폴더 분석")
        output_content.append("=" * 50)
        
        # 하위 폴더들 순회
        for subfolder in folder_path.iterdir():
            if subfolder.is_dir():
                output_content.append(f"\n📁 {subfolder.name}/")
                
                # JSON 파일들 확인
                json_files = list(subfolder.glob("*.json"))
                if not json_files:
                    output_content.append("  ❌ JSON 파일이 없습니다.")
                    continue
                
                for json_file in json_files:
                    output_content.append(f"\n  📄 {json_file.name}")
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 데이터 타입 및 구조 분석
                        if isinstance(data, dict):
                            output_content.append(f"    📊 타입: Dictionary (키 개수: {len(data)})")
                            
                            # 모든 키 출력
                            keys = list(data.keys())
                            output_content.append(f"    🔑 키 목록: {keys}")
                            
                            # 첫 번째 키의 구조를 자세히 분석
                            if keys:
                                first_key = keys[0]
                                first_value = data[first_key]
                                output_content.append(f"    📝 첫 번째 키 ({first_key})의 구조:")
                                
                                def analyze_nested_structure(obj, indent="        "):
                                    if isinstance(obj, dict):
                                        for sub_key, sub_value in obj.items():
                                            output_content.append(f"{indent}{sub_key}: {type(sub_value).__name__}")
                                            if isinstance(sub_value, dict):
                                                analyze_nested_structure(sub_value, indent + "    ")
                                            elif isinstance(sub_value, list) and sub_value:
                                                output_content.append(f"{indent}    리스트 길이: {len(sub_value)}, 첫 요소: {type(sub_value[0]).__name__}")
                                            elif isinstance(sub_value, (int, float)):
                                                output_content.append(f"{indent}    값: {sub_value}")
                                            elif isinstance(sub_value, str):
                                                preview = sub_value[:50] + "..." if len(sub_value) > 50 else sub_value
                                                output_content.append(f"{indent}    값: {preview}")
                                    elif isinstance(obj, list) and obj:
                                        output_content.append(f"{indent}리스트 길이: {len(obj)}")
                                        if isinstance(obj[0], dict):
                                            output_content.append(f"{indent}첫 요소 구조:")
                                            analyze_nested_structure(obj[0], indent + "    ")
                                
                                analyze_nested_structure(first_value)
                        
                        elif isinstance(data, list):
                            output_content.append(f"    📊 타입: List (길이: {len(data)})")
                            if data:
                                first_item = data[0]
                                output_content.append(f"    📝 첫 요소 타입: {type(first_item).__name__}")
                                if isinstance(first_item, dict):
                                    output_content.append(f"    🔑 첫 요소 키들: {list(first_item.keys())}")
                                    # 첫 요소의 각 키-값 확인
                                    for key, value in list(first_item.items())[:3]:
                                        output_content.append(f"        {key}: {type(value).__name__} = {value}")
                        
                        else:
                            output_content.append(f"    📊 타입: {type(data).__name__}")
                            output_content.append(f"    📝 값: {data}")
                    
                    except json.JSONDecodeError as e:
                        output_content.append(f"    ❌ JSON 파싱 오류: {e}")
                    except Exception as e:
                        output_content.append(f"    ❌ 파일 읽기 오류: {e}")
    
    # 결과를 TXT 파일로 저장
    output_filename = "json_structure_analysis.txt"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_content))
        
        print(f"✅ 분석 결과가 '{output_filename}' 파일로 저장되었습니다.")
        print(f"📄 총 {len(output_content)} 줄이 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 파일 저장 오류: {e}")
        # 콘솔에 출력
        print('\n'.join(output_content))

# 사용법
if __name__ == "__main__":
    # 현재 YAICON2025 디렉토리에서 실행
    base_path = "."  # 또는 절대 경로 사용
    
    print("🚀 JSON 파일 구조 분석 및 저장 시작")
    analyze_and_save_json_structure(base_path)