import json
import numpy as np
from datetime import datetime

def create_recognition_summary_table():
    """Individual Recognition 결과를 표 형태로 정리하는 함수"""
    
    print("📊 Individual Recognition (Yes/No 비율) 종합 분석")
    print("=" * 80)
    
    # 기존 분석 결과 (최신 CSV에서 추출한 데이터)
    models_data = {
        'Qwen': {
            'yes_mean': 0.902643,
            'yes_std': 0.061225,
            'no_mean': 0.097357,
            'no_std': 0.061225
        },
        'Xsum': {
            'yes_mean': 0.900243,
            'yes_std': 0.045930,
            'no_mean': 0.099757,
            'no_std': 0.045930
        },
        'Deepseek': {
            'yes_mean': 0.896431,
            'yes_std': 0.048857,
            'no_mean': 0.103569,
            'no_std': 0.048857
        },
        'Llama': {
            'yes_mean': 0.895771,
            'yes_std': 0.043157,
            'no_mean': 0.104229,
            'no_std': 0.043157
        },
        'Gemini': {  # 새로 추가된 결과
            'yes_mean': 0.908755,
            'yes_std': 0.041712,
            'no_mean': 0.091245,
            'no_std': 0.041712
        }
    }
    
    # 표 헤더 출력
    print(f"{'모델':<12} {'Yes 평균':<20} {'Yes 표준편차':<12} {'No 평균':<20} {'No 표준편차':<12}")
    print("=" * 80)
    
    # 각 모델별 결과 출력
    for model, data in models_data.items():
        yes_percent = data['yes_mean'] * 100
        no_percent = data['no_mean'] * 100
        
        print(f"{model:<12} {data['yes_mean']:.6f} ({yes_percent:.2f}%){'':<2} {data['yes_std']:.6f}{'':<4} "
              f"{data['no_mean']:.6f} ({no_percent:.2f}%){'':<3} {data['no_std']:.6f}")
    
    print("=" * 80)
    
    # 순위 정리
    print("\n🏆 모델 순위 (Yes 인식률 기준):")
    sorted_models = sorted(models_data.items(), key=lambda x: x[1]['yes_mean'], reverse=True)
    
    for i, (model, data) in enumerate(sorted_models, 1):
        yes_percent = data['yes_mean'] * 100
        print(f"{i}. {model}: {yes_percent:.2f}%")
    
    # 마크다운 테이블 형태로도 출력
    print("\n📋 마크다운 표 형태:")
    print("| 모델 | Yes 평균 | Yes 표준편차 | No 평균 | No 표준편차 |")
    print("|------|----------|-------------|---------|------------|")
    
    for model, data in sorted_models:
        yes_percent = data['yes_mean'] * 100
        no_percent = data['no_mean'] * 100
        print(f"| {model} | {data['yes_mean']:.6f} ({yes_percent:.2f}%) | {data['yes_std']:.6f} | "
              f"{data['no_mean']:.6f} ({no_percent:.2f}%) | {data['no_std']:.6f} |")
    
    # 결과를 파일로 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"recognition_summary_table_{timestamp}.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("Individual Recognition (Yes/No 비율) 종합 분석\n")
        f.write("=" * 80 + "\n")
        f.write(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"{'모델':<12} {'Yes 평균':<20} {'Yes 표준편차':<12} {'No 평균':<20} {'No 표준편차':<12}\n")
        f.write("=" * 80 + "\n")
        
        for model, data in models_data.items():
            yes_percent = data['yes_mean'] * 100
            no_percent = data['no_mean'] * 100
            
            f.write(f"{model:<12} {data['yes_mean']:.6f} ({yes_percent:.2f}%)     {data['yes_std']:.6f}    "
                   f"{data['no_mean']:.6f} ({no_percent:.2f}%)     {data['no_std']:.6f}\n")
        
        f.write("=" * 80 + "\n\n")
        
        f.write("모델 순위 (Yes 인식률 기준):\n")
        for i, (model, data) in enumerate(sorted_models, 1):
            yes_percent = data['yes_mean'] * 100
            f.write(f"{i}. {model}: {yes_percent:.2f}%\n")
        
        f.write("\n마크다운 표 형태:\n")
        f.write("| 모델 | Yes 평균 | Yes 표준편차 | No 평균 | No 표준편차 |\n")
        f.write("|------|----------|-------------|---------|------------|\n")
        
        for model, data in sorted_models:
            yes_percent = data['yes_mean'] * 100
            no_percent = data['no_mean'] * 100
            f.write(f"| {model} | {data['yes_mean']:.6f} ({yes_percent:.2f}%) | {data['yes_std']:.6f} | "
                   f"{data['no_mean']:.6f} ({no_percent:.2f}%) | {data['no_std']:.6f} |\n")
    
    print(f"\n✅ 종합 분석 결과가 '{output_filename}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    create_recognition_summary_table() 