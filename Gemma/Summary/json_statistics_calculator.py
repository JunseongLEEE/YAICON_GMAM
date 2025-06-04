import json
import os
import numpy as np
# import pandas as pd  # 제거
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics
import csv

def extract_numeric_values(data, target_keys=None):
    """JSON 데이터에서 숫자 값들을 추출하는 함수"""
    if target_keys is None:
        target_keys = ['Yes', 'No', 'raw_yes', 'raw_no', 'total_yes_no']
    
    extracted_values = defaultdict(list)
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                # 각 항목의 recognition과 raw_probabilities 값들 추출
                if 'recognition' in value:
                    for rec_key, rec_value in value['recognition'].items():
                        if isinstance(rec_value, (int, float)):
                            extracted_values[f'recognition_{rec_key}'].append(rec_value)
                
                if 'raw_probabilities' in value:
                    for raw_key, raw_value in value['raw_probabilities'].items():
                        if isinstance(raw_value, (int, float)):
                            extracted_values[f'raw_probabilities_{raw_key}'].append(raw_value)
                
                # Summary 폴더의 Pairwise 파일의 confidence 필드들 추출
                if 'recognition_confidence' in value and isinstance(value['recognition_confidence'], (int, float)):
                    extracted_values['recognition_confidence'].append(value['recognition_confidence'])
                
                if 'preference_confidence' in value and isinstance(value['preference_confidence'], (int, float)):
                    extracted_values['preference_confidence'].append(value['preference_confidence'])
                
                # Story 폴더의 Pairwise 파일: first_order와 second_order의 gemma_prob 평균 계산
                if 'details' in value and isinstance(value['details'], dict):
                    details = value['details']
                    if ('first_order' in details and 'second_order' in details and
                        isinstance(details['first_order'], dict) and isinstance(details['second_order'], dict)):
                        
                        first_gemma_prob = details['first_order'].get('gemma_prob')
                        second_gemma_prob = details['second_order'].get('gemma_prob')
                        
                        if (isinstance(first_gemma_prob, (int, float)) and 
                            isinstance(second_gemma_prob, (int, float))):
                            # first_order와 second_order의 gemma_prob 평균 계산
                            avg_gemma_prob = (first_gemma_prob + second_gemma_prob) / 2
                            extracted_values['pairwise_gemma_confidence'].append(avg_gemma_prob)
                
                # 다른 숫자 값들도 재귀적으로 추출
                nested_values = extract_numeric_values(value, target_keys)
                for nested_key, nested_list in nested_values.items():
                    extracted_values[nested_key].extend(nested_list)
            
            elif isinstance(value, (int, float)):
                # confidence, score, probability가 포함된 키들 추출
                if any(target in key.lower() for target in ['confidence', 'score', 'probability']):
                    extracted_values[key].append(value)
                # 특정 키 이름들도 직접 추출
                elif key in ['recognition_confidence', 'preference_confidence', 'average_recognition_confidence', 'average_preference_confidence', 'high_confidence_ratio']:
                    extracted_values[key].append(value)
    
    elif isinstance(data, list):
        for item in data:
            nested_values = extract_numeric_values(item, target_keys)
            for nested_key, nested_list in nested_values.items():
                extracted_values[nested_key].extend(nested_list)
    
    return extracted_values

def calculate_statistics(values):
    """값들의 평균, 표준편차, 최소값, 최대값을 계산"""
    if not values:
        return None
    
    return {
        'count': len(values),
        'mean': np.mean(values),
        'std': np.std(values, ddof=1) if len(values) > 1 else 0,
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values)
    }

def analyze_json_files(base_path):
    """JSON 파일들을 분석하여 통계를 계산하는 메인 함수"""
    
    results = {}
    folders_to_check = ['Summary', 'Story']
    
    for main_folder in folders_to_check:
        folder_path = Path(base_path) / main_folder
        if not folder_path.exists():
            print(f"❌ {main_folder} 폴더가 존재하지 않습니다.")
            continue
        
        results[main_folder] = {}
        print(f"\n🔍 {main_folder} 폴더 분석 중...")
        
        # 하위 폴더들 순회
        for subfolder in folder_path.iterdir():
            if subfolder.is_dir():
                subfolder_name = subfolder.name
                results[main_folder][subfolder_name] = {}
                print(f"  📁 {subfolder_name}/ 분석 중...")
                
                # JSON 파일들 확인
                json_files = list(subfolder.glob("*.json"))
                if not json_files:
                    print(f"    ❌ JSON 파일이 없습니다.")
                    continue
                
                for json_file in json_files:
                    file_name = json_file.name
                    print(f"    📄 {file_name} 처리 중...")
                    
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 숫자 값들 추출
                        extracted_values = extract_numeric_values(data)
                        
                        # 각 키에 대한 통계 계산
                        file_stats = {}
                        for key, values in extracted_values.items():
                            if values:  # 값이 있는 경우만
                                stats = calculate_statistics(values)
                                if stats:
                                    file_stats[key] = stats
                        
                        if file_stats:
                            results[main_folder][subfolder_name][file_name] = file_stats
                        
                    except json.JSONDecodeError as e:
                        print(f"    ❌ JSON 파싱 오류 ({file_name}): {e}")
                    except Exception as e:
                        print(f"    ❌ 파일 읽기 오류 ({file_name}): {e}")
        
        # 메인 폴더의 JSON 파일들도 확인
        main_json_files = list(folder_path.glob("*.json"))
        if main_json_files:
            results[main_folder]['main_folder'] = {}
            for json_file in main_json_files:
                file_name = json_file.name
                print(f"  📄 {file_name} (메인 폴더) 처리 중...")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    extracted_values = extract_numeric_values(data)
                    
                    file_stats = {}
                    for key, values in extracted_values.items():
                        if values:
                            stats = calculate_statistics(values)
                            if stats:
                                file_stats[key] = stats
                    
                    if file_stats:
                        results[main_folder]['main_folder'][file_name] = file_stats
                
                except Exception as e:
                    print(f"    ❌ 오류 ({file_name}): {e}")
    
    return results

def save_results_to_files(results):
    """결과를 여러 형태로 저장하는 함수"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 상세 결과를 JSON으로 저장
    detailed_filename = f"detailed_statistics_{timestamp}.json"
    with open(detailed_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 상세 결과가 '{detailed_filename}'에 저장되었습니다.")
    
    # 2. 요약 결과를 텍스트로 저장
    summary_filename = f"statistics_summary_{timestamp}.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(f"JSON 파일 통계 분석 요약\n")
        f.write(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for main_folder, subfolders in results.items():
            f.write(f"📁 {main_folder} 폴더\n")
            f.write("-" * 50 + "\n")
            
            for subfolder_name, files in subfolders.items():
                f.write(f"\n  📂 {subfolder_name}/\n")
                
                for file_name, stats in files.items():
                    f.write(f"\n    📄 {file_name}\n")
                    
                    for metric_name, metric_stats in stats.items():
                        f.write(f"      🔢 {metric_name}:\n")
                        f.write(f"        개수: {metric_stats['count']}\n")
                        f.write(f"        평균: {metric_stats['mean']:.6f}\n")
                        f.write(f"        표준편차: {metric_stats['std']:.6f}\n")
                        f.write(f"        최소값: {metric_stats['min']:.6f}\n")
                        f.write(f"        최대값: {metric_stats['max']:.6f}\n")
                        f.write(f"        중앙값: {metric_stats['median']:.6f}\n")
                        f.write("\n")
            f.write("\n")
    
    print(f"✅ 요약 결과가 '{summary_filename}'에 저장되었습니다.")
    
    # 3. CSV 형태로도 저장 (pandas 대신 csv 모듈 사용)
    csv_data = []
    for main_folder, subfolders in results.items():
        for subfolder_name, files in subfolders.items():
            for file_name, stats in files.items():
                for metric_name, metric_stats in stats.items():
                    csv_data.append({
                        'main_folder': main_folder,
                        'subfolder': subfolder_name,
                        'file_name': file_name,
                        'metric': metric_name,
                        'count': metric_stats['count'],
                        'mean': metric_stats['mean'],
                        'std': metric_stats['std'],
                        'min': metric_stats['min'],
                        'max': metric_stats['max'],
                        'median': metric_stats['median']
                    })
    
    if csv_data:
        csv_filename = f"statistics_data_{timestamp}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['main_folder', 'subfolder', 'file_name', 'metric', 'count', 'mean', 'std', 'min', 'max', 'median']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"✅ CSV 데이터가 '{csv_filename}'에 저장되었습니다.")

def print_summary_statistics(results):
    """주요 통계를 콘솔에 출력하는 함수"""
    print("\n" + "="*80)
    print("📊 주요 통계 요약")
    print("="*80)
    
    for main_folder, subfolders in results.items():
        print(f"\n🔍 {main_folder} 폴더")
        print("-" * 50)
        
        # Recognition Yes/No 통계 수집
        all_yes_values = []
        all_no_values = []
        
        # Pairwise confidence 통계 수집
        recognition_confidence_values = []
        preference_confidence_values = []
        pairwise_gemma_confidence_values = []  # Story 폴더용
        
        for subfolder_name, files in subfolders.items():
            for file_name, stats in files.items():
                if 'recognition_Yes' in stats:
                    # 이 파일의 모든 Yes 값들을 가져와서 평균 계산
                    yes_mean = stats['recognition_Yes']['mean']
                    all_yes_values.append(yes_mean)
                
                if 'recognition_No' in stats:
                    no_mean = stats['recognition_No']['mean']
                    all_no_values.append(no_mean)
                
                # Summary 폴더의 Pairwise confidence 값들 수집
                if 'recognition_confidence' in stats:
                    recognition_confidence_values.append(stats['recognition_confidence']['mean'])
                
                if 'preference_confidence' in stats:
                    preference_confidence_values.append(stats['preference_confidence']['mean'])
                
                # Story 폴더의 Pairwise confidence 값들 수집
                if 'pairwise_gemma_confidence' in stats:
                    pairwise_gemma_confidence_values.append(stats['pairwise_gemma_confidence']['mean'])
        
        if all_yes_values:
            print(f"  🎯 Recognition 'Yes' 점수:")
            print(f"    전체 평균: {np.mean(all_yes_values):.4f}")
            print(f"    표준편차: {np.std(all_yes_values, ddof=1):.4f}")
            print(f"    범위: {np.min(all_yes_values):.4f} ~ {np.max(all_yes_values):.4f}")
        
        if all_no_values:
            print(f"  🎯 Recognition 'No' 점수:")
            print(f"    전체 평균: {np.mean(all_no_values):.4f}")
            print(f"    표준편차: {np.std(all_no_values, ddof=1):.4f}")
            print(f"    범위: {np.min(all_no_values):.4f} ~ {np.max(all_no_values):.4f}")
        
        if recognition_confidence_values:
            print(f"  🎯 Pairwise Recognition Confidence:")
            print(f"    전체 평균: {np.mean(recognition_confidence_values):.4f}")
            print(f"    표준편차: {np.std(recognition_confidence_values, ddof=1):.4f}")
            print(f"    범위: {np.min(recognition_confidence_values):.4f} ~ {np.max(recognition_confidence_values):.4f}")
        
        if preference_confidence_values:
            print(f"  🎯 Pairwise Preference Confidence:")
            print(f"    전체 평균: {np.mean(preference_confidence_values):.4f}")
            print(f"    표준편차: {np.std(preference_confidence_values, ddof=1):.4f}")
            print(f"    범위: {np.min(preference_confidence_values):.4f} ~ {np.max(preference_confidence_values):.4f}")
        
        if pairwise_gemma_confidence_values:
            print(f"  🎯 Pairwise Gemma Confidence (Story):")
            print(f"    전체 평균: {np.mean(pairwise_gemma_confidence_values):.4f}")
            print(f"    표준편차: {np.std(pairwise_gemma_confidence_values, ddof=1):.4f}")
            print(f"    범위: {np.min(pairwise_gemma_confidence_values):.4f} ~ {np.max(pairwise_gemma_confidence_values):.4f}")

if __name__ == "__main__":
    print("🚀 JSON 파일 통계 분석 시작")
    print("=" * 50)
    
    # 현재 디렉토리에서 실행
    base_path = "."
    
    try:
        # 분석 실행
        results = analyze_json_files(base_path)
        
        if results:
            # 결과 출력
            print_summary_statistics(results)
            
            # 파일로 저장
            save_results_to_files(results)
            
            print(f"\n✅ 분석 완료!")
        else:
            print("❌ 분석할 데이터가 없습니다.")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc() 