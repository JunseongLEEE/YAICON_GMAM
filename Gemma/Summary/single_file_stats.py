import json
import numpy as np
from datetime import datetime
from collections import defaultdict

def analyze_single_file(file_path):
    """단일 JSON 파일의 통계를 계산하는 함수"""
    
    print(f"🔍 분석 파일: {file_path}")
    print("=" * 60)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 통계 수집을 위한 리스트들
        yes_values = []
        no_values = []
        raw_yes_values = []
        raw_no_values = []
        raw_total_values = []
        
        # results 안의 데이터 순회
        if 'results' in data:
            results = data['results']
            
            for item_id, item_data in results.items():
                # recognition 값들 수집
                if 'recognition' in item_data:
                    recognition = item_data['recognition']
                    if 'Yes' in recognition:
                        yes_values.append(recognition['Yes'])
                    if 'No' in recognition:
                        no_values.append(recognition['No'])
                
                # raw_probabilities 값들 수집
                if 'raw_probabilities' in item_data:
                    raw_probs = item_data['raw_probabilities']
                    if 'yes' in raw_probs:
                        raw_yes_values.append(raw_probs['yes'])
                    if 'no' in raw_probs:
                        raw_no_values.append(raw_probs['no'])
                    if 'total' in raw_probs:
                        raw_total_values.append(raw_probs['total'])
        
        # 통계 계산 및 출력
        print(f"📊 분석 결과 (총 {len(yes_values)}개 항목)")
        print("-" * 40)
        
        if yes_values:
            print(f"🎯 Recognition 'Yes' 점수:")
            print(f"  개수: {len(yes_values)}")
            print(f"  평균: {np.mean(yes_values):.6f}")
            print(f"  표준편차: {np.std(yes_values, ddof=1):.6f}")
            print(f"  최소값: {np.min(yes_values):.6f}")
            print(f"  최대값: {np.max(yes_values):.6f}")
            print(f"  중앙값: {np.median(yes_values):.6f}")
            print()
        
        if no_values:
            print(f"🎯 Recognition 'No' 점수:")
            print(f"  개수: {len(no_values)}")
            print(f"  평균: {np.mean(no_values):.6f}")
            print(f"  표준편차: {np.std(no_values, ddof=1):.6f}")
            print(f"  최소값: {np.min(no_values):.6f}")
            print(f"  최대값: {np.max(no_values):.6f}")
            print(f"  중앙값: {np.median(no_values):.6f}")
            print()
        
        if raw_yes_values:
            print(f"🎯 Raw Probabilities 'Yes':")
            print(f"  개수: {len(raw_yes_values)}")
            print(f"  평균: {np.mean(raw_yes_values):.8f}")
            print(f"  표준편차: {np.std(raw_yes_values, ddof=1):.8f}")
            print(f"  최소값: {np.min(raw_yes_values):.8f}")
            print(f"  최대값: {np.max(raw_yes_values):.8f}")
            print(f"  중앙값: {np.median(raw_yes_values):.8f}")
            print()
        
        if raw_no_values:
            print(f"🎯 Raw Probabilities 'No':")
            print(f"  개수: {len(raw_no_values)}")
            print(f"  평균: {np.mean(raw_no_values):.8f}")
            print(f"  표준편차: {np.std(raw_no_values, ddof=1):.8f}")
            print(f"  최소값: {np.min(raw_no_values):.8f}")
            print(f"  최대값: {np.max(raw_no_values):.8f}")
            print(f"  중앙값: {np.median(raw_no_values):.8f}")
            print()
        
        if raw_total_values:
            print(f"🎯 Raw Probabilities 'Total':")
            print(f"  개수: {len(raw_total_values)}")
            print(f"  평균: {np.mean(raw_total_values):.8f}")
            print(f"  표준편차: {np.std(raw_total_values, ddof=1):.8f}")
            print(f"  최소값: {np.min(raw_total_values):.8f}")
            print(f"  최대값: {np.max(raw_total_values):.8f}")
            print(f"  중앙값: {np.median(raw_total_values):.8f}")
            print()
        
        # 결과를 파일로 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"gemini_recognition_stats_{timestamp}.txt"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"Gemini Summary Individual Recognition 통계 분석\n")
            f.write(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"파일: {file_path}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"총 분석 항목 수: {len(yes_values)}\n\n")
            
            if yes_values:
                f.write("Recognition 'Yes' 점수:\n")
                f.write(f"  평균: {np.mean(yes_values):.6f}\n")
                f.write(f"  표준편차: {np.std(yes_values, ddof=1):.6f}\n")
                f.write(f"  최소값: {np.min(yes_values):.6f}\n")
                f.write(f"  최대값: {np.max(yes_values):.6f}\n")
                f.write(f"  중앙값: {np.median(yes_values):.6f}\n\n")
            
            if no_values:
                f.write("Recognition 'No' 점수:\n")
                f.write(f"  평균: {np.mean(no_values):.6f}\n")
                f.write(f"  표준편차: {np.std(no_values, ddof=1):.6f}\n")
                f.write(f"  최소값: {np.min(no_values):.6f}\n")
                f.write(f"  최대값: {np.max(no_values):.6f}\n")
                f.write(f"  중앙값: {np.median(no_values):.6f}\n\n")
        
        print(f"✅ 통계 결과가 '{output_filename}' 파일로 저장되었습니다.")
        return True
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    file_path = "/Users/ijunseong/Desktop/YAI/YAICON2025/Summary/Individual_Recognition_result/gemini_summary_individual_recognition.json"
    analyze_single_file(file_path) 