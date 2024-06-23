import numpy as np

# 파일에서 데이터 읽기
with open('processTime.txt', 'r') as file:
    data = file.readlines()

# 데이터 정리 및 정수형으로 변환
data = [int(line.strip()) for line in data]

# 데이터 배열을 numpy 배열로 변환
data_array = np.array(data)

# 통계 정보 계산
mean = np.mean(data_array)
median = np.median(data_array)
std_dev = np.std(data_array)
min_value = np.min(data_array)
max_value = np.max(data_array)

# 결과 출력
print(f'Length: {len(data_array)}')
print(f'Mean: {mean}')
print(f'Median: {median}')
print(f'Standard Deviation: {std_dev}')
print(f'Min: {min_value}')
print(f'Max: {max_value}')
