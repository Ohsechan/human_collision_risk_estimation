import os
from ament_index_python.packages import get_package_prefix

def find_package_path(package_name):
    package_path = get_package_prefix(package_name)
    package_path = os.path.dirname(package_path)
    package_path = os.path.dirname(package_path)
    package_path = os.path.join(package_path, "src", package_name)
    return package_path

# 예제 사용
package_name = 'image_processing'
package_path = find_package_path('image_processing')

if package_path:
    print(f"패키지 '{package_name}'의 경로: {package_path}")
else:
    print(f"패키지 '{package_name}'의 경로를 찾을 수 없습니다.")